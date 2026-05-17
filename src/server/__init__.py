"""
FastAPI Server — production-ready REST API for the Gemma agent.

Endpoints:
  POST /query               — process a user query
  GET  /health              — check model availability
  GET  /tools               — list registered tools
  POST /documents           — add a document to the knowledge base
  POST /upload-document     — upload & index a file (PDF/TXT/MD) with SSE progress
  POST /export-training-data — export interaction logs
  POST /train              — live fine-tuning with SSE progress
  GET  /train/status       — current training status
  POST /eval               — run model evaluation (intent accuracy)
  GET  /eval/results       — stored before/after eval snapshots
  POST /eval/reset         — clear stored eval results
  GET  /gpu                 — real-time GPU statistics
  GET  /energy              — accumulated energy consumption for the session
  GET  /privacy             — zero-exfiltration proof
  POST /models/swap         — hot-swap base / fine-tuned models
  GET  /models/mode         — current model mode
  POST /compare             — side-by-side local vs cloud LLM
  POST /escalate            — HITL cloud escalation (user-approved)
  POST /network-mode        — toggle online/offline (Kill the WiFi)
  POST /routing-mode        — toggle local-only/hybrid routing
  POST /voice/chat          — full voice round-trip (STT → agent → TTS) with SSE
  GET  /voice/audio/{id}    — serve generated TTS audio (WAV)
  POST /voice/synthesize    — standalone text-to-speech
  WS   /ws/stats             — real-time GPU + energy push (WebSocket)

Run:  uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs

Architecture:
  src/server/__init__.py        — App wiring, lifespan, CORS (this file)
  src/server/state.py           — AppState singleton, port configuration
  src/server/models.py          — Pydantic request/response models
  src/server/agent_routes.py    — Core agent: /query, /health, /tools, /documents
  src/server/training_routes.py — /train, /eval SSE endpoints
  src/server/voice_routes.py    — /voice/chat, /voice/audio, /voice/synthesize
  src/server/cloud_routes.py    — /compare, /escalate
  src/server/system_routes.py   — /gpu, /privacy, /network-mode, /models/swap
"""

import os
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.engine.inference.client import SmallLanguageModelClient, SmallLanguageModelRole
from src.engine.inference.config import (
    CORS_ORIGINS, QWEN_PORT, QWEN_MODEL, OPENAI_COMPARE_MODEL,
    OCR_PORT, OCR_URL, OCR_MODEL, OCR_MAX_TOKENS, OCR_TIMEOUT,
    SCENARIO_CONFIG,
)
from src.engine.knowledge.vector_store import VectorStore
from src.engine.tools import create_default_registry
from src.engine.agent import SmallLanguageModelAgentOrchestrator

from src.engine.agent.cloud_orchestrator import CloudOrchestrator

from .state import state, BASE_PORTS
from .agent_routes import router as agent_router
from .training_routes import router as training_router
from .voice_routes import router as voice_router
from .cloud_routes import router as cloud_router
from .system_routes import router as system_router

# Re-export for backward compatibility with tests that access server._state
_state = state


# ---------------------------------------------------------------------------
# Lifespan — initialise once at startup, clean up on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up all components when the server starts."""
    state.client       = SmallLanguageModelClient.create_with_auto_detection()
    _chroma = SCENARIO_CONFIG.chroma_dir
    state.vector_store = VectorStore(persist_dir=_chroma)  # curated KB
    state.vector_store.set_client(state.client)
    state.upload_store = VectorStore(collection_name="uploads", persist_dir=_chroma)  # user uploads
    state.upload_store.set_client(state.client)

    # ChromaDB health check — verify collections are accessible.
    # Catches corruption early (e.g., from external scripts writing to the same DB).
    try:
        kb_count = await state.vector_store.count()
        up_count = await state.upload_store.count()
        print(f"  ChromaDB: KB={kb_count} docs, uploads={up_count} chunks ✓")
    except Exception as exc:
        print(f"  ⚠ ChromaDB health check failed: {exc}")
        print("    Recreating uploads collection...")
        await state.upload_store.clear()
        print("    ✓ uploads collection recreated")
    state.tools = create_default_registry(
        vector_store=state.vector_store,
        upload_store=state.upload_store,
    )

    state.agent = SmallLanguageModelAgentOrchestrator(state.client, state.tools)

    # Verify LogReg model file is present and loadable at startup.
    # Without this check a missing file causes a silent crash on the first request.
    # Trust: joblib.load runs pickle and will execute code in a malicious file.
    # The committed model.joblib is locally produced — never swap in one from
    # an untrusted source. See SECURITY.md and intent_classifier_logreg._load_model.
    from pathlib import Path as _Path
    import joblib as _joblib
    _logreg_path = _Path("models/intent-logreg/model.joblib")
    if _logreg_path.exists():
        try:
            _joblib.load(_logreg_path)
            print(f"✓ LogReg model loaded ({_logreg_path})")
        except Exception as _exc:
            print(f"⚠ LogReg model file corrupt or unreadable ({_logreg_path}): {_exc}")
    else:
        print(f"⚠ LogReg model not found ({_logreg_path}) — generative fallback active")

    # Determinism check: llama-server must run with --parallel 1 (PARALLEL_SLOTS=1).
    # Higher slot counts cause batch-size-dependent FP variance that silently
    # breaks the deterministic-routing guarantee on which the eval numbers depend.
    # Probe /props on each upstream URL; warn loudly if any exposes n_parallel > 1.
    try:
        import httpx
        _seen_urls: set[str] = set()
        _violators: list[tuple[str, int]] = []
        for _role in (
            SmallLanguageModelRole.INFERENCE,
            SmallLanguageModelRole.FUNCTION,
            SmallLanguageModelRole.EMBEDDING,
            SmallLanguageModelRole.VISION,
        ):
            _url = state.client.urls.get(_role, "")
            if not _url:
                continue
            _base = _url.rstrip("/v1").rstrip("/")
            if _base in _seen_urls:
                continue
            _seen_urls.add(_base)
            try:
                _r = httpx.get(f"{_base}/props", timeout=1.5)
                _n_par = int((_r.json() or {}).get("n_parallel", 1))
                if _n_par > 1:
                    _violators.append((_base, _n_par))
            except Exception:
                pass  # /props unavailable — skip
        if _violators:
            print(
                "⚠ llama-server is running with --parallel > 1 on:\n"
                + "\n".join(f"    {url}  (n_parallel={n})" for url, n in _violators)
                + "\n  Determinism is NOT guaranteed in this configuration — eval"
                  " numbers and e2e determinism tests will diverge. Restart with"
                  " --parallel 1 (see scripts/start_servers.sh)."
            )
        else:
            print("✓ llama-server parallel-slots probe: all upstreams n_parallel=1")
    except Exception as _exc:
        print(f"⚠ Could not probe llama-server parallel-slots: {_exc}")

    # Detect model mode: check the actual model name served on the inference port.
    # Handles both FT-on-dedicated-ports (9094-9096) and FT-on-base-ports (USE_FT=true).
    inf_url = state.client.urls.get(SmallLanguageModelRole.INFERENCE, "")
    try:
        import httpx
        base_url = inf_url.rstrip("/v1").rstrip("/")
        resp = httpx.get(f"{base_url}/v1/models", timeout=2.0)
        model_name = resp.json().get("models", [{}])[0].get("model", "")
        state.model_mode = "finetuned" if "-ft" in model_name else "base"
    except Exception:
        # Fallback: port-based detection. Catches httpx connect errors when the
        # inference server hasn't fully come up yet, plus the original parse
        # errors (KeyError/IndexError/ValueError).
        ft_port = str(os.getenv("INFERENCE_PORT_FT", "9094"))
        state.model_mode = "finetuned" if f":{ft_port}/" in inf_url else "base"

    print(f"✓ {SCENARIO_CONFIG.label} Agent ready (mode: {state.model_mode})")

    # --- Three-path comparison: Qwen (local) + Cloud ---

    # Qwen: probe dedicated port, create separate client + orchestrator (no builders)
    state.qwen_available = False
    state.qwen_client = None
    state.qwen_agent = None
    try:
        import httpx
        qwen_health = httpx.get(f"http://localhost:{QWEN_PORT}/health", timeout=2.0)
        if qwen_health.status_code == 200:
            qwen_url = f"http://localhost:{QWEN_PORT}/v1"
            embedding_url = state.client.urls.get(SmallLanguageModelRole.EMBEDDING, "")
            state.qwen_client = SmallLanguageModelClient(
                inference_url=qwen_url,
                function_url=qwen_url,
                embedding_url=embedding_url,
                vision_url=qwen_url,
                inference_model=QWEN_MODEL,
                function_model=QWEN_MODEL,
                vision_model=QWEN_MODEL,
            )
            state.qwen_agent = SmallLanguageModelAgentOrchestrator(
                state.qwen_client,
                state.tools,
            )
            state.qwen_available = True
            print(f"✓ Qwen comparison ready ({QWEN_MODEL} on port {QWEN_PORT})")
    except Exception as exc:
        print(f"⚠ Qwen comparison not available: {exc}")

    # Cloud: always create (gated by API key at call time)
    state.cloud_orchestrator = CloudOrchestrator(tools=state.tools)
    if state.cloud_orchestrator.available:
        print(f"✓ Cloud comparison ready ({OPENAI_COMPARE_MODEL})")

    # --- OCR: probe GLM-OCR server (optional, upload-time only) ---
    state.ocr_available = False
    state.ocr_client = None
    try:
        import httpx
        ocr_health = httpx.get(f"http://localhost:{OCR_PORT}/health", timeout=2.0)
        if ocr_health.status_code == 200:
            from src.engine.knowledge.ocr_client import OCRClient
            state.ocr_client = OCRClient(
                base_url=OCR_URL,
                model=OCR_MODEL,
                max_tokens=OCR_MAX_TOKENS,
                timeout=OCR_TIMEOUT,
            )
            state.ocr_available = True
            print(f"✓ GLM-OCR ready ({OCR_MODEL} on port {OCR_PORT})")
    except Exception as exc:
        print(f"⚠ GLM-OCR not available: {exc}")

    # --- Semantic chunking: init LlamaServerEmbeddings for chonkie ---
    from src.engine.inference.config import SEMANTIC_CHUNKING_ENABLED
    from src.server.state import BASE_PORTS, FT_PORTS
    state.semantic_embeddings = None
    if SEMANTIC_CHUNKING_ENABLED:
        try:
            from src.engine.knowledge.semantic_embeddings import LlamaServerEmbeddings
            # Try base embedding port (always available)
            embedding_port = BASE_PORTS["embedding"]
            sem_emb = LlamaServerEmbeddings(
                base_url=f"http://localhost:{embedding_port}",
            )
            if sem_emb.dimension > 0:
                state.semantic_embeddings = sem_emb
                print(f"✓ Semantic chunking ready (chonkie + embeddinggemma, dim={sem_emb.dimension})")
            else:
                print("⚠ Semantic chunking: embedding probe failed, using fixed-size chunking")
        except Exception as exc:
            print(f"⚠ Semantic chunking not available: {exc}")

    yield
    print("Shutting down…")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Model Local AI Agent",
    description="Privacy-preserving agentic AI with specialized small language models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route modules
app.include_router(agent_router)
app.include_router(training_router)
app.include_router(voice_router)
app.include_router(cloud_router)
app.include_router(system_router)


# ---------------------------------------------------------------------------
# Static / root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name":    "Multi-Model Local AI Agent",
        "docs":    "/docs",
        "health":  "/health",
        "ui":      "/app",
        "version": "1.0.0",
    }


# React Observatory (Vite build)
_react_dist = Path("src/clients/observatory-react/dist")
_react_public = Path("src/clients/observatory-react/public")

@app.get("/app", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(_react_dist / "index.html", headers={"Cache-Control": "no-cache, no-store"})

@app.get("/app/dashboard", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    return FileResponse(_react_public / "dashboard.html", headers={"Cache-Control": "no-cache, no-store"})

if _react_dist.exists():
    app.mount("/app/assets", StaticFiles(directory=_react_dist / "assets"), name="react-assets")
    # Serve root-level public files (e.g. openwakeword ONNX models for wake word)
    app.mount("/app/static", StaticFiles(directory=_react_dist), name="react-static")


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=["*.py", "*.html", "*.css", "*.js"],
    )
