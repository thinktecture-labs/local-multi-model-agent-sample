"""
Core agent routes — query processing, health, tools, documents, export.

Routes:
  POST /query               — process a user query
  GET  /health              — check model availability
  GET  /tools               — list registered tools
  POST /documents           — add a document to the knowledge base
  POST /upload-document     — upload & index a file (PDF/TXT/MD) with SSE progress
  POST /export-training-data — export interaction logs
"""

import asyncio
import json as _json
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Header
from fastapi.responses import StreamingResponse

from src.engine.inference.config import PIPELINE_TIMEOUT
from src.engine.knowledge.vector_store import Document
from src.engine.scaffolding.confidence_router import score_confidence
from src.engine.knowledge.document_processor import DocumentProcessor

from .models import (
    QueryRequest,
    QueryResponse,
    ExecutionStepOut,
    DocumentIn,
    HealthResponse,
    ThreePathResponse,
    ExtractionRequest,
    ExtractionResponse,
)
from .state import state, QWEN_PORT

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_response_to_query_response(result) -> QueryResponse:
    """Convert an internal AgentResponse to the API QueryResponse. Pure function."""
    steps = [
        ExecutionStepOut(
            action=s.action,
            model=s.model,
            details=s.details,
            duration_ms=s.duration_ms,
            tokens_used=s.tokens_used,
            prompt_tokens=s.prompt_tokens,
            completion_tokens=s.completion_tokens,
        )
        for s in result.steps
    ]

    # Extract cloud cost from cloud_inference step (if present)
    cloud_cost = next(
        (s.details["cost"] for s in result.steps
         if s.action == "cloud_inference" and isinstance(s.details.get("cost"), (int, float))),
        None,
    )

    return QueryResponse(
        request_id=result.request_id,
        intent=result.intent.value,
        response=result.response,
        execution_time_ms=result.execution_time_ms,
        steps=steps,
        models_used=list({s.model for s in result.steps}),
        total_tokens=result.total_tokens,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        confidence=None,
        cloud_cost=cloud_cost,
    )


def _append_confidence_step(qr: QueryResponse, result, query: str) -> QueryResponse:
    """Compute and append a confidence_assessment step. Mutates qr in place."""
    context_docs = [
        doc.get("content", "")
        for s in result.steps
        if s.action == "vector_search" and s.details.get("documents")
        for doc in s.details["documents"]
    ]
    conf = score_confidence(result.response, query, context_docs=context_docs or None)
    qr.confidence = conf.score
    qr.steps.append(ExecutionStepOut(
        action="confidence_assessment",
        model="heuristic",
        details={
            "score": conf.score,
            "score_pct": conf.score_pct,
            "should_escalate": conf.should_escalate,
            "factors": {k: round(v, 3) for k, v in conf.factors.items() if v != 0},
        },
        duration_ms=0.1,
    ))
    return qr


async def _run_backend(backend: str, query: str, images: list[str] | None = None, timeout: float = PIPELINE_TIMEOUT):
    """Run a query on the specified backend, returning an AgentResponse."""
    if backend == "qwen":
        if not state.qwen_available or state.qwen_agent is None:
            raise HTTPException(status_code=503, detail="Qwen backend not available")
        return await asyncio.wait_for(state.qwen_agent.process(query, images=images), timeout=timeout)
    elif backend == "cloud":
        if state.cloud_orchestrator is None or not state.cloud_orchestrator.available:
            raise HTTPException(status_code=503, detail="Cloud backend not available (no API key)")
        if state.network_mode == "offline":
            raise HTTPException(status_code=503, detail="Cloud backend blocked in offline mode")
        return await asyncio.wait_for(state.cloud_orchestrator.process(query), timeout=timeout)
    else:  # multi-models (default)
        return await asyncio.wait_for(state.agent.process(query, images=images), timeout=timeout)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, tags=["Agent"])
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process a query through the selected backend pipeline.

    When document_id is set, the query is scoped to a specific uploaded
    document — skips intent classification and goes directly to RAG search
    against the uploads collection. This is the "chat with uploaded doc" mode.

    Backends (when document_id is not set):
    - multi-models (default): 4-model Gemma pipeline with deterministic builders
    - qwen: Single Qwen 3.5 model (no builders, raw model capability)
    - cloud: GPT-5.4 via OpenAI API
    """
    # Document chat mode: query scoped to a specific uploaded document
    if request.document_id:
        return await _query_uploaded_document(request.query, request.document_id)

    try:
        result = await _run_backend(
            request.backend,
            request.query,
            images=request.images or None,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Pipeline timed out after {PIPELINE_TIMEOUT}s")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    qr = _agent_response_to_query_response(result)
    if state.routing_mode == "hybrid" and not request.images and request.backend == "multi-models":
        _append_confidence_step(qr, result, request.query)
    return qr


async def _query_uploaded_document(query: str, document_id: str) -> QueryResponse:
    """Chat with an uploaded document — direct RAG, no intent classification.

    Searches the uploads collection filtered by document_id, then synthesizes
    an answer using the 4B model. Fast path: no classifier, no tool routing.
    """
    import time as _time
    from src.engine.agent.types import ExecutionStep as _Step, _new_request_id
    from src.engine.inference.config import RAG_SYNTHESIS_MAX_TOKENS, RAG_SYNTHESIS_TEMPERATURE, DOC_CHAT_TOP_K, DOC_CHAT_CONTEXT_DOCS
    from src.engine.inference.prompts import build_rag_messages
    from src.engine.inference.client import SmallLanguageModelRole

    start = _time.perf_counter()
    steps: list[ExecutionStepOut] = []

    # Step 1: Search uploads filtered by document_id (ChromaDB where clause)
    t0 = _time.perf_counter()
    try:
        filtered = await state.upload_store.search(
            query, top_k=DOC_CHAT_TOP_K,
            where={"document_id": document_id},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload search failed: {exc}") from exc

    search_ms = (_time.perf_counter() - t0) * 1000
    retrieved_docs = [
        {"id": doc.id, "content": doc.content, "metadata": doc.metadata,
         "score": round(doc.score, 4) if doc.score is not None else None}
        for doc in filtered
    ]
    steps.append(ExecutionStepOut(
        action="document_search",
        model="embeddinggemma",
        details={
            "document_id": document_id,
            "query": query,
            "documents": retrieved_docs,
        },
        duration_ms=round(search_ms, 1),
    ))

    if not filtered:
        return QueryResponse(
            request_id=_new_request_id(),
            intent="document_chat",
            response=f"No content found for document '{document_id}'. Has it been uploaded?",
            execution_time_ms=round((_time.perf_counter() - start) * 1000, 1),
            steps=steps,
            models_used=["embeddinggemma"],
        )

    # Step 2: Synthesize answer from retrieved chunks
    t0 = _time.perf_counter()
    messages, _ = build_rag_messages(filtered[:DOC_CHAT_CONTEXT_DOCS], query)

    synth_response = await state.client.generate_synthesis(
        messages=messages,
        temperature=RAG_SYNTHESIS_TEMPERATURE,
        max_tokens=RAG_SYNTHESIS_MAX_TOKENS,
    )
    synth_ms = (_time.perf_counter() - t0) * 1000

    synth_model = state.client.models[SmallLanguageModelRole.VISION].replace("-vision", "")
    steps.append(ExecutionStepOut(
        action="synthesize_response",
        model=synth_model,
        details={"context_docs": len(filtered), "response": synth_response.content.strip()},
        duration_ms=round(synth_ms, 1),
        tokens_used=synth_response.tokens_used,
        prompt_tokens=synth_response.prompt_tokens,
        completion_tokens=synth_response.completion_tokens,
    ))

    elapsed_ms = (_time.perf_counter() - start) * 1000
    response_text = synth_response.content.strip()
    qr = QueryResponse(
        request_id=_new_request_id(),
        intent="document_chat",
        response=response_text,
        execution_time_ms=round(elapsed_ms, 1),
        steps=steps,
        models_used=["embeddinggemma", synth_model],
        total_tokens=synth_response.tokens_used,
        prompt_tokens=synth_response.prompt_tokens,
        completion_tokens=synth_response.completion_tokens,
    )

    # Hybrid mode: append confidence_assessment step + populate qr.confidence
    # so the UI can render the BELOW THRESHOLD escalation banner if the doc-chat
    # answer scored low (e.g. cross-reference query against the uploaded doc).
    if state.routing_mode == "hybrid":
        context_docs = [d.content for d in filtered]
        conf = score_confidence(response_text, query, context_docs=context_docs or None)
        qr.confidence = conf.score
        qr.steps.append(ExecutionStepOut(
            action="confidence_assessment",
            model="heuristic",
            details={
                "score": conf.score,
                "score_pct": conf.score_pct,
                "should_escalate": conf.should_escalate,
                "factors": {k: round(v, 3) for k, v in conf.factors.items() if v != 0},
            },
            duration_ms=0.1,
        ))
    return qr


async def _stream_document_chat(query: str, document_id: str):
    """Stream a document chat response via SSE — direct RAG against uploaded doc."""
    import time as _time
    from src.engine.agent.types import _new_request_id
    from src.engine.inference.config import RAG_SYNTHESIS_MAX_TOKENS, RAG_SYNTHESIS_TEMPERATURE, DOC_CHAT_TOP_K, DOC_CHAT_CONTEXT_DOCS
    from src.engine.inference.prompts import build_rag_messages
    from src.engine.inference.client import SmallLanguageModelRole

    async def event_stream():
        start = _time.perf_counter()

        # Step 1: Search
        t0 = _time.perf_counter()
        try:
            filtered = await state.upload_store.search(
                query, top_k=DOC_CHAT_TOP_K, where={"document_id": document_id},
            )
        except Exception as exc:
            yield _sse_event("error", {"message": f"Upload search failed: {exc}"})
            return

        retrieved_docs = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata,
             "score": round(doc.score, 4) if doc.score is not None else None}
            for doc in filtered
        ]
        yield _sse_event("step", {
            "action": "document_search", "model": "embeddinggemma",
            "details": {"document_id": document_id, "query": query, "documents": retrieved_docs},
            "duration_ms": round((_time.perf_counter() - t0) * 1000, 1),
            "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0,
        })

        if not filtered:
            yield _sse_event("token", {"text": f"No content found for document '{document_id}'."})
            yield _sse_event("done", {"intent": "document_chat", "execution_time_ms": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "models_used": []})
            return

        # Step 2: Stream synthesis
        messages, _ = build_rag_messages(filtered[:DOC_CHAT_CONTEXT_DOCS], query)

        t0 = _time.perf_counter()
        tokens_used = prompt_tokens = completion_tokens = 0
        synth_model = state.client.models[SmallLanguageModelRole.VISION].replace("-vision", "")
        accumulated_text = ""

        stream = state.client.generate_synthesis_stream(
            messages=messages,
            temperature=RAG_SYNTHESIS_TEMPERATURE,
            max_tokens=RAG_SYNTHESIS_MAX_TOKENS,
        )
        async for chunk in stream:
            if chunk.done:
                tokens_used = chunk.tokens_used
                prompt_tokens = chunk.prompt_tokens
                completion_tokens = chunk.completion_tokens
            elif chunk.text:
                accumulated_text += chunk.text
                yield _sse_event("token", {"text": chunk.text})

        synth_ms = (_time.perf_counter() - t0) * 1000
        yield _sse_event("step", {
            "action": "synthesize_response", "model": synth_model,
            "details": {"context_docs": len(filtered)},
            "duration_ms": round(synth_ms, 1),
            "tokens_used": tokens_used, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        })

        # Hybrid mode: emit confidence_assessment step so the UI can render
        # the BELOW THRESHOLD escalation banner for low-confidence doc-chat answers.
        if state.routing_mode == "hybrid" and accumulated_text:
            context_docs = [d.content for d in filtered]
            conf = score_confidence(
                accumulated_text, query, context_docs=context_docs or None,
            )
            yield _sse_event("step", {
                "action": "confidence_assessment", "model": "heuristic",
                "details": {
                    "score": conf.score,
                    "score_pct": conf.score_pct,
                    "should_escalate": conf.should_escalate,
                    "factors": {k: round(v, 3) for k, v in conf.factors.items() if v != 0},
                },
                "duration_ms": 0.1,
                "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0,
            })

        elapsed_ms = (_time.perf_counter() - start) * 1000
        yield _sse_event("done", {
            "intent": "document_chat",
            "execution_time_ms": round(elapsed_ms, 1),
            "total_tokens": tokens_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "models_used": ["embeddinggemma", synth_model],
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {_json.dumps(data, default=str)}\n\n"


@router.post("/query/stream", tags=["Agent"])
async def process_query_stream(request: QueryRequest):
    """
    Stream a query response via SSE.

    When document_id is set, streams a document chat response (direct RAG,
    no intent classification). Otherwise streams from the normal agent pipeline.

    Events:
    - step: an ExecutionStep (trace panel update)
    - token: incremental response text {"text": "..."}
    - done: final metadata (request_id, intent, timing, tokens)
    - error: error message
    """
    # Document chat mode: stream synthesis from uploaded document
    if request.document_id:
        return await _stream_document_chat(request.query, request.document_id)

    from src.engine.agent.types import ExecutionStep as _Step
    import time as _time

    async def event_stream():
        start = _time.perf_counter()
        accumulated_text = ""
        steps = []

        try:
            # Select streaming source based on backend
            if request.backend == "cloud" and state.cloud_orchestrator and state.cloud_orchestrator.available:
                gen = state.cloud_orchestrator.process_stream(
                    request.query, images=request.images or None,
                )
            elif request.backend == "qwen":
                # Verify Qwen is actually reachable before using it
                qwen_live = False
                if state.qwen_available and state.qwen_client is not None:
                    try:
                        qwen_health = await state.qwen_client.check_health()
                        qwen_live = any(qwen_health.values())
                    except Exception:
                        pass
                if qwen_live and state.qwen_agent is not None:
                    gen = state.qwen_agent.process_stream(
                        request.query, images=request.images or None,
                    )
                else:
                    yield _sse_event("error", {"message": "MoE backend not available"})
                    return
            else:
                gen = state.agent.process_stream(
                    request.query, images=request.images or None,
                )

            async for item in gen:
                if isinstance(item, _Step):
                    steps.append(item)
                    yield _sse_event("step", {
                        "action": item.action,
                        "model": item.model,
                        "details": item.details,
                        "duration_ms": item.duration_ms,
                        "tokens_used": item.tokens_used,
                        "prompt_tokens": item.prompt_tokens,
                        "completion_tokens": item.completion_tokens,
                    })
                elif isinstance(item, str):
                    # <think>...</think> stripping is handled upstream in
                    # SmallLanguageModelClient.generate_stream() and
                    # generate_synthesis_stream() — all string tokens here are clean.
                    accumulated_text += item
                    yield _sse_event("token", {"text": item})

            elapsed_ms = (_time.perf_counter() - start) * 1000
            total_tokens = sum(s.tokens_used for s in steps)
            prompt_tokens = sum(s.prompt_tokens for s in steps)
            completion_tokens = sum(s.completion_tokens for s in steps)

            # Determine intent from classify_intent step
            intent = "direct_answer"
            for s in steps:
                if s.action == "classify_intent":
                    intent = s.details.get("intent", "direct_answer")
                    break

            # Confidence assessment for hybrid routing mode
            confidence_score = None
            if (
                state.routing_mode == "hybrid"
                and not request.images
                and request.backend == "multi-models"
                and accumulated_text
            ):
                context_docs = []
                for s in steps:
                    if s.action == "vector_search" and s.details.get("documents"):
                        for doc in s.details["documents"]:
                            context_docs.append(doc.get("content", ""))
                conf = score_confidence(
                    accumulated_text, request.query,
                    context_docs=context_docs or None,
                )
                confidence_score = conf.score
                yield _sse_event("step", {
                    "action": "confidence_assessment",
                    "model": "heuristic",
                    "details": {
                        "score": conf.score,
                        "score_pct": conf.score_pct,
                        "should_escalate": conf.should_escalate,
                        "factors": {k: round(v, 3) for k, v in conf.factors.items() if v != 0},
                    },
                    "duration_ms": 0.1,
                    "tokens_used": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                })

            cloud_cost = next(
                (s.details["cost"] for s in steps
                 if s.action == "cloud_inference" and isinstance(s.details.get("cost"), (int, float))),
                None,
            )

            yield _sse_event("done", {
                "intent": intent,
                "execution_time_ms": round(elapsed_ms, 1),
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "models_used": list(dict.fromkeys(s.model for s in steps)),
                "cloud_cost": cloud_cost,
            })

        except asyncio.TimeoutError:
            yield _sse_event("error", {"message": f"Pipeline timed out after {PIPELINE_TIMEOUT}s"})
        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/compare-all", response_model=ThreePathResponse, tags=["Comparison"])
async def compare_all(request: QueryRequest) -> ThreePathResponse:
    """
    Run the same query through all three backends in parallel.

    Returns results from multi-models (always), qwen (if available),
    and cloud (if online + API key configured).
    """
    async def _safe_run(backend: str):
        try:
            return await _run_backend(backend, request.query, timeout=15.0)
        except Exception:
            return None

    # Always run multi-models; conditionally run qwen and cloud
    tasks = [_run_backend("multi-models", request.query, images=request.images or None)]
    # Re-check Qwen health (startup probe may be stale if Qwen started/stopped since)
    run_qwen = False
    if state.qwen_available and state.qwen_agent is not None:
        try:
            qwen_health = await state.qwen_client.check_health()
            run_qwen = any(qwen_health.values())
        except Exception:
            run_qwen = False
    run_cloud = (
        state.cloud_orchestrator is not None
        and state.cloud_orchestrator.available
        and state.network_mode != "offline"
    )
    if run_qwen:
        tasks.append(_safe_run("qwen"))
    if run_cloud:
        tasks.append(_safe_run("cloud"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results
    multi_result = results[0]
    if isinstance(multi_result, Exception):
        raise HTTPException(status_code=500, detail=str(multi_result))

    idx = 1
    qwen_result = results[idx] if run_qwen and not isinstance(results[idx], Exception) else None
    if run_qwen:
        idx += 1
    cloud_result = results[idx] if run_cloud and idx < len(results) and not isinstance(results[idx], Exception) else None

    return ThreePathResponse(
        multi_models=_agent_response_to_query_response(multi_result),
        qwen=_agent_response_to_query_response(qwen_result) if qwen_result else None,
        cloud=_agent_response_to_query_response(cloud_result) if cloud_result else None,
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check which models are available (Gemma stack + optional Qwen + cloud)."""
    model_health    = await state.client.check_health()
    document_count  = await state.vector_store.count()
    upload_count    = await state.upload_store.count()

    # Add Qwen health (probe port if we think it's available)
    if state.qwen_available and state.qwen_client is not None:
        try:
            qwen_health = await state.qwen_client.check_health()
            model_health["QWEN"] = any(qwen_health.values())
        except Exception:
            model_health["QWEN"] = False
    else:
        model_health["QWEN"] = False

    # Add cloud availability
    model_health["CLOUD"] = (
        state.cloud_orchestrator is not None
        and state.cloud_orchestrator.available
        and state.network_mode != "offline"
    )

    # Add OCR availability (optional — upload-time only)
    model_health["OCR"] = state.ocr_available

    # Add LogReg classifier availability — primary intent path; degrades to
    # gemma3-ft generative fallback if the model.joblib file is absent.
    try:
        model_health["LOGREG"] = bool(state.agent._classifier.using_logreg)
    except AttributeError:
        model_health["LOGREG"] = False

    all_healthy = all(v for k, v in model_health.items() if k not in ("QWEN", "CLOUD", "WHISPER", "OCR", "LOGREG"))

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        models=model_health,
        document_count=document_count,
        interaction_count=state.agent.interaction_count,
    )


@router.get("/tools", tags=["System"])
async def list_tools() -> dict:
    """List all registered tools and their schemas."""
    return {
        "tools": state.tools.list_tools(),
        "schemas": state.tools.get_all_schemas(),
    }


@router.post("/documents", tags=["Knowledge Base"])
async def add_document(doc: DocumentIn) -> dict:
    """
    Add a document to the knowledge base.

    The document is embedded with embeddinggemma and stored in ChromaDB.
    It will be immediately available for semantic search.
    """
    try:
        await state.vector_store.add_document(Document(
            id=doc.id,
            content=doc.content,
            metadata=doc.metadata,
        ))
        total = await state.vector_store.count()
        return {"status": "indexed", "id": doc.id, "total_documents": total}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/uploads", tags=["Knowledge Base"])
async def clear_uploads() -> dict:
    """Clear all uploaded documents from the uploads collection."""
    count = await state.upload_store.count()
    await state.upload_store.clear()
    state.active_upload = None
    return {"status": "cleared", "deleted_chunks": count}


@router.get("/uploads/status", tags=["Knowledge Base"])
async def upload_status() -> dict:
    """Check status of an active upload (for reconnect after SSE drop)."""
    if state.active_upload is None:
        return {"status": "idle", "filename": None}
    task = state.active_upload.get("task")
    filename = state.active_upload.get("filename", "")
    if task is None or task.done():
        chunks = await state.upload_store.count()
        state.active_upload = None
        return {"status": "completed", "filename": filename, "chunks": chunks}
    return {"status": "processing", "filename": filename}


@router.post("/upload-document", tags=["Knowledge Base"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF, TXT, MD).

    Returns Server-Sent Events with real-time progress for each pipeline
    stage: parsing → chunking → embedding → indexed.
    The document is immediately queryable after indexing completes.
    """
    from pathlib import Path as _Path

    filename = file.filename or "document"
    suffix = _Path(filename).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Use PDF, TXT, or MD.",
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    processor = DocumentProcessor(
        state.upload_store,
        ocr_client=state.ocr_client,
        semantic_embeddings=state.semantic_embeddings,
    )

    # Run the upload in a background task so SSE disconnects don't kill
    # processing. The event_stream reads from an asyncio.Queue that the
    # background task feeds. If the client disconnects, the task continues
    # and the document still gets indexed.
    event_queue: asyncio.Queue = asyncio.Queue()

    async def _process_in_background():
        try:
            async for event in processor.process_file(filename, content):
                await event_queue.put(event)
        except Exception as exc:
            from src.engine.knowledge.document_processor import ProcessingEvent
            await event_queue.put(ProcessingEvent(
                stage="error",
                message=f"Upload failed: {exc}",
                detail={"error": str(exc)},
            ))
        finally:
            await event_queue.put(None)  # sentinel

    # Start processing — continues even if SSE client disconnects
    task = asyncio.create_task(_process_in_background())

    # Track active upload for /uploads/status
    state.active_upload = {"filename": filename, "task": task}

    async def event_stream():
        while True:
            event = await event_queue.get()
            if event is None:
                break
            data = _json.dumps({
                "stage": event.stage,
                "message": event.message,
                "detail": event.detail,
            })
            yield f"event: progress\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_DB_PATH = os.getenv("DB_PATH", "./data/business.db")


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

@router.post("/extract", response_model=ExtractionResponse, tags=["Knowledge Base"])
async def extract_data(request: ExtractionRequest):
    """
    Extract structured financial data from an uploaded document.

    Reads the document's chunks from the uploads collection, sends them to
    gemma3-4B for structured extraction, parses the JSON output, and stores
    the result in the competitors SQL table.

    The extracted JSON is returned in the response for the UI debug view.
    """
    import time as _time
    from src.engine.knowledge.data_extractor import DataExtractor

    t0 = _time.perf_counter()

    # Get document text from uploads collection
    try:
        docs = await state.upload_store.search(
            query="revenue customers growth fiscal year",
            top_k=10,
            where={"document_id": request.document_id},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to search uploads: {exc}")

    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded document found with id '{request.document_id}'",
        )

    text = "\n\n".join(d.content for d in docs)

    # Extract
    extractor = DataExtractor(client=state.client, db_path=_DB_PATH)
    result = await extractor.extract(text, source_document=request.document_id)

    elapsed = (_time.perf_counter() - t0) * 1000

    return ExtractionResponse(
        success=result.success,
        extracted=result.extracted,
        raw_output=result.raw_output,
        stored=result.stored,
        error=result.error,
        execution_time_ms=round(elapsed, 1),
    )


@router.get("/competitors", tags=["Knowledge Base"])
async def list_competitors():
    """List all extracted competitor data from the database."""
    import aiosqlite

    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM competitors ORDER BY fiscal_year DESC") as c:
                rows = await c.fetchall()
                return {"competitors": [dict(r) for r in rows], "count": len(rows)}
    except Exception as exc:
        return {"competitors": [], "count": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Fine-tuning data export
# ---------------------------------------------------------------------------

_EXPORT_DIR = os.path.abspath("./data")
_EXPORT_TOKEN = os.getenv("EXPORT_TOKEN", "")  # empty = disabled (local-only default)


def _check_export_auth(authorization: str | None) -> None:
    """Raise 401 if EXPORT_TOKEN is set and the request doesn't match."""
    if not _EXPORT_TOKEN:
        return  # token not configured → unauthenticated access allowed (localhost only)
    if authorization != f"Bearer {_EXPORT_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing EXPORT_TOKEN")


@router.post("/export-training-data", tags=["Fine-Tuning"])
async def export_training_data(
    filename: str = "interactions.json",
    authorization: str | None = Header(default=None),
) -> dict:
    """
    Export all logged agent interactions as fine-tuning data.

    Use this data to improve the text Gemma models for your domain.
    """
    _check_export_auth(authorization)
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = os.path.join(_EXPORT_DIR, safe_name)

    count = state.agent.export_training_data(filepath)
    return {
        "status":       "exported",
        "filepath":     filepath,
        "interactions": count,
        "evicted":      state.agent.eviction_count,
    }
