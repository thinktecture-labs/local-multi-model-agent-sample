"""
Shared application state for the FastAPI server.

Single source of truth for AppState, port configuration, and module-level
singletons. All route modules import state from here.
"""

import asyncio
import os
import time as _time

from src.engine.inference.client import SmallLanguageModelClient
from src.engine.knowledge.vector_store import VectorStore
from src.engine.tools import ToolRegistry
from src.engine.agent import SmallLanguageModelAgentOrchestrator
from src.engine.agent.cloud_orchestrator import CloudOrchestrator


# ---------------------------------------------------------------------------
# Port sets for dual-port swap (zero-downtime model switching)
# ---------------------------------------------------------------------------

BASE_PORTS = {
    "inference": int(os.getenv("INFERENCE_PORT", "9090")),
    "function":  int(os.getenv("FUNCTION_PORT",  "9091")),
    "embedding": int(os.getenv("EMBEDDING_PORT", "9092")),
    "vision":    int(os.getenv("VISION_PORT",     "9093")),
}
FT_PORTS = {
    "inference": int(os.getenv("INFERENCE_PORT_FT", "9094")),
    "function":  int(os.getenv("FUNCTION_PORT_FT",  "9095")),
    "embedding": int(os.getenv("EMBEDDING_PORT_FT", "9096")),
    "vision":    int(os.getenv("VISION_PORT",        "9093")),  # shared, no FT variant
}

# Qwen comparison port — single llama-server handles inference + function calling
QWEN_PORT = int(os.getenv("QWEN_PORT", "9100"))


# ---------------------------------------------------------------------------
# Application state (shared across requests)
# ---------------------------------------------------------------------------
#
# Concurrency model: uvicorn runs a single-threaded asyncio event loop.
# Within one event loop tick, only one coroutine executes — so plain attribute
# reads (e.g. `state.network_mode`) are safe without locks. Locks are only
# needed for read-modify-write sequences that span multiple awaits:
#
#   _bytes_lock   — guards cloud_bytes_sent (incremented from multiple routes)
#   _energy_lock  — guards energy_wh, energy_samples, energy_last_* (polled)
#   _mode_lock    — guards network_mode, routing_mode, model_mode (toggled)
#
# If uvicorn is ever run with --workers >1 (multiprocess), this model breaks.
# For the demo/keynote single-worker setup, this is correct and intentional.
# ---------------------------------------------------------------------------

class AppState:
    __slots__ = (
        'client', 'vector_store', 'upload_store', 'tools', 'agent',
        'qwen_client', 'qwen_agent', 'qwen_available',
        'cloud_orchestrator',
        'ocr_available', 'ocr_client', 'active_upload',
        'semantic_embeddings',
        'model_mode', 'cloud_bytes_sent', 'network_mode', 'routing_mode',
        'training_running', 'training_stage', 'eval_results',
        'energy_wh', 'energy_samples', 'energy_last_sample_time',
        'energy_last_gpu_w', 'energy_last_sys_w', 'energy_backend',
        # Concurrency locks — one per logical group of mutable fields
        '_mode_lock',    # guards: network_mode, routing_mode, model_mode
        '_energy_lock',  # guards: energy_wh, energy_samples, energy_last_*
        '_bytes_lock',   # guards: cloud_bytes_sent
    )

    def __init__(self) -> None:
        # Core modules (assigned during server startup lifespan)
        self.client:       SmallLanguageModelClient
        self.vector_store: VectorStore       # curated KB (13 seeded docs)
        self.upload_store: VectorStore       # user uploads (OCR/PDF, separate collection)
        self.tools:        ToolRegistry
        self.agent:        SmallLanguageModelAgentOrchestrator
        # Three-path comparison (assigned during startup if available)
        self.qwen_client:  SmallLanguageModelClient | None
        self.qwen_agent:   SmallLanguageModelAgentOrchestrator | None
        self.qwen_available: bool = False
        self.cloud_orchestrator: CloudOrchestrator | None = None
        # OCR (optional — auto-detected at startup)
        self.ocr_available: bool = False
        self.ocr_client = None  # OCRClient | None
        self.active_upload: dict | None = None  # {filename, task} for in-progress upload
        # Semantic chunking (optional — LlamaServerEmbeddings for chonkie)
        self.semantic_embeddings = None
        # Mode flags
        self.model_mode:   str = "finetuned"
        self.cloud_bytes_sent: int = 0
        self.network_mode: str = "online"       # "online" | "offline"
        self.routing_mode: str = "local-only"   # "local-only" | "hybrid"
        # Training state
        self.training_running: bool = False
        self.training_stage:   str = "idle"
        # Eval results storage (before/after snapshots)
        self.eval_results: dict = {}
        # Energy accumulation (sampled from /gpu polls)
        self.energy_wh: float = 0.0
        self.energy_samples: int = 0
        self.energy_last_sample_time: float = 0.0
        self.energy_last_gpu_w: float = 0.0
        self.energy_last_sys_w: float = 0.0
        self.energy_backend: str = "none"
        # Locks (asyncio — must be created inside the event loop)
        self._mode_lock   = asyncio.Lock()
        self._energy_lock = asyncio.Lock()
        self._bytes_lock  = asyncio.Lock()


state = AppState()
training_lock = asyncio.Lock()
STARTUP_TIME = _time.time()
