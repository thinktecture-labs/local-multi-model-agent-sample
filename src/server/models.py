"""
Pydantic request/response models for the FastAPI server.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB per image (decoded base64)
MAX_IMAGES = 3


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question or instruction")
    images: list[str] = Field(default_factory=list, description="Base64-encoded images (PNG/JPEG) for vision queries")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_IMAGES:
            raise ValueError(f"Too many images: {len(v)} (max {MAX_IMAGES})")
        import base64
        for i, img in enumerate(v):
            decoded_size = len(base64.b64decode(img, validate=True))
            if decoded_size > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image {i + 1} too large: {decoded_size / 1024 / 1024:.1f} MB "
                    f"(max {MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB)"
                )
        return v
    backend: str = Field(
        default="multi-models",
        pattern=r"^(multi-models|qwen|cloud)$",
        description="Backend path: multi-models (default), qwen (single model), or cloud (GPT-5.4)",
    )
    document_id: Optional[str] = Field(
        default=None,
        description=(
            "Scope the query to a specific uploaded document. "
            "Skips intent classification — goes directly to RAG search "
            "against the uploads collection filtered by this document_id. "
            "Set by the UI after document upload; cleared on reset."
        ),
    )


class ExecutionStepOut(BaseModel):
    action: str
    model:  str
    details: dict[str, Any] = {}
    duration_ms: float = 0.0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class QueryResponse(BaseModel):
    request_id:        str = ""
    intent:            str
    response:          str
    execution_time_ms: float
    steps:             list[ExecutionStepOut]
    models_used:       list[str]
    total_tokens:      int = 0
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    # Hybrid routing fields (populated when routing_mode=hybrid)
    confidence:        Optional[float] = None
    escalated:         bool = False
    cloud_response:    Optional[str] = None
    cloud_model:       Optional[str] = None
    cloud_latency_ms:  Optional[float] = None
    cloud_cost:        Optional[float] = None


class DocumentIn(BaseModel):
    id:       str
    content:  str
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status:       str
    models:       dict[str, bool]
    document_count: int
    interaction_count: int = 0


class EscalateRequest(BaseModel):
    query: str = Field(..., description="The original query to escalate to cloud")


class EscalateDisclosure(BaseModel):
    """What left the machine on this escalation — the privacy receipt.

    The user has opted in by clicking escalate, but the implicit content
    of "what got sent" needs to be explicit: curated KB chunks are always
    safe to share (they're the canonical answer set), but uploaded-document
    chunks may contain confidential material the user put on disk for a
    different reason. Surfacing this lets a privacy-conscious user verify
    after the fact (and the UI can show it before the call).
    """
    kb_chunks: int = Field(0, description="Curated KB chunks included in the prompt")
    upload_chunks: int = Field(0, description="User-uploaded document chunks included")
    context_chars: int = Field(0, description="Total chars of LOCAL CONTEXT block sent")
    upload_chunk_ids: list[str] = Field(default_factory=list, description="IDs of the uploaded chunks sent (so the user can audit)")


class EscalateResponse(BaseModel):
    cloud_response: str
    cloud_model: str
    cloud_latency_ms: float
    cloud_cost: float
    cloud_tokens: int
    cloud_bytes_sent: int = 0
    disclosure: EscalateDisclosure = Field(default_factory=EscalateDisclosure)


class TrainRequest(BaseModel):
    model: str = Field(default="gemma3", pattern=r"^(gemma3)$")
    task: str = Field(default="intent", pattern=r"^(intent|synthesis|both)$")
    demo_mode: bool = Field(default=False, description="Simulate training without GPU")


class EvalRequest(BaseModel):
    model: str = Field(default="gemma3", pattern=r"^(gemma3)$")
    save_as: str = Field(default="", description="Label: 'before' or 'after' (auto-assigned if empty)")


class GpuStats(BaseModel):
    available: bool
    backend: str = "none"
    name: str = ""
    vram_used_mb: float = 0
    vram_total_mb: float = 0
    utilization_pct: float = 0
    temperature_c: float = 0
    gpu_power_w: float = 0
    system_power_w: float = 0


class EnergyStats(BaseModel):
    """Accumulated energy consumption for the session."""
    total_wh: float = 0.0
    total_queries: int = 0
    wh_per_query: float = 0.0
    gpu_power_now_w: float = 0.0
    system_power_now_w: float = 0.0
    co2_local_g: float = 0.0
    co2_cloud_g: float = 0.0
    electricity_cost_local: float = 0.0
    estimated_cloud_wh: float = 0.0
    backend: str = "none"
    sample_count: int = 0
    uptime_seconds: float = 0.0


class PrivacyStats(BaseModel):
    total_queries: int
    total_tokens_generated: int
    external_bytes_sent: int = 0
    uptime_seconds: float
    active_connections: str = "localhost only"
    network_mode: str = "online"
    routing_mode: str = "local-only"
    cloud_queries: int = 0


class SwapRequest(BaseModel):
    mode: str = Field(..., pattern=r"^(base|finetuned)$", description="'base' or 'finetuned'")


class SwapResponse(BaseModel):
    status: str
    mode: str
    message: str


class CompareRequest(BaseModel):
    query: str = Field(..., description="Query to compare local vs cloud")


class CompareResponse(BaseModel):
    intent: str = ""
    local_response: str
    local_latency_ms: float
    local_tokens: int
    local_cost: float = 0.0
    cloud_response: Optional[str] = None
    cloud_latency_ms: Optional[float] = None
    cloud_tokens: Optional[int] = None
    cloud_bytes_sent: Optional[int] = None
    cloud_cost: Optional[float] = None
    cloud_available: bool = False
    cloud_model: str = ""
    estimated_cloud_cost: float = 0.0


class ThreePathResponse(BaseModel):
    """Side-by-side results from all three backend paths."""
    multi_models: QueryResponse
    qwen: Optional[QueryResponse] = None
    cloud: Optional[QueryResponse] = None


class ExtractionRequest(BaseModel):
    document_id: str = Field(..., description="Document ID to extract data from (must be uploaded first)")


class ExtractionResponse(BaseModel):
    success: bool
    extracted: Optional[dict] = None
    raw_output: Optional[str] = None
    stored: bool = False
    error: Optional[str] = None
    execution_time_ms: float = 0.0
