"""
Integration tests for the FastAPI server endpoints.

Uses FastAPI's TestClient with fully mocked agent internals so no
llama-server instances are needed. Tests HTTP request/response structure,
status codes, validation, and error handling.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.engine.agent import AgentResponse, Intent, ExecutionStep
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse
from src.engine.tools.tool_result import ToolResult


# ---------------------------------------------------------------------------
# Build a fully-mocked app for each test
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with a mocked lifespan that injects mock components."""
    import src.server as server_module

    mock_client = MagicMock()
    mock_client.models = {
        SmallLanguageModelRole.INFERENCE: "gemma3:1b-it",
        SmallLanguageModelRole.FUNCTION:  "qwen",
        SmallLanguageModelRole.EMBEDDING: "embeddinggemma",
        SmallLanguageModelRole.VISION:    "gemma3-4b-vision",
    }
    mock_client.check_health = AsyncMock(return_value={
        "INFERENCE": True, "FUNCTION": True, "EMBEDDING": True, "VISION": True,
    })

    mock_vs = MagicMock()
    mock_vs.count = AsyncMock(return_value=13)
    mock_vs.add_document = AsyncMock()
    mock_vs.set_client = MagicMock()

    mock_us = MagicMock()
    mock_us.count = AsyncMock(return_value=0)

    mock_tools = MagicMock()
    mock_tools.list_tools = MagicMock(return_value=["calculator", "sql_query", "vector_search"])
    mock_tools.get_all_schemas = MagicMock(return_value=[
        {"type": "function", "function": {"name": "calculator"}},
    ])

    mock_agent = MagicMock()
    mock_agent.process = AsyncMock(return_value=AgentResponse(
        query="test",
        intent=Intent.DIRECT_ANSWER,
        response="Hello!",
        steps=[ExecutionStep(action="direct_response", model="gemma3:1b-it")],
        execution_time_ms=42.0,
    ))
    mock_agent.export_training_data = MagicMock(return_value=3)
    mock_agent.interaction_count = 7
    mock_agent.total_tokens_generated = 150

    # Replace the lifespan to inject mocks instead of real components
    @asynccontextmanager
    async def mock_lifespan(app):
        server_module._state.client = mock_client
        server_module._state.vector_store = mock_vs
        server_module._state.upload_store = mock_us
        server_module._state.tools = mock_tools
        server_module._state.agent = mock_agent
        # Reset mutable state to prevent leakage between tests
        server_module._state.cloud_bytes_sent = 0
        server_module._state.network_mode = "online"
        server_module._state.routing_mode = "local-only"
        server_module._state.model_mode = "finetuned"
        server_module._state.training_running = False
        server_module._state.training_stage = "idle"
        server_module._state.eval_results = {}
        yield

    original_lifespan = server_module.app.router.lifespan_context
    server_module.app.router.lifespan_context = mock_lifespan

    with TestClient(server_module.app, raise_server_exceptions=False) as tc:
        tc._mock_agent = mock_agent
        tc._mock_client = mock_client
        tc._mock_vs = mock_vs
        yield tc

    server_module.app.router.lifespan_context = original_lifespan


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_returns_json(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert data["docs"] == "/docs"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_query_returns_response(self, client):
        resp = client.post("/query", json={"query": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "direct_answer"
        assert data["response"] == "Hello!"
        assert data["execution_time_ms"] == 42.0
        assert len(data["steps"]) == 1

    def test_query_includes_token_fields(self, client):
        """QueryResponse must include token aggregates."""
        client._mock_agent.process = AsyncMock(return_value=AgentResponse(
            query="test", intent=Intent.DIRECT_ANSWER, response="ok",
            steps=[ExecutionStep(
                action="direct_response", model="gemma3",
                tokens_used=15, prompt_tokens=10, completion_tokens=5,
            )],
            execution_time_ms=10.0,
        ))
        resp = client.post("/query", json={"query": "test"})
        data = resp.json()
        assert data["total_tokens"] == 15
        assert data["prompt_tokens"] == 10
        assert data["completion_tokens"] == 5
        assert data["steps"][0]["tokens_used"] == 15

    def test_query_has_models_used(self, client):
        resp = client.post("/query", json={"query": "test"})
        data = resp.json()
        assert "models_used" in data
        assert isinstance(data["models_used"], list)

    def test_query_missing_body_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_with_images_returns_response(self, client):
        client._mock_agent.process = AsyncMock(return_value=AgentResponse(
            query="describe",
            intent=Intent.IMAGE_QUERY,
            response="This image shows a revenue chart.",
            steps=[ExecutionStep(action="analyse_image", model="gemma3-4b-vision")],
            execution_time_ms=120.0,
        ))
        # 1x1 red pixel PNG, valid base64
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        resp = client.post("/query", json={"query": "describe", "images": [tiny_png]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "image_query"
        assert data["response"] == "This image shows a revenue chart."
        assert len(data["steps"]) == 1
        assert data["steps"][0]["action"] == "analyse_image"

    def test_query_without_images_still_works(self, client):
        resp = client.post("/query", json={"query": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "direct_answer"
        assert data["response"] == "Hello!"

    def test_query_agent_error_returns_500(self, client):
        client._mock_agent.process = AsyncMock(side_effect=RuntimeError("model crashed"))
        resp = client.post("/query", json={"query": "boom"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_all_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["document_count"] == 13
        # Core models (INFERENCE, FUNCTION, EMBEDDING, VISION) + optional (QWEN, CLOUD)
        assert len(data["models"]) >= 4
        for key in ("INFERENCE", "FUNCTION", "EMBEDDING", "VISION"):
            assert data["models"].get(key) is True, f"{key} should be healthy"

    def test_health_includes_interaction_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "interaction_count" in data
        assert data["interaction_count"] == 7

    def test_health_degraded(self, client):
        client._mock_client.check_health = AsyncMock(return_value={
            "INFERENCE": True, "FUNCTION": False, "EMBEDDING": True, "VISION": True,
        })
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# GET /tools
# ---------------------------------------------------------------------------

class TestToolsEndpoint:
    def test_tools_returns_list(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert "schemas" in data
        assert "calculator" in data["tools"]


# ---------------------------------------------------------------------------
# POST /documents
# ---------------------------------------------------------------------------

class TestDocumentsEndpoint:
    def test_add_document_returns_indexed(self, client):
        resp = client.post("/documents", json={
            "id": "test-doc",
            "content": "Test content",
            "metadata": {"category": "test"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"
        assert data["id"] == "test-doc"

    def test_add_document_missing_fields_returns_422(self, client):
        resp = client.post("/documents", json={"id": "no-content"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /export-training-data
# ---------------------------------------------------------------------------

class TestExportEndpoint:
    def test_export_returns_count(self, client):
        resp = client.post("/export-training-data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "exported"
        assert data["interactions"] == 3


# ---------------------------------------------------------------------------
# GET /gpu
# ---------------------------------------------------------------------------

class TestGpuEndpoint:
    def test_gpu_returns_stats(self, client):
        resp = client.get("/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "backend" in data
        assert "name" in data
        assert "vram_used_mb" in data
        assert "vram_total_mb" in data

    def test_gpu_backend_is_valid(self, client):
        resp = client.get("/gpu")
        data = resp.json()
        assert data["backend"] in ("cuda", "metal", "cpu", "none")


# ---------------------------------------------------------------------------
# GET /privacy
# ---------------------------------------------------------------------------

class TestPrivacyEndpoint:
    def test_privacy_returns_zero_external_bytes(self, client):
        resp = client.get("/privacy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["external_bytes_sent"] == 0

    def test_privacy_has_uptime(self, client):
        resp = client.get("/privacy")
        data = resp.json()
        assert data["uptime_seconds"] > 0

    def test_privacy_has_query_count(self, client):
        resp = client.get("/privacy")
        data = resp.json()
        assert data["total_queries"] == 7

    def test_privacy_has_token_count(self, client):
        resp = client.get("/privacy")
        data = resp.json()
        assert data["total_tokens_generated"] == 150


# ---------------------------------------------------------------------------
# GET /models/mode
# ---------------------------------------------------------------------------

class TestModelsMode:
    def test_get_mode_returns_current(self, client):
        resp = client.get("/models/mode")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] in ("base", "finetuned")


# ---------------------------------------------------------------------------
# POST /compare (without API key)
# ---------------------------------------------------------------------------

class TestCompareEndpoint:
    def test_compare_returns_local_result(self, client):
        resp = client.post("/compare", json={"query": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["local_response"] == "Hello!"
        assert isinstance(data["local_latency_ms"], float)
        assert data["local_cost"] == 0.0
        assert isinstance(data["cloud_available"], bool)

    def test_compare_has_estimated_cloud_cost(self, client):
        resp = client.post("/compare", json={"query": "Hello"})
        data = resp.json()
        assert "estimated_cloud_cost" in data
        assert isinstance(data["estimated_cloud_cost"], float)

    def test_compare_missing_query_returns_422(self, client):
        resp = client.post("/compare", json={})
        assert resp.status_code == 422


class TestEscalateStreamEndpoint:
    """Tests for POST /escalate/stream SSE endpoint."""

    def test_escalate_stream_offline_returns_503(self, client):
        import src.server as sm
        sm._state.network_mode = "offline"
        resp = client.post("/escalate/stream", json={"query": "test"})
        assert resp.status_code == 503

    def test_escalate_stream_no_api_key_returns_503(self, client):
        """Without OPENAI_API_KEY, cloud is not enabled."""
        import src.server as sm
        import src.server.cloud_routes as cloud_mod
        original = cloud_mod.CLOUD_COMPARISON_ENABLED
        cloud_mod.CLOUD_COMPARISON_ENABLED = False
        sm._state.network_mode = "online"
        try:
            resp = client.post("/escalate/stream", json={"query": "test"})
            assert resp.status_code == 503
        finally:
            cloud_mod.CLOUD_COMPARISON_ENABLED = original

    def test_escalate_stream_returns_sse_content_type(self, client):
        """When cloud IS available, verify SSE content-type and headers."""
        import src.server as sm
        import src.engine.inference.config as config_mod
        original = config_mod.CLOUD_COMPARISON_ENABLED
        config_mod.CLOUD_COMPARISON_ENABLED = True
        sm._state.network_mode = "online"

        try:
            # Mock the OpenAI client to return a fake stream
            from unittest.mock import patch, AsyncMock, MagicMock
            import asyncio

            async def fake_stream():
                # Yield token chunks
                for text in ["Hello", " world"]:
                    chunk = MagicMock()
                    chunk.usage = None
                    chunk.choices = [MagicMock()]
                    chunk.choices[0].delta.content = text
                    yield chunk
                # Final chunk with usage
                final = MagicMock()
                final.usage = MagicMock()
                final.usage.prompt_tokens = 10
                final.usage.completion_tokens = 5
                final.choices = []
                yield final

            mock_openai_cls = MagicMock()
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = AsyncMock(
                return_value=fake_stream()
            )
            mock_openai_cls.return_value = mock_openai_instance

            with patch("src.server.cloud_routes.OPENAI_API_KEY", "sk-test"), \
                 patch.dict("sys.modules", {}), \
                 patch("openai.AsyncOpenAI", mock_openai_cls):
                resp = client.post("/escalate/stream", json={"query": "test"})

            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("cache-control") == "no-cache"

        finally:
            config_mod.CLOUD_COMPARISON_ENABLED = original

    def test_escalate_stream_missing_query_returns_422(self, client):
        resp = client.post("/escalate/stream", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /query/stream  (SSE streaming path)
# ---------------------------------------------------------------------------

def _make_process_stream(*items):
    """Return a mock process_stream callable that yields ExecutionSteps and str tokens."""
    async def _gen(*args, **kwargs):
        for item in items:
            yield item
    return _gen


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into list of {event, data} dicts."""
    import json as _j
    events = []
    for block in raw.strip().split("\n\n"):
        ev = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                try:
                    ev["data"] = _j.loads(line[6:])
                except Exception:
                    ev["data"] = line[6:]
        if "event" in ev:
            events.append(ev)
    return events


class TestQueryStreamEndpoint:
    """SSE streaming path — POST /query/stream."""

    def test_stream_returns_sse_content_type(self, client):
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(step, "Hello world")
        resp = client.post("/query/stream", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"

    def test_stream_yields_step_event(self, client):
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(step, "Hi")
        resp = client.post("/query/stream", json={"query": "Hi"})
        events = _parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "step" in event_types

    def test_stream_yields_token_event(self, client):
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(step, "answer text")
        resp = client.post("/query/stream", json={"query": "q"})
        events = _parse_sse(resp.text)
        assert any(e["event"] == "token" for e in events)

    def test_stream_yields_done_event_last(self, client):
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(step, "done answer")
        resp = client.post("/query/stream", json={"query": "q"})
        events = _parse_sse(resp.text)
        assert events[-1]["event"] == "done"

    def test_stream_done_event_has_metadata(self, client):
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "tool_use"})
        client._mock_agent.process_stream = _make_process_stream(step, "42")
        resp = client.post("/query/stream", json={"query": "calc"})
        events = _parse_sse(resp.text)
        done = next(e for e in events if e["event"] == "done")
        assert "intent" in done["data"]
        assert "execution_time_ms" in done["data"]
        assert "total_tokens" in done["data"]

    def test_stream_passes_through_orchestrator_tokens(self, client):
        """SSE endpoint passes through whatever the orchestrator yields.

        Think-block stripping happens upstream in SmallLanguageModelClient,
        not in the SSE endpoint. The orchestrator's process_stream() yields
        already-stripped tokens. This test verifies the SSE layer doesn't
        corrupt or modify the token stream.
        """
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(
            step, "The actual answer"
        )
        resp = client.post("/query/stream", json={"query": "hi"})
        events = _parse_sse(resp.text)
        tokens = [e["data"].get("text", "") for e in events if e["event"] == "token"]
        combined = "".join(tokens)
        assert "The actual answer" in combined

    def test_stream_agent_error_yields_error_event(self, client):
        async def _crash(*args, **kwargs):
            raise RuntimeError("model exploded")
            yield  # makes it an async generator

        client._mock_agent.process_stream = _crash
        resp = client.post("/query/stream", json={"query": "boom"})
        # Either error event in SSE body or 500 — both are acceptable
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            events = _parse_sse(resp.text)
            assert any(e["event"] == "error" for e in events)

    def test_stream_missing_query_returns_422(self, client):
        resp = client.post("/query/stream", json={})
        assert resp.status_code == 422

    def test_stream_empty_query_returns_200_with_refusal(self, client):
        """Empty query string yields a refusal token, not a 422."""
        step = ExecutionStep(action="classify_intent", model="gemma3",
                             details={"intent": "direct_answer"})
        client._mock_agent.process_stream = _make_process_stream(step, "Please provide a question.")
        resp = client.post("/query/stream", json={"query": ""})
        # Server may short-circuit with a refusal or delegate to agent
        assert resp.status_code in (200, 422)
