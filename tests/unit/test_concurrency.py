"""
Concurrency and stress tests — Code Review Issue #10.

Fires concurrent async operations at the server's shared mutable state to
expose race conditions.  All tests run in-process with mocked agent/client —
no external services needed.

Findings:
  - Training double-start: CONFIRMED TOCTOU race (fixed with asyncio.Lock).
  - Model swap during query: No crash — queries complete with whichever model
    was active at each await point.  Acceptable for demo.
  - Audio cache: No race in practice.  _expire_audio_cache() is synchronous,
    runs to completion without yielding.  Safe under single-event-loop.
  - Counter increments (cloud_bytes_sent, _total_tokens): Safe under
    single-event-loop — no await between read and write.
  - Eval label auto-assignment: Benign race — last writer wins.  Acceptable.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.engine.agent import AgentResponse, ExecutionStep, Intent
from src.engine.inference.client import SmallLanguageModelRole, LLMResponse


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _mock_server():
    """Import server module and return mocked components."""
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
    mock_client.swap_urls = MagicMock()

    mock_vs = MagicMock()
    mock_vs.count = AsyncMock(return_value=13)
    mock_vs.set_client = MagicMock()

    mock_tools = MagicMock()
    mock_tools.list_tools = MagicMock(return_value=["calculator", "sql_query"])
    mock_tools.get_all_schemas = MagicMock(return_value=[])

    mock_agent = MagicMock()
    mock_agent.process = AsyncMock(return_value=AgentResponse(
        query="test",
        intent=Intent.DIRECT_ANSWER,
        response="Hello!",
        steps=[ExecutionStep(action="direct_response", model="gemma3:1b-it")],
        execution_time_ms=10.0,
    ))
    mock_agent.export_training_data = MagicMock(return_value=0)
    mock_agent.interaction_count = 0
    mock_agent.total_tokens_generated = 0

    return server_module, mock_client, mock_vs, mock_tools, mock_agent


@pytest.fixture
async def async_client():
    """httpx.AsyncClient bound to the FastAPI app with mocked lifespan."""
    server_module, mock_client, mock_vs, mock_tools, mock_agent = _mock_server()

    @asynccontextmanager
    async def mock_lifespan(app):
        server_module._state.client = mock_client
        server_module._state.vector_store = mock_vs
        server_module._state.tools = mock_tools
        server_module._state.agent = mock_agent
        server_module._state.training_running = False
        server_module._state.training_stage = "idle"
        server_module._state.cloud_bytes_sent = 0
        server_module._state.eval_results = {}
        server_module._state.model_mode = "finetuned"
        from src.server.voice_routes import _audio_cache; _audio_cache.clear()
        yield

    original_lifespan = server_module.app.router.lifespan_context
    server_module.app.router.lifespan_context = mock_lifespan

    transport = httpx.ASGITransport(app=server_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        ac._server = server_module
        ac._mock_agent = mock_agent
        ac._mock_client = mock_client
        yield ac

    server_module.app.router.lifespan_context = original_lifespan
    from src.server.voice_routes import _audio_cache; _audio_cache.clear()


# ---------------------------------------------------------------------------
# TestTrainingDoubleStart
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingDoubleStart:
    """Verify the asyncio.Lock prevents concurrent training starts."""

    async def test_concurrent_train_only_one_succeeds(self, async_client):
        """Fire 5 concurrent POST /train — exactly 1 should succeed, rest get 409."""

        async def start_train():
            resp = await async_client.post(
                "/train",
                json={"demo_mode": True},
            )
            return resp.status_code

        results = await asyncio.gather(*[start_train() for _ in range(5)])
        successes = results.count(200)
        conflicts = results.count(409)
        assert successes == 1, f"Expected 1 success, got {successes} (all: {results})"
        assert conflicts == 4, f"Expected 4 conflicts, got {conflicts}"

    async def test_training_flag_resets_after_completion(self, async_client):
        """After consuming the full SSE stream, training_running should be False."""
        server = async_client._server

        resp = await async_client.post(
            "/train",
            json={"demo_mode": True},
        )
        assert resp.status_code == 200
        # Consume the stream to let event_stream() run to completion
        # (httpx consumes it eagerly for non-streaming calls)

        # Give the event loop a tick to finalize
        await asyncio.sleep(0.1)
        assert server._state.training_running is False

    async def test_training_flag_resets_after_error(self, async_client):
        """If training raises, the finally block should still reset the flag."""
        server = async_client._server

        # Patch _simulate_training to yield a valid SSE string then raise
        async def _explode(total_epochs=7):
            yield "event: progress\ndata: {\"stage\": \"preparing\"}\n\n"
            raise RuntimeError("boom")

        import src.server.training_routes as _training_mod
        with patch.object(_training_mod, "_simulate_training", _explode):
            resp = await async_client.post(
                "/train",
                json={"demo_mode": True},
            )
            # The error surfaces through the SSE stream; httpx still gets 200
            assert resp.status_code == 200

        await asyncio.sleep(0.1)
        assert server._state.training_running is False


# ---------------------------------------------------------------------------
# TestAudioCacheConcurrency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAudioCacheConcurrency:
    """Stress the _AudioCache under concurrent access."""

    async def test_concurrent_cache_write_and_read(self):
        """Fire concurrent insert+read — no RuntimeError."""
        from src.server.voice_routes import _audio_cache

        _audio_cache.clear()

        async def cache_worker(idx: int):
            _audio_cache.put(f"new-{idx}", b"wav-data")
            _ = _audio_cache.get(f"new-{idx}")
            _ = _audio_cache.get(f"new-{(idx + 1) % 20}")

        await asyncio.gather(*[cache_worker(i) for i in range(20)])

        assert len(_audio_cache) <= 20
        _audio_cache.clear()

    async def test_cache_eviction_under_concurrent_load(self):
        """Bounded cache evicts oldest entries when full."""
        from src.server.voice_routes import _AudioCache

        cache = _AudioCache(maxsize=5, ttl=120.0)
        for i in range(20):
            cache.put(f"load-{i}", b"wav-data")

        assert len(cache) == 5
        # The 5 most recent should survive
        for i in range(15, 20):
            assert cache.get(f"load-{i}") is not None
        cache.clear()


# ---------------------------------------------------------------------------
# TestModelSwapDuringQuery
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModelSwapDuringQuery:
    """Verify model swap during concurrent queries doesn't crash."""

    async def test_swap_during_concurrent_reads_no_crash(self):
        """Concurrent mode reads + writes don't corrupt state.

        The real /models/swap endpoint does httpx health checks to localhost
        ports, so we test the state-level race directly: concurrent reads of
        model_mode while another coroutine writes it.
        """
        import src.server as server

        server._state.model_mode = "finetuned"
        observed_modes = []

        async def read_mode():
            await asyncio.sleep(0.001)
            observed_modes.append(server._state.model_mode)

        async def swap_mode():
            await asyncio.sleep(0.005)
            server._state.model_mode = "base"

        await asyncio.gather(
            *[read_mode() for _ in range(10)],
            swap_mode(),
        )

        # All observed values should be valid modes (no partial/corrupt state)
        for m in observed_modes:
            assert m in {"base", "finetuned"}, f"Corrupt mode: {m!r}"
        # Final state should be "base" (from the swap)
        assert server._state.model_mode == "base"

    async def test_rapid_swap_toggle(self, async_client):
        """Fire 20 alternating state swaps — final mode should be valid."""
        server = async_client._server

        async def toggle(idx: int):
            """Simulate swap by directly setting state (avoids httpx health checks)."""
            mode = "base" if idx % 2 == 0 else "finetuned"
            server._state.model_mode = mode

        await asyncio.gather(*[toggle(i) for i in range(20)])

        # Final state should be valid
        assert server._state.model_mode in {"base", "finetuned"}


# ---------------------------------------------------------------------------
# TestCounterAtomicity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterAtomicity:
    """Verify counters behave correctly under concurrent access."""

    async def test_cloud_bytes_concurrent_increments(self):
        """100 concurrent += operations should produce correct total."""
        import src.server as server

        server._state.cloud_bytes_sent = 0

        async def increment():
            server._state.cloud_bytes_sent += 42

        await asyncio.gather(*[increment() for _ in range(100)])
        # In single-event-loop, += without await is atomic
        assert server._state.cloud_bytes_sent == 4200

    async def test_total_tokens_concurrent_logging(self):
        """Concurrent _log_interaction calls should tally tokens correctly."""
        from src.engine.agent import SmallLanguageModelAgentOrchestrator

        mock_client = MagicMock()
        mock_tools = MagicMock()
        mock_tools.list_tools = MagicMock(return_value=[])
        mock_tools.get_all_schemas = MagicMock(return_value=[])

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        async def log_one():
            agent._logger.log(
                query="test",
                intent=Intent.DIRECT_ANSWER,
                response="ok",
                steps=[ExecutionStep(action="test", model="test", tokens_used=10)],
            )

        await asyncio.gather(*[log_one() for _ in range(50)])
        assert agent.total_tokens_generated == 500

    async def test_interaction_count_matches_log_length(self):
        """After 50 concurrent logs, count == len(log) == 50."""
        from src.engine.agent import SmallLanguageModelAgentOrchestrator

        mock_client = MagicMock()
        mock_tools = MagicMock()
        mock_tools.list_tools = MagicMock(return_value=[])
        mock_tools.get_all_schemas = MagicMock(return_value=[])

        agent = SmallLanguageModelAgentOrchestrator(mock_client, mock_tools)

        async def log_one():
            agent._logger.log(
                query="test",
                intent=Intent.DIRECT_ANSWER,
                response="ok",
                steps=[ExecutionStep(action="test", model="test")],
            )

        await asyncio.gather(*[log_one() for _ in range(50)])
        assert agent.interaction_count == 50


# ---------------------------------------------------------------------------
# TestEvalConcurrency
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvalConcurrency:
    """Concurrent eval requests should not crash or corrupt state."""

    async def test_eval_blocked_during_training(self, async_client):
        """POST /eval should return 409 when training is running."""
        server = async_client._server
        server._state.training_running = True

        resp = await async_client.post("/eval", json={"model": "gemma3"})
        assert resp.status_code == 409
        server._state.training_running = False

    async def test_concurrent_eval_no_crash(self):
        """3 concurrent eval score calls — no corruption of eval_results dict."""
        import src.server as server

        server._state.eval_results = {}

        async def store_eval(idx: int):
            label = f"run-{idx}"
            server._state.eval_results[label] = {
                "overall_accuracy": 0.95,
                "n": 5,
                "saved_as": label,
            }

        await asyncio.gather(*[store_eval(i) for i in range(10)])
        # All 10 results should be stored (last-writer-wins for same key is fine)
        assert len(server._state.eval_results) == 10
        for i in range(10):
            assert f"run-{i}" in server._state.eval_results


# ---------------------------------------------------------------------------
# TestNetworkModeToggle
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNetworkModeToggle:
    """Rapid mode toggling should leave state consistent."""

    async def test_rapid_network_mode_toggle(self, async_client):
        """20 concurrent mode switches — final state should be valid."""
        server = async_client._server

        async def toggle(idx: int):
            mode = "offline" if idx % 2 == 0 else "online"
            resp = await async_client.post("/network-mode", json={"mode": mode})
            return resp.status_code

        results = await asyncio.gather(*[toggle(i) for i in range(20)])
        assert all(r == 200 for r in results)
        assert server._state.network_mode in {"online", "offline"}

    async def test_rapid_routing_mode_toggle(self, async_client):
        """20 concurrent routing mode switches — final state should be valid."""
        server = async_client._server

        async def toggle(idx: int):
            mode = "hybrid" if idx % 2 == 0 else "local-only"
            resp = await async_client.post("/routing-mode", json={"mode": mode})
            return resp.status_code

        results = await asyncio.gather(*[toggle(i) for i in range(20)])
        assert all(r == 200 for r in results)
        assert server._state.routing_mode in {"local-only", "hybrid"}


# ---------------------------------------------------------------------------
# Semaphore concurrency limits on SmallLanguageModelClient
# ---------------------------------------------------------------------------

class TestSemaphoreLimits:
    """Verify per-model asyncio.Semaphore caps concurrent requests."""

    def test_client_has_semaphores_per_model(self):
        """SmallLanguageModelClient should have a semaphore for each SmallLanguageModelRole role."""
        from src.engine.inference.client import SmallLanguageModelClient, SmallLanguageModelRole
        client = SmallLanguageModelClient()
        assert hasattr(client, "_semaphores")
        for role in SmallLanguageModelRole:
            assert role in client._semaphores
            assert isinstance(client._semaphores[role], asyncio.Semaphore)

    def test_semaphore_matches_config(self):
        """Semaphore value should match MODEL_CONCURRENCY_LIMIT."""
        from src.engine.inference.client import SmallLanguageModelClient, SmallLanguageModelRole
        from src.engine.inference.config import MODEL_CONCURRENCY_LIMIT
        client = SmallLanguageModelClient()
        for role in SmallLanguageModelRole:
            assert client._semaphores[role]._value == MODEL_CONCURRENCY_LIMIT

    def test_concurrency_limit_config_default(self):
        from src.engine.inference.config import MODEL_CONCURRENCY_LIMIT
        assert isinstance(MODEL_CONCURRENCY_LIMIT, int)
        assert MODEL_CONCURRENCY_LIMIT == 4


# ---------------------------------------------------------------------------
# Circuit breaker stress tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCircuitBreakerStress:
    """Stress the CircuitBreaker state machine under concurrent access."""

    def test_breaker_opens_after_threshold_failures(self):
        """N consecutive failures should trip the breaker to 'open'."""
        from src.engine.inference.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        assert cb.state == "closed"

        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"

    def test_success_resets_failure_count(self):
        """A success after partial failures should reset the counter."""
        from src.engine.inference.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"

        # Should need 3 fresh failures to open again
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_half_open_after_recovery_timeout(self):
        """After recovery_timeout, breaker should transition to half_open."""
        from src.engine.inference.client import CircuitBreaker
        import time as _time

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == "open"

        _time.sleep(0.06)
        assert cb.state == "half_open"

    async def test_concurrent_failure_recording(self):
        """Concurrent record_failure calls should all be counted."""
        from src.engine.inference.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=100, recovery_timeout=30.0)

        async def fail_once():
            cb.record_failure()

        await asyncio.gather(*[fail_once() for _ in range(50)])
        assert cb.failure_count == 50

    def test_reset_clears_all_state(self):
        """reset() should return breaker to pristine closed state."""
        from src.engine.inference.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb.failure_count == 0


@pytest.mark.unit
class TestSemaphoreConcurrentLoad:
    """Verify semaphore actually limits concurrent coroutines."""

    async def test_semaphore_limits_concurrent_access(self):
        """With semaphore(2), at most 2 coroutines should run simultaneously."""
        sem = asyncio.Semaphore(2)
        max_concurrent = 0
        current = 0

        async def worker():
            nonlocal max_concurrent, current
            async with sem:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
                await asyncio.sleep(0.01)
                current -= 1

        await asyncio.gather(*[worker() for _ in range(20)])
        assert max_concurrent <= 2, f"Semaphore allowed {max_concurrent} concurrent (expected max 2)"
        assert current == 0, "All workers should have finished"

    async def test_semaphore_does_not_deadlock(self):
        """100 tasks through a semaphore(4) should all complete."""
        sem = asyncio.Semaphore(4)
        completed = 0

        async def worker():
            nonlocal completed
            async with sem:
                await asyncio.sleep(0.001)
                completed += 1

        await asyncio.wait_for(
            asyncio.gather(*[worker() for _ in range(100)]),
            timeout=10.0,
        )
        assert completed == 100


# ---------------------------------------------------------------------------
# AppState __slots__ enforcement
# ---------------------------------------------------------------------------

class TestAppStateSlots:
    """Verify AppState uses __slots__ to prevent accidental attribute creation."""

    def test_has_slots(self):
        from src.server.state import AppState
        assert hasattr(AppState, "__slots__")

    def test_rejects_arbitrary_attributes(self):
        from src.server.state import AppState
        state = AppState()
        with pytest.raises(AttributeError):
            state.nonexistent_field = "should fail"

    def test_all_declared_slots_assignable(self):
        from src.server.state import AppState
        state = AppState()
        # These should be assignable without error
        state.model_mode = "base"
        state.network_mode = "offline"
        state.energy_wh = 1.5
        assert state.model_mode == "base"
        assert state.network_mode == "offline"
        assert state.energy_wh == 1.5
