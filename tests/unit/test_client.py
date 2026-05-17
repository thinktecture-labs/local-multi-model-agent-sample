"""
Unit tests for SmallLanguageModelClient.

Mocks the underlying AsyncOpenAI SDK to test response parsing, message
construction, error handling, and health check logic — without any servers.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.engine.inference.client import CircuitBreaker, CircuitOpenError, SmallLanguageModelClient, SmallLanguageModelRole, LLMResponse


# ---------------------------------------------------------------------------
# Helpers to build mock OpenAI SDK responses
# ---------------------------------------------------------------------------

def _make_chat_response(content="test", model="gemma3", total_tokens=10, tool_calls=None):
    """Build a mock that looks like openai.types.chat.ChatCompletion."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.delta = MagicMock(content=None)

    usage = MagicMock()
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_embedding_response(embedding, model="embeddinggemma"):
    """Build a mock that looks like openai.types.CreateEmbeddingResponse."""
    item = MagicMock()
    item.embedding = embedding

    response = MagicMock()
    response.data = [item]
    return response


def _make_batch_embedding_response(embeddings):
    items = []
    for emb in embeddings:
        item = MagicMock()
        item.embedding = emb
        items.append(item)
    response = MagicMock()
    response.data = items
    return response


def _make_tool_call(name="calculator", arguments='{"expression": "1+1"}'):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class TestGenerate:
    async def test_returns_llm_response(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="rag_query", total_tokens=5)
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.generate(prompt="classify this")
        assert isinstance(result, LLMResponse)
        assert result.content == "rag_query"
        assert result.tokens_used == 5

    async def test_system_prompt_included(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test", system_prompt="You are helpful.")
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    async def test_no_system_prompt(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test")
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    async def test_json_mode_sets_response_format(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content='{"intent": "rag_query"}')
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test", json_mode=True)
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_temperature_and_max_tokens_passed(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response()
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="x", temperature=0.5, max_tokens=100)
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    async def test_none_content_returns_empty_string(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content=None)
        mock_resp.choices[0].message.content = None
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.generate(prompt="test")
        assert result.content == ""

    async def test_no_usage_returns_zero_tokens(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response()
        mock_resp.usage = None
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.generate(prompt="test")
        assert result.tokens_used == 0

    async def test_deterministic_mode_sets_greedy_params(self):
        """deterministic=True must set temp=0, seed=42, top_p=1, top_k=1."""
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test", temperature=0.5, deterministic=True)
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0, "deterministic must override temperature to 0"
        assert call_kwargs["seed"] == 42
        assert call_kwargs["top_p"] == 1.0
        assert call_kwargs["extra_body"] == {"top_k": 1}

    async def test_deterministic_false_does_not_set_seed(self):
        """Default (non-deterministic) must not set seed or top_k."""
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test")
        call_kwargs = client._inference.chat.completions.create.call_args[1]
        assert "seed" not in call_kwargs
        assert "extra_body" not in call_kwargs


# ---------------------------------------------------------------------------
# Generate vision
# ---------------------------------------------------------------------------

class TestGenerateVision:
    async def test_returns_llm_response(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="A cat sitting on a mat", total_tokens=15)
        client._vision.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.generate_vision(
            prompt="What is in this image?",
            images=["base64imagedata"],
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "A cat sitting on a mat"
        assert result.tokens_used == 15

    async def test_multipart_message_format(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._vision.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate_vision(
            prompt="Describe this",
            images=["img1base64", "img2base64"],
        )
        call_kwargs = client._vision.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"

        content_parts = user_msg["content"]
        # Two image_url parts followed by one text part
        assert len(content_parts) == 3
        assert content_parts[0]["type"] == "image_url"
        assert content_parts[1]["type"] == "image_url"
        assert content_parts[2]["type"] == "text"
        assert content_parts[2]["text"] == "Describe this"
        assert "base64,img1base64" in content_parts[0]["image_url"]["url"]
        assert "base64,img2base64" in content_parts[1]["image_url"]["url"]

    async def test_system_prompt_included(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._vision.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate_vision(
            prompt="Describe",
            images=["imgdata"],
            system_prompt="You are a visual assistant.",
        )
        call_kwargs = client._vision.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a visual assistant."
        assert messages[1]["role"] == "user"

    async def test_uses_vision_model(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="description")
        client._vision.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.generate_vision(
            prompt="What do you see?",
            images=["imgdata"],
        )
        assert result.model == client.models[SmallLanguageModelRole.VISION]


# ---------------------------------------------------------------------------
# Call function
# ---------------------------------------------------------------------------

class TestCallFunction:
    async def test_returns_function_call(self):
        client = SmallLanguageModelClient()
        tc = _make_tool_call(name="calculator", arguments='{"expression": "6*7"}')
        mock_resp = _make_chat_response(content="", tool_calls=[tc])
        client._function.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.call_function(
            messages=[{"role": "user", "content": "6 * 7"}],
            tools=[{"type": "function", "function": {"name": "calculator"}}],
        )
        assert result.function_call is not None
        assert result.function_call["name"] == "calculator"
        assert result.function_call["arguments"] == {"expression": "6*7"}

    async def test_no_tool_calls_returns_none(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="I don't know", tool_calls=None)
        mock_resp.choices[0].message.tool_calls = None
        client._function.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.call_function(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )
        assert result.function_call is None
        assert result.content == "I don't know"

    async def test_uses_function_model(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="", tool_calls=None)
        mock_resp.choices[0].message.tool_calls = None
        client._function.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.call_function(messages=[], tools=[])
        assert result.model == client.models[SmallLanguageModelRole.FUNCTION]

    async def test_deterministic_mode_sets_greedy_params(self):
        """call_function(deterministic=True) must set temp=0, seed=42, top_k=1."""
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="", tool_calls=None)
        mock_resp.choices[0].message.tool_calls = None
        client._function.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.call_function(messages=[], tools=[], deterministic=True)
        call_kwargs = client._function.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["seed"] == 42
        assert call_kwargs["top_p"] == 1.0
        assert call_kwargs["extra_body"] == {"top_k": 1}


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

class TestEmbed:
    async def test_returns_embedding_vector(self):
        client = SmallLanguageModelClient()
        expected = [0.1, 0.2, 0.3]
        mock_resp = _make_embedding_response(expected)
        client._embedding.embeddings.create = AsyncMock(return_value=mock_resp)

        result = await client.embed("test text")
        assert result == expected

    async def test_embed_batch_returns_list(self):
        client = SmallLanguageModelClient()
        expected = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_resp = _make_batch_embedding_response(expected)
        client._embedding.embeddings.create = AsyncMock(return_value=mock_resp)

        result = await client.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]

    async def test_embed_uses_embedding_model(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_embedding_response([0.1])
        client._embedding.embeddings.create = AsyncMock(return_value=mock_resp)

        await client.embed("test")
        call_kwargs = client._embedding.embeddings.create.call_args[1]
        assert call_kwargs["model"] == client.models[SmallLanguageModelRole.EMBEDDING]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestCheckHealth:
    async def test_all_healthy(self):
        client = SmallLanguageModelClient(
            inference_url="http://localhost:19090/v1",
            function_url="http://localhost:19091/v1",
            embedding_url="http://localhost:19092/v1",
            vision_url="http://localhost:19093/v1",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockHttpx:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockHttpx.return_value = mock_http

            health = await client.check_health()

        assert all(v is True for v in health.values())
        assert len(health) == 5  # 4 models + WHISPER

    async def test_one_unhealthy(self):
        client = SmallLanguageModelClient(
            inference_url="http://localhost:19090/v1",
            function_url="http://localhost:19091/v1",
            embedding_url="http://localhost:19092/v1",
            vision_url="http://localhost:19093/v1",
        )

        async def mock_get(url):
            resp = MagicMock()
            # Return 503 for the function port on every call (including retries)
            resp.status_code = 503 if "19091" in url else 200
            return resp

        with patch("httpx.AsyncClient") as MockHttpx:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=mock_get)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockHttpx.return_value = mock_http

            health = await client.check_health()

        assert not all(v is True for v in health.values())

    async def test_connection_error_marks_unhealthy(self):
        client = SmallLanguageModelClient(
            inference_url="http://localhost:19090/v1",
            function_url="http://localhost:19091/v1",
            embedding_url="http://localhost:19092/v1",
            vision_url="http://localhost:19093/v1",
        )

        with patch("httpx.AsyncClient") as MockHttpx:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockHttpx.return_value = mock_http

            health = await client.check_health()

        assert all(v is False for v in health.values())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_default_model_names(self):
        client = SmallLanguageModelClient()
        assert SmallLanguageModelRole.INFERENCE in client.models
        assert SmallLanguageModelRole.FUNCTION in client.models
        assert SmallLanguageModelRole.EMBEDDING in client.models

    def test_custom_model_names(self):
        client = SmallLanguageModelClient(
            inference_model="custom-inf",
            function_model="custom-fn",
            embedding_model="custom-emb",
        )
        assert client.models[SmallLanguageModelRole.INFERENCE] == "custom-inf"
        assert client.models[SmallLanguageModelRole.FUNCTION] == "custom-fn"
        assert client.models[SmallLanguageModelRole.EMBEDDING] == "custom-emb"

    def test_vision_model_in_models(self):
        client = SmallLanguageModelClient()
        assert SmallLanguageModelRole.VISION in client.models

    def test_custom_vision_url(self):
        client = SmallLanguageModelClient(
            vision_url="http://vision-host:9999/v1",
        )
        assert client._urls[SmallLanguageModelRole.VISION] == "http://vision-host:9999/v1"

    def test_custom_urls(self):
        client = SmallLanguageModelClient(
            inference_url="http://host1:8080/v1",
            function_url="http://host2:8081/v1",
            embedding_url="http://host3:8082/v1",
        )
        assert client._urls[SmallLanguageModelRole.INFERENCE] == "http://host1:8080/v1"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 2

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.failure_count == 3

    def test_check_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check("test-model")
        assert exc_info.value.model == "test-model"
        assert exc_info.value.failures == 1

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.check("test-model")  # should not raise

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_transitions_to_half_open_after_recovery(self):
        import time
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"

    def test_half_open_allows_check(self):
        import time
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.check("test-model")  # should not raise (half_open allows probe)

    def test_reset_clears_state(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_partial_failures_reset_on_success(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"


class TestCircuitBreakerIntegration:
    async def test_generate_records_success(self):
        client = SmallLanguageModelClient()
        mock_resp = _make_chat_response(content="ok")
        client._inference.chat.completions.create = AsyncMock(return_value=mock_resp)

        await client.generate(prompt="test")
        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "closed"
        assert client._breakers[SmallLanguageModelRole.INFERENCE].failure_count == 0

    async def test_generate_records_connection_failure(self):
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        with pytest.raises(openai.APIConnectionError):
            await client.generate(prompt="test")
        assert client._breakers[SmallLanguageModelRole.INFERENCE].failure_count == 1

    async def test_generate_opens_circuit_after_threshold(self):
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        for _ in range(3):
            with pytest.raises(openai.APIConnectionError):
                await client.generate(prompt="test")

        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "open"
        # Next call should fail fast with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await client.generate(prompt="test")

    async def test_call_function_circuit_breaker(self):
        import openai
        client = SmallLanguageModelClient()
        client._function.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        for _ in range(3):
            with pytest.raises(openai.APITimeoutError):
                await client.call_function(messages=[], tools=[])

        with pytest.raises(CircuitOpenError) as exc_info:
            await client.call_function(messages=[], tools=[])
        assert exc_info.value.model == client.models[SmallLanguageModelRole.FUNCTION]

    async def test_embed_circuit_breaker(self):
        import openai
        client = SmallLanguageModelClient()
        client._embedding.embeddings.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        for _ in range(3):
            with pytest.raises(openai.APIConnectionError):
                await client.embed("test")

        with pytest.raises(CircuitOpenError):
            await client.embed("test")

    async def test_http_error_does_not_trip_breaker(self):
        """A 400 error from a live server should NOT trip the circuit breaker."""
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="bad request", response=MagicMock(), body=None
            )
        )

        for _ in range(5):
            with pytest.raises(openai.BadRequestError):
                await client.generate(prompt="test")

        # Circuit should still be closed — BadRequestError is not a connection error
        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "closed"
        assert client._breakers[SmallLanguageModelRole.INFERENCE].failure_count == 0

    async def test_swap_urls_resets_breakers(self):
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        for _ in range(3):
            with pytest.raises(openai.APIConnectionError):
                await client.generate(prompt="test")
        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "open"

        client.swap_urls({})
        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "closed"

    def test_breaker_states_property(self):
        client = SmallLanguageModelClient()
        states = client.breaker_states
        assert set(states.keys()) == {"INFERENCE", "FUNCTION", "EMBEDDING", "VISION"}
        assert all(s == "closed" for s in states.values())

    async def test_vision_circuit_breaker(self):
        import openai
        client = SmallLanguageModelClient()
        client._vision.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )

        for _ in range(3):
            with pytest.raises(openai.APIConnectionError):
                await client.generate_vision(prompt="test", images=["base64data"])

        with pytest.raises(CircuitOpenError):
            await client.generate_vision(prompt="test", images=["base64data"])

    async def test_internal_server_error_trips_breaker(self):
        """HTTP 500 from llama-server (e.g. OOM) must trip the circuit breaker."""
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.InternalServerError(
                message="internal server error",
                response=MagicMock(status_code=500),
                body=None,
            )
        )

        for _ in range(3):
            with pytest.raises(openai.InternalServerError):
                await client.generate(prompt="test")

        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "open"
        with pytest.raises(CircuitOpenError):
            await client.generate(prompt="test")

    async def test_bad_request_does_not_trip_breaker_still(self):
        """Regression: 400 must still NOT trip the circuit breaker after adding 500."""
        import openai
        client = SmallLanguageModelClient()
        client._inference.chat.completions.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="bad request", response=MagicMock(), body=None
            )
        )

        for _ in range(5):
            with pytest.raises(openai.BadRequestError):
                await client.generate(prompt="test")

        assert client._breakers[SmallLanguageModelRole.INFERENCE].state == "closed"
        assert client._breakers[SmallLanguageModelRole.INFERENCE].failure_count == 0


# ---------------------------------------------------------------------------
# Qwen tool-caller native format parser
# (The `_parse_qwen_native` helper was retired when Qwen3.5-4B FT switched
# off the b8117-era <start_function_call> format; tool-call parsing now
# goes through the OpenAI tool-calls API path. The legacy tests for that
# helper were removed alongside the function.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Think-token stripping
# ---------------------------------------------------------------------------

class TestStripThinking:
    """Tests for _strip_thinking() — Qwen 3.5 thinking token removal."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.engine.inference.client import _strip_thinking
        self._strip = _strip_thinking

    def test_removes_full_think_block(self):
        result = self._strip("<think>I am reasoning here...</think>\nActual answer")
        assert "<think>" not in result
        assert "</think>" not in result
        assert "Actual answer" in result

    def test_removes_bare_closing_tag(self):
        result = self._strip("</think>\nActual answer")
        assert "</think>" not in result
        assert "Actual answer" in result

    def test_removes_multiple_think_blocks(self):
        result = self._strip("<think>step 1</think>\n<think>step 2</think>\nAnswer")
        assert "<think>" not in result
        assert "Answer" in result

    def test_leaves_clean_text_unchanged(self):
        text = "The Enterprise plan costs $3,500 per month."
        assert self._strip(text) == text

    def test_multiline_think_block(self):
        result = self._strip("<think>\nLine one\nLine two\n</think>\nFinal answer")
        assert "Line one" not in result
        assert "Final answer" in result

    def test_leading_newlines_stripped(self):
        result = self._strip("<think>reasoning</think>\n\n\nAnswer")
        assert not result.startswith("\n")
        assert "Answer" in result

    def test_bare_tag_only(self):
        result = self._strip("</think>")
        assert result.strip() == ""

    def test_empty_string(self):
        assert self._strip("") == ""
