"""
SmallLanguageModelClient — Unified interface to all local SLM models via llama-server.

Each model runs as an independent llama-server instance (OpenAI-compatible).
Ports and model names are configured via .env (or .env.local for overrides):

  Port 9090 — inference model  (intent classification, synthesis)
  Port 9091 — function model   (tool selection, parameter extraction)
  Port 9092 — embedding model  (semantic search, retrieval)
  Port 9093 — vision model     (multimodal image understanding)

Start all servers:
  bash scripts/start_servers.sh          # base model
  bash scripts/start_servers.sh --ft     # fine-tuned model

Key insight: llama-server exposes an OpenAI-compatible API, meaning the
standard OpenAI Python SDK works unchanged against fully local models.
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

import httpx
import openai
from openai import AsyncOpenAI

from .config import (
    AGENT_TIMEOUT,
    CIRCUIT_BREAKER_RECOVERY,
    CIRCUIT_BREAKER_THRESHOLD,
    MODEL_CONCURRENCY_LIMIT,
    N_KEEP_DIRECT_ANSWER,
    N_KEEP_RAG_SYNTHESIS,
    N_KEEP_VISION,
    VISION_MAX_TOKENS,
    VISION_TEMPERATURE,
    WHISPER_URL,
)


# Errors that trip the circuit breaker — connection failures and server-side
# 5xx responses (llama-server returns HTTP 500 on OOM conditions).
# Client errors (4xx) are NOT included: a 400 from a live server is a caller
# bug, not a server-down event.
_BREAKER_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,  # HTTP 500 — e.g. llama-server OOM
)


class CircuitOpenError(Exception):
    """Raised when a model server's circuit breaker is open (server presumed down)."""

    def __init__(self, model: str, failures: int, retry_after: float):
        self.model = model
        self.failures = failures
        self.retry_after = retry_after
        super().__init__(
            f"Circuit open for {model}: {failures} consecutive failures, "
            f"retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Per-model circuit breaker — fail fast when a llama-server instance is down.

    States:
      closed    — normal operation, requests pass through
      open      — server presumed down, requests fail immediately
      half_open — recovery window expired, next request is a test probe

    Connection errors, timeouts, and HTTP 5xx responses trip the breaker.
    HTTP 4xx errors (client bugs against a live server) do not.
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"

    @property
    def state(self) -> str:
        if (
            self._state == "open"
            and time.monotonic() - self._last_failure_time >= self._recovery_timeout
        ):
            self._state = "half_open"
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self._failure_threshold:
            self._state = "open"

    def reset(self) -> None:
        """Reset to closed state (e.g. after a model swap)."""
        self._failures = 0
        self._state = "closed"
        self._last_failure_time = 0.0

    def check(self, model: str) -> None:
        """Raise CircuitOpenError if the circuit is open."""
        state = self.state
        if state == "open":
            retry_after = self._recovery_timeout - (
                time.monotonic() - self._last_failure_time
            )
            raise CircuitOpenError(model, self._failures, max(0.0, retry_after))


class SmallLanguageModelRole(Enum):
    """
    Model roles in the multi-model agent — each optimized for a specific task.

    Rather than one monolithic model doing everything (and doing it
    mediocrely), we assign each cognitive task to a specialized small model.

    These values are canonical role identifiers (stable strings used as dict
    keys). Actual model names are resolved from env vars in the client
    constructor — not here — so that importing this module has no side effects
    and tests can override names without monkeypatching the enum.
    """
    INFERENCE  = "gemma3"           # intent classification, synthesis
    FUNCTION   = "qwen"              # tool selection, argument extraction (Qwen 3.5-4B fine-tuned)
    EMBEDDING  = "embeddinggemma"   # semantic search, retrieval
    VISION     = "gemma3-4b-vision" # multimodal (4B + mmproj); also RAG synthesis


def _url(port_env: str, default_port: int) -> str:
    port = int(os.getenv(port_env, default_port))
    return f"http://localhost:{port}/v1"


# llama-server endpoints — one server per model (configured via .env)
_DEFAULT_URLS = {
    SmallLanguageModelRole.INFERENCE: _url("INFERENCE_PORT", 9090),
    SmallLanguageModelRole.FUNCTION:  _url("FUNCTION_PORT",  9091),
    SmallLanguageModelRole.EMBEDDING: _url("EMBEDDING_PORT", 9092),
    SmallLanguageModelRole.VISION:    _url("VISION_PORT",    9093),
}



_THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)
_THINK_BARE_PATTERN = re.compile(r"</think>\s*")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and bare </think> tags from model output.

    Qwen 3.5 leaks thinking tokens even with enable_thinking=false:
    sometimes full <think>...</think> blocks, sometimes just bare </think> tags.
    """
    text = _THINK_BLOCK_PATTERN.sub("", text)
    text = _THINK_BARE_PATTERN.sub("", text)
    return text.lstrip("\n")


async def _stream_with_think_filter(
    stream,
    breaker: "CircuitBreaker",
) -> "AsyncIterator[StreamChunk]":
    """Drain a raw completions stream, stripping think blocks and recording breaker state.

    Handles the three-phase think-block pattern shared by all streaming methods:
    1. Buffer tokens until we know whether output starts with <think>.
    2. Once a full <think>...</think> block is seen, strip it and emit the remainder.
    3. If no think block is present, emit the buffered prefix immediately.

    Yields StreamChunk(text=...) for content, then a final StreamChunk(done=True)
    carrying token-usage counts.  The circuit breaker is recorded as successful on
    the first non-empty chunk, so callers only need to handle record_failure() in
    the except block before the generator is entered.
    """
    got_first_chunk = False
    prompt_tokens = 0
    completion_tokens = 0
    think_buffer = ""
    think_done = False

    async for chunk in stream:
        if not got_first_chunk:
            breaker.record_success()
            got_first_chunk = True

        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                if think_done:
                    yield StreamChunk(text=delta)
                else:
                    think_buffer += delta
                    stripped = think_buffer.lstrip()
                    if not stripped.startswith("<think") and not stripped.startswith("</think"):
                        think_done = True
                        yield StreamChunk(text=think_buffer)
                        think_buffer = ""
                    elif "</think>" in think_buffer:
                        think_done = True
                        remainder = _strip_thinking(think_buffer)
                        if remainder:
                            yield StreamChunk(text=remainder)
                        think_buffer = ""

    if think_buffer:
        cleaned = _strip_thinking(think_buffer)
        if cleaned:
            yield StreamChunk(text=cleaned)

    yield StreamChunk(
        done=True,
        tokens_used=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@dataclass
class LLMResponse:
    """Standardized, type-safe response from any local SLM."""
    content: str
    model: str
    tokens_used: int
    function_call: Optional[dict] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    text: str = ""
    done: bool = False
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SmallLanguageModelClient:
    """
    Unified async client for local small language models via llama-server.

    Design decisions:
    - OpenAI-compatible API  → familiar SDK, easy cloud/local switching
    - Async throughout       → non-blocking, production-grade performance
    - Per-model endpoints    → each model has its own llama-server instance
    - Configurable endpoints → swap ports without touching call sites
    """

    def __init__(
        self,
        inference_url: str = _DEFAULT_URLS[SmallLanguageModelRole.INFERENCE],
        function_url:  str = _DEFAULT_URLS[SmallLanguageModelRole.FUNCTION],
        embedding_url: str = _DEFAULT_URLS[SmallLanguageModelRole.EMBEDDING],
        vision_url:    str = _DEFAULT_URLS[SmallLanguageModelRole.VISION],
        whisper_url:   str = WHISPER_URL,
        # Actual model names are resolved here from env vars (not in the enum),
        # so importing this module has no side effects.
        inference_model: str = os.getenv("INFERENCE_MODEL", SmallLanguageModelRole.INFERENCE.value),
        function_model:  str = os.getenv("FUNCTION_MODEL",  SmallLanguageModelRole.FUNCTION.value),
        embedding_model: str = os.getenv("EMBEDDING_MODEL", SmallLanguageModelRole.EMBEDDING.value),
        vision_model:    str = os.getenv("VISION_MODEL",    SmallLanguageModelRole.VISION.value),
        timeout: float = AGENT_TIMEOUT,
    ):
        self._timeout   = timeout
        self._inference = AsyncOpenAI(base_url=inference_url, api_key="no-key", timeout=timeout)
        self._function  = AsyncOpenAI(base_url=function_url,  api_key="no-key", timeout=timeout)
        self._embedding = AsyncOpenAI(base_url=embedding_url, api_key="no-key", timeout=timeout)
        self._vision    = AsyncOpenAI(base_url=vision_url,    api_key="no-key", timeout=timeout)
        self._whisper_url = whisper_url
        self.models = {
            SmallLanguageModelRole.INFERENCE: inference_model,
            SmallLanguageModelRole.FUNCTION:  function_model,
            SmallLanguageModelRole.EMBEDDING: embedding_model,
            SmallLanguageModelRole.VISION:    vision_model,
        }
        self._urls = {
            SmallLanguageModelRole.INFERENCE: inference_url,
            SmallLanguageModelRole.FUNCTION:  function_url,
            SmallLanguageModelRole.EMBEDDING: embedding_url,
            SmallLanguageModelRole.VISION:    vision_url,
        }
        self._breakers = {
            role: CircuitBreaker() for role in SmallLanguageModelRole
        }
        self._semaphores = {
            role: asyncio.Semaphore(MODEL_CONCURRENCY_LIMIT) for role in SmallLanguageModelRole
        }
        # Separate semaphore for text-only 4B calls (RAG synthesis) so they
        # don't starve vision (image) queries that share the same model server.
        # Reserves at least 1 slot for vision by capping text-only concurrency.
        self._vision_text_semaphore = asyncio.Semaphore(
            max(1, MODEL_CONCURRENCY_LIMIT - 1)
        )

    @property
    def urls(self) -> dict[SmallLanguageModelRole, str]:
        """Base URLs for each model role (read-only)."""
        return dict(self._urls)

    # ------------------------------------------------------------------
    # Factory — auto-detect FT servers
    # ------------------------------------------------------------------

    @classmethod
    def create_with_auto_detection(cls) -> "SmallLanguageModelClient":
        """Create a SmallLanguageModelClient, preferring FT servers (9094-9096) if available.

        Probes FT ports via /health. Falls back to base ports (9090-9092).
        """
        ft_ports = {
            "inference": int(os.getenv("INFERENCE_PORT_FT", 9094)),
            "function":  int(os.getenv("FUNCTION_PORT_FT",  9095)),
            "embedding": int(os.getenv("EMBEDDING_PORT_FT", 9096)),
        }
        use_ft = True
        for port in ft_ports.values():
            try:
                resp = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if resp.status_code != 200:
                    use_ft = False
                    break
            except Exception:
                use_ft = False
                break

        if use_ft:
            return cls(
                inference_url=f"http://localhost:{ft_ports['inference']}/v1",
                function_url=f"http://localhost:{ft_ports['function']}/v1",
                embedding_url=f"http://localhost:{ft_ports['embedding']}/v1",
            )
        return cls()

    # ------------------------------------------------------------------
    # Circuit breaker state
    # ------------------------------------------------------------------

    @property
    def breaker_states(self) -> dict[str, str]:
        """Current circuit breaker state per model role."""
        return {role.name: self._breakers[role].state for role in SmallLanguageModelRole}

    # ------------------------------------------------------------------
    # Dual-port swap (zero-downtime model switching)
    # ------------------------------------------------------------------

    def swap_urls(self, urls: dict[str, str]) -> None:
        """Swap model URLs for dual-port zero-downtime switching."""
        self._urls = {
            SmallLanguageModelRole.INFERENCE: urls.get("inference", self._urls[SmallLanguageModelRole.INFERENCE]),
            SmallLanguageModelRole.FUNCTION:  urls.get("function",  self._urls[SmallLanguageModelRole.FUNCTION]),
            SmallLanguageModelRole.EMBEDDING: urls.get("embedding", self._urls[SmallLanguageModelRole.EMBEDDING]),
            SmallLanguageModelRole.VISION:    urls.get("vision",    self._urls[SmallLanguageModelRole.VISION]),
        }
        common = dict(api_key="no-key", timeout=self._timeout)
        self._inference = AsyncOpenAI(base_url=self._urls[SmallLanguageModelRole.INFERENCE], **common)
        self._function  = AsyncOpenAI(base_url=self._urls[SmallLanguageModelRole.FUNCTION],  **common)
        self._embedding = AsyncOpenAI(base_url=self._urls[SmallLanguageModelRole.EMBEDDING], **common)
        self._vision    = AsyncOpenAI(base_url=self._urls[SmallLanguageModelRole.VISION],    **common)
        # Reset breakers — new servers get a fresh slate
        for breaker in self._breakers.values():
            breaker.reset()

    # ------------------------------------------------------------------
    # Text generation  (inference model)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = False,
        deterministic: bool = False,
        role: SmallLanguageModelRole = SmallLanguageModelRole.INFERENCE,
    ) -> LLMResponse:
        """
        Generate text with the model assigned to `role`.

        Default role is INFERENCE (gemma3-ft on :9090). Pass role=FUNCTION
        to route through the Qwen tool-calling model on :9091 — useful for
        sub-tasks that need stronger reasoning over structured context
        (e.g. concretize_step substituting SQL-result values into a
        decomposed step). FUNCTION-role calls still go through plain chat
        completions (no tools); Qwen handles plain prompts fine.

        Best for: intent classification, query rewriting, response synthesis,
        general question answering.

        temperature=0.1 by default for consistent, focused outputs.
        json_mode=True forces structured JSON output (great for classification).
        deterministic=True forces greedy decoding: temp=0, seed=42, top_k=1, top_p=1.
          See: https://www.linkedin.com/pulse/achieving-determinism-local-slm-llm-deployments-using-christian-weyer-quoxe/
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model":       self.models[role],
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if deterministic:
            kwargs["temperature"] = 0.0
            kwargs["seed"] = 42
            kwargs["top_p"] = 1.0
            kwargs["extra_body"] = {"top_k": 1}
        if system_prompt:
            # Protect the system-prompt tokens from context-window eviction.
            # n_keep pins the first N tokens so cache stays warm across turns.
            kwargs.setdefault("extra_body", {})["n_keep"] = N_KEEP_DIRECT_ANSWER
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        openai_client = self._function if role == SmallLanguageModelRole.FUNCTION else self._inference

        async with self._semaphores[role]:
            breaker = self._breakers[role]
            breaker.check(self.models[role])
            try:
                response = await openai_client.chat.completions.create(**kwargs)
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise
        return LLMResponse(
            content=_strip_thinking(response.choices[0].message.content or ""),
            model=self.models[role],
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        deterministic: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response tokens from the inference model.

        Holds the semaphore for the full streaming duration to prevent
        overloading llama-server. Yields StreamChunk objects; the final
        chunk has done=True with token usage.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.models[SmallLanguageModelRole.INFERENCE],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if deterministic:
            kwargs["temperature"] = 0.0
            kwargs["seed"] = 42
            kwargs["top_p"] = 1.0
            kwargs["extra_body"] = {"top_k": 1}
        if system_prompt:
            kwargs.setdefault("extra_body", {})["n_keep"] = N_KEEP_DIRECT_ANSWER

        async with self._semaphores[SmallLanguageModelRole.INFERENCE]:
            breaker = self._breakers[SmallLanguageModelRole.INFERENCE]
            breaker.check(self.models[SmallLanguageModelRole.INFERENCE])
            try:
                stream = await self._inference.chat.completions.create(**kwargs)
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise

            async for chunk in _stream_with_think_filter(stream, breaker):
                yield chunk

    async def generate_synthesis_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream synthesis tokens from the vision/synthesis model (4B).

        Same as generate_synthesis() but streaming. Uses the vision_text
        semaphore to prevent starving image analysis requests.
        """
        model_name = self.models[SmallLanguageModelRole.VISION]

        # Qwen 3.5 needs higher temperature for non-thinking mode
        if "qwen" in model_name.lower() and temperature < 0.5:
            temperature = 0.7

        async with self._vision_text_semaphore:
            breaker = self._breakers[SmallLanguageModelRole.VISION]
            breaker.check(model_name)
            try:
                stream = await self._vision.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"n_keep": N_KEEP_RAG_SYNTHESIS},
                )
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise

            async for chunk in _stream_with_think_filter(stream, breaker):
                yield chunk

    # ------------------------------------------------------------------
    # Function / tool calling  (function model)
    # ------------------------------------------------------------------

    async def call_function(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        deterministic: bool = False,
        recommended_sampling: bool = False,
    ) -> LLMResponse:
        """
        Select and parameterize tools using the function-calling specialist.

        Three modes (mutually exclusive — `deterministic` wins if both flags set):

          deterministic=True   — true greedy: temp=0, seed=42, top_p=1, top_k=1.
                                 Used in production by ToolUseHandler — same
                                 query → byte-identical output, every time.

          recommended_sampling=True — Qwen 3.5 *documented* recommendation for
                                 instruct mode: temp=0.7, top_p=0.95, top_k=20,
                                 no seed. Used by eval_tool_routing to quantify
                                 the accuracy cost of greedy vs Qwen-recommended
                                 sampling on the FT model.

          default              — temp=`temperature` (caller's value), or for Qwen
                                 the config-overridden QWEN_FUNCTION_TEMPERATURE
                                 (0.0 in shipping config). Effectively greedy
                                 without the seed — what most callers want.

        Thinking is disabled server-side via --reasoning-budget 0.
        """
        model_name = self.models[SmallLanguageModelRole.FUNCTION]
        is_qwen = "qwen" in model_name.lower()

        # Qwen 3.5 recommended params (from HuggingFace model cards)
        if is_qwen and not deterministic and not recommended_sampling:
            from .config import (
                QWEN_FUNCTION_TEMPERATURE,
                QWEN_FUNCTION_TOP_P,
                QWEN_FUNCTION_TOP_K,
                QWEN_FUNCTION_PRESENCE_PENALTY,
            )
            temperature = QWEN_FUNCTION_TEMPERATURE

        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": 200,  # Prevent runaway generation on slow backends (MPS)
        }

        if is_qwen and not deterministic and not recommended_sampling:
            kwargs["top_p"] = QWEN_FUNCTION_TOP_P
            kwargs["extra_body"] = {"top_k": QWEN_FUNCTION_TOP_K}
            if QWEN_FUNCTION_PRESENCE_PENALTY > 0:
                kwargs["presence_penalty"] = QWEN_FUNCTION_PRESENCE_PENALTY

        if recommended_sampling and not deterministic:
            # Qwen's published recommendation for instruct (non-thinking) mode.
            # Differs from `default` only in temperature (0.7 vs 0.0) — surfaces
            # the stochasticity the FT model was trained against.
            kwargs["temperature"] = 0.7
            kwargs["top_p"] = 0.95
            kwargs["extra_body"] = {"top_k": 20}

        if deterministic:
            kwargs["temperature"] = 0.0
            kwargs["seed"] = 42
            kwargs["top_p"] = 1.0
            kwargs["extra_body"] = {"top_k": 1}

        async with self._semaphores[SmallLanguageModelRole.FUNCTION]:
            breaker = self._breakers[SmallLanguageModelRole.FUNCTION]
            breaker.check(self.models[SmallLanguageModelRole.FUNCTION])
            try:
                response = await self._function.chat.completions.create(**kwargs)
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise

        message = response.choices[0].message
        function_call: Optional[dict] = None
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": tc.function.arguments}
            function_call = {
                "name":      tc.function.name,
                "arguments": args,
            }
        return LLMResponse(
            content=message.content or "",
            model=self.models[SmallLanguageModelRole.FUNCTION],
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            function_call=function_call,
        )

    # ------------------------------------------------------------------
    # Vision / multimodal  (vision model)
    # ------------------------------------------------------------------

    async def generate_vision(
        self,
        prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = VISION_TEMPERATURE,
        max_tokens: int = VISION_MAX_TOKENS,
    ) -> LLMResponse:
        """
        Analyse images with the vision model.

        images: list of base64-encoded image data (PNG/JPEG).
        Constructs OpenAI-compatible multipart content with image_url entries.
        """
        content: list[dict] = []
        for b64 in images:
            # Sniff MIME type from base64 magic bytes
            try:
                header = base64.b64decode(b64[:12])
                if header[:2] == b"\xff\xd8":
                    mime = "image/jpeg"
                elif header[:4] == b"\x89PNG":
                    mime = "image/png"
                else:
                    mime = "image/png"
            except Exception:
                mime = "image/png"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        async with self._semaphores[SmallLanguageModelRole.VISION]:
            breaker = self._breakers[SmallLanguageModelRole.VISION]
            breaker.check(self.models[SmallLanguageModelRole.VISION])
            extra: dict = {}
            if system_prompt:
                extra["n_keep"] = N_KEEP_VISION
            try:
                response = await self._vision.chat.completions.create(
                    model=self.models[SmallLanguageModelRole.VISION],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **({"extra_body": extra} if extra else {}),
                )
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise
        return LLMResponse(
            content=_strip_thinking(response.choices[0].message.content or ""),
            model=self.models[SmallLanguageModelRole.VISION],
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    async def generate_synthesis(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> LLMResponse:
        """
        Text synthesis using the highest-capacity model.

        In multi-model mode this is gemma3-4B-vision; in single-model mode
        (Qwen) it's the same model used for everything.

        Uses a separate semaphore from generate_vision() to prevent synthesis
        calls from starving image analysis requests.
        """
        async with self._vision_text_semaphore:
            breaker = self._breakers[SmallLanguageModelRole.VISION]
            model_name = self.models[SmallLanguageModelRole.VISION]
            breaker.check(model_name)

            # Qwen 3.5 produces terse output with low temperature (near-greedy).
            # The Qwen team recommends temp=0.7 for non-thinking mode.
            if "qwen" in model_name.lower() and temperature < 0.5:
                temperature = 0.7

            try:
                response = await self._vision.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"n_keep": N_KEEP_RAG_SYNTHESIS},
                )
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise
        return LLMResponse(
            content=_strip_thinking(response.choices[0].message.content or ""),
            model=self.models[SmallLanguageModelRole.VISION],
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    # ------------------------------------------------------------------
    # Embeddings  (embedding model)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """
        Generate a semantic embedding vector for a single text.

        Best for: query-time retrieval, similarity comparisons.
        """
        async with self._semaphores[SmallLanguageModelRole.EMBEDDING]:
            breaker = self._breakers[SmallLanguageModelRole.EMBEDDING]
            breaker.check(self.models[SmallLanguageModelRole.EMBEDDING])
            try:
                response = await self._embedding.embeddings.create(
                    model=self.models[SmallLanguageModelRole.EMBEDDING],
                    input=text,
                )
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed multiple texts in a single API call.

        Significantly faster than calling embed() in a loop when indexing
        large document collections.
        """
        async with self._semaphores[SmallLanguageModelRole.EMBEDDING]:
            breaker = self._breakers[SmallLanguageModelRole.EMBEDDING]
            breaker.check(self.models[SmallLanguageModelRole.EMBEDDING])
            try:
                response = await self._embedding.embeddings.create(
                    model=self.models[SmallLanguageModelRole.EMBEDDING],
                    input=texts,
                )
                breaker.record_success()
            except _BREAKER_ERRORS:
                breaker.record_failure()
                raise
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Speech-to-text  (whisper.cpp server)
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_data: bytes,
        filename: str = "audio.wav",
    ) -> dict:
        """
        Transcribe audio via whisper-server.

        Returns dict with "text" and "language" keys.
        whisper-server auto-detects the language from audio content.
        """
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{self._whisper_url}/inference",
                files={"file": (filename, audio_data, "audio/wav")},
                data={"response_format": "json", "temperature": "0.0"},
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def check_health(self) -> dict[str, bool]:
        """
        Verify all llama-server instances are reachable.

        Returns a dict mapping model role names to availability booleans.
        Includes optional whisper-server for voice features.
        Used at startup to give clear feedback when servers need to be started.

        Retries once on failure (1s delay) to handle the transient window where
        llama-server responds to /health before its CUDA kernels are fully warm.
        """
        async def _probe(http: httpx.AsyncClient, url: str) -> bool:
            for attempt in range(2):
                try:
                    resp = await http.get(url)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                if attempt == 0:
                    await asyncio.sleep(1.0)
            return False

        result = {}
        async with httpx.AsyncClient(timeout=3.0) as http:
            for role in SmallLanguageModelRole:
                base = self._urls[role].replace("/v1", "")
                result[role.name] = await _probe(http, f"{base}/health")
            # Whisper is optional — voice features disabled if not running
            result["WHISPER"] = await _probe(http, f"{self._whisper_url}/health")
        return result
