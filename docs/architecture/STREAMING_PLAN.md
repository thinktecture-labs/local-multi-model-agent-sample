# End-to-End Output Streaming — Implementation Plan

> **STATUS: IMPLEMENTED — 2026-03-19**
> All phases completed. `POST /query/stream` SSE endpoint live, all handlers have `handle_stream()`,
> `process_stream()` in orchestrator, `/escalate/stream` for cloud, full frontend streaming
> (token-by-token, trace panel updates incrementally). See `src/engine/agent/handlers/`,
> `src/engine/agent/orchestrator.py`, `src/server/agent_routes.py`, `src/server/cloud_routes.py`.
>
> **Goal:** Replace all blocking LLM calls with streamed SSE output so the user
> sees tokens arrive immediately instead of waiting for the entire response.

---

## Current State

| Component | Streaming? |
|-----------|-----------|
| Local llama-server (gemma3-ft, qwen3.5-4b, vision) | `generate_stream()` exists in client.py line 350 — **never called** |
| Cloud GPT-5.4 (`/escalate`, `/compare`) | No — blocking `await` |
| Cloud orchestrator (`/query?backend=cloud`) | `CloudOrchestrator` exists (`cloud_orchestrator.py`) — no streaming |
| Qwen path | No — same non-streaming client |
| Frontend | Waits for complete `QueryResult`, renders all at once |

**Impact:** GPT-5.4 responses block 10-15s before any text appears.
Local synthesis (gemma3-4b) blocks 500-1200ms (M5 Max Metal) — noticeable on RAG queries.

---

## 1. SSE Protocol Design

**New endpoint:** `POST /query/stream`
**Existing `POST /query` stays unchanged** for backward compatibility.

### Event Types

```
event: step
data: {"action":"classify_intent","model":"logreg","details":{"intent":"rag_query"},"duration_ms":24.8,"tokens_used":0}

event: step
data: {"action":"vector_search","model":"embeddinggemma","details":{"query":"...","documents":[...]},"duration_ms":7.1,...}

event: token
data: {"text":"The"}

event: token
data: {"text":" Enterprise"}

event: token
data: {"text":" plan"}

event: done
data: {"request_id":"abc123","intent":"rag_query","execution_time_ms":742.7,"total_tokens":450,"prompt_tokens":320,"completion_tokens":130,"models_used":["logreg","gemma3-ft","embeddinggemma","gemma3-4b-vision"],"confidence":0.82}

event: error
data: {"message":"Pipeline timed out after 60s","code":"timeout"}
```

**Design decisions:**
- `step` events are full `ExecutionStepOut` objects — frontend appends them to the trace panel incrementally.
- `token` events carry only `{"text":"..."}` — minimal payload for high-frequency events.
- `done` carries final metadata (minus `response` and `steps` which were already streamed). Frontend reconstructs the full `QueryResult`.
- `error` can appear at any point and terminates the stream.
- SSE guarantees ordering — no sequence numbers needed.

---

## 2. Backend Changes

### 2.1 Extend `SmallLanguageModelClient` streaming

**File:** `src/engine/inference/client.py`

The existing `generate_stream()` (line 350) only works with the INFERENCE model role. **Bug:** the semaphore `async with` wraps only the `create()` call — it releases before token iteration, allowing overload. Needs:

1. **Parameterize `generate_stream()`** — add `temperature`, `max_tokens`, `deterministic` matching `generate()`.
2. **Add `generate_synthesis_stream()`** — stream from VISION client (port 9093).
3. **Token usage reporting** — set `stream_options={"include_usage": True}`, yield usage from final chunk.

Introduce a small wrapper:

```python
@dataclass
class StreamChunk:
    text: str = ""
    done: bool = False
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

Both methods yield `StreamChunk` objects. Final chunk has `done=True` with token counts.

**Semaphore note:** Currently `generate_stream()` releases the semaphore before tokens arrive (line 366). New methods must hold the semaphore for the full streaming duration to avoid overloading llama-server.

### 2.2 Add streaming variants to handlers

**Protocol:** Add alongside existing `handle()` — not a replacement.

```python
class StreamingHandler(Protocol):
    async def handle_stream(self, query: str, **kwargs) -> AsyncIterator[ExecutionStep | str]:
        """Yield ExecutionStep for completed steps, then yield str tokens for final response."""
```

**`src/engine/agent/handlers/rag.py` — `RAGHandler.handle_stream()`**
- Steps 1-2 (rewrite, vector_search) stay non-streaming — yield `ExecutionStep` objects as they complete.
- Step 3 (synthesis) uses `generate_synthesis_stream()` — yields `str` tokens.
- After stream completes, yield final `ExecutionStep` for synthesis with token counts.

**`src/engine/agent/handlers/direct_answer.py` — `DirectAnswerHandler.handle_stream()`**
- Switch from `client.generate()` to `client.generate_stream()`.
- Yield `str` tokens.

**`src/engine/agent/handlers/tool_use.py` — `ToolUseHandler.handle_stream()`**
- All steps up to tool execution stay non-streaming (fast, deterministic). Each yields `ExecutionStep`.
- Final formatting step uses `client.generate_stream()`.
- Calculator path returns static string — no streaming needed.
- Multi-step synthesis also switches to streaming.

**`src/engine/agent/handlers/vision.py`** — Deferred to Phase 3 (vision responses are typically short).

### 2.3 Streaming orchestrator method

**File:** `src/engine/agent/orchestrator.py`

Add `process_stream()` to `SmallLanguageModelAgentOrchestrator`:

```python
async def process_stream(
    self, query: str, images: list[str] | None = None,
) -> AsyncIterator[ExecutionStep | str]:
```

- Classification step is non-streaming (fast). Yields `ExecutionStep` immediately.
- Dispatches to `handler.handle_stream()` instead of `handler.handle()`.
- Forwards all yielded items (steps and tokens) to caller.
- After stream completes, logs interaction.

### 2.4 Streaming CloudOrchestrator

**File:** `src/engine/agent/cloud_orchestrator.py` (already exists — add streaming variant)

Add `process_stream()`:
- Vector search step yields `ExecutionStep`.
- Tool-calling loop rounds yield `ExecutionStep` objects for each `cloud_inference` and `execute_tool` action.
- Final cloud inference uses `stream=True` on OpenAI API. Yields `str` tokens.
- After stream completes, yields final `ExecutionStep` with cost/token metadata.
- Note: `generate_synthesis()` was renamed from `generate_with_vision_model()` (commit 4a37cdd).

**This is the highest-impact change** — GPT-5.4 TTFT drops from 15s to ~200-500ms.

### 2.5 SSE route

**File:** `src/server/agent_routes.py`

```python
@router.post("/query/stream", tags=["Agent"])
async def process_query_stream(request: QueryRequest):
    async def event_stream():
        try:
            # Select orchestrator based on backend (same as _run_backend)
            gen = state.agent.process_stream(request.query, images=...)

            accumulated_text = ""
            steps = []
            async for item in gen:
                if isinstance(item, ExecutionStep):
                    steps.append(item)
                    yield _sse_event("step", step_to_dict(item))
                elif isinstance(item, str):
                    accumulated_text += item
                    yield _sse_event("token", {"text": item})

            yield _sse_event("done", { ...metadata... })

        except asyncio.TimeoutError:
            yield _sse_event("error", {"message": "Pipeline timed out"})
        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`_sse_event()` helper already exists in `voice_routes.py` — extract to shared utility.

### 2.6 Streaming escalation route

**File:** `src/server/cloud_routes.py`

Add `POST /escalate/stream` — uses `stream=True` on OpenAI `chat.completions.create()`, yields `token` events followed by `done` event with cost/latency metadata.

---

## 3. Frontend Changes

### 3.1 SSE client function

**File:** `src/clients/observatory-react/src/api/client.ts`

Add `queryAgentStream()` — follows the existing SSE parsing pattern from `voiceChat()` (line 160-196):

```typescript
export async function queryAgentStream(
  query: string,
  callbacks: {
    onStep: (step: ExecutionStep) => void
    onToken: (text: string) => void
    onDone: (meta: QueryResult) => void
    onError: (msg: string) => void
  },
  images?: string[],
  backend?: DemoMode,
): Promise<void>
```

Also add `escalateQueryStream()` for the HITL escalation path.

### 3.2 New reducer actions

**File:** `src/clients/observatory-react/src/state/reducer.ts`

```typescript
| { type: 'APPEND_STEP'; idx: number; step: ExecutionStep }
| { type: 'APPEND_TOKEN'; idx: number; text: string }
| { type: 'FINALIZE_STREAM'; idx: number; meta: Partial<QueryResult> }
```

**File:** `src/clients/observatory-react/src/types/state.ts`

Add to `Exchange`:
```typescript
streamingText?: string  // accumulated tokens during streaming
```

Reducer logic:
- `APPEND_STEP` — appends step to `exchange.result.steps`.
- `APPEND_TOKEN` — appends text to `exchange.streamingText`.
- `FINALIZE_STREAM` — merges metadata into `exchange.result`, sets `response` to accumulated `streamingText`, clears `streamingText`.

### 3.3 Update ConversationPane

**File:** `src/clients/observatory-react/src/components/Conversation/ConversationPane.tsx`

In `handleSend()`, for the default path, replace `await queryAgent()` with:

```typescript
// Create partial result so ExchangeCard starts rendering
dispatch({ type: 'UPDATE_EXCHANGE', idx, updates: {
  result: { intent: '', response: '', execution_time_ms: 0, steps: [], models_used: [], total_tokens: 0 }
}})

await queryAgentStream(text, {
  onStep: (step) => dispatch({ type: 'APPEND_STEP', idx, step }),
  onToken: (text) => dispatch({ type: 'APPEND_TOKEN', idx, text }),
  onDone: (meta) => {
    dispatch({ type: 'FINALIZE_STREAM', idx, meta })
    dispatch({ type: 'UPDATE_TOKENS', ... })
  },
  onError: (msg) => dispatch({ type: 'UPDATE_EXCHANGE', idx, updates: {
    result: { intent: 'direct_answer', response: `Error: ${msg}`, ... }
  }}),
}, images, demoMode)
```

### 3.4 Update ExchangeCard rendering

**File:** `src/clients/observatory-react/src/components/Conversation/ExchangeCard.tsx`

```tsx
<div className="exchange-response">
  {!result ? (
    <div className="typing-dots">...</div>
  ) : exchange.streamingText != null ? (
    /* Streaming in progress — render accumulated text with blinking cursor */
    <div className="exchange-text streaming">
      <span dangerouslySetInnerHTML={{ __html: formatResponse(exchange.streamingText) }} />
      <span className="cursor-blink" />
    </div>
  ) : (
    /* Complete — render final response */
    <div className="exchange-text" dangerouslySetInnerHTML={{ __html: formatResponse(result.response) }} />
  )}
</div>
```

Trace panel updates incrementally as `APPEND_STEP` adds steps.

### 3.5 Streaming escalation

Update `handleEscalate` in `ExchangeCard` to use `escalateQueryStream()` — the cloud response area renders tokens incrementally with a blinking cursor.

---

## 4. Backward Compatibility

| What | Status |
|------|--------|
| `POST /query` | **Unchanged** — non-streaming clients, CLI, tests all work |
| `POST /escalate` | **Unchanged** |
| `POST /compare`, `/query/compare-all` | **Unchanged** |
| Handler `handle()` methods | **Unchanged** — `handle_stream()` is additive |
| Orchestrator `process()` | **Unchanged** — `process_stream()` is additive |
| `queryAgent()` in frontend | **Unchanged** — `queryAgentStream()` is a parallel function |
| All existing tests | **Pass without modification** |

---

## 5. Phase Ordering

### Phase 1 — Cloud Streaming (highest impact, lowest risk)

**Why first:** GPT-5.4 has the worst latency (10-15s). OpenAI SDK natively supports `stream=True`. No local model code changes needed.

1. Add `POST /escalate/stream` in `cloud_routes.py`
2. Add `escalateQueryStream()` to frontend `client.ts`
3. Update `ExchangeCard.handleEscalate` to stream
4. Add blinking cursor CSS
5. Test via UI: HYBRID → trigger escalation → see tokens stream

**Files:** 3 backend, 2 frontend. ~150 lines.

### Phase 2 — Local Pipeline Streaming (core feature)

1. Extend `SmallLanguageModelClient` — parameterized `generate_stream()`, new `generate_synthesis_stream()`
2. Add `handle_stream()` to `DirectAnswerHandler` (simplest)
3. Add `handle_stream()` to `RAGHandler`
4. Add `handle_stream()` to `ToolUseHandler`
5. Add `process_stream()` to orchestrator
6. Add `POST /query/stream` SSE route
7. Add `queryAgentStream()` to frontend client
8. Add `APPEND_STEP`, `APPEND_TOKEN`, `FINALIZE_STREAM` to reducer
9. Update `ConversationPane.handleSend()` and `ExchangeCard` rendering

**Files:** ~10 backend, ~5 frontend. ~500 lines.

### Phase 3 — Cloud Orchestrator + Comparison Modes

1. Add `process_stream()` to `CloudOrchestrator`
2. Wire `/query/stream?backend=cloud` to it
3. Streaming for `/query/compare-all` — interleaved SSE with `backend` field in events (more complex protocol)
4. Streaming for `VisionHandler`

**Files:** ~5 backend, ~3 frontend. ~300 lines.

---

## 6. Edge Cases

### Timeouts
- Wrap the async generator with `asyncio.wait_for` on each `__anext__()` call.
- Handlers check `time.perf_counter() < deadline` before each yield.
- Emit `error` event and close stream on timeout.

### ngrok compatibility
- SSE works through ngrok. `X-Accel-Buffering: no` + `Cache-Control: no-cache` headers prevent buffering.
- ngrok has 60s connection timeout. Send SSE keepalive comments (`:keepalive\n\n`) every 15s during idle periods.

### Circuit breaker
- Existing `CircuitBreaker` records success on first chunk in `generate_stream()`. New streaming methods must do the same.

### Error mid-stream
- If error occurs after tokens have been sent, emit `error` event. Frontend shows error banner below partial text. `done` event is NOT sent.
- Partial text remains visible — user can still read what arrived.

### Token batching
- llama-server sends tokens one at a time — potentially hundreds of tiny SSE events.
- Consider batching on a 50ms timer: accumulate tokens, send one `token` event with the batch.
- Reduces event overhead without perceptible latency.

### Reconnection
- SSE natively supports reconnection via `Last-Event-Id`. Not meaningful for one-shot query streams.
- On disconnect, frontend shows error + "Retry" button that resubmits.

---

## File Summary

| File | Phase | Action | Lines est. |
|------|-------|--------|-----------|
| `src/engine/inference/client.py` | 2 | Modify — parameterized streaming, vision streaming | +80 |
| `src/engine/agent/handlers/rag.py` | 2 | Modify — `handle_stream()` | +50 |
| `src/engine/agent/handlers/direct_answer.py` | 2 | Modify — `handle_stream()` | +25 |
| `src/engine/agent/handlers/tool_use.py` | 2 | Modify — `handle_stream()` | +60 |
| `src/engine/agent/orchestrator.py` | 2 | Modify — `process_stream()` | +40 |
| `src/engine/agent/cloud_orchestrator.py` | 3 | Modify — `process_stream()` | +60 |
| `src/server/agent_routes.py` | 2 | Modify — `POST /query/stream` | +50 |
| `src/server/cloud_routes.py` | 1 | Modify — `POST /escalate/stream` | +40 |
| `observatory-react/src/api/client.ts` | 1+2 | Modify — streaming client functions | +80 |
| `observatory-react/src/state/reducer.ts` | 2 | Modify — new action types | +20 |
| `observatory-react/src/types/state.ts` | 2 | Modify — `streamingText` field | +2 |
| `observatory-react/src/components/Conversation/ConversationPane.tsx` | 2 | Modify — streaming dispatch | +30 |
| `observatory-react/src/components/Conversation/ExchangeCard.tsx` | 1+2 | Modify — incremental rendering | +20 |
| `observatory-react/src/styles/global.css` | 1 | Modify — cursor-blink animation | +10 |

**Total: ~14 files modified, ~570 lines added, across 3 phases.**
