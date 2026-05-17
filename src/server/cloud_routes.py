"""
Cloud comparison and escalation routes.

Routes:
  POST /compare  — side-by-side local vs cloud LLM
  POST /escalate — HITL cloud escalation (user-approved)
"""

import asyncio
import json as _json
import time as _time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.engine.inference.config import (
    CLOUD_COMPARISON_ENABLED,
    CLOUD_INPUT_COST_PER_1M,
    CLOUD_OUTPUT_COST_PER_1M,
    OPENAI_API_KEY,
    OPENAI_COMPARE_MODEL,
    PIPELINE_TIMEOUT,
)
from .models import (
    CompareRequest,
    CompareResponse,
    EscalateDisclosure,
    EscalateRequest,
    EscalateResponse,
)
from .state import state


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"

router = APIRouter()


# Cloud escalation system prompt — frames the cloud as a collaborator that
# anchors on the user's local context and adds external knowledge.
_ESCALATE_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user has explicitly opted in to send "
    "this question to the cloud after the local agent flagged low confidence. "
    "If LOCAL CONTEXT is provided below, treat it as the user's authoritative "
    "anchor for facts about their own data — quote and cite it where relevant. "
    "Combine that anchor with your broader world knowledge for facts the local "
    "context doesn't cover. Be honest about which parts come from the local "
    "context versus your general knowledge."
)


async def _fetch_local_context_block(
    query: str, top_k: int = 10,
) -> tuple[str, EscalateDisclosure]:
    """Build a 'LOCAL CONTEXT' block + disclosure of what's actually being sent.

    Queries the curated KB and the user uploads *separately* so we can tell
    the caller (and ultimately the UI) how many chunks of each kind are
    leaving the machine. The merged-and-sorted list mirrors what
    vector_search would have returned with include_all_uploads=True, but
    we keep the split-by-source count for the privacy receipt.

    Returns ("", empty-disclosure) on any failure — escalation degrades to
    no-context but the receipt still shows nothing left the machine.
    """
    empty = EscalateDisclosure()
    if not getattr(state, "tools", None):
        return "", empty
    try:
        registry = state.tools
        vs_tool = registry.get("vector_search") if hasattr(registry, "get") else None
        kb_store = getattr(vs_tool, "vector_store", None) if vs_tool else None
        upload_store = getattr(vs_tool, "upload_store", None) if vs_tool else None
        if kb_store is None:
            return "", empty

        kb_results = await kb_store.search(query, top_k=top_k)
        upload_results = []
        if upload_store is not None and await upload_store.count() > 0:
            upload_results = await upload_store.search(query, top_k=top_k)

        combined = (kb_results or []) + (upload_results or [])
        combined.sort(key=lambda d: d.score if d.score is not None else 0, reverse=True)
        sent = combined[:top_k]
        if not sent:
            return "", empty

        upload_ids = {id(d) for d in (upload_results or [])}
        kb_count = sum(1 for d in sent if id(d) not in upload_ids)
        upload_count = sum(1 for d in sent if id(d) in upload_ids)
        upload_chunk_ids = [str(getattr(d, "id", "")) for d in sent if id(d) in upload_ids]

        docs_text = "\n\n".join(
            f"[{getattr(d, 'id', '')}] {getattr(d, 'content', '')}" for d in sent
        )
        context_block = f"\n\nLOCAL CONTEXT (anchor your answer in this):\n\n{docs_text}"

        disclosure = EscalateDisclosure(
            kb_chunks=kb_count,
            upload_chunks=upload_count,
            context_chars=len(context_block),
            upload_chunk_ids=upload_chunk_ids,
        )
        return context_block, disclosure
    except Exception:
        return "", empty


@router.post("/escalate", response_model=EscalateResponse, tags=["Agent"])
async def escalate_to_cloud(request: EscalateRequest) -> EscalateResponse:
    """
    Human-in-the-loop cloud escalation.

    Called when the user approves sending a query to the cloud LLM after
    the confidence assessment recommended escalation. This is the only
    path that sends user data externally — and only with explicit consent.

    The user's local RAG context (curated KB + uploaded documents — uploads
    bypass the UPLOAD_MERGE_MIN_SCORE filter via include_all_uploads=True)
    is injected into the cloud's system prompt as an "anchor", so the cloud
    builds on what local found rather than answering from training data alone.
    """
    if state.network_mode == "offline":
        raise HTTPException(status_code=503, detail="Offline mode — cloud escalation blocked.")
    if not CLOUD_COMPARISON_ENABLED:
        raise HTTPException(status_code=503, detail="No cloud API key configured.")

    from openai import AsyncOpenAI as _CloudOpenAI
    cloud_client = _CloudOpenAI(api_key=OPENAI_API_KEY)

    context_block, disclosure = await _fetch_local_context_block(request.query)
    messages = [
        {"role": "system", "content": _ESCALATE_SYSTEM_PROMPT + context_block},
        {"role": "user", "content": request.query},
    ]

    t0 = _time.perf_counter()
    cloud_resp = await cloud_client.chat.completions.create(
        model=OPENAI_COMPARE_MODEL,
        messages=messages,
        max_completion_tokens=1000,
        reasoning_effort="none",
    )
    cloud_ms = (_time.perf_counter() - t0) * 1000

    cloud_text = cloud_resp.choices[0].message.content or ""
    cloud_tokens = (
        (cloud_resp.usage.prompt_tokens + cloud_resp.usage.completion_tokens)
        if cloud_resp.usage else 0
    )
    cost = (
        (cloud_resp.usage.prompt_tokens / 1_000_000) * CLOUD_INPUT_COST_PER_1M
        + (cloud_resp.usage.completion_tokens / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
    ) if cloud_resp.usage else 0

    payload_bytes = len(_json.dumps(messages).encode("utf-8"))
    async with state._bytes_lock:
        state.cloud_bytes_sent += payload_bytes

    return EscalateResponse(
        cloud_response=cloud_text,
        cloud_model=OPENAI_COMPARE_MODEL,
        cloud_latency_ms=round(cloud_ms, 1),
        cloud_cost=round(cost, 6),
        cloud_tokens=cloud_tokens,
        cloud_bytes_sent=payload_bytes,
        disclosure=disclosure,
    )


@router.post("/escalate/stream", tags=["Agent"])
async def escalate_to_cloud_stream(request: EscalateRequest):
    """
    Streaming cloud escalation via SSE.

    Same as /escalate but streams tokens as they arrive from GPT-5.4,
    reducing perceived latency from 10-15s to ~200-500ms TTFT.

    Events: token (incremental text), done (metadata), error.
    """
    if state.network_mode == "offline":
        raise HTTPException(status_code=503, detail="Offline mode — cloud escalation blocked.")
    if not CLOUD_COMPARISON_ENABLED:
        raise HTTPException(status_code=503, detail="No cloud API key configured.")

    async def event_stream():
        from openai import AsyncOpenAI as _CloudOpenAI
        cloud_client = _CloudOpenAI(api_key=OPENAI_API_KEY)

        context_block, disclosure = await _fetch_local_context_block(request.query)
        messages = [
            {"role": "system", "content": _ESCALATE_SYSTEM_PROMPT + context_block},
            {"role": "user", "content": request.query},
        ]

        # Privacy receipt: emit BEFORE any cloud token so the UI can show
        # "Sending N curated + M upload chunks (~X KB)" *before* the user
        # sees a reply roll in.
        yield _sse_event("disclosure", disclosure.model_dump())

        t0 = _time.perf_counter()
        try:
            stream = await cloud_client.chat.completions.create(
                model=OPENAI_COMPARE_MODEL,
                messages=messages,
                max_completion_tokens=1000,
                reasoning_effort="none",
                stream=True,
                stream_options={"include_usage": True},
            )

            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            async for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        accumulated_text += delta
                        yield _sse_event("token", {"text": delta})

            cloud_ms = (_time.perf_counter() - t0) * 1000
            total_tokens = prompt_tokens + completion_tokens
            cost = (
                (prompt_tokens / 1_000_000) * CLOUD_INPUT_COST_PER_1M
                + (completion_tokens / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
            )

            payload_bytes = len(_json.dumps(messages).encode("utf-8"))
            async with state._bytes_lock:
                state.cloud_bytes_sent += payload_bytes

            yield _sse_event("done", {
                "cloud_response": accumulated_text,
                "cloud_model": OPENAI_COMPARE_MODEL,
                "cloud_latency_ms": round(cloud_ms, 1),
                "cloud_cost": round(cost, 6),
                "cloud_tokens": total_tokens,
                "cloud_bytes_sent": payload_bytes,
                "disclosure": disclosure.model_dump(),
            })

        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/compare", response_model=CompareResponse, tags=["Comparison"])
async def compare_query(request: CompareRequest) -> CompareResponse:
    """Run the same query through local agent and optionally a cloud LLM."""
    # 1. Local query (always)
    t0 = _time.perf_counter()
    try:
        local_result = await asyncio.wait_for(
            state.agent.process(request.query),
            timeout=PIPELINE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {PIPELINE_TIMEOUT}s",
        )
    local_ms = (_time.perf_counter() - t0) * 1000

    est_cost = (
        (local_result.prompt_tokens / 1_000_000) * CLOUD_INPUT_COST_PER_1M
        + (local_result.completion_tokens / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
    )

    resp = CompareResponse(
        intent=local_result.intent.value,
        local_response=local_result.response,
        local_latency_ms=round(local_ms, 1),
        local_tokens=local_result.total_tokens,
        cloud_available=CLOUD_COMPARISON_ENABLED and state.network_mode == "online",
        cloud_model=OPENAI_COMPARE_MODEL if (CLOUD_COMPARISON_ENABLED and state.network_mode == "online") else "",
        estimated_cloud_cost=round(est_cost, 6),
    )

    # 2. Cloud query — same pipeline, swap the LLM
    if CLOUD_COMPARISON_ENABLED and state.network_mode == "online":
        from openai import AsyncOpenAI as _CloudOpenAI
        cloud_client = _CloudOpenAI(api_key=OPENAI_API_KEY)

        messages: list[dict] = []
        intent = local_result.intent.value

        if intent == "rag_query":
            context_parts = []
            for step in local_result.steps:
                if step.action == "vector_search" and step.details.get("documents"):
                    for doc in step.details["documents"]:
                        title = doc.get("metadata", {}).get("title", doc.get("id", ""))
                        context_parts.append(f"[Source: {title}]\n{doc['content']}")
            if context_parts:
                context = "\n\n---\n\n".join(context_parts[:3])
                messages = [
                    {"role": "system", "content": (
                        "You are a precise knowledge-base assistant. Answer questions using "
                        "ONLY the provided source documents. Structure your answer clearly: "
                        "use bullet points for lists, cite [Source: title] for each fact. "
                        "If the documents don't contain the answer, say so explicitly. "
                        "Be concise — 2-4 sentences for simple questions, structured lists for complex ones."
                    )},
                    {"role": "user", "content": (
                        f"Based on the following documents, answer the question.\n\n"
                        f"DOCUMENTS:\n{context}\n\n"
                        f"QUESTION: {request.query}"
                    )},
                ]
            else:
                messages = [{"role": "user", "content": request.query}]

        elif intent == "tool_use":
            tool_result_data = None
            tool_name = ""
            for step in local_result.steps:
                if step.action == "execute_tool" and step.details.get("result"):
                    tool_result_data = step.details["result"]
                    tool_name = step.details.get("tool", "")
            if tool_result_data:
                result_str = _json.dumps(tool_result_data, indent=2, default=str)
                messages = [
                    {"role": "system", "content": (
                        "You are a data analyst assistant. Format tool results into clear, "
                        "human-readable answers. Use exact numbers from the data — never "
                        "approximate. For tables, use a clean list format."
                    )},
                    {"role": "user", "content": (
                        f"The user asked: \"{request.query}\"\n\n"
                        f"A {tool_name} tool returned this result:\n{result_str}\n\n"
                        f"Provide a clear, direct answer."
                    )},
                ]
            else:
                messages = [{"role": "user", "content": request.query}]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful, concise AI assistant. Keep answers brief and direct."},
                {"role": "user", "content": request.query},
            ]

        try:
            t0 = _time.perf_counter()
            cloud_resp = await cloud_client.chat.completions.create(
                model=OPENAI_COMPARE_MODEL,
                messages=messages,
                max_completion_tokens=1000,
                reasoning_effort="none",
            )
            cloud_ms = (_time.perf_counter() - t0) * 1000
            cloud_prompt = cloud_resp.usage.prompt_tokens if cloud_resp.usage else 0
            cloud_completion = cloud_resp.usage.completion_tokens if cloud_resp.usage else 0
            cloud_cost = (
                (cloud_prompt / 1_000_000) * CLOUD_INPUT_COST_PER_1M
                + (cloud_completion / 1_000_000) * CLOUD_OUTPUT_COST_PER_1M
            )
            resp.cloud_response = cloud_resp.choices[0].message.content or ""
            resp.cloud_latency_ms = round(cloud_ms, 1)
            resp.cloud_tokens = (cloud_prompt + cloud_completion)
            resp.cloud_cost = round(cloud_cost, 6)
            payload_bytes = len(_json.dumps(messages).encode("utf-8"))
            resp.cloud_bytes_sent = payload_bytes
            async with state._bytes_lock:
                state.cloud_bytes_sent += payload_bytes
        except Exception as exc:
            resp.cloud_response = f"Cloud error: {exc}"
            resp.cloud_latency_ms = 0

    return resp
