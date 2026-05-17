#!/usr/bin/env python3
"""
Prompt Cache Benchmark — measure TTFT improvement from KV cache reuse.

Captures time-to-first-token (TTFT) for the two paths most affected by
prompt caching: RAG synthesis (fixed system prompt, varying context) and
multi-turn document chat (warm slot across follow-up questions).

Run BEFORE applying caching flags to capture the baseline:
  python scripts/benchmark_prompt_cache.py --save results/cache_pre.json

Run AFTER applying --swa-full + --cache-reuse + n_keep to measure improvement:
  python scripts/benchmark_prompt_cache.py --save results/cache_post.json

Compare:
  python scripts/benchmark_prompt_cache.py --compare results/cache_pre.json results/cache_post.json

Scenarios:
  A — RAG synthesis: same system prompt, different context each turn.
      First request = cold (system prompt not cached).
      Subsequent requests = warm (system prompt 65 tokens cached in slot).

  B — Multi-turn document chat: same session, follow-up questions.
      Turn 1 = cold (nothing cached).
      Turn 2+ = warm (system prompt + shared context prefix cached).
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.inference.config import SCENARIO_CONFIG

INFERENCE_PORT = int(os.getenv("INFERENCE_PORT", 9090))
VISION_PORT    = int(os.getenv("VISION_PORT",    9093))

# Demo document — pre-loaded context for Scenario B
DOCS_DIR    = Path(SCENARIO_CONFIG.demo_documents_dir)
NEXTERA_PDF = DOCS_DIR / "nextera_quarterly_report.pdf"

# Scenario A: varied user content paired with a fixed RAG context block
RAG_CONTEXTS = [
    ("What was Q1 2024 revenue?",  "Q1 2024: revenue €55,100, 7 new customers, churn 1.0%, ARR growth 28.7%."),
    ("What was Q2 2024 revenue?",  "Q2 2024: revenue €68,300, 8 new customers, churn 0.9%, ARR growth 23.9%."),
    ("What was Q3 2024 revenue?",  "Q3 2024: revenue €84,900, 9 new customers, churn 0.8%, ARR growth 24.3%."),
    ("What was Q4 2024 revenue?",  "Q4 2024: revenue €103,200, 11 new customers, churn 0.7%, ARR growth 21.6%."),
    ("Total revenue in 2024?",     "Full year 2024: total revenue €311,500 across Q1-Q4."),
    ("Which quarter had the most growth?", "Highest ARR growth was Q1 2024 at 28.7%."),
    ("How many new customers in Q3?",      "Q3 2024: 9 new customers joined."),
    ("What was the best churn rate?",      "Best churn rate was Q4 2024 at 0.7%."),
]

# Scenario B: follow-up questions about the same document session
MULTI_TURN_SESSION = [
    "What was total revenue in Q4 2024?",
    "And what about Q3?",
    "Which quarter had the highest number of new customers?",
    "What was the churn rate trend across the year?",
]

SHARED_CONTEXT = (
    "Nextera Platform Q4 2024 Business Review\n\n"
    "Revenue by quarter:\n"
    "- Q1 2024: €55,100 | 7 new customers | 1.0% churn | 28.7% ARR growth\n"
    "- Q2 2024: €68,300 | 8 new customers | 0.9% churn | 23.9% ARR growth\n"
    "- Q3 2024: €84,900 | 9 new customers | 0.8% churn | 24.3% ARR growth\n"
    "- Q4 2024: €103,200 | 11 new customers | 0.7% churn | 21.6% ARR growth\n\n"
    "Top customers: BrightHealth GmbH (Enterprise, €7,000/mo), "
    "Acme Corp (Enterprise, €3,500/mo), CodeStack Ltd (Professional, €999/mo).\n"
    "Products: Starter €299/mo, Professional €999/mo, Enterprise €3,500/mo."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_platform_info() -> dict:
    info = {
        "hostname": platform.node(),
        "system":   platform.system(),
        "machine":  platform.machine(),
        "python":   platform.python_version(),
    }
    try:
        result = __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
            info["backend"] = "CUDA"
    except Exception:
        info["backend"] = "Metal" if platform.system() == "Darwin" else "CPU"
    return info


def server_available(port: int) -> bool:
    try:
        resp = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_slot_n_past(port: int) -> int:
    """Query /slots to see how many prompt tokens are resident in slot 0."""
    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=3.0)
        if resp.status_code == 200:
            slots = resp.json()
            if slots:
                return slots[0].get("n_past", 0)
    except Exception:
        pass
    return -1  # -1 = endpoint not available (older llama.cpp build)


async def measure_ttft(
    port: int,
    model_name: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int = 80,
) -> tuple[float, float]:
    """
    Send a streaming request and return (ttft_ms, total_ms).

    ttft_ms  = time from request start to first non-empty content token.
    total_ms = time from request start to stream completion.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="no-key",
        timeout=30.0,
    )

    ttft_ms: float | None = None
    t_start = time.perf_counter()

    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if (
            ttft_ms is None
            and chunk.choices
            and chunk.choices[0].delta.content
        ):
            ttft_ms = (time.perf_counter() - t_start) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    await client.close()

    return ttft_ms or total_ms, total_ms


def _build_rag_messages(context: str, query: str) -> list[dict]:
    """Mirror build_rag_messages() exactly so benchmark uses the same prompt structure."""
    from src.engine.inference.prompts import (
        RAG_SYNTHESIS_SYSTEM_PROMPT,
        RAG_SYNTHESIS_USER_TEMPLATE,
    )
    return [
        {"role": "system", "content": RAG_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user",   "content": RAG_SYNTHESIS_USER_TEMPLATE.format(
            context=f"[Source: nextera_quarterly_report.pdf]\n{context}",
            query=query,
        )},
    ]


# ---------------------------------------------------------------------------
# Scenario A: RAG synthesis, fixed system prompt, varying context
# ---------------------------------------------------------------------------

async def benchmark_scenario_a(n_rounds: int = 3, warmup: int = 2) -> dict:
    """
    Scenario A — RAG Synthesis.

    Cycles through RAG_CONTEXTS N rounds. First request is always cold
    (different context + question). Subsequent requests reuse the system
    prompt from the slot cache.

    Returns per-request TTFT measurements split into:
      cold   — first request after server restart (no cache)
      warm   — requests 2+ (system prompt cached in slot)
    """
    port = VISION_PORT
    if not server_available(port):
        print(f"  Vision/synthesis server not available on port {port} — skipping Scenario A")
        return {}

    # Discover the model name from /props
    try:
        resp = httpx.get(f"http://localhost:{port}/props", timeout=3.0)
        model_name = resp.json().get("default_generation_settings", {}).get("model", "gemma3-4b-vision")
        model_name = Path(model_name).stem  # strip path/extension
    except Exception:
        model_name = "gemma3-4b-vision"

    print(f"  Scenario A: RAG synthesis TTFT (model={model_name}, port={port})")
    print(f"  {warmup} warmup + {n_rounds} rounds × {len(RAG_CONTEXTS)} prompts = "
          f"{n_rounds * len(RAG_CONTEXTS)} measurements")

    results: list[dict] = []

    # Warmup — not recorded
    print("  Warming up…", end="", flush=True)
    for q, ctx in RAG_CONTEXTS[:warmup]:
        msgs = _build_rag_messages(ctx, q)
        await measure_ttft(port, model_name, msgs, temperature=0.1)
        print(".", end="", flush=True)
    print(" done")

    for rnd in range(n_rounds):
        for idx, (query, context) in enumerate(RAG_CONTEXTS):
            msgs = _build_rag_messages(context, query)
            n_past_before = get_slot_n_past(port)
            ttft_ms, total_ms = await measure_ttft(port, model_name, msgs, temperature=0.1)
            n_past_after = get_slot_n_past(port)

            # Request is "warm" if the slot had cached tokens before the call
            is_warm = n_past_before > 0

            results.append({
                "round":        rnd,
                "idx":          idx,
                "query":        query[:50],
                "ttft_ms":      round(ttft_ms, 1),
                "total_ms":     round(total_ms, 1),
                "n_past_before": n_past_before,
                "n_past_after":  n_past_after,
                "warm":          is_warm,
            })
            marker = "W" if is_warm else "C"
            print(f"    [{marker}] rnd={rnd} idx={idx:02d}  TTFT={ttft_ms:6.0f}ms  "
                  f"n_past={n_past_before}→{n_past_after}  {query[:40]}")

    cold_ttfts = [r["ttft_ms"] for r in results if not r["warm"]]
    warm_ttfts = [r["ttft_ms"] for r in results if r["warm"]]

    return {
        "model":      model_name,
        "port":       port,
        "n_rounds":   n_rounds,
        "n_prompts":  len(RAG_CONTEXTS),
        "measurements": results,
        "cold": _stats(cold_ttfts) if cold_ttfts else {},
        "warm": _stats(warm_ttfts) if warm_ttfts else {},
        "improvement_pct": _pct_improvement(cold_ttfts, warm_ttfts),
    }


# ---------------------------------------------------------------------------
# Scenario B: Multi-turn document chat
# ---------------------------------------------------------------------------

async def benchmark_scenario_b(n_sessions: int = 5) -> dict:
    """
    Scenario B — Multi-turn document chat.

    Simulates N complete sessions of the MULTI_TURN_SESSION questions,
    all using the same SHARED_CONTEXT (the Nextera quarterly data).

    Turn 1 of each session is cold (slot evicted by end of previous session
    if context changes enough). Turns 2+ benefit from the system prompt
    and shared context prefix being cached.

    Returns per-turn TTFT statistics across all sessions.
    """
    port = VISION_PORT
    if not server_available(port):
        print(f"  Vision/synthesis server not available on port {port} — skipping Scenario B")
        return {}

    try:
        resp = httpx.get(f"http://localhost:{port}/props", timeout=3.0)
        model_name = resp.json().get("default_generation_settings", {}).get("model", "gemma3-4b-vision")
        model_name = Path(model_name).stem
    except Exception:
        model_name = "gemma3-4b-vision"

    print(f"\n  Scenario B: Multi-turn document chat TTFT (model={model_name})")
    print(f"  {n_sessions} sessions × {len(MULTI_TURN_SESSION)} turns")

    per_turn: dict[int, list[float]] = {i: [] for i in range(len(MULTI_TURN_SESSION))}
    per_turn_n_past: dict[int, list[int]] = {i: [] for i in range(len(MULTI_TURN_SESSION))}

    for session in range(n_sessions):
        print(f"  Session {session + 1}/{n_sessions}:")
        for turn_idx, question in enumerate(MULTI_TURN_SESSION):
            msgs = _build_rag_messages(SHARED_CONTEXT, question)
            n_past_before = get_slot_n_past(port)
            ttft_ms, total_ms = await measure_ttft(port, model_name, msgs, temperature=0.1)
            n_past_after = get_slot_n_past(port)

            per_turn[turn_idx].append(ttft_ms)
            if n_past_before >= 0:
                per_turn_n_past[turn_idx].append(n_past_before)

            print(f"    Turn {turn_idx + 1}: TTFT={ttft_ms:6.0f}ms  "
                  f"n_past={n_past_before}→{n_past_after}  {question[:45]}")

    turn_stats = []
    for turn_idx in range(len(MULTI_TURN_SESSION)):
        ttfts = per_turn[turn_idx]
        n_pasts = per_turn_n_past[turn_idx]
        turn_stats.append({
            "turn":           turn_idx + 1,
            "question":       MULTI_TURN_SESSION[turn_idx][:50],
            "ttft_stats":     _stats(ttfts),
            "n_past_mean":    round(mean(n_pasts), 1) if n_pasts else -1,
        })
        print(f"  Turn {turn_idx + 1} summary: mean={mean(ttfts):.0f}ms  "
              f"median={median(ttfts):.0f}ms  n_past≈{mean(n_pasts) if n_pasts else '?':.0f}")

    turn1_ttfts = per_turn[0]
    later_ttfts  = [t for i in range(1, len(MULTI_TURN_SESSION)) for t in per_turn[i]]

    return {
        "model":            model_name,
        "port":             port,
        "n_sessions":       n_sessions,
        "n_turns":          len(MULTI_TURN_SESSION),
        "per_turn":         turn_stats,
        "turn1_cold":       _stats(turn1_ttfts),
        "turn2plus_warm":   _stats(later_ttfts),
        "improvement_pct":  _pct_improvement(turn1_ttfts, later_ttfts),
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = {
        "n":      len(values),
        "mean":   round(mean(values), 1),
        "median": round(median(values), 1),
        "min":    round(min(values), 1),
        "max":    round(max(values), 1),
    }
    if len(values) > 1:
        s["stdev"] = round(stdev(values), 1)
        s["p95"] = round(sorted(values)[int(len(values) * 0.95)], 1)
    return s


def _pct_improvement(baseline: list[float], improved: list[float]) -> float | None:
    if not baseline or not improved:
        return None
    b, i = mean(baseline), mean(improved)
    if b == 0:
        return None
    return round((b - i) / b * 100, 1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(bench: dict) -> None:
    w = 70
    print("\n" + "=" * w)
    print(f"  Prompt Cache Benchmark — {bench['platform']['hostname']}")
    print(f"  {bench['platform'].get('backend', '?')} | "
          f"{bench['platform'].get('gpu', bench['platform'].get('machine', ''))}")
    print(f"  {bench['timestamp']}")
    print("=" * w)

    if a := bench.get("scenario_a"):
        print("\n  Scenario A — RAG Synthesis (system prompt caching)")
        cold = a.get("cold", {})
        warm = a.get("warm", {})
        if cold:
            print(f"    Cold TTFT  mean={cold.get('mean','?')}ms  "
                  f"median={cold.get('median','?')}ms  n={cold.get('n','?')}")
        if warm:
            print(f"    Warm TTFT  mean={warm.get('mean','?')}ms  "
                  f"median={warm.get('median','?')}ms  n={warm.get('n','?')}")
        if (imp := a.get("improvement_pct")) is not None:
            print(f"    TTFT improvement: {imp:+.1f}% (warm vs cold)")

    if b := bench.get("scenario_b"):
        print("\n  Scenario B — Multi-turn Document Chat")
        for t in b.get("per_turn", []):
            s = t["ttft_stats"]
            print(f"    Turn {t['turn']}: mean={s.get('mean','?')}ms  "
                  f"median={s.get('median','?')}ms  "
                  f"n_past≈{t.get('n_past_mean','?')}")
        t1 = b.get("turn1_cold", {})
        t2 = b.get("turn2plus_warm", {})
        if t1 and t2:
            print(f"    Turn 1 (cold): mean={t1.get('mean','?')}ms")
            print(f"    Turn 2+ (warm): mean={t2.get('mean','?')}ms")
        if (imp := b.get("improvement_pct")) is not None:
            print(f"    TTFT improvement turn 2+: {imp:+.1f}% vs turn 1")

    print()


def print_compare(pre: dict, post: dict) -> None:
    w = 70
    print("\n" + "=" * w)
    print(f"  Prompt Cache Comparison")
    print(f"  PRE:  {pre['timestamp']}  ({pre['platform']['hostname']})")
    print(f"  POST: {post['timestamp']}  ({post['platform']['hostname']})")
    print("=" * w)

    def _row(label: str, pre_val, post_val, suffix: str = "ms") -> None:
        if pre_val is None or post_val is None:
            return
        delta = post_val - pre_val
        sign = "▼" if delta < 0 else "▲"
        pct = (pre_val - post_val) / pre_val * 100 if pre_val else 0
        print(f"  {label:<35s}  {pre_val:>7.1f}{suffix}  →  {post_val:>7.1f}{suffix}  "
              f"{sign}{abs(delta):.1f}{suffix} ({pct:+.1f}%)")

    for scenario_key, label in [("scenario_a", "Scenario A"), ("scenario_b", "Scenario B")]:
        pre_s  = pre.get(scenario_key, {})
        post_s = post.get(scenario_key, {})
        if not pre_s or not post_s:
            continue
        print(f"\n  {label}:")
        if scenario_key == "scenario_a":
            _row("Cold TTFT mean", pre_s.get("cold", {}).get("mean"), post_s.get("cold", {}).get("mean"))
            _row("Warm TTFT mean", pre_s.get("warm", {}).get("mean"), post_s.get("warm", {}).get("mean"))
        elif scenario_key == "scenario_b":
            pre_turns  = {t["turn"]: t for t in pre_s.get("per_turn", [])}
            post_turns = {t["turn"]: t for t in post_s.get("per_turn", [])}
            for turn in sorted(set(pre_turns) & set(post_turns)):
                _row(
                    f"Turn {turn} TTFT mean",
                    pre_turns[turn]["ttft_stats"].get("mean"),
                    post_turns[turn]["ttft_stats"].get("mean"),
                )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prompt cache benchmark")
    parser.add_argument("--save",    type=str, help="Save results to JSON file (e.g. results/cache_pre.json)")
    parser.add_argument("--rounds",  type=int, default=3,  help="Rounds per scenario (default 3)")
    parser.add_argument("--sessions",type=int, default=5,  help="Sessions for Scenario B (default 5)")
    parser.add_argument("--warmup",  type=int, default=2,  help="Warmup requests before measuring (default 2)")
    parser.add_argument("--compare", nargs=2, metavar=("PRE", "POST"),
                        help="Compare two saved result files")
    args = parser.parse_args()

    if args.compare:
        pre_path, post_path = args.compare
        with open(pre_path) as f: pre  = json.load(f)
        with open(post_path) as f: post = json.load(f)
        print_compare(pre, post)
        return

    bench: dict = {
        "timestamp": datetime.now().isoformat(),
        "platform":  get_platform_info(),
    }

    print("\n1. Scenario A — RAG Synthesis")
    bench["scenario_a"] = await benchmark_scenario_a(
        n_rounds=args.rounds, warmup=args.warmup,
    )

    print("\n2. Scenario B — Multi-turn document chat")
    bench["scenario_b"] = await benchmark_scenario_b(n_sessions=args.sessions)

    print_report(bench)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(bench, f, indent=2)
        print(f"  Results saved → {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
