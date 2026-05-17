"""
Performance benchmark — measure end-to-end latency across all intent paths.

Runs a fixed set of queries (same ones used in demo.py and the Observatory UI),
collects per-query timing, and prints a summary table grouped by intent type.

Usage:
    python scripts/benchmark.py                  # 1 warmup + 3 measured runs
    python scripts/benchmark.py --runs 5         # 1 warmup + 5 measured runs
    python scripts/benchmark.py --json results/bench.json  # save raw data
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rich import box
from rich.console import Console
from rich.table import Table

from src.engine.inference.client import SmallLanguageModelClient
from src.engine.inference.config import SCENARIO_CONFIG, DEMO_IMAGES_DIR
from src.engine.knowledge.vector_store import VectorStore
from src.engine.tools import create_default_registry
from src.engine.agent import SmallLanguageModelAgentOrchestrator

import importlib
_loader = importlib.import_module(SCENARIO_CONFIG.data_loader_module)
seed_vector_store = _loader.seed_vector_store
seed_sql_database = _loader.seed_sql_database

console = Console()

# ---------------------------------------------------------------------------
# Benchmark queries — cover every intent path
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEMO_IMAGES_DIR)

# (query, expected_type, image_filename_or_None)
_BENCH_QUERIES_NEXTERA: list[tuple[str, str, str | None]] = [
    # RAG queries
    ("What are the features included in the Enterprise plan?",    "rag_query",  None),
    ("What integrations does the platform support?",              "rag_query",  None),
    ("What are the support SLAs?",                                "rag_query",  None),
    ("Which plan should a 15-person startup choose?",             "rag_query",  None),
    # Tool — calculator
    ("If I have 50 customers paying €999/month, what is my ARR?", "tool_calc",  None),
    ("What is 23% of 84900?",                                     "tool_calc",  None),
    ("What's 15% of $45,000?",                                    "tool_calc",  None),
    # Tool — SQL
    ("How many customers do we have?",                            "tool_sql",   None),
    ("What were the total sales revenue figures for 2024?",       "tool_sql",   None),
    ("How many new customers joined in Q3 and Q4 of 2024?",      "tool_sql",   None),
    ("Show top 3 customers by revenue",                           "tool_sql",   None),
    # Direct
    ("Hello! What can you help me with?",                         "direct",     None),
    ("How are you?",                                              "direct",     None),
    # Image queries (only run if vision server is available)
    ("What trends do you see in this revenue chart?",             "image_query", "revenue_chart.png"),
    ("Summarize the pricing tiers shown in this table",           "image_query", "pricing_table.png"),
    ("Explain what this system diagram shows",                    "image_query", "architecture_diagram.png"),
]

# Add per-scenario query lists here as new scenarios/<name>.json are added.
BENCH_QUERIES = _BENCH_QUERIES_NEXTERA


def _preload_images(queries: list[tuple[str, str, str | None]]) -> dict[str, list[str]]:
    """Pre-load and base64-encode images needed by benchmark queries."""
    cache: dict[str, list[str]] = {}
    for _, _, img_file in queries:
        if img_file and img_file not in cache:
            path = os.path.join(IMAGES_DIR, img_file)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    cache[img_file] = [base64.b64encode(f.read()).decode("utf-8")]
    return cache


async def run_benchmark(agent: SmallLanguageModelAgentOrchestrator, runs: int) -> list[dict]:
    """Run all queries for the specified number of measured runs (plus 1 warmup)."""
    results = []

    # Pre-load images
    image_cache = _preload_images(BENCH_QUERIES)

    # Filter: skip image queries if images aren't available
    active_queries = [
        (q, t, img) for q, t, img in BENCH_QUERIES
        if img is None or img in image_cache
    ]

    # Warmup run (not counted)
    console.print("  [dim]Warmup run…[/dim]")
    for query, _, img_file in active_queries:
        images = image_cache.get(img_file) if img_file else None
        await agent.process(query, images=images)

    # Measured runs
    for run_idx in range(1, runs + 1):
        console.print(f"  [bold]Run {run_idx}/{runs}[/bold]")
        for query, expected_type, img_file in active_queries:
            images = image_cache.get(img_file) if img_file else None
            start = time.perf_counter()
            result = await agent.process(query, images=images)
            elapsed_ms = (time.perf_counter() - start) * 1000

            step_breakdown = [
                {
                    "action": s.action,
                    "model": s.model,
                    "duration_ms": round(s.duration_ms, 1),
                    "tokens_used": s.tokens_used,
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                }
                for s in result.steps
            ]
            results.append({
                "run": run_idx,
                "query": query,
                "expected_type": expected_type,
                "actual_intent": result.intent.value,
                "latency_ms": round(elapsed_ms, 1),
                "steps": len(result.steps),
                "step_breakdown": step_breakdown,
                "success": result.success,
            })
            console.print(
                f"    {result.intent.value:<15} {elapsed_ms:>8.0f} ms  "
                f"[dim]{query[:50]}{'…' if len(query) > 50 else ''}[/dim]"
            )

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table grouped by query type."""
    console.print()

    # Group by expected_type
    groups: dict[str, list[float]] = {}
    for r in results:
        groups.setdefault(r["expected_type"], []).append(r["latency_ms"])

    table = Table(
        title="Latency Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Path", style="bold")
    table.add_column("Queries", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Stdev", justify="right", style="dim")

    order = ["rag_query", "tool_sql", "tool_calc", "direct", "image_query"]
    for key in order:
        if key not in groups:
            continue
        vals = groups[key]
        n = len(vals)
        table.add_row(
            key,
            str(n),
            f"{min(vals):.0f} ms",
            f"{statistics.median(vals):.0f} ms",
            f"{statistics.mean(vals):.0f} ms",
            f"{max(vals):.0f} ms",
            f"{statistics.stdev(vals):.0f} ms" if n > 1 else "—",
        )

    console.print(table)

    # Per-step breakdown table
    step_groups: dict[str, list[float]] = {}
    for r in results:
        for s in r.get("step_breakdown", []):
            step_groups.setdefault(s["action"], []).append(s["duration_ms"])

    if step_groups:
        step_table = Table(
            title="Per-Step Breakdown (all queries)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        step_table.add_column("Step", style="bold")
        step_table.add_column("Count", justify="right")
        step_table.add_column("Min", justify="right")
        step_table.add_column("Median", justify="right")
        step_table.add_column("Mean", justify="right")
        step_table.add_column("Max", justify="right")

        step_order = [
            "classify_intent", "rewrite_query", "vector_search",
            "synthesize_response", "select_tool", "execute_tool",
            "format_response", "direct_response",
            "analyse_image",
        ]
        for action in step_order:
            if action not in step_groups:
                continue
            vals = step_groups[action]
            step_table.add_row(
                action,
                str(len(vals)),
                f"{min(vals):.0f} ms",
                f"{statistics.median(vals):.0f} ms",
                f"{statistics.mean(vals):.0f} ms",
                f"{max(vals):.0f} ms",
            )
        # Any steps not in step_order
        for action in sorted(step_groups):
            if action not in step_order:
                vals = step_groups[action]
                step_table.add_row(
                    action,
                    str(len(vals)),
                    f"{min(vals):.0f} ms",
                    f"{statistics.median(vals):.0f} ms",
                    f"{statistics.mean(vals):.0f} ms",
                    f"{max(vals):.0f} ms",
                )

        console.print()
        console.print(step_table)

    # Overall
    all_vals = [r["latency_ms"] for r in results]
    console.print(
        f"\n  Total queries: {len(all_vals)}  |  "
        f"Overall mean: {statistics.mean(all_vals):.0f} ms  |  "
        f"Overall median: {statistics.median(all_vals):.0f} ms"
    )


async def main(args: argparse.Namespace) -> None:
    console.print()
    console.print("[bold]Performance Benchmark[/bold]")
    console.print(f"  Runs: {args.runs} (+ 1 warmup)  |  Queries per run: {len(BENCH_QUERIES)}")
    console.print()

    # Initialise (same as demo.py)
    client = SmallLanguageModelClient.create_with_auto_detection()
    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    tools = create_default_registry(vector_store=vector_store)
    agent = SmallLanguageModelAgentOrchestrator(client, tools)

    # Health check
    console.print("  [dim]Checking servers…[/dim]")
    health = await client.check_health()
    for model, ok in health.items():
        icon = "[green]●[/green]" if ok else "[red]○[/red]"
        console.print(f"    {icon} {model}")
    # Whisper is optional (STT only) — don't block benchmarks on it
    required = {k: v for k, v in health.items() if k != "WHISPER"}
    if not all(required.values()):
        console.print("\n  [red]Not all required servers are healthy. Aborting.[/red]")
        sys.exit(1)

    # Seed data
    console.print("  [dim]Seeding data…[/dim]")
    vector_store.set_client(client)
    await seed_vector_store(client, vector_store)
    await seed_sql_database()
    console.print()

    # Run
    results = await run_benchmark(agent, args.runs)

    # Summary
    print_summary(results)

    # JSON output
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "runs": args.runs,
                "queries_per_run": len(BENCH_QUERIES),
                "results": results,
            }, f, indent=2)
        console.print(f"\n  [dim]Raw data saved → {args.json}[/dim]")

    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark agent latency")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs (default: 3)")
    parser.add_argument("--json", type=str, default="", help="Save raw results to JSON file")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        console.print("\n  [dim]Interrupted.[/dim]")
