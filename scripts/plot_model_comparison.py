"""
Radar chart comparing Gemma FT, Qwen 4B-only, and Hybrid model configurations.

Usage:
    python scripts/plot_model_comparison.py
    # Opens the chart and saves to results/model_comparison_radar.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json
from pathlib import Path


# ── Data: normalized 0-100 (higher = better for all axes) ────────────────────

CATEGORIES = [
    "Tool Routing\nAccuracy",
    "Expression\nAccuracy",
    "Vision\nAccuracy",
    "Multi-Step\nTool Chain",
    "Speed\n(inverse latency)",
    "Simplicity\n(fewer models)",
]

# Raw values for reference in annotations
RAW = {
    "Gemma FT": {
        "tool_routing": 92.5,
        "expression": 98.8,
        "vision": 100.0,
        "multi_step": 70.0,
        "speed_p50_ms": 484,
        "n_models": 4,
    },
    "Qwen 4B Only": {
        "tool_routing": 100.0,
        "expression": 100.0,
        "vision": 90.0,  # effectively 100 (false neg)
        "multi_step": 68.8,
        "speed_p50_ms": 2411,
        "n_models": 2,  # Qwen 4B + embeddinggemma
    },
    "Hybrid": {
        "tool_routing": 100.0,  # Qwen for function calling
        "expression": 98.8,     # Gemma scaffolding still active
        "vision": 100.0,        # Gemma 4B vision unchanged
        "multi_step": 72.5,     # Gemma decomposition + Qwen tools
        "speed_p50_ms": None,   # will be filled from benchmark
        "n_models": 4,          # gemma3-1B + qwen-4B + embeddinggemma + gemma3-4B
    },
}

# Try to load hybrid benchmark results
hybrid_bench = Path("results/hybrid-gemma-qwen-bench.json")
if hybrid_bench.exists():
    with open(hybrid_bench) as f:
        hb = json.load(f)
    import statistics
    all_lats = [r["latency_ms"] for r in hb["results"]]
    RAW["Hybrid"]["speed_p50_ms"] = statistics.median(all_lats)
    print(f"  Loaded hybrid benchmark: p50 = {RAW['Hybrid']['speed_p50_ms']:.0f} ms")
else:
    # Estimate: Gemma speed for non-tool queries, Qwen for tool queries
    # ~40% tool queries at 1200ms + 60% Gemma at 484ms ≈ 770ms
    RAW["Hybrid"]["speed_p50_ms"] = 770
    print("  Hybrid benchmark not found — using estimate (770ms)")


def normalize_speed(ms):
    """Convert latency to 0-100 score (lower ms = higher score)."""
    # 100ms → 100, 500ms → 80, 1000ms → 60, 2500ms → 20, 5000ms → 0
    return max(0, min(100, 100 - (ms - 100) / 49))


def normalize_simplicity(n_models):
    """Fewer models = higher score."""
    return {1: 100, 2: 85, 3: 60, 4: 40}[n_models]


CONFIGS = {}
for name, raw in RAW.items():
    CONFIGS[name] = [
        raw["tool_routing"],
        raw["expression"],
        raw["vision"],
        raw["multi_step"],
        normalize_speed(raw["speed_p50_ms"]),
        normalize_simplicity(raw["n_models"]),
    ]


# ── Radar chart ──────────────────────────────────────────────────────────────

COLORS = {
    "Gemma FT":     ("#2196F3", "#2196F320"),  # blue
    "Qwen 4B Only": ("#FF5722", "#FF572220"),  # orange-red
    "Hybrid":       ("#4CAF50", "#4CAF5020"),  # green
}

N = len(CATEGORIES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

for name, values in CONFIGS.items():
    vals = values + values[:1]  # close
    line_color, fill_color = COLORS[name]
    ax.plot(angles, vals, "o-", linewidth=2.5, label=name, color=line_color, markersize=7)
    ax.fill(angles, vals, alpha=0.08, color=line_color)

# Style
ax.set_xticks(angles[:-1])
ax.set_xticklabels(CATEGORIES, size=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8, color="gray")
ax.yaxis.grid(True, color="lightgray", linewidth=0.5)
ax.xaxis.grid(True, color="lightgray", linewidth=0.5)

# Legend with raw values
legend_text = []
for name, raw in RAW.items():
    speed = f"{raw['speed_p50_ms']:.0f}ms" if raw['speed_p50_ms'] else "?"
    legend_text.append(
        f"{name}: tools={raw['tool_routing']}% | expr={raw['expression']}% | "
        f"vision={raw['vision']}% | multi={raw['multi_step']}% | "
        f"p50={speed} | models={raw['n_models']}"
    )

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.12),
    fontsize=10,
    framealpha=0.9,
)

# Title
ax.set_title(
    "Model Configuration Comparison\nGemma FT vs Qwen 3.5-4B vs Hybrid",
    size=16,
    fontweight="bold",
    pad=30,
)

# Raw values annotation box
box_text = "\n".join(legend_text)
fig.text(
    0.5, 0.02, box_text,
    ha="center", va="bottom", fontsize=8, fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray"),
)

plt.tight_layout()
out_path = Path("results/model_comparison_radar.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
print(f"\n  Saved → {out_path}")
plt.show()
