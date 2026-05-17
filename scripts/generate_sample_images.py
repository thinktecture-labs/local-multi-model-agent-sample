"""
Generate sample images for the multimodal/vision demo scenarios.

Creates 3 images in data/demo-images/ using matplotlib, all based on actual
Nextera Platform demo data so they're internally consistent with SQL/RAG results.

Usage:
    python scripts/generate_sample_images.py

Images generated:
    data/demo-images/revenue_chart.png       — Quarterly revenue bar chart
    data/demo-images/pricing_table.png       — Nextera tier comparison table
    data/demo-images/architecture_diagram.png — Four-model agent architecture
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from src.engine.inference.config import SCENARIO_CONFIG

OUTPUT_DIR = SCENARIO_CONFIG.demo_images_dir


def generate_revenue_chart() -> str:
    """Quarterly revenue bar chart from the actual sales seed data."""
    # Data matches data/loader.py SQL_SEED exactly
    quarters = ["Q1\n2023", "Q2\n2023", "Q3\n2023", "Q4\n2023",
                "Q1\n2024", "Q2\n2024", "Q3\n2024", "Q4\n2024"]
    revenue = [18500, 24700, 31200, 42800, 55100, 68300, 84900, 103200]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4285F4"] * 4 + ["#0F9D58"] * 4  # Blue for 2023, Green for 2024
    bars = ax.bar(quarters, revenue, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, revenue):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                f"€{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Revenue (€)", fontsize=12)
    ax.set_title("Nextera Platform — Quarterly Revenue", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 120000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    ax.legend(
        [mpatches.Patch(color="#4285F4"), mpatches.Patch(color="#0F9D58")],
        ["2023", "2024"],
        loc="upper left",
    )

    path = os.path.join(OUTPUT_DIR, "revenue_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_pricing_table() -> str:
    """Nextera tier comparison table matching the product/pricing docs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(6, 9.5, "Nextera Platform — Pricing Plans", ha="center", va="center",
            fontsize=16, fontweight="bold")

    # Column positions — shifted right so Starter values don't overlap feature labels
    cols = [4.0, 7.0, 10.0]
    tiers = [
        ("Starter", "€299/mo", "#E8F0FE", "#4285F4"),
        ("Professional", "€999/mo", "#E6F4EA", "#0F9D58"),
        ("Enterprise", "€3,500/mo", "#FEF7E0", "#F4B400"),
    ]

    features = [
        ("Concurrent users",    ["5",       "25",             "Unlimited"]),
        ("Vector storage",      ["10 GB",   "100 GB",         "Unlimited"]),
        ("Max model size",      ["7B",      "70B",            "405B+"]),
        ("Fine-tuning",         ["—",       "Basic",          "Dedicated cluster"]),
        ("Support SLA",         ["Email",   "8h response",    "1h / 24-7 phone"]),
        ("Deployment",          ["Cloud",   "Cloud + On-prem", "Air-gapped"]),
        ("Compliance",          ["—",       "SOC 2",          "SOC 2 Type II"]),
    ]

    # Draw tier headers
    for i, (name, price, bg_color, accent) in enumerate(tiers):
        x = cols[i]
        header_box = FancyBboxPatch(
            (x - 1.2, 7.8), 2.4, 1.2,
            boxstyle="round,pad=0.1", facecolor=accent, edgecolor="none", alpha=0.9,
        )
        ax.add_patch(header_box)
        ax.text(x, 8.55, name, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(x, 8.1, price, ha="center", va="center",
                fontsize=11, color="white", alpha=0.9)

    # Draw feature rows
    row_y_start = 7.2
    row_height = 0.85

    for row_idx, (feature, values) in enumerate(features):
        y = row_y_start - row_idx * row_height
        bg = "#F8F9FA" if row_idx % 2 == 0 else "white"

        # Row background
        row_bg = FancyBboxPatch(
            (0.2, y - 0.3), 11.6, 0.7,
            boxstyle="round,pad=0.05", facecolor=bg, edgecolor="#E0E0E0", linewidth=0.5,
        )
        ax.add_patch(row_bg)

        # Feature name (left column)
        ax.text(0.5, y + 0.05, feature, ha="left", va="center", fontsize=10,
                fontweight="bold", color="#333")

        # Tier values (right columns)
        for i, val in enumerate(values):
            color = "#333" if val != "—" else "#AAA"
            weight = "bold" if val in ("Unlimited", "405B+", "Dedicated cluster") else "normal"
            ax.text(cols[i], y + 0.05, val, ha="center", va="center", fontsize=10,
                    color=color, fontweight=weight)

    path = os.path.join(OUTPUT_DIR, "pricing_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_architecture_diagram() -> str:
    """Four-model agent architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Title
    ax.text(6.5, 7.1, "Nextera Agent — Four-Model Architecture", ha="center", va="center",
            fontsize=15, fontweight="bold")

    # --- User query box ---
    user_box = FancyBboxPatch(
        (4.5, 5.8), 4, 0.8,
        boxstyle="round,pad=0.15", facecolor="#E8EAF6", edgecolor="#3F51B5", linewidth=2,
    )
    ax.add_patch(user_box)
    ax.text(6.5, 6.2, "User Query", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#3F51B5")

    # --- Router ---
    router_box = FancyBboxPatch(
        (5.0, 4.4), 3.0, 0.8,
        boxstyle="round,pad=0.15", facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=2,
    )
    ax.add_patch(router_box)
    ax.text(6.5, 4.95, "has image?", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#E65100")
    ax.text(6.5, 4.65, "Deterministic Router", ha="center", va="center",
            fontsize=8, color="#999")

    # Arrow: User bottom edge → Router top edge
    ax.annotate("", xy=(6.5, 5.35), xytext=(6.5, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    # --- Four model boxes (spaced wider to avoid label overlap) ---
    # Box coords: (x, y, w, h) — visual edges are at x-pad..x+w+pad, y-pad..y+h+pad
    PAD = 0.15
    seer_box      = (0.3,  1.2, 2.2, 1.7)   # top: 3.05, center: 1.4
    thinker_box   = (3.2,  1.2, 2.2, 1.7)   # top: 3.05, right: 5.55, center: 4.3
    doer_box      = (6.0,  1.2, 2.4, 1.7)   # left: 5.85, center: 7.2
    librarian_box = (9.5,  1.2, 2.4, 1.7)   # left: 9.35, center: 10.7

    models = [
        (*seer_box,      "Seer",      "gemma3-4B\n(vision)",      "#E1BEE7", "#9C27B0"),
        (*thinker_box,   "Thinker",   "gemma3-1B\n(fine-tuned)",  "#BBDEFB", "#1976D2"),
        (*doer_box,      "Doer",      "Qwen3.5\n4B",              "#C8E6C9", "#388E3C"),
        (*librarian_box, "Librarian", "embeddinggemma\n300M",     "#FFF9C4", "#F9A825"),
    ]

    for x, y, w, h, role, model, bg, edge in models:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={PAD}", facecolor=bg, edgecolor=edge, linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.35, role, ha="center", va="center",
                fontsize=11, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + 0.5, model, ha="center", va="center",
                fontsize=8, color="#555", linespacing=1.4)

    # --- Arrows: all start/end at box EDGES (not inside) ---

    # Router bottom-left edge → Seer top-center
    # Router: (5.0, 4.4) → bottom-left = (5.0-PAD, 4.4-PAD) = (4.85, 4.25)
    # Seer top center = (0.3+2.2/2, 1.2+1.7+PAD) = (1.4, 3.05)
    ax.annotate("", xy=(1.4, 3.05), xytext=(4.85, 4.25),
                arrowprops=dict(arrowstyle="-|>", color="#9C27B0", lw=1.5))
    ax.text(2.7, 3.85, "image", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#9C27B0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#9C27B0", alpha=0.9))

    # Router bottom → Thinker top-center
    # Thinker top center = (3.2+2.2/2, 1.2+1.7+PAD) = (4.3, 3.05)
    ax.annotate("", xy=(4.3, 3.05), xytext=(5.8, 4.25),
                arrowprops=dict(arrowstyle="-|>", color="#1976D2", lw=1.5))
    ax.text(4.7, 3.85, "text", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1976D2",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1976D2", alpha=0.9))

    # Thinker right edge → Doer left edge (upper)
    # Thinker right: 3.2+2.2+PAD = 5.55, Doer left: 6.0-PAD = 5.85
    ax.annotate("", xy=(5.85, 2.35), xytext=(5.55, 2.35),
                arrowprops=dict(arrowstyle="-|>", color="#388E3C", lw=1.5))
    ax.text(5.7, 2.65, "tool_use", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#388E3C",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#388E3C", alpha=0.9))

    # Thinker right edge → Librarian left edge (lower)
    # Librarian left: 9.5-PAD = 9.35
    ax.annotate("", xy=(9.35, 2.05), xytext=(5.55, 2.05),
                arrowprops=dict(arrowstyle="-|>", color="#F9A825", lw=1.5))
    ax.text(7.5, 2.35, "rag_query", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#F9A825",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#F9A825", alpha=0.9))

    # --- Bottom labels ---
    ax.text(6.5, 0.7, "Combined: ~5.9B parameters  |  ~5-6 GB VRAM  |  All local, no cloud dependency",
            ha="center", va="center", fontsize=9, color="#888", fontstyle="italic")

    path = os.path.join(OUTPUT_DIR, "architecture_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating sample images for multimodal demo…\n")

    path = generate_revenue_chart()
    print(f"  Created: {path}")

    path = generate_pricing_table()
    print(f"  Created: {path}")

    path = generate_architecture_diagram()
    print(f"  Created: {path}")

    print(f"\nDone. Images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
