"""
Data Preparation — convert agent interaction logs into model-specific training sets.

Preparers (one per training objective in the production stack):

  Gemma3DataPreparer         — intent classification + response synthesis examples
  EmbeddingGemmaDataPreparer — positive/negative passage pairs for contrastive learning

Qwen3.5-4B tool-calling data is prepared separately by `data_prep_qwen35_toolcalling.py`,
which generates hand-curated tool-call examples (not derived from interaction logs).

Input:  ./data/interactions.json  (produced by agent.export_training_data())
Output: ./data/training-data/      (JSON-Lines files, one per model)

Split into per-model modules for maintainability:
  data_prep_shared.py           — load_interactions(), save_jsonl()
  data_prep_gemma3.py           — Gemma3DataPreparer (~170 lines)
  data_prep_embeddinggemma.py   — EmbeddingGemmaDataPreparer (~260 lines)
"""

from __future__ import annotations

# Re-export preparers so existing imports (pipeline.py, tests) keep working
from finetune.data_prep_shared import load_interactions as _load_interactions, save_jsonl as _save_jsonl
from finetune.data_prep_gemma3 import Gemma3DataPreparer, Gemma3Example
from finetune.data_prep_embeddinggemma import EmbeddingGemmaDataPreparer

__all__ = [
    "Gemma3DataPreparer",
    "Gemma3Example",
    "EmbeddingGemmaDataPreparer",
    "_load_interactions",
    "_save_jsonl",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare fine-tuning datasets from interaction logs")
    parser.add_argument("--interactions", default="./data/interactions.json")
    parser.add_argument("--output-dir",   default="./data/training-data")
    args = parser.parse_args()

    print("Preparing fine-tuning datasets…\n")

    g3 = Gemma3DataPreparer(args.interactions, args.output_dir)
    eg = EmbeddingGemmaDataPreparer(args.interactions, args.output_dir)

    g3.prepare()
    eg.prepare()

    print("\nDone. Datasets ready in", args.output_dir)
