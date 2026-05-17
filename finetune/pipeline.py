"""
Fine-Tuning Pipeline — orchestrate the full multi-model improvement workflow.

This script ties together data preparation and training for the production
model stack: EmbeddingGemma (retrieval), Gemma3-1B (intent), Gemma3-4B (RAG
synthesis), Qwen3.5-4B (tool calling).

Typical schedule after a conference:
  Day 1-2: Collect 500+ interactions via demo.py or the FastAPI server
  Day 3:   python -m finetune.pipeline --export-data
  Day 3-4: python -m finetune.pipeline --train-all (overnight if needed)
  Day 5:   Test fine-tuned models, update SmallLanguageModelClient model names
  Day 6:   Re-run demo — observe improvements across all intent types

Usage:
  python -m finetune.pipeline --status              # check readiness
  python -m finetune.pipeline --export-data         # prepare datasets only
  python -m finetune.pipeline --train-gemma3        # fine-tune gemma3 only
  python -m finetune.pipeline --train-qwen35        # fine-tune qwen35 tool caller only
  python -m finetune.pipeline --train-embedding     # fine-tune embeddinggemma only
  python -m finetune.pipeline --train-all           # fine-tune the full stack
  python -m finetune.pipeline --train-all --qlora   # use 4-bit QLoRA
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

from finetune._scenario import SCENARIO_NAME, TRAINING_DIR, SUFFIX

_REQUIRED_FILES = {
    "gemma3 intent data":       f"./{TRAINING_DIR}/gemma3_intent{SUFFIX}.jsonl",
    "gemma3 synthesis data":    f"./{TRAINING_DIR}/gemma3_synthesis{SUFFIX}.jsonl",
    "tool routing data":        f"./{TRAINING_DIR}/qwen35_toolcalling{SUFFIX}.jsonl",
    "embeddinggemma pairs":     f"./{TRAINING_DIR}/embeddinggemma_retrieval{SUFFIX}.jsonl",
}

# Interaction logs are an *optional* augmentation source — the pipeline ships
# curated training JSONL files in `data/training-data/` and runs without them.
# Create one by running the agent (`python demo.py --interactive`) or by
# enabling logging in `src/server/agent_routes.py`; the file is gitignored.
_OPTIONAL_INTERACTIONS = "./data/interactions.json"

_OPTIONAL_MODELS = {
    "gemma3 fine-tuned":        "./models/gemma3-1b-ft-merged",
    "qwen35 tool-calling FT":   "./models/qwen3.5-4b-toolcalling-ft-merged",
    "embeddinggemma fine-tuned":"./models/embeddinggemma-300m-ft-merged",
}


def _count_records(filepath: str) -> int:
    p = Path(filepath)
    if not p.exists():
        return 0
    suffix = p.suffix.lower()
    with p.open() as f:
        if suffix == ".jsonl":
            return sum(1 for line in f if line.strip())
        elif suffix == ".json":
            try:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
            except json.JSONDecodeError:
                return 0
    return 0


def show_status() -> None:
    print("\n" + "="*60)
    print("Fine-Tuning Pipeline — Status")
    print("="*60)

    print("\n  Data files:")
    for label, path in _REQUIRED_FILES.items():
        exists = Path(path).exists()
        count  = _count_records(path) if exists else 0
        icon   = "✓" if exists else "✗"
        color  = "" if exists else "  ← missing"
        if count > 0:
            print(f"  [{icon}] {label:<30s} {count:>5d} records{color}")
        else:
            print(f"  [{icon}] {label:<30s}{color}")

    print("\n  Models:")
    for label, path in _OPTIONAL_MODELS.items():
        exists = Path(path).exists()
        icon   = "✓" if exists else "○"
        status = "trained" if exists else "not trained"
        print(f"  [{icon}] {label:<30s} {status}")

    # Optional interaction logs — augment the curated training sets
    n_interactions = _count_records(_OPTIONAL_INTERACTIONS)
    print(f"\n  Interaction logs (optional augmentation): {n_interactions}")
    if n_interactions == 0:
        print("  ○  None — pipeline runs on curated training data only.")
        print("     Run `python demo.py --interactive` to collect logs for additional signal.")
    elif n_interactions < 200:
        print(f"  ○  {n_interactions} interactions: useful augmentation.")
    else:
        print(f"  ✓  {n_interactions} interactions: strong augmentation signal.")

    print()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def export_data(interactions_path: str = "./data/interactions.json") -> None:
    """Prepare gemma3 + embedding training datasets from interaction logs.

    Qwen3.5 tool-calling data is generated separately via the curated dataset
    in data_prep_qwen35_toolcalling.py (not derived from interaction logs).
    """
    from finetune.data_prep import (
        Gemma3DataPreparer,
        EmbeddingGemmaDataPreparer,
    )

    print("\n" + "="*60)
    print("Preparing fine-tuning datasets")
    print("="*60 + "\n")

    g3 = Gemma3DataPreparer(interactions_path)
    eg = EmbeddingGemmaDataPreparer(interactions_path)

    g3.prepare()
    eg.prepare()

    print("\n  Done. Datasets written to ./data/training-data/")


def train_gemma3(qlora: bool = False, task: str = "both") -> None:
    from finetune.train_gemma3 import train, Gemma3TrainConfig
    cfg = Gemma3TrainConfig(use_qlora=qlora)
    train(task=task, config=cfg)


def train_qwen35_toolcalling(qlora: bool = True) -> None:
    """Qwen3.5-4B tool-calling FT. QLoRA is the only supported mode
    (see train_qwen35_toolcalling.py — full FT was deliberately removed
    as it overfits at this dataset size). The qlora kwarg is kept for
    call-site signature parity with the other trainers."""
    from finetune.train_qwen35_toolcalling import train, Qwen35TrainConfig
    cfg = Qwen35TrainConfig()
    train(config=cfg)


def train_embeddinggemma() -> None:
    from finetune.train_embeddinggemma import train, EmbeddingGemmaTrainConfig
    cfg = EmbeddingGemmaTrainConfig()
    train(config=cfg)


def train_all(qlora: bool = False) -> None:
    """Fine-tune the production model stack in sequence."""
    print("\n" + "="*60)
    print("Fine-Tuning Pipeline — Training Production Stack")
    print("="*60)
    print("\nOrder: embeddinggemma → gemma3 → qwen35-toolcalling")
    print("(embedding first so RAG quality improves before synthesis)")

    # 1. Embedding (no GPU memory required for sentence-transformers)
    train_embeddinggemma()

    # 2. gemma3 (intent classification + synthesis)
    train_gemma3(qlora=qlora, task="both")

    # 3. qwen35 (tool calling)
    train_qwen35_toolcalling(qlora=qlora)

    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print("""
Next steps:
  1. Convert and deploy the fine-tuned models:
       bash finetune/convert_gemma3_to_gguf.sh
       bash finetune/convert_qwen35_to_gguf.sh
       bash finetune/convert_embeddinggemma_to_gguf.sh
  2. Restart inference servers: bash scripts/start_servers.sh --bg --ft
  3. Rebuild the vector index with the fine-tuned embedding model:
       python -m data.loader
  4. Re-run the demo and observe quality improvements:
       python demo.py --interactive
""")


# ---------------------------------------------------------------------------
# SmallLanguageModelFineTuningPipeline — object-oriented interface to the pipeline
# ---------------------------------------------------------------------------

class SmallLanguageModelFineTuningPipeline:
    """
    High-level orchestrator for fine-tuning the production model stack.

    Usage:
        pipeline = SmallLanguageModelFineTuningPipeline(qlora=True)
        pipeline.run()

    The pipeline automatically:
      1. Checks interaction log size — enables synthetic augmentation when < 500
      2. Exports datasets for the production stack
      3. Fine-tunes in the optimal order: embedding → gemma3 → qwen35
      4. Converts the fine-tuned gemma3 model to GGUF for llama-server
    """

    # Minimum interactions for quality fine-tuning before we rely on synthetic data
    _MIN_INTERACTIONS_FOR_REAL_DATA = 500

    def __init__(
        self,
        interactions_path: str = "./data/interactions.json",
        qlora: bool = False,
    ) -> None:
        self.interactions_path = interactions_path
        self.qlora             = qlora
        self._n_interactions   = _count_records(interactions_path)

    def run(self, steps: Optional[list[str]] = None) -> None:
        """
        Execute the full fine-tuning pipeline.

        Args:
            steps: subset of ["export", "embedding", "gemma3", "function"].
                   Defaults to all four steps.
        """
        if steps is None:
            steps = ["export", "embedding", "gemma3", "qwen35"]

        print("\n" + "="*60)
        print("SmallLanguageModelFineTuningPipeline — Starting")
        print("="*60)
        print(f"  Interactions: {self._n_interactions}")

        if self._n_interactions < self._MIN_INTERACTIONS_FOR_REAL_DATA:
            print(
                f"  ⚠  Fewer than {self._MIN_INTERACTIONS_FOR_REAL_DATA} interactions "
                f"— synthetic augmentation will supplement real data."
            )
        else:
            print(f"  ✓  Sufficient data for high-quality fine-tuning.")

        # Step 1 — Data preparation
        # Skip export when the scenario ships with curated training data —
        # export_data() regenerates from interaction logs and would overwrite
        # the curated JSONL files with near-empty data.
        if "export" in steps:
            if self._prebuilt_data_exists():
                print(f"\n  ⏭  Skipping data export — using pre-prepared {SCENARIO_NAME} training data")
            else:
                export_data(self.interactions_path)

        # Step 2 — embeddinggemma first: better retrieval improves RAG quality
        #          which feeds higher-quality synthesis examples for gemma3
        if "embedding" in steps:
            train_embeddinggemma()

        # Step 3 — gemma3: intent classification + synthesis
        if "gemma3" in steps:
            train_gemma3(qlora=self.qlora, task="both")

        # Step 4 — qwen35: tool calling
        if "qwen35" in steps:
            train_qwen35_toolcalling(qlora=self.qlora)

        # Convert gemma3 fine-tuned model to GGUF for llama-server
        self._convert_gemma3_to_gguf()

        print("\n" + "="*60)
        print("SmallLanguageModelFineTuningPipeline — Complete")
        print("="*60)
        print("""
Next steps:
  1. Convert and deploy the fine-tuned model: bash finetune/convert_gemma3_to_gguf.sh
  2. Restart inference server: bash scripts/start_servers.sh --bg --ft
  3. Rebuild the vector index with the fine-tuned embedding model:
       python -m data.loader
  4. Re-run the demo and observe quality improvements:
       python demo.py --interactive
""")

    def _prebuilt_data_exists(self) -> bool:
        """Check if pre-prepared training data files exist with real content."""
        for path in _REQUIRED_FILES.values():
            if "interactions" in path:
                continue  # skip interaction logs check
            if not Path(path).exists() or _count_records(path) < 10:
                return False
        return True

    def _convert_gemma3_to_gguf(self) -> None:
        """
        Convert the merged gemma3 fine-tuned model to GGUF for llama-server.
        """
        import subprocess
        gguf_script = Path("finetune/convert_gemma3_to_gguf.sh")
        if not gguf_script.exists():
            print("  convert_gemma3_to_gguf.sh not found — skipping GGUF conversion")
            return
        if not Path("models/gemma3-1b-ft-merged/model.safetensors").exists():
            print("  gemma3-1b-ft-merged not found — skipping GGUF conversion")
            return
        print("\n  Converting gemma3-1b-ft-merged to GGUF…")
        result = subprocess.run(["bash", str(gguf_script)], capture_output=True, text=True)
        if result.returncode == 0:
            print("  GGUF conversion complete")
            print("  Start inference server: bash scripts/start_servers.sh --bg --ft")
        else:
            print(f"  GGUF conversion failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning pipeline for the production model stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--status",           action="store_true", help="Show pipeline status")
    parser.add_argument("--export-data",      action="store_true", help="Prepare training datasets")
    parser.add_argument("--train-gemma3",     action="store_true", help="Fine-tune gemma3")
    parser.add_argument("--train-qwen35",     action="store_true", help="Fine-tune qwen35 tool caller")
    parser.add_argument("--train-embedding",  action="store_true", help="Fine-tune embeddinggemma")
    parser.add_argument("--train-all",        action="store_true", help="Fine-tune the full stack")
    parser.add_argument("--qlora",            action="store_true", help="Use 4-bit QLoRA (CUDA only)")
    parser.add_argument("--interactions",     default="./data/interactions.json")
    args = parser.parse_args()

    if args.status or not any([
        args.export_data, args.train_gemma3, args.train_qwen35,
        args.train_embedding, args.train_all,
    ]):
        show_status()

    if args.export_data:
        export_data(args.interactions)

    if args.train_all:
        SmallLanguageModelFineTuningPipeline(
            interactions_path=args.interactions,
            qlora=args.qlora,
        ).run()
    else:
        if args.train_embedding:
            train_embeddinggemma()
        if args.train_gemma3:
            train_gemma3(qlora=args.qlora)
        if args.train_qwen35:
            train_qwen35_toolcalling(qlora=args.qlora)
