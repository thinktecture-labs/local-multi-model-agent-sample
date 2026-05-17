"""
Fine-Tune gemma3 with LoRA — intent classification & response synthesis.

Uses HuggingFace transformers + PEFT (LoRA) to improve gemma3's ability to:
  1. Classify user queries into the correct intent bucket
  2. Synthesize high-quality responses from retrieved context

Training format: Gemma chat template
  <start_of_turn>user
  {instruction}

  {input}<end_of_turn>
  <start_of_turn>model
  {output}<end_of_turn>

Architecture choice: LoRA r=8 targeting all four attention projection layers
for maximum coverage without overfitting on the 1B model.

Run:
  python -m finetune.train_gemma3 --task intent
  python -m finetune.train_gemma3 --task intent --epochs 20
  python -m finetune.train_gemma3 --task synthesis
  python -m finetune.train_gemma3 --task both

Output: ./models/gemma3-1b-ft-merged/  (merged model ready for GGUF conversion)

Prerequisites:
  pip install -r requirements-finetune.txt
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from finetune._scenario import TRAINING_DIR as _TRAINING_DIR, SUFFIX as _SUFFIX

# Guard: these are optional heavy deps
try:
    import torch
    from datasets import Dataset
    from peft import EvaConfig, LoraConfig, TaskType, get_peft_model, initialize_lora_eva_weights, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Gemma3TrainConfig:
    # Model — instruct-tuned 1B model (retains synthesis ability after LoRA)
    base_model:       str  = "google/gemma-3-1b-it"

    # LoRA — all four attention projections for best coverage at this model size
    lora_r:           int  = 8
    lora_alpha:       int  = 16
    lora_dropout:     float = 0.05
    lora_target:      list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    num_epochs:       int  = 7
    batch_size:       int  = 4
    grad_accumulation: int = 4
    learning_rate:    float = 1e-4
    max_seq_length:   int  = 512
    warmup_ratio:     float = 0.1   # fraction of total steps used for warm-up
    logging_steps:    int  = 10
    save_steps:       int  = 100

    # Full fine-tuning — train all weights, no LoRA adapter
    full_ft:          bool = False

    # QLoRA — opt-in only. QLoRA weights cannot be converted to GGUF, so the
    # default is full-precision LoRA which produces GGUF-ready merged weights.
    # Enable with --qlora for experimentation on low-VRAM GPUs (no deployment).
    use_qlora:        bool = False
    bnb_4bit_compute: str  = "bfloat16"

    # Paths (loaded from scenarios/<name>.json)
    intent_data:      str  = field(default_factory=lambda: f"./{_TRAINING_DIR}/gemma3_intent{_SUFFIX}.jsonl")
    synthesis_data:   str  = field(default_factory=lambda: f"./{_TRAINING_DIR}/gemma3_synthesis{_SUFFIX}.jsonl")
    output_dir:       str  = "./models/gemma3-1b-ft-merged"

    def output_path(self, task: str) -> str:
        return os.path.join(self.output_dir, task)


# ---------------------------------------------------------------------------
# Dataset helpers — Gemma chat format
# ---------------------------------------------------------------------------

def _format_gemma_chat(instruction: str, user_input: str, output: str) -> str:
    """
    Format one example using the Gemma chat template.

    The Gemma tokenizer's apply_chat_template() produces this exact format.
    We replicate it directly so we can tokenize with a plain tokenizer call.
    """
    user_turn = instruction
    if user_input:
        user_turn = f"{instruction}\n\n{user_input}"
    return (
        f"<start_of_turn>user\n{user_turn}<end_of_turn>\n"
        f"<start_of_turn>model\n{output}<end_of_turn>"
    )


def _load_gemma_chat_dataset(filepath: str) -> Dataset:
    """
    Load Alpaca-format JSONL and re-format as Gemma chat sequences.

    Returns a Dataset with a 'text' column — SFTTrainer handles tokenization.
    Expected JSONL fields: instruction, input (optional), output
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts = [
        _format_gemma_chat(r["instruction"], r.get("input", ""), r["output"])
        for r in records
    ]

    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(task: str = "intent", config: Optional[Gemma3TrainConfig] = None) -> None:
    """
    Fine-tune gemma3 for the given task ("intent" or "synthesis").

    Args:
        task:   "intent" | "synthesis" | "both"
        config: TrainConfig (uses defaults if None)

    When task="both", the intent and synthesis datasets are concatenated into
    a single training set so one unified LoRA adapter learns all prompt patterns.
    Training them sequentially would cause catastrophic forgetting — the second
    task's adapter overwrites the first.
    """
    if not _DEPS_AVAILABLE:
        print(
            "ERROR: Fine-tuning dependencies not installed.\n"
            "Run: pip install -r requirements-finetune.txt"
        )
        return

    if config is None:
        config = Gemma3TrainConfig()

    if task == "both":
        # Concatenate both datasets into one training set so a single LoRA
        # adapter covers all prompt patterns (intent + synthesis).
        data_paths = []
        for label, path in [("intent", config.intent_data), ("synthesis", config.synthesis_data)]:
            if Path(path).exists():
                data_paths.append((label, path))
            else:
                print(f"  [skip] {label}: dataset not found at {path}")
        if not data_paths:
            print("  No datasets found. Run: python -m finetune.data_prep first")
            return
        _train_combined("both", data_paths, config)
    else:
        data_path = config.intent_data if task == "intent" else config.synthesis_data
        if not Path(data_path).exists():
            print(f"  [skip] {task}: dataset not found at {data_path}")
            print("  Run: python -m finetune.data_prep first")
            return
        _train_single(task, data_path, config)


def _train_combined(task: str, data_paths: list[tuple[str, str]], config: Gemma3TrainConfig) -> None:
    """Train on concatenated datasets (used by --task both)."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning gemma3 — task: {task} (combined datasets)")
    print(f"{'='*60}")
    for label, path in data_paths:
        count = sum(1 for line in open(path) if line.strip())
        print(f"  {label}: {path} ({count} examples)")

    # Concatenate all JSONL files into one dataset
    all_records: list[dict] = []
    for _, path in data_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    texts = [
        _format_gemma_chat(r["instruction"], r.get("input", ""), r["output"])
        for r in all_records
    ]
    combined_dataset = Dataset.from_dict({"text": texts})
    print(f"  Combined: {len(combined_dataset)} total examples")

    _train_single(task, None, config, dataset_override=combined_dataset)


def _train_single(task: str, data_path: str | None, config: Gemma3TrainConfig, *, dataset_override: Dataset | None = None) -> None:
    print(f"\n{'='*60}")
    print(f"Fine-tuning gemma3 ({config.base_model}) — task: {task}")
    print(f"{'='*60}")

    # --- Deterministic training for reproducibility ---
    import random, numpy as np
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_qlora = config.use_qlora and device == "cuda" and not config.full_ft
    method = "Full FT" if config.full_ft else ("QLoRA" if use_qlora else "LoRA")
    print(f"  Device:  {device}")
    print(f"  Format:  Gemma chat template")
    print(f"  Method:  {method}")
    print(f"  Deterministic: ON (CUBLAS_WORKSPACE_CONFIG + cudnn)")

    # --- QLoRA: 4-bit quantization when available ---
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute),
        )

    # --- Load model + tokenizer ---
    print(f"\n  Loading {config.base_model}…")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    if not config.full_ft:
        # --- LoRA: EVA init + rsLoRA for stable training ---
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target,
            bias="none",
            use_rslora=True,              # α/√r scaling — stabilizes gradient norms
            init_lora_weights="eva",      # data-driven SVD init — most stable method
            eva_config=EvaConfig(rho=2.0),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Full FT: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # --- Dataset ---
    if dataset_override is not None:
        dataset = dataset_override
    else:
        print(f"  Loading dataset: {data_path}")
        dataset = _load_gemma_chat_dataset(data_path)
    print(f"  Examples: {len(dataset)}")

    # 90/10 train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # --- EVA initialization (data-driven SVD on activations) ---
    if not config.full_ft and lora_cfg.init_lora_weights == "eva":
        print("  Initializing LoRA weights with EVA (SVD on activations)…")

        def _collate(batch):
            texts = [b["text"] for b in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_seq_length)
            return dict(enc)  # EVA requires plain dict, not BatchEncoding

        eva_loader = DataLoader(split["train"], batch_size=8, collate_fn=_collate, shuffle=False)
        initialize_lora_eva_weights(model, dataloader=eva_loader)
        print("  EVA initialization complete ✓")

    # --- Training arguments ---
    output_path = config.output_path(task)
    sft_config = SFTConfig(
        output_dir=output_path,
        seed=42,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",       # cosine decay — ~5% better than linear
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=device in ("cuda", "mps"),
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="none",
        max_length=config.max_seq_length,
        dataset_text_field="text",
    )

    # --- Convergence monitoring callbacks ---
    from finetune.training_utils import (
        LossHistoryCallback,
        DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_EARLY_STOPPING_THRESHOLD,
    )
    loss_cb = LossHistoryCallback(output_dir=output_path, model_name="gemma3")
    early_cb = EarlyStoppingCallback(
        early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=DEFAULT_EARLY_STOPPING_THRESHOLD,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[loss_cb, early_cb],
    )

    print(f"\n  Training {task} model for {config.num_epochs} epochs…")
    print(f"  Early stopping: patience={DEFAULT_EARLY_STOPPING_PATIENCE}, threshold={DEFAULT_EARLY_STOPPING_THRESHOLD}")
    trainer.train()

    # --- Backup existing model before overwriting ---
    from finetune._scenario import INFERENCE_GGUF_FT
    gguf_path = INFERENCE_GGUF_FT
    if os.path.isfile(gguf_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{gguf_path}.bak.{ts}"
        shutil.copy2(gguf_path, backup)
        print(f"\n  Backed up existing GGUF → {backup}")

    # --- Save trained model ---
    os.makedirs(config.output_dir, exist_ok=True)
    if not config.full_ft:
        print(f"\n  Merging LoRA adapter into base weights…")
        model = model.merge_and_unload()
    else:
        print(f"\n  Saving full fine-tuned model…")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"  Model saved → {config.output_dir}")

    _print_next_steps(config)


def _print_next_steps(config: Gemma3TrainConfig) -> None:
    print(f"""
  ─────────────────────────────────────────────────────
  Next: convert to GGUF and serve with llama-server

    bash finetune/convert_gemma3_to_gguf.sh

  Restart all servers with fine-tuned models:

    bash scripts/start_servers.sh --bg --ft

  Evaluate improvement:

    python -m finetune.eval_gemma3 --save results/finetuned_gemma3.json
    python -m finetune.eval_gemma3 --compare \\
        results/baseline_gemma3.json results/finetuned_gemma3.json
  ─────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune gemma3")
    parser.add_argument(
        "--task", choices=["intent", "synthesis", "both"], default="intent",
        help="Which task to fine-tune for (default: intent)",
    )
    parser.add_argument("--epochs",     type=int,   default=7)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--qlora",      action="store_true", help="Use 4-bit QLoRA (CUDA only, cannot convert to GGUF)")
    parser.add_argument("--full-ft",    action="store_true", help="Full fine-tuning (no LoRA) — needs more VRAM")
    args = parser.parse_args()

    cfg = Gemma3TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_qlora=args.qlora,
        full_ft=args.full_ft,
    )
    train(task=args.task, config=cfg)
