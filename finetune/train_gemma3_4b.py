"""
Fine-Tune gemma3-4b with LoRA — RAG synthesis (extractive QA).

Teaches the 4B model to quote exact numbers and facts from provided context
chunks instead of hallucinating plausible-sounding answers.

Training format: Gemma chat template (same as train_gemma3.py)
  <start_of_turn>user
  {instruction}

  {input}<end_of_turn>
  <start_of_turn>model
  {output}<end_of_turn>

Architecture: LoRA r=16 on all linear layers in the language model.
Vision encoder (SigLIP) and multi_modal_projector are excluded from LoRA
since Google froze vision during pretraining.

The model is loaded as Gemma3ForConditionalGeneration (multimodal) because
loading as Gemma3ForCausalLM causes weight mapping issues from the HF
checkpoint. A custom data collator provides the required token_type_ids.

Run:
  python -m finetune.train_gemma3_4b
  python -m finetune.train_gemma3_4b --epochs 3 --lr 5e-5

Output: ./models/gemma3-4b-ft-merged/  (merged model ready for GGUF conversion)

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
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
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
class Gemma3_4BTrainConfig:
    # Model — instruct-tuned 4B model for RAG synthesis
    base_model:       str  = "google/gemma-3-4b-it"

    # LoRA — all linear layers for best domain QA quality (research-backed)
    lora_r:           int  = 16
    lora_alpha:       int  = 32
    lora_dropout:     float = 0.0
    lora_target:      list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Exclude vision components from LoRA — SigLIP was frozen during
    # Google's pretraining and is identical across 4B/12B/27B.
    lora_exclude:     list[str] = field(
        default_factory=lambda: ["vision_tower", "multi_modal_projector"]
    )

    # Training — reduced batch size for 4B, longer context for RAG chunks
    num_epochs:       int  = 3
    batch_size:       int  = 2
    grad_accumulation: int = 8
    learning_rate:    float = 5e-5
    max_seq_length:   int  = 1024
    warmup_ratio:     float = 0.1
    logging_steps:    int  = 10
    save_steps:       int  = 50

    # QLoRA — opt-in for low-VRAM GPUs
    use_qlora:        bool = False
    bnb_4bit_compute: str  = "bfloat16"

    # Paths
    synthesis_data:   str  = field(default_factory=lambda: f"./{_TRAINING_DIR}/gemma3_4b_synthesis{_SUFFIX}.jsonl")
    output_dir:       str  = "./models/gemma3-4b-ft-merged"


# ---------------------------------------------------------------------------
# Dataset helpers — Gemma chat format (identical to train_gemma3.py)
# ---------------------------------------------------------------------------

def _format_gemma_chat(instruction: str, user_input: str, output: str) -> str:
    user_turn = instruction
    if user_input:
        user_turn = f"{instruction}\n\n{user_input}"
    return (
        f"<start_of_turn>user\n{user_turn}<end_of_turn>\n"
        f"<start_of_turn>model\n{output}<end_of_turn>"
    )


def _load_gemma_chat_dataset(filepath: str) -> Dataset:
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
# Custom data collator — injects token_type_ids for Gemma3 multimodal model
# ---------------------------------------------------------------------------

class _Gemma3TextCollator:
    """Wraps the default SFTTrainer collation to add token_type_ids=0.

    Gemma3ForConditionalGeneration requires token_type_ids during training
    to distinguish text vs image tokens. For text-only fine-tuning, all
    tokens are text, so token_type_ids is all zeros.
    """
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Labels = input_ids with padding masked to -100
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # token_type_ids = all zeros (text-only, no image tokens)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(config: Optional[Gemma3_4BTrainConfig] = None) -> None:
    if not _DEPS_AVAILABLE:
        print(
            "ERROR: Fine-tuning dependencies not installed.\n"
            "Run: pip install -r requirements-finetune.txt"
        )
        return

    if config is None:
        config = Gemma3_4BTrainConfig()

    data_path = config.synthesis_data
    if not Path(data_path).exists():
        print(f"  [skip] Training data not found at {data_path}")
        print("  Create data/training-data/gemma3_4b_synthesis.jsonl first")
        return

    print(f"\n{'='*60}")
    print(f"Fine-tuning gemma3-4b ({config.base_model}) — extractive QA synthesis")
    print(f"{'='*60}")

    # --- Deterministic training ---
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
    use_qlora = config.use_qlora and device == "cuda"
    method = "QLoRA" if use_qlora else "LoRA"
    print(f"  Device:  {device}")
    print(f"  Format:  Gemma chat template")
    print(f"  Method:  {method}")
    print(f"  Data:    {data_path}")
    print(f"  Deterministic: ON")

    # --- QLoRA ---
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute),
        )

    # --- Load model + tokenizer ---
    # Load as Gemma3ForConditionalGeneration (the full multimodal model).
    # Gemma3ForCausalLM causes weight mapping issues from the HF checkpoint
    # (model outputs garbage after LoRA merge). The multimodal model works
    # correctly with token_type_ids=0 for text-only inputs.
    print(f"\n  Loading {config.base_model} (multimodal, vision frozen)…")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    # Target all linear layers in the language model, but exclude vision
    # tower and multi_modal_projector (frozen during Google's pretraining).
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target,
        modules_to_save=None,
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset ---
    print(f"  Loading dataset: {data_path}")
    dataset = _load_gemma_chat_dataset(data_path)
    print(f"  Examples: {len(dataset)}")

    split = dataset.train_test_split(test_size=0.1, seed=42)

    # --- Custom collator for token_type_ids ---
    collator = _Gemma3TextCollator(tokenizer, max_length=config.max_seq_length)

    # --- Training ---
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        seed=42,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
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
        remove_unused_columns=False,
    )

    from finetune.training_utils import (
        LossHistoryCallback,
        DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_EARLY_STOPPING_THRESHOLD,
    )
    loss_cb = LossHistoryCallback(output_dir=config.output_dir, model_name="gemma3-4b")
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
        data_collator=collator,
        callbacks=[loss_cb, early_cb],
    )

    print(f"\n  Training for {config.num_epochs} epochs…")
    print(f"  Early stopping: patience={DEFAULT_EARLY_STOPPING_PATIENCE}, threshold={DEFAULT_EARLY_STOPPING_THRESHOLD}")
    trainer.train()

    # --- Save ---
    # Merge LoRA into base and save. The saved model will be
    # Gemma3ForConditionalGeneration but only the language model weights
    # changed. GGUF conversion extracts just the language model.
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"\n  Merging LoRA adapter into base weights…")
    model = model.merge_and_unload()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"  Model saved → {config.output_dir}")

    print(f"""
  ─────────────────────────────────────────────────────
  Next: convert to GGUF

    bash finetune/convert_gemma3_4b_to_gguf.sh

  Then restart servers:

    bash scripts/start_servers.sh --scenario <name>

  Evaluate:

    python -m finetune.eval_rag_groundtruth
  ─────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune gemma3-4b for extractive QA synthesis")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--qlora",      action="store_true", help="Use 4-bit QLoRA (CUDA only)")
    args = parser.parse_args()

    cfg = Gemma3_4BTrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_qlora=args.qlora,
    )
    train(config=cfg)
