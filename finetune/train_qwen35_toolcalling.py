"""
Fine-tune Qwen3.5-4B for tool calling.

Architecture decision:
  Qwen3.5-4B zero-shot: 95.0% routing / 94.4% expression correctness.
  Fine-tuning on the scenario's domain data targets ≥97% routing / ≥95% expression.

Training approach:
  - QLoRA (4-bit base + LoRA adapters) — NOT full FT
  - Full FT on ~1,300 examples overfits; QLoRA generalises better
  - Unsloth for 2x speed + 70% less VRAM + built-in GGUF export
  - enable_thinking=False — disables <think> blocks, avoids llama.cpp parser
    conflict (issue #20260) and keeps tool call output clean
  - train_on_responses_only — loss only on model's tool call output, not on
    system/user/tool-result tokens

Configuration knobs:
  - QLoRA r=16 — overfitting prevention at this dataset size
  - LR=2e-4 — standard for QLoRA / LoRA fine-tuning
  - 2 epochs max — overfitting risk is higher for larger models on small data
  - Unsloth handles GGUF export in-band — no separate convert script needed
  - enable_thinking=False in apply_chat_template throughout

Run:
  # On CUDA machine (RTX Pro 6000 BW recommended):
  python -m finetune.train_qwen35_toolcalling

  # Options:
  python -m finetune.train_qwen35_toolcalling --epochs 1        # quick test
  python -m finetune.train_qwen35_toolcalling --lora-r 32       # higher capacity
  python -m finetune.train_qwen35_toolcalling --no-gguf         # skip GGUF export

Output:
  models/qwen3.5-4b-toolcalling-ft-merged/          — merged HF model (safetensors)
  models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-<scenario>-q4_k_m.gguf  — deployment GGUF (Q4_K_M)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from finetune._scenario import TRAINING_DIR as _TRAINING_DIR, SUFFIX as _SUFFIX


# ---------------------------------------------------------------------------
# Tool schemas — MUST match src/engine/tools/{calculator,sql_query}.py exactly
# Any description mismatch = training/inference vocabulary mismatch = accuracy loss
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "Query the database for numbers, counts, totals, averages, and lists "
                "from the sales, customers, and products tables. Use when the user asks "
                "for specific data records, revenue figures, customer lists, or aggregations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQL SELECT statement",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression with specific numbers. "
                "Use ONLY when the user provides concrete numbers to compute. "
                "Do NOT use for data lookups, aggregations, or product questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A valid Python math expression. "
                            "Use Python syntax: ** for exponents, // for integer division."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to two tools: "
    "sql_query (for database lookups) and calculator (for arithmetic on given numbers). "
    "Always use a tool when the user's request requires data retrieval or computation. "
    "Select the correct tool and provide the exact arguments."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Qwen35TrainConfig:
    base_model:        str   = "Qwen/Qwen3.5-4B"   # official instruct model (no -Instruct suffix in Qwen3.5)
    data_path:         str   = field(default_factory=lambda: f"./{_TRAINING_DIR}/qwen35_toolcalling{_SUFFIX}.jsonl")
    output_dir:        str   = "./models/qwen3.5-4b-toolcalling-ft-merged"

    # LoRA — QLoRA (4-bit base) is the correct choice at this dataset size.
    # Full FT risks overfitting on 1,300 examples and uses unnecessary VRAM.
    lora_r:            int   = 16
    lora_alpha:        int   = 16     # equal to r is safe starting point
    lora_dropout:      float = 0.0    # Unsloth recommends 0 for speed
    lora_target:       list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    seed:              int   = 42
    num_epochs:        int   = 2      # 1-2 max at this dataset size; beyond 2 overfits
    batch_size:        int   = 1      # per device; use grad_accumulation for effective batch
    grad_accumulation: int   = 8      # effective batch = 8
    learning_rate:     float = 2e-4   # standard for QLoRA; 1e-5 (full FT) is too low here
    lr_scheduler:      str   = "cosine"
    warmup_steps:      int   = 10
    weight_decay:      float = 0.01
    max_seq_length:    int   = 2048   # tool call conversations rarely exceed this
    logging_steps:     int   = 5

    # Qwen3.5 specific
    enable_thinking:   bool  = False  # disable <think> blocks — avoids llama.cpp
                                      # parser conflict (issue #20260), keeps
                                      # tool call output clean for deployment

    # Output
    export_gguf:       bool  = True
    gguf_quantization: str   = "q4_k_m"  # best quality/size tradeoff for Apple Silicon


# ---------------------------------------------------------------------------
# Dataset — convert raw tool-call format to Qwen3.5 messages format
# ---------------------------------------------------------------------------

def _raw_to_messages(example: dict) -> tuple[list[dict], list[dict]]:
    """
    Convert a training example to Qwen3.5 messages list + tools list.

    Handles two formats:
    1. Single-turn: {"query": str, "tool_call": {"name": str, "arguments": dict}}
    2. Multi-turn:  {"multi_turn_messages": [list of role dicts]}
       Multi-turn examples teach step-2 behaviour (sql_query → calculator).
       The system prompt is prepended automatically.
    """
    if "multi_turn_messages" in example:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + example["multi_turn_messages"]
        return messages, _TOOL_SCHEMAS

    tool_call = example["tool_call"]
    call_id = f"call_{uuid.uuid4().hex[:8]}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": example["query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id":   call_id,
                "type": "function",
                "function": {
                    "name":      tool_call["name"],
                    # Pass as dict — Qwen3.5's Jinja template iterates arguments
                    # as a mapping. Passing a JSON string causes "Can only get item
                    # pairs from a mapping" error (Qwen3.5 template behaviour differs
                    # from OpenAI spec which uses JSON-encoded strings).
                    "arguments": tool_call["arguments"],
                },
            }],
        },
    ]
    return messages, _TOOL_SCHEMAS


def _load_dataset(path: str, tokenizer, config: Qwen35TrainConfig):
    """Load JSONL, convert to chat template text, return HF Dataset."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    texts = []
    skipped = 0
    for rec in records:
        messages, tools = _raw_to_messages(rec)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=config.enable_thinking,
            )
            texts.append(text)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  WARNING: apply_chat_template failed for: {rec['query'][:60]}... — {e}")

    if skipped:
        print(f"  Skipped {skipped}/{len(records)} examples due to template errors")

    print(f"  Formatted {len(texts)} examples ({len(records) - skipped} usable)")
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(config: Optional[Qwen35TrainConfig] = None) -> None:
    if config is None:
        config = Qwen35TrainConfig()

    # Check Unsloth availability — required for CUDA training + GGUF export
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        UNSLOTH = True
    except ImportError:
        UNSLOTH = False
        print("\n  WARNING: Unsloth not installed.")
        print("  Install: pip install unsloth")
        print("  Training will fall back to standard PEFT (slower, more VRAM).\n")

    try:
        import torch
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install -r requirements-finetune.txt && pip install unsloth")
        return

    if not Path(config.data_path).exists():
        print(f"ERROR: Training data not found: {config.data_path}")
        print("Run: python -m finetune.data_prep_qwen35_toolcalling")
        return

    print(f"\n{'='*60}")
    print(f"Fine-tuning Qwen3.5-4B for tool calling")
    print(f"  Base model: {config.base_model}")
    print(f"  Method:     QLoRA r={config.lora_r} via {'Unsloth' if UNSLOTH else 'standard PEFT'}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  LR:         {config.learning_rate}")
    print(f"  Thinking:   {'enabled' if config.enable_thinking else 'disabled (non-thinking mode)'}")
    print(f"{'='*60}\n")

    # Deterministic seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=True,
            dtype=None,         # auto-detect
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # enables longer context
            random_state=config.seed,
        )
    else:
        # Fallback: standard PEFT (no 4-bit on MPS/CPU without bitsandbytes)
        from transformers import BitsAndBytesConfig
        bnb_config = None
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto" if device != "cpu" else None,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Explicit EOS token — critical for Qwen3.5, prevents runaway generation
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"  Loading dataset: {config.data_path}")
    dataset = _load_dataset(config.data_path, tokenizer, config)
    split = dataset.train_test_split(test_size=0.1, seed=config.seed)
    print(f"  Split: {len(split['train'])} train / {len(split['test'])} eval")

    # ── Training args ─────────────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)
    use_bf16 = (device == "cuda") and (UNSLOTH and is_bfloat16_supported() or not UNSLOTH)

    sft_config = SFTConfig(
        output_dir=os.path.join(config.output_dir, "_adapter"),
        seed=config.seed,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=use_bf16,
        optim="adamw_8bit" if UNSLOTH else "adamw_torch",
        report_to="none",
        max_length=config.max_seq_length,
        dataset_text_field="text",
        # packing=True: concatenates examples into fixed-length chunks of max_seq_length.
        # All batches are identical length → CUDA kernels compile once and stay cached.
        # Without packing, variable sequence lengths (single-turn ~300t vs multi-turn ~500t)
        # force kernel recompilation every step → 3x slowdown.
        # train_on_responses_only (Unsloth) handles loss masking correctly with packing.
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_config,
    )

    # Unsloth's train_on_responses_only masks non-assistant tokens from loss
    if UNSLOTH:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Training for {config.num_epochs} epochs…")
    trainer.train()

    # ── Save merged model ─────────────────────────────────────────────────────
    print(f"\n  Saving merged model → {config.output_dir}")
    if UNSLOTH:
        model.save_pretrained_merged(
            config.output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        # Standard merge: load base again in float16, apply adapter, merge
        from peft import PeftModel
        print("  Merging LoRA adapter into base weights (float16)…")
        from transformers import AutoModelForCausalLM as AMCL
        base = AMCL.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto" if device != "cpu" else None,
        )
        merged = PeftModel.from_pretrained(base, os.path.join(config.output_dir, "_adapter"))
        merged = merged.merge_and_unload()
        merged.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"  Merged model saved → {config.output_dir}")

    # ── Export GGUF ───────────────────────────────────────────────────────────
    if config.export_gguf and UNSLOTH:
        from finetune._scenario import FUNCTION_GGUF_FT
        gguf_path = FUNCTION_GGUF_FT
        # Derive the stem name (without .gguf) for Unsloth's save_pretrained_gguf
        gguf_name = os.path.splitext(os.path.basename(gguf_path))[0]

        # Backup existing GGUF
        if os.path.exists(gguf_path):
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = f"{gguf_path}.bak.{ts}"
            import shutil
            shutil.copy2(gguf_path, backup)
            print(f"  Backed up existing GGUF → {backup}")

        print(f"  Exporting GGUF ({config.gguf_quantization}) → {gguf_path}…")
        # No fallback: the only correct production artifact is Unsloth's
        # Q4_K_M output. `convert_qwen35_to_gguf.sh` produces a different
        # artifact (real f16, debug-suffixed path) — using it as a silent
        # fallback would have shipped an 8 GB f16 GGUF under the
        # production path, which is what the misleading `-f16` filename in
        # earlier revisions actually was. If Unsloth fails, surface it.
        model.save_pretrained_gguf(
            os.path.join(os.path.dirname(gguf_path), gguf_name),
            tokenizer,
            quantization_method=config.gguf_quantization,
        )
        print(f"  GGUF ready → {gguf_path}")
    elif config.export_gguf and not UNSLOTH:
        print("\n  NOTE: GGUF export requires Unsloth.")
        print("  Run manually: bash finetune/convert_qwen35_to_gguf.sh")

    _print_next_steps(config)


def _print_next_steps(config: Qwen35TrainConfig) -> None:
    from finetune._scenario import FUNCTION_GGUF_FT
    gguf = FUNCTION_GGUF_FT
    print(f"""
  ─────────────────────────────────────────────────────
  Next steps:

  1. Copy GGUF to deployment machine (M5 Max):
       scp {gguf} <m5max>:~/sources/local-multi-model-agent-slm/models/qwen3.5-4b-toolcalling-ft-merged/

  2. Activate Qwen tool-calling config:
       # In .env.local, set (replace <scenario> with the active scenario, e.g. nextera):
       FUNCTION_GGUF=models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-<scenario>-{config.gguf_quantization}.gguf
       FUNCTION_MODEL=qwen3.5-4b-toolcalling-ft

  3. Restart function server:
       bash scripts/start_servers.sh --bg

  4. Evaluate:
       python -m finetune.eval_tool_routing --save results/qwen35_finetuned.json

  5. Compare against baseline:
       python -m finetune.eval_tool_routing --compare \\
           results/qwen35_baseline.json results/qwen35_finetuned.json

  6. If ≥97% routing + ≥95% expression correctness:
       The deterministic scaffolding pre-routers are already retired —
       ToolUseHandler routes everything through Qwen FT (see scaffolding/README.md).
  ─────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B for tool calling (QLoRA)")
    parser.add_argument("--epochs",        type=int,   default=2)
    parser.add_argument("--lora-r",        type=int,   default=16)
    parser.add_argument("--lora-alpha",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--batch-size",    type=int,   default=1)
    parser.add_argument("--grad-accum",    type=int,   default=8)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--base-model",    type=str,   default=None)
    parser.add_argument("--data",          type=str,   default=None)
    parser.add_argument("--output-dir",    type=str,   default=None)
    parser.add_argument("--no-gguf",       action="store_true", help="Skip GGUF export")
    parser.add_argument("--quant",         type=str,   default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization method (default: q4_k_m)")
    args = parser.parse_args()

    cfg = Qwen35TrainConfig(
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        seed=args.seed,
        export_gguf=not args.no_gguf,
        gguf_quantization=args.quant,
    )
    if args.base_model:
        cfg.base_model = args.base_model
    if args.data:
        cfg.data_path = args.data
    if args.output_dir:
        cfg.output_dir = args.output_dir

    train(config=cfg)
