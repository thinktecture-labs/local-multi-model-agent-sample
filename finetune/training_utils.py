"""
Convergence monitoring utilities for all training scripts.

Provides:
  - LossHistoryCallback  — captures train/eval loss per logging step
  - save_training_curves  — writes loss history to JSON
  - print_convergence_summary — prints convergence diagnostics after training

No external dependencies beyond transformers (already required by training scripts).
Import-guarded so tests can run without GPU deps.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

# Import guard — transformers is only available in the training environment
try:
    from transformers import TrainerCallback
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

    class TrainerCallback:  # type: ignore[no-redef]
        """Stub so LossHistoryCallback can be defined without transformers."""
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_EARLY_STOPPING_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Callback — captures train/eval loss from HuggingFace Trainer
# ---------------------------------------------------------------------------

class LossHistoryCallback(TrainerCallback):
    """
    Captures train_loss and eval_loss at each logging step.

    Usage:
        cb = LossHistoryCallback(output_dir="./models/foo", model_name="gemma3")
        trainer = SFTTrainer(..., callbacks=[cb])
        trainer.train()
        # on_train_end fires automatically — saves curves + prints summary
    """

    def __init__(self, output_dir: str = ".", model_name: str = "model"):
        self.output_dir = output_dir
        self.model_name = model_name
        self.history: list[dict[str, Any]] = []
        self._last_eval_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry: dict[str, Any] = {}
        if "loss" in logs:
            entry["step"] = state.global_step
            entry["train_loss"] = logs["loss"]
            entry["epoch"] = round(state.epoch, 2) if state.epoch else 0
        if "eval_loss" in logs:
            entry["step"] = state.global_step
            entry["eval_loss"] = logs["eval_loss"]
            entry["epoch"] = round(state.epoch, 2) if state.epoch else 0
            self._last_eval_loss = logs["eval_loss"]
        if entry:
            # Merge with existing entry for same step if present
            if self.history and self.history[-1].get("step") == entry.get("step"):
                self.history[-1].update(entry)
            else:
                self.history.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        save_training_curves(self.history, self.output_dir, self.model_name)
        print_convergence_summary(self.history, self.model_name)


# ---------------------------------------------------------------------------
# Save / Load curves
# ---------------------------------------------------------------------------

def save_training_curves(
    history: list[dict[str, Any]],
    output_dir: str,
    model_name: str,
) -> str:
    """
    Save training loss history to JSON.

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_training_curves.json")
    data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "steps": history,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Training curves saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Convergence summary
# ---------------------------------------------------------------------------

def print_convergence_summary(
    history: list[dict[str, Any]],
    model_name: str,
) -> None:
    """Print a convergence diagnostic after training."""
    print(f"\n  {'='*50}")
    print(f"  Convergence Summary — {model_name}")
    print(f"  {'='*50}")

    if not history:
        print("  No training history recorded.")
        return

    # Train loss trajectory
    train_entries = [e for e in history if "train_loss" in e]
    if train_entries:
        initial = train_entries[0]["train_loss"]
        final = train_entries[-1]["train_loss"]
        best_train = min(e["train_loss"] for e in train_entries)
        print(f"  Train loss:  {initial:.4f} -> {final:.4f}  (best: {best_train:.4f})")
    else:
        print("  Train loss:  not recorded")

    # Eval loss trajectory
    eval_entries = [e for e in history if "eval_loss" in e]
    if eval_entries:
        initial_eval = eval_entries[0]["eval_loss"]
        final_eval = eval_entries[-1]["eval_loss"]
        best_eval = min(e["eval_loss"] for e in eval_entries)
        best_step = next(e["step"] for e in eval_entries if e["eval_loss"] == best_eval)
        print(f"  Eval loss:   {initial_eval:.4f} -> {final_eval:.4f}  (best: {best_eval:.4f} @ step {best_step})")

        # Overfitting detection
        if final_eval > best_eval * 1.1:
            gap = ((final_eval - best_eval) / best_eval) * 100
            print(f"  WARNING: Final eval loss is {gap:.1f}% above best — possible overfitting")
            print(f"           Best checkpoint was at step {best_step}; consider fewer epochs or early stopping")
    else:
        print("  Eval loss:   not recorded (eval_strategy may be disabled)")

    total_steps = max((e.get("step", 0) for e in history), default=0)
    print(f"  Total steps: {total_steps}")
    print(f"  {'='*50}")
