"""
Unit tests for finetune/training_utils.py.

Tests the convergence monitoring utilities: LossHistoryCallback, save/load
curves, and convergence summary printing.

All tests are pure-Python — no GPU, no external services.
"""

import json
import pytest

from finetune.training_utils import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EARLY_STOPPING_THRESHOLD,
    LossHistoryCallback,
    print_convergence_summary,
    save_training_curves,
)


# ---------------------------------------------------------------------------
# Helpers — minimal State/Args stubs for callback testing
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self, global_step: int = 0, epoch: float = 0.0):
        self.global_step = global_step
        self.epoch = epoch


class _FakeArgs:
    pass


class _FakeControl:
    pass


def _make_history(
    steps: int = 5,
    train_start: float = 2.5,
    train_end: float = 0.3,
    eval_start: float = 2.8,
    eval_end: float = 0.5,
    include_eval: bool = True,
) -> list[dict]:
    """Generate a synthetic loss history."""
    history = []
    for i in range(steps):
        frac = i / max(1, steps - 1)
        entry = {
            "step": (i + 1) * 10,
            "train_loss": train_start + (train_end - train_start) * frac,
            "epoch": round(frac * 3, 2),
        }
        if include_eval:
            entry["eval_loss"] = eval_start + (eval_end - eval_start) * frac
        history.append(entry)
    return history


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstants:
    def test_patience_value(self):
        assert DEFAULT_EARLY_STOPPING_PATIENCE == 3

    def test_threshold_value(self):
        assert DEFAULT_EARLY_STOPPING_THRESHOLD == 0.01

    def test_patience_is_int(self):
        assert isinstance(DEFAULT_EARLY_STOPPING_PATIENCE, int)

    def test_threshold_is_positive(self):
        assert DEFAULT_EARLY_STOPPING_THRESHOLD > 0


# ---------------------------------------------------------------------------
# LossHistoryCallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLossHistoryCallback:
    def test_captures_train_loss(self):
        cb = LossHistoryCallback()
        state = _FakeState(global_step=10, epoch=0.5)
        cb.on_log(_FakeArgs(), state, _FakeControl(), logs={"loss": 1.23})
        assert len(cb.history) == 1
        assert cb.history[0]["train_loss"] == 1.23
        assert cb.history[0]["step"] == 10

    def test_captures_eval_loss(self):
        cb = LossHistoryCallback()
        state = _FakeState(global_step=20, epoch=1.0)
        cb.on_log(_FakeArgs(), state, _FakeControl(), logs={"eval_loss": 0.89})
        assert len(cb.history) == 1
        assert cb.history[0]["eval_loss"] == 0.89

    def test_merges_same_step(self):
        cb = LossHistoryCallback()
        state = _FakeState(global_step=10, epoch=0.5)
        cb.on_log(_FakeArgs(), state, _FakeControl(), logs={"loss": 1.5})
        cb.on_log(_FakeArgs(), state, _FakeControl(), logs={"eval_loss": 1.8})
        assert len(cb.history) == 1
        assert cb.history[0]["train_loss"] == 1.5
        assert cb.history[0]["eval_loss"] == 1.8

    def test_separate_steps(self):
        cb = LossHistoryCallback()
        cb.on_log(_FakeArgs(), _FakeState(10, 0.5), _FakeControl(), logs={"loss": 1.5})
        cb.on_log(_FakeArgs(), _FakeState(20, 1.0), _FakeControl(), logs={"loss": 1.2})
        assert len(cb.history) == 2

    def test_ignores_none_logs(self):
        cb = LossHistoryCallback()
        cb.on_log(_FakeArgs(), _FakeState(10, 0.5), _FakeControl(), logs=None)
        assert len(cb.history) == 0

    def test_ignores_irrelevant_logs(self):
        cb = LossHistoryCallback()
        cb.on_log(_FakeArgs(), _FakeState(10, 0.5), _FakeControl(), logs={"learning_rate": 1e-4})
        assert len(cb.history) == 0

    def test_epoch_captured(self):
        cb = LossHistoryCallback()
        cb.on_log(_FakeArgs(), _FakeState(10, 2.75), _FakeControl(), logs={"loss": 0.5})
        assert cb.history[0]["epoch"] == 2.75

    def test_on_train_end_calls_save_and_print(self, tmp_path, capsys):
        cb = LossHistoryCallback(output_dir=str(tmp_path), model_name="test_model")
        cb.on_log(_FakeArgs(), _FakeState(10, 0.5), _FakeControl(), logs={"loss": 2.0})
        cb.on_log(_FakeArgs(), _FakeState(10, 0.5), _FakeControl(), logs={"eval_loss": 2.2})
        cb.on_log(_FakeArgs(), _FakeState(20, 1.0), _FakeControl(), logs={"loss": 0.5})
        cb.on_log(_FakeArgs(), _FakeState(20, 1.0), _FakeControl(), logs={"eval_loss": 0.6})
        cb.on_train_end(_FakeArgs(), _FakeState(20, 1.0), _FakeControl())

        # Check JSON file was saved
        curves_path = tmp_path / "test_model_training_curves.json"
        assert curves_path.exists()

        # Check summary was printed
        captured = capsys.readouterr()
        assert "Convergence Summary" in captured.out
        assert "test_model" in captured.out


# ---------------------------------------------------------------------------
# save_training_curves
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveTrainingCurves:
    def test_creates_json_file(self, tmp_path):
        history = _make_history(steps=3)
        path = save_training_curves(history, str(tmp_path), "gemma3")
        assert path.endswith("gemma3_training_curves.json")
        assert (tmp_path / "gemma3_training_curves.json").exists()

    def test_json_structure(self, tmp_path):
        history = _make_history(steps=3)
        path = save_training_curves(history, str(tmp_path), "gemma3")
        with open(path) as f:
            data = json.load(f)
        assert data["model"] == "gemma3"
        assert "timestamp" in data
        assert len(data["steps"]) == 3

    def test_creates_parent_dirs(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c")
        path = save_training_curves([], nested, "test")
        assert (tmp_path / "a" / "b" / "c" / "test_training_curves.json").exists()

    def test_empty_history(self, tmp_path):
        path = save_training_curves([], str(tmp_path), "empty")
        with open(path) as f:
            data = json.load(f)
        assert data["steps"] == []

    def test_roundtrip_values(self, tmp_path):
        history = [{"step": 10, "train_loss": 1.234, "eval_loss": 1.567, "epoch": 0.5}]
        path = save_training_curves(history, str(tmp_path), "rt")
        with open(path) as f:
            data = json.load(f)
        step = data["steps"][0]
        assert step["train_loss"] == pytest.approx(1.234)
        assert step["eval_loss"] == pytest.approx(1.567)


# ---------------------------------------------------------------------------
# print_convergence_summary
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrintConvergenceSummary:
    def test_normal_convergence(self, capsys):
        history = _make_history(steps=5)
        print_convergence_summary(history, "gemma3")
        captured = capsys.readouterr()
        assert "gemma3" in captured.out
        assert "Train loss:" in captured.out
        assert "Eval loss:" in captured.out
        assert "Total steps:" in captured.out

    def test_empty_history(self, capsys):
        print_convergence_summary([], "empty")
        captured = capsys.readouterr()
        assert "No training history" in captured.out

    def test_train_only(self, capsys):
        history = _make_history(steps=3, include_eval=False)
        print_convergence_summary(history, "no_eval")
        captured = capsys.readouterr()
        assert "Train loss:" in captured.out
        assert "not recorded" in captured.out

    def test_overfitting_warning(self, capsys):
        # Eval loss gets worse: best at step 20, final at step 50 is 20% higher
        history = [
            {"step": 10, "train_loss": 2.0, "eval_loss": 2.0, "epoch": 0.5},
            {"step": 20, "train_loss": 1.0, "eval_loss": 1.0, "epoch": 1.0},
            {"step": 30, "train_loss": 0.5, "eval_loss": 1.1, "epoch": 1.5},
            {"step": 40, "train_loss": 0.3, "eval_loss": 1.3, "epoch": 2.0},
            {"step": 50, "train_loss": 0.2, "eval_loss": 1.5, "epoch": 2.5},
        ]
        print_convergence_summary(history, "overfit_test")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "overfitting" in captured.out

    def test_no_warning_when_converged(self, capsys):
        history = _make_history(steps=5, eval_start=2.0, eval_end=0.5)
        print_convergence_summary(history, "converged")
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_shows_best_step(self, capsys):
        history = [
            {"step": 10, "eval_loss": 2.0, "epoch": 0.5},
            {"step": 20, "eval_loss": 1.0, "epoch": 1.0},
            {"step": 30, "eval_loss": 1.5, "epoch": 1.5},
        ]
        print_convergence_summary(history, "best_step_test")
        captured = capsys.readouterr()
        assert "step 20" in captured.out
