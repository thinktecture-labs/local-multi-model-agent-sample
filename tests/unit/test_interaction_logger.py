"""Unit tests for InteractionLogger (extracted from agent.py)."""

import json
import os
import threading

import pytest

from src.engine.agent.types import ExecutionStep, Intent
from src.engine.agent.interaction_logger import InteractionLogger


@pytest.fixture
def logger_inst():
    return InteractionLogger()


@pytest.mark.unit
class TestInteractionLogger:
    def test_initial_state(self, logger_inst):
        assert logger_inst.interaction_count == 0
        assert logger_inst.total_tokens_generated == 0

    def test_log_increments_count(self, logger_inst):
        logger_inst.log(
            query="test", intent=Intent.DIRECT_ANSWER,
            response="ok", steps=[],
        )
        assert logger_inst.interaction_count == 1

    def test_log_accumulates_tokens(self, logger_inst):
        steps = [
            ExecutionStep(action="a", model="m", tokens_used=10),
            ExecutionStep(action="b", model="m", tokens_used=20),
        ]
        logger_inst.log(query="q", intent=Intent.TOOL_USE, response="r", steps=steps)
        assert logger_inst.total_tokens_generated == 30

    def test_log_entry_structure(self, logger_inst):
        steps = [ExecutionStep(action="test", model="test_model", tokens_used=5)]
        logger_inst.log(
            query="Hello", intent=Intent.DIRECT_ANSWER,
            response="Hi", steps=steps,
        )
        entry = logger_inst._log[0]
        assert "timestamp" in entry
        assert entry["query"] == "Hello"
        assert entry["intent"] == "direct_answer"
        assert entry["response"] == "Hi"
        assert len(entry["steps"]) == 1
        assert "models_used" in entry
        assert "total_tokens" in entry

    def test_export_creates_file(self, logger_inst, tmp_path):
        logger_inst.log(
            query="q", intent=Intent.DIRECT_ANSWER,
            response="r", steps=[],
        )
        filepath = str(tmp_path / "export.json")
        count = logger_inst.export(filepath)
        assert count == 1
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["query"] == "q"

    def test_export_creates_directories(self, logger_inst, tmp_path):
        logger_inst.log(
            query="q", intent=Intent.DIRECT_ANSWER,
            response="r", steps=[],
        )
        filepath = str(tmp_path / "deep" / "nested" / "out.json")
        logger_inst.export(filepath)
        assert os.path.exists(filepath)

    def test_thread_safety(self, logger_inst):
        """Concurrent logging should not corrupt state."""
        def log_n(n):
            for _ in range(n):
                logger_inst.log(
                    query="t", intent=Intent.DIRECT_ANSWER,
                    response="r",
                    steps=[ExecutionStep(action="a", model="m", tokens_used=1)],
                )

        threads = [threading.Thread(target=log_n, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert logger_inst.interaction_count == 200
        assert logger_inst.total_tokens_generated == 200

    def test_eviction_count_increments_on_overflow(self):
        log = InteractionLogger(max_size=3)
        for i in range(5):
            log.log(query=f"q{i}", intent=Intent.DIRECT_ANSWER, response="r", steps=[])
        assert log.eviction_count == 2
        assert log.interaction_count == 3
        assert log.total_interactions_logged == 5

    def test_no_eviction_below_capacity(self):
        log = InteractionLogger(max_size=10)
        for i in range(5):
            log.log(query=f"q{i}", intent=Intent.DIRECT_ANSWER, response="r", steps=[])
        assert log.eviction_count == 0
        assert log.total_interactions_logged == 5

    def test_export_warns_on_eviction(self, tmp_path, caplog):
        import logging
        log = InteractionLogger(max_size=2)
        for i in range(4):
            log.log(query=f"q{i}", intent=Intent.DIRECT_ANSWER, response="r", steps=[])
        with caplog.at_level(logging.WARNING, logger="src.engine.agent.interaction_logger"):
            log.export(str(tmp_path / "out.json"))
        assert "evicted" in caplog.text.lower()

    def test_export_no_warning_when_no_eviction(self, tmp_path, caplog):
        import logging
        log = InteractionLogger(max_size=100)
        log.log(query="q", intent=Intent.DIRECT_ANSWER, response="r", steps=[])
        with caplog.at_level(logging.WARNING, logger="src.engine.agent.interaction_logger"):
            log.export(str(tmp_path / "out.json"))
        assert "evicted" not in caplog.text.lower()
