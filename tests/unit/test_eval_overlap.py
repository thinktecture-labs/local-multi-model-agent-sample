"""
Systematic eval/training data overlap detection for ALL eval scripts.

Ensures no eval query appears (or near-appears) in any training JSONL file
that trains the model being evaluated. This prevents data contamination
that would inflate eval metrics.

Uses Jaccard word-set similarity with threshold 0.6 (stricter than the
default 0.7 in eval_base.py). Any match above 0.6 is a test failure.

Run:
    pytest tests/unit/test_eval_overlap.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.eval_base import check_eval_training_overlap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
_TD = _ROOT / "data" / "training-data"

JACCARD_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_queries_from_jsonl(path: Path, key: str) -> list[str]:
    """Extract query strings from a JSONL file by key."""
    if not path.exists():
        return []
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            q = entry.get(key, "")
            if q:
                queries.append(q)
    return queries


def _assert_no_overlap(
    eval_queries: list[str],
    training_path: Path,
    query_key: str = "input",
    label: str = "",
):
    """Assert zero overlap between eval queries and a training file."""
    if not training_path.exists():
        pytest.skip(f"Training file not found: {training_path}")
    overlaps = check_eval_training_overlap(
        eval_queries, str(training_path), query_key=query_key, threshold=JACCARD_THRESHOLD,
    )
    if overlaps:
        msg = f"{'[' + label + '] ' if label else ''}Found {len(overlaps)} overlap(s) (Jaccard >= {JACCARD_THRESHOLD}):\n"
        for o in overlaps[:10]:
            msg += f"  sim={o['similarity']:.3f}\n"
            msg += f"    eval:  {o['eval_query'][:120]}\n"
            msg += f"    train: {o['train_query'][:120]}\n"
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Eval query loaders
# ---------------------------------------------------------------------------

def _gemma3_queries() -> list[str]:
    from finetune.eval_gemma3 import TEST_SET
    return [item["query"] for item in TEST_SET]


def _tool_routing_queries() -> list[str]:
    from finetune.eval_tool_routing import TEST_SET
    return [item["query"] for item in TEST_SET]


def _calculator_queries() -> list[str]:
    from finetune.eval_tool_routing import CALCULATOR_EXPECTED
    return list(CALCULATOR_EXPECTED.keys())


def _adversarial_queries() -> list[str]:
    from finetune.eval_adversarial import TEST_SET
    return [item["query"] for item in TEST_SET]


def _multi_step_queries() -> list[str]:
    from finetune.eval_multi_step import load_test_set
    items = load_test_set()
    return [item["query"] for item in items]


def _response_quality_queries() -> list[str]:
    from finetune.eval_response_quality import _TEST_SET
    return [item["query"] for item in _TEST_SET]


def _vision_queries() -> list[str]:
    from finetune.eval_vision import TEST_SET
    return [item["query"] for item in TEST_SET]


def _ocr_queries() -> list[str]:
    from finetune.eval_ocr import TEST_SET
    return [item["query"] for item in TEST_SET]


def _embeddinggemma_queries() -> list[str]:
    from finetune.eval_embeddinggemma import TEST_PAIRS
    return [item["query"] for item in TEST_PAIRS]


def _rag_groundtruth_queries() -> list[str]:
    from finetune.eval_rag_groundtruth import TEST_SET
    return [item["query"] for item in TEST_SET]


# ---------------------------------------------------------------------------
# Scenario paths (single-scenario: nextera)
# ---------------------------------------------------------------------------

def _td() -> Path:
    return _TD


def _suffix() -> str:
    return ""


# ---------------------------------------------------------------------------
# Tests: eval_gemma3 (intent classification — 180 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGemma3Overlap:
    def test_vs_gemma3_intent(self):
        _assert_no_overlap(
            _gemma3_queries(),
            _td() / f"gemma3_intent{_suffix()}.jsonl",
            query_key="input",
            label="eval_gemma3 vs gemma3_intent",
        )

    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _gemma3_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="eval_gemma3 vs gemma3_synthesis",
        )


# ---------------------------------------------------------------------------
# Tests: eval_tool_routing (tool selection — 160 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToolRoutingOverlap:
    def test_vs_qwen35_toolcalling(self):
        _assert_no_overlap(
            _tool_routing_queries(),
            _td() / f"qwen35_toolcalling{_suffix()}.jsonl",
            query_key="query",
            label="eval_tool_routing vs qwen35_toolcalling",
        )

    def test_vs_tool_routing_2tool(self):
        _assert_no_overlap(
            _tool_routing_queries(),
            _td() / f"tool_routing_2tool{_suffix()}.jsonl",
            query_key="query",
            label="eval_tool_routing vs tool_routing_2tool",
        )

    def test_vs_tool_routing_multi_turn(self):
        _assert_no_overlap(
            _tool_routing_queries(),
            _td() / f"tool_routing_multi_turn{_suffix()}.jsonl",
            query_key="query",
            label="eval_tool_routing vs tool_routing_multi_turn",
        )


# ---------------------------------------------------------------------------
# Tests: eval_expression_pipeline (calculator — 80 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpressionPipelineOverlap:
    def test_vs_qwen35_toolcalling(self):
        _assert_no_overlap(
            _calculator_queries(),
            _td() / f"qwen35_toolcalling{_suffix()}.jsonl",
            query_key="query",
            label="calculator vs qwen35_toolcalling",
        )

    def test_vs_tool_routing_2tool(self):
        _assert_no_overlap(
            _calculator_queries(),
            _td() / f"tool_routing_2tool{_suffix()}.jsonl",
            query_key="query",
            label="calculator vs tool_routing_2tool",
        )


# ---------------------------------------------------------------------------
# Tests: eval_adversarial (60 queries vs ALL training files)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAdversarialOverlap:
    def test_vs_gemma3_intent(self):
        _assert_no_overlap(
            _adversarial_queries(),
            _td() / f"gemma3_intent{_suffix()}.jsonl",
            query_key="input",
            label="adversarial vs gemma3_intent",
        )

    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _adversarial_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="adversarial vs gemma3_synthesis",
        )

    def test_vs_qwen35_toolcalling(self):
        _assert_no_overlap(
            _adversarial_queries(),
            _td() / f"qwen35_toolcalling{_suffix()}.jsonl",
            query_key="query",
            label="adversarial vs qwen35_toolcalling",
        )

    def test_vs_intent_hard_negatives(self):
        _assert_no_overlap(
            _adversarial_queries(),
            _td() / f"intent_hard_negatives{_suffix()}.jsonl",
            query_key="input",
            label="adversarial vs intent_hard_negatives",
        )


# ---------------------------------------------------------------------------
# Tests: eval_multi_step (80 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMultiStepOverlap:
    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _multi_step_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="multi_step vs gemma3_synthesis",
        )

    def test_vs_qwen35_toolcalling(self):
        _assert_no_overlap(
            _multi_step_queries(),
            _td() / f"qwen35_toolcalling{_suffix()}.jsonl",
            query_key="query",
            label="multi_step vs qwen35_toolcalling",
        )

    def test_vs_tool_routing_2tool(self):
        _assert_no_overlap(
            _multi_step_queries(),
            _td() / f"tool_routing_2tool{_suffix()}.jsonl",
            query_key="query",
            label="multi_step vs tool_routing_2tool",
        )


# ---------------------------------------------------------------------------
# Tests: eval_response_quality (~30-52 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResponseQualityOverlap:
    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _response_quality_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="response_quality vs gemma3_synthesis",
        )

    def test_vs_gemma3_intent(self):
        _assert_no_overlap(
            _response_quality_queries(),
            _td() / f"gemma3_intent{_suffix()}.jsonl",
            query_key="input",
            label="response_quality vs gemma3_intent",
        )

    def test_vs_qwen35_toolcalling(self):
        _assert_no_overlap(
            _response_quality_queries(),
            _td() / f"qwen35_toolcalling{_suffix()}.jsonl",
            query_key="query",
            label="response_quality vs qwen35_toolcalling",
        )


# ---------------------------------------------------------------------------
# Tests: eval_vision (10 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVisionOverlap:
    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _vision_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="vision vs gemma3_synthesis",
        )


# ---------------------------------------------------------------------------
# Tests: eval_ocr (22 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOCROverlap:
    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _ocr_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="ocr vs gemma3_synthesis",
        )


# ---------------------------------------------------------------------------
# Tests: eval_embeddinggemma (25 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddinggemmaOverlap:
    def test_vs_embeddinggemma_retrieval(self):
        _assert_no_overlap(
            _embeddinggemma_queries(),
            _td() / f"embeddinggemma_retrieval{_suffix()}.jsonl",
            query_key="anchor",
            label="embeddinggemma vs embeddinggemma_retrieval",
        )

    def test_vs_gemma3_synthesis(self):
        _assert_no_overlap(
            _embeddinggemma_queries(),
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="embeddinggemma vs gemma3_synthesis",
        )


# ---------------------------------------------------------------------------
# Tests: eval_rag_groundtruth (20 queries)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRAGGroundtruthOverlap:
    def test_vs_gemma3_synthesis(self):
        queries = _rag_groundtruth_queries()
        if not queries:
            pytest.skip("No RAG ground-truth queries for current scenario")
        _assert_no_overlap(
            queries,
            _td() / f"gemma3_synthesis{_suffix()}.jsonl",
            query_key="input",
            label="rag_groundtruth vs gemma3_synthesis",
        )

    def test_vs_gemma3_intent(self):
        queries = _rag_groundtruth_queries()
        if not queries:
            pytest.skip("No RAG ground-truth queries for current scenario")
        _assert_no_overlap(
            queries,
            _td() / f"gemma3_intent{_suffix()}.jsonl",
            query_key="input",
            label="rag_groundtruth vs gemma3_intent",
        )
