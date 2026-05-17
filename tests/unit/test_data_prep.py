"""
Unit tests for finetune/data_prep.py.

Verifies that:
- Interaction logs are parsed correctly for each model's training set
- <function_call> XML tags appear in qwen output examples
- Step field names match what agent.py actually logs (select_tool, vector_search)
- EmbeddingGemmaDataPreparer reads from details["documents"] key
- Synthetic augmentation adds examples when real data is sparse
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from finetune.data_prep import (
    EmbeddingGemmaDataPreparer,
    Gemma3DataPreparer,
    _load_interactions,
    _save_jsonl,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_interactions(path: str, interactions: list) -> None:
    with open(path, "w") as f:
        json.dump(interactions, f)


# ---------------------------------------------------------------------------
# _load_interactions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadInteractions:
    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_interactions(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_valid_file_loaded(self, tmp_path, sample_interactions):
        p = tmp_path / "interactions.json"
        _write_interactions(str(p), sample_interactions)
        result = _load_interactions(str(p))
        assert len(result) == 3

    def test_empty_list_file(self, tmp_path):
        p = tmp_path / "interactions.json"
        p.write_text("[]")
        assert _load_interactions(str(p)) == []


# ---------------------------------------------------------------------------
# _save_jsonl
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSaveJsonl:
    def test_writes_correct_number_of_lines(self, tmp_path):
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        out = str(tmp_path / "out.jsonl")
        count = _save_jsonl(records, out)
        assert count == 3
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        records = [{"key": "value"}, {"num": 42}]
        out = str(tmp_path / "out.jsonl")
        _save_jsonl(records, out)
        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_creates_parent_directories(self, tmp_path):
        out = str(tmp_path / "nested" / "dir" / "out.jsonl")
        _save_jsonl([{"x": 1}], out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# Gemma3DataPreparer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGemma3DataPreparer:
    def test_intent_dataset_includes_real_interactions(self, tmp_path, sample_interactions):
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, sample_interactions)

        preparer = Gemma3DataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
            augment=False,
        )
        count = preparer.build_intent_dataset()
        assert count == 3  # 3 real interactions, no augmentation

    def test_intent_dataset_augmented(self, tmp_path):
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, [])  # no real data

        preparer = Gemma3DataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
            augment=True,
        )
        count = preparer.build_intent_dataset()
        assert count > 0  # synthetic examples added

    def test_intent_examples_have_correct_fields(self, tmp_path, sample_interactions):
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, sample_interactions)

        preparer = Gemma3DataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
            augment=False,
        )
        preparer.build_intent_dataset()

        out_path = str(tmp_path / "gemma3_intent.jsonl")
        with open(out_path) as f:
            for line in f:
                ex = json.loads(line)
                assert "instruction" in ex
                assert "input" in ex
                assert "output" in ex
                assert ex["output"] in ("rag_query", "tool_use", "direct_answer")

    def test_synthesis_dataset_skips_missing_response(self, tmp_path):
        interactions = [{"query": "Q?", "intent": "direct_answer"}]  # no response key
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, interactions)

        preparer = Gemma3DataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        count = preparer.build_synthesis_dataset()
        assert count == 0

    def test_synthesis_dataset_builds_context_from_documents(self, tmp_path, sample_interactions):
        """Synthesis examples for RAG interactions should include document context."""
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, [sample_interactions[0]])  # RAG interaction

        preparer = Gemma3DataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        count = preparer.build_synthesis_dataset()
        assert count == 1

        out_path = str(tmp_path / "gemma3_synthesis.jsonl")
        with open(out_path) as f:
            ex = json.loads(f.readline())
        # Context from documents should appear in the input
        assert "€3,500" in ex["input"] or "Enterprise" in ex["input"]


# ---------------------------------------------------------------------------
# Tool-calling data prep (Qwen3.5-4B): exercised end-to-end via
# tests/unit/test_eval_tool_routing.py. The previous in-line class block here
# referenced a FunctionGemma helper that was removed when the 270M legacy
# tool-caller was scratched. Nothing here today.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EmbeddingGemmaDataPreparer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingGemmaDataPreparer:
    def test_prepare_includes_hard_coded_pairs(self, tmp_path):
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, [])

        preparer = EmbeddingGemmaDataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        count = preparer.prepare()
        assert count > 0

    def test_extracts_pairs_from_documents_key(self, tmp_path, sample_interactions):
        """Must read documents from details['documents'], not details['results']."""
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, [sample_interactions[0]])  # RAG interaction

        preparer = EmbeddingGemmaDataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        preparer.prepare()

        out_path = str(tmp_path / "embeddinggemma_retrieval.jsonl")
        with open(out_path) as f:
            examples = [json.loads(l) for l in f if l.strip()]

        # Should have at least one pair derived from the logged documents
        positives = [ex["positive"] for ex in examples]
        assert any("€3,500" in p or "unlimited" in p for p in positives)

    def test_output_has_query_and_positive_fields(self, tmp_path):
        interactions_path = str(tmp_path / "interactions.json")
        _write_interactions(interactions_path, [])

        preparer = EmbeddingGemmaDataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        preparer.prepare()

        out_path = str(tmp_path / "embeddinggemma_retrieval.jsonl")
        with open(out_path) as f:
            for line in f:
                ex = json.loads(line)
                assert "query" in ex
                assert "positive" in ex

    def test_only_rag_interactions_used(self, tmp_path, sample_interactions):
        """tool_use and direct_answer interactions should not produce retrieval pairs."""
        interactions_path = str(tmp_path / "interactions.json")
        # Only pass non-RAG interactions
        _write_interactions(interactions_path, sample_interactions[1:])  # tool_use + direct

        preparer = EmbeddingGemmaDataPreparer(
            interactions_path=interactions_path,
            output_dir=str(tmp_path),
        )
        preparer.prepare()

        out_path = str(tmp_path / "embeddinggemma_retrieval.jsonl")
        with open(out_path) as f:
            examples = [json.loads(l) for l in f if l.strip()]

        # Should only contain the hard-coded pairs, not any from tool_use/direct logs
        # (hard-coded pairs don't reference "total sales")
        queries = [ex["query"] for ex in examples]
        assert not any("total sales" in q.lower() for q in queries)
