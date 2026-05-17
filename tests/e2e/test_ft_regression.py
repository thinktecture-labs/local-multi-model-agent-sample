"""
Fine-tuned model regression tests — 351 golden queries against live servers.

Validates that FT models meet minimum accuracy thresholds for:
  1. Intent classification (180 queries from eval_gemma3)
  2. Tool routing (160 queries from eval_tool_routing)
  3. Response quality (garbage detection, empty response checks)
  4. Pipeline golden queries (11 showcase queries from demo.py)
  5. Synthesis smoke tests (response content spot checks)

These tests are designed for fine-tuned model servers. They will:
  - SKIP if no llama-server instances are running
  - Auto-detect FT ports (9094-9096) when available (--all cheat mode)
  - PASS with fine-tuned models loaded (meet accuracy thresholds)
  - FAIL with base models loaded (thresholds too high for base models)

Run:
  bash scripts/start_servers.sh --all --bg
  pytest tests/e2e/test_ft_regression.py -v

Addresses Code Review Issue #3: "No real-model regression tests"
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter

import pytest

from src.engine.agent import SmallLanguageModelAgentOrchestrator, Intent
from src.engine.inference.client import SmallLanguageModelClient
from src.engine.inference.config import SCENARIO_CONFIG
from src.engine.tools.tool_registry import create_default_registry
from src.engine.knowledge.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Accuracy thresholds — conservative to avoid flakiness, but catch regressions
# Current production: 95% intent, 95% tool routing
# These catch catastrophic failures (e.g., synthesis-only FT → 17% intent)
# ---------------------------------------------------------------------------

INTENT_THRESHOLD_OVERALL = 0.80
INTENT_THRESHOLD_RAG = 0.80
INTENT_THRESHOLD_TOOL = 0.85
INTENT_THRESHOLD_DIRECT = 0.65

TOOL_THRESHOLD_OVERALL = 0.80
TOOL_THRESHOLD_CALC = 0.65  # qwen misroutes some calc queries to sql_query
TOOL_THRESHOLD_SQL = 0.75

PIPELINE_SUCCESS_THRESHOLD = 0.90

MAX_RESPONSE_LENGTH = 2000

# Known garbage patterns from FT model failures (documented in MEMORY.md)
GARBAGE_PATTERNS = [
    re.compile(r"0{20,}"),            # Long zero strings (FT qwen)
    re.compile(r"(.)\1{50,}"),        # Any char repeated 50+ times
    re.compile(r"<\w+_of_turn>"),     # Leaked Gemma control tokens
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent(servers_available):
    """Build a fully-wired agent using FT ports when available.

    FT regression tests are designed for fine-tuned models. With --all cheat mode,
    FT servers run on 9094/9095/9096; this fixture auto-detects and uses them.
    Falls back to base ports (9090/9091/9092) if FT servers aren't running.
    """
    if not servers_available:
        pytest.skip(
            "FT regression tests require llama-servers.\n"
            "Start with: bash scripts/start_servers.sh --all --bg"
        )

    client = SmallLanguageModelClient.create_with_auto_detection()

    vector_store = VectorStore(persist_dir=SCENARIO_CONFIG.chroma_dir)
    vector_store.set_client(client)
    tools = create_default_registry(vector_store=vector_store)
    return SmallLanguageModelAgentOrchestrator(client=client, tools=tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent_to_str(intent: Intent) -> str:
    """Convert Intent enum to the string label used in eval test sets."""
    return intent.value


def _format_failures(failures: list[dict]) -> str:
    """Format misclassified queries for assertion messages."""
    lines = []
    for f in failures[:10]:  # Show first 10
        lines.append(f"  expected={f['expected']:<16} got={f['got']!r:<16} {f['query']!r}")
    if len(failures) > 10:
        lines.append(f"  ... and {len(failures) - 10} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Intent Classification Accuracy (180 queries)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestIntentAccuracy:
    """Run all 180 eval queries and assert per-class accuracy thresholds."""

    @pytest.fixture(scope="class")
    def intent_results(self, agent):
        """Run all 180 intent queries once, cache for all tests in this class."""
        from finetune.eval_gemma3 import TEST_SET

        async def _run():
            results = []
            for item in TEST_SET:
                response = await agent.process(item["query"])
                results.append({
                    "query": item["query"],
                    "expected": item["intent"],
                    "got": _intent_to_str(response.intent),
                    "correct": _intent_to_str(response.intent) == item["intent"],
                    "response": response,
                })
            return results

        return asyncio.run(_run())

    def test_overall_accuracy_above_threshold(self, intent_results):
        correct = sum(1 for r in intent_results if r["correct"])
        accuracy = correct / len(intent_results)
        failures = [r for r in intent_results if not r["correct"]]
        assert accuracy >= INTENT_THRESHOLD_OVERALL, (
            f"Intent accuracy {accuracy:.1%} ({correct}/{len(intent_results)}) "
            f"below threshold {INTENT_THRESHOLD_OVERALL:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_rag_query_accuracy(self, intent_results):
        rag = [r for r in intent_results if r["expected"] == "rag_query"]
        correct = sum(1 for r in rag if r["correct"])
        accuracy = correct / len(rag)
        failures = [r for r in rag if not r["correct"]]
        assert accuracy >= INTENT_THRESHOLD_RAG, (
            f"rag_query accuracy {accuracy:.1%} ({correct}/{len(rag)}) "
            f"below threshold {INTENT_THRESHOLD_RAG:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_tool_use_accuracy(self, intent_results):
        tool = [r for r in intent_results if r["expected"] == "tool_use"]
        correct = sum(1 for r in tool if r["correct"])
        accuracy = correct / len(tool)
        failures = [r for r in tool if not r["correct"]]
        assert accuracy >= INTENT_THRESHOLD_TOOL, (
            f"tool_use accuracy {accuracy:.1%} ({correct}/{len(tool)}) "
            f"below threshold {INTENT_THRESHOLD_TOOL:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_direct_answer_accuracy(self, intent_results):
        direct = [r for r in intent_results if r["expected"] == "direct_answer"]
        correct = sum(1 for r in direct if r["correct"])
        accuracy = correct / len(direct)
        failures = [r for r in direct if not r["correct"]]
        assert accuracy >= INTENT_THRESHOLD_DIRECT, (
            f"direct_answer accuracy {accuracy:.1%} ({correct}/{len(direct)}) "
            f"below threshold {INTENT_THRESHOLD_DIRECT:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_no_unknown_intents(self, intent_results):
        unknowns = [
            r for r in intent_results
            if r["got"] not in ("rag_query", "tool_use", "direct_answer", "image_query")
        ]
        assert len(unknowns) == 0, (
            f"{len(unknowns)} queries returned unknown intents:\n"
            f"{_format_failures(unknowns)}"
        )


# ---------------------------------------------------------------------------
# 2. Tool Routing Accuracy (160 queries)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestToolRouting:
    """Run 160 tool queries through the full pipeline and check tool selection."""

    @pytest.fixture(scope="class")
    def tool_results(self, agent):
        """Run all 40 tool queries, extract which tool was selected."""
        from finetune.eval_tool_routing import TEST_SET

        async def _run():
            results = []
            for item in TEST_SET:
                response = await agent.process(item["query"])

                # Extract tool from execution steps
                tool_selected = None
                for step in response.steps:
                    if step.action == "select_tool":
                        tool_selected = step.details.get("tool", "")
                        break
                    if step.action == "execute_tool":
                        tool_selected = step.details.get("tool", "")
                        break

                results.append({
                    "query": item["query"],
                    "expected": item["expected_tool"],
                    "got": tool_selected,
                    "correct": tool_selected == item["expected_tool"],
                    "intent": _intent_to_str(response.intent),
                    "response": response,
                })
            return results

        return asyncio.run(_run())

    def test_overall_tool_accuracy(self, tool_results):
        correct = sum(1 for r in tool_results if r["correct"])
        accuracy = correct / len(tool_results)
        failures = [r for r in tool_results if not r["correct"]]
        assert accuracy >= TOOL_THRESHOLD_OVERALL, (
            f"Tool routing accuracy {accuracy:.1%} ({correct}/{len(tool_results)}) "
            f"below threshold {TOOL_THRESHOLD_OVERALL:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_calculator_accuracy(self, tool_results):
        calc = [r for r in tool_results if r["expected"] == "calculator"]
        correct = sum(1 for r in calc if r["correct"])
        accuracy = correct / len(calc)
        failures = [r for r in calc if not r["correct"]]
        assert accuracy >= TOOL_THRESHOLD_CALC, (
            f"calculator accuracy {accuracy:.1%} ({correct}/{len(calc)}) "
            f"below threshold {TOOL_THRESHOLD_CALC:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )

    def test_sql_query_accuracy(self, tool_results):
        sql = [r for r in tool_results if r["expected"] == "sql_query"]
        correct = sum(1 for r in sql if r["correct"])
        accuracy = correct / len(sql)
        failures = [r for r in sql if not r["correct"]]
        assert accuracy >= TOOL_THRESHOLD_SQL, (
            f"sql_query accuracy {accuracy:.1%} ({correct}/{len(sql)}) "
            f"below threshold {TOOL_THRESHOLD_SQL:.0%}.\n"
            f"Misclassified:\n{_format_failures(failures)}"
        )


# ---------------------------------------------------------------------------
# 3. Response Quality (garbage detection)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestResponseQuality:
    """Check all intent query responses for quality issues."""

    @pytest.fixture(scope="class")
    def all_responses(self, agent):
        """Run a representative subset of queries and collect responses."""
        from finetune.eval_gemma3 import TEST_SET

        async def _run():
            results = []
            for item in TEST_SET:
                response = await agent.process(item["query"])
                results.append({
                    "query": item["query"],
                    "intent": item["intent"],
                    "response_text": response.response,
                    "success": response.success,
                    "response": response,
                })
            return results

        return asyncio.run(_run())

    def test_no_empty_responses(self, all_responses):
        empty = [
            r for r in all_responses
            if not r["response_text"] or not r["response_text"].strip()
        ]
        assert len(empty) == 0, (
            f"{len(empty)} queries returned empty responses:\n"
            + "\n".join(f"  [{r['intent']}] {r['query']!r}" for r in empty[:10])
        )

    def test_no_garbage_patterns(self, all_responses):
        garbage_hits = []
        for r in all_responses:
            text = r["response_text"] or ""
            for pattern in GARBAGE_PATTERNS:
                match = pattern.search(text)
                if match:
                    garbage_hits.append({
                        "query": r["query"],
                        "pattern": pattern.pattern,
                        "match": match.group()[:50],
                    })
                    break
        assert len(garbage_hits) == 0, (
            f"{len(garbage_hits)} responses contain garbage patterns:\n"
            + "\n".join(
                f"  pattern={h['pattern']!r} match={h['match']!r} query={h['query']!r}"
                for h in garbage_hits[:10]
            )
        )

    def test_responses_under_max_length(self, all_responses):
        too_long = [
            r for r in all_responses
            if r["response_text"] and len(r["response_text"]) > MAX_RESPONSE_LENGTH
        ]
        assert len(too_long) == 0, (
            f"{len(too_long)} responses exceed {MAX_RESPONSE_LENGTH} chars (runaway generation):\n"
            + "\n".join(
                f"  len={len(r['response_text'])} query={r['query']!r}"
                for r in too_long[:5]
            )
        )

    def test_pipeline_success_rate(self, all_responses):
        succeeded = sum(1 for r in all_responses if r["success"])
        rate = succeeded / len(all_responses)
        failed = [r for r in all_responses if not r["success"]]
        assert rate >= PIPELINE_SUCCESS_THRESHOLD, (
            f"Pipeline success rate {rate:.1%} ({succeeded}/{len(all_responses)}) "
            f"below threshold {PIPELINE_SUCCESS_THRESHOLD:.0%}.\n"
            f"Failed queries:\n"
            + "\n".join(f"  [{r['intent']}] {r['query']!r}" for r in failed[:10])
        )


# ---------------------------------------------------------------------------
# 4. Pipeline Golden Queries (11 showcase)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestPipelineGoldenQueries:
    """Run the 11 demo showcase queries — these are the on-stage demo queries."""

    # Map demo description prefixes to expected Intent
    _INTENT_MAP = {
        "RAG": Intent.RAG_QUERY,
        "TOOL": Intent.TOOL_USE,
        "MULTI-STEP": Intent.TOOL_USE,  # Multi-step is a sub-type of tool_use
        "DIRECT": Intent.DIRECT_ANSWER,
        "IMAGE": Intent.IMAGE_QUERY,
    }

    @pytest.fixture(scope="class")
    def showcase_results(self, agent):
        """Run all text-only showcase queries (skip image queries — need actual images)."""
        from demo import SHOWCASE_QUERIES

        async def _run():
            results = []
            for query, description, images in SHOWCASE_QUERIES:
                # Skip image queries — IMAGE_QUERY intent is deterministic (requires
                # images param), and we can't send actual image files in this test.
                if description.startswith("IMAGE"):
                    continue

                # Determine expected intent from description prefix
                prefix = description.split("→")[0].strip().split(" ")[0]
                expected_intent = self._INTENT_MAP.get(prefix)

                response = await agent.process(query)
                results.append({
                    "query": query,
                    "description": description,
                    "expected_intent": expected_intent,
                    "actual_intent": response.intent,
                    "success": response.success,
                    "response_text": response.response,
                })
            return results

        return asyncio.run(_run())

    def test_all_showcase_queries_succeed(self, showcase_results):
        failed = [r for r in showcase_results if not r["success"]]
        assert len(failed) == 0, (
            f"{len(failed)}/{len(showcase_results)} showcase queries failed:\n"
            + "\n".join(f"  {r['description']}: {r['query']!r}" for r in failed)
        )

    def test_showcase_queries_correct_intents(self, showcase_results):
        wrong = [
            r for r in showcase_results
            if r["expected_intent"] and r["actual_intent"] != r["expected_intent"]
        ]
        correct = len(showcase_results) - len(wrong)
        assert len(wrong) == 0, (
            f"{correct}/{len(showcase_results)} showcase queries got correct intent.\n"
            f"Wrong:\n"
            + "\n".join(
                f"  expected={r['expected_intent'].value} got={r['actual_intent'].value} "
                f"{r['query']!r}"
                for r in wrong
            )
        )


# ---------------------------------------------------------------------------
# 5. Synthesis Smoke Tests (response content spot checks)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestSynthesisSmoke:
    """Spot-check that responses contain meaningful content, not just correct intents."""

    def test_calculator_response_contains_number(self, agent):
        response = asyncio.run(agent.process("What is 15% of 45000?"))
        assert response.success
        # Expected: 6750 (or 6,750)
        assert re.search(r"6[,.]?750", response.response), (
            f"Calculator response doesn't contain expected result 6750.\n"
            f"Response: {response.response!r}"
        )

    def test_sql_response_references_data(self, agent):
        q = "How many customers joined in 2024?"
        response = asyncio.run(agent.process(q))
        assert response.success
        # Response should contain at least one number (the count)
        assert re.search(r"\d+", response.response), (
            f"SQL response contains no numbers.\nResponse: {response.response!r}"
        )

    def test_direct_answer_is_conversational(self, agent):
        response = asyncio.run(agent.process("Hello!"))
        assert response.success
        text = response.response.strip()
        assert len(text) > 5, f"Direct answer too short: {text!r}"
        # Should not contain raw SQL or JSON
        assert "SELECT" not in text.upper(), f"Direct answer contains SQL: {text!r}"
        assert not text.startswith("{"), f"Direct answer is raw JSON: {text!r}"

    def test_rag_response_has_substance(self, agent):
        q = "What is the Enterprise plan?"
        response = asyncio.run(agent.process(q))
        assert response.success
        assert len(response.response.strip()) > 50, (
            f"RAG response too short ({len(response.response.strip())} chars).\n"
            f"Response: {response.response!r}"
        )
