"""
Data preparation for gemma3 — intent classification + response synthesis.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict

from finetune.data_prep_shared import load_interactions, save_jsonl

from finetune._scenario import SCENARIO_NAME as _SCENARIO


# ---------------------------------------------------------------------------
# Intent classification + synthesis dataset (gemma3)
# ---------------------------------------------------------------------------

@dataclass
class Gemma3Example:
    """One supervised example for gemma3 fine-tuning (Alpaca chat format)."""
    instruction: str   # system context
    input:       str   # user query
    output:      str   # expected response


# The three intent labels gemma3 must classify
_INTENT_SYSTEM = (
    "You are an intent classifier. Given a user query, respond with exactly one "
    "label from: rag_query, tool_use, direct_answer. "
    "No explanation. Output only the label."
)

# Synthetic augmentation queries per intent — used when real logs are sparse
_AUGMENTATION_NEXTERA: dict[str, list[str]] = {
    "rag_query": [
        "What features does the Enterprise plan include?",
        "How does Nextera handle document retrieval?",
        "Is Nextera HIPAA compliant?",
        "What vector database does Nextera use?",
        "What is the difference between Starter and Professional plans?",
        "Does Nextera support fine-tuning on the Starter plan?",
        "What deployment options does Nextera offer?",
        "How does Nextera compare to cloud AI providers for privacy?",
    ],
    "tool_use": [
        "What were the total sales in Q4 2024?",
        "How many customers joined in the first half of 2024?",
        "Show me the revenue breakdown by quarter for 2023.",
        "Which customer has the highest MRR?",
        "List all Enterprise customers.",
        "If I have 30 users on the Professional plan, what is my annual spend?",
        "What is 15% of 84900?",
        "Calculate 50 * 999 * 12",
        "If ARR is €103200 and I add 5 new Enterprise customers, what is the new ARR?",
        "What is the monthly cost difference between Professional and Enterprise?",
    ],
    "direct_answer": [
        "Hello! What can you help me with?",
        "What is Nextera Platform?",
        "Can you summarise what you can do?",
        "Thanks for the help!",
        "Who built this agent?",
    ],
}

_AUGMENTATION = _AUGMENTATION_NEXTERA


class Gemma3DataPreparer:
    """
    Builds two training datasets for gemma3:

    1. **Intent classification** — teaches gemma3 to output the correct label
       (rag_query | tool_use | direct_answer).

    2. **Response synthesis** — teaches gemma3 to produce high-quality final
       responses given retrieved context or tool results.
    """

    def __init__(
        self,
        interactions_path:  str = "./data/interactions.json",
        output_dir:         str = "./data/training-data",
        min_examples:       int = 30,
        augment:            bool = True,
    ) -> None:
        self.interactions   = load_interactions(interactions_path)
        self.output_dir     = output_dir
        self.min_examples   = min_examples
        self.augment        = augment

    # --- Intent classification dataset ------------------------------------

    def build_intent_dataset(self) -> int:
        """Build intent-classification JSONL from logs + augmentation."""
        examples: list[dict] = []

        # Real interaction logs
        for interaction in self.interactions:
            if "query" not in interaction or "intent" not in interaction:
                continue
            ex = Gemma3Example(
                instruction=_INTENT_SYSTEM,
                input=interaction["query"],
                output=interaction["intent"],
            )
            examples.append(asdict(ex))

        # Synthetic augmentation for underrepresented classes
        if self.augment:
            for intent, queries in _AUGMENTATION.items():
                for q in queries:
                    ex = Gemma3Example(
                        instruction=_INTENT_SYSTEM,
                        input=q,
                        output=intent,
                    )
                    examples.append(asdict(ex))

        random.shuffle(examples)
        out = os.path.join(self.output_dir, "gemma3_intent.jsonl")
        count = save_jsonl(examples, out)
        print(f"  [gemma3] Intent dataset: {count} examples → {out}")
        return count

    # --- Response synthesis dataset --------------------------------------

    def build_synthesis_dataset(self) -> int:
        """
        Build a synthesis dataset: (query + context) → final answer.
        Uses the actual responses from logged interactions.
        """
        examples: list[dict] = []

        for interaction in self.interactions:
            if not interaction.get("response") or not interaction.get("query"):
                continue

            # Build context string from retrieval steps if available
            # agent.py stores retrieved docs under details["documents"]
            context_parts = []
            for step in interaction.get("steps", []):
                if step.get("action") == "vector_search":
                    for doc in step.get("details", {}).get("documents", [])[:3]:
                        if isinstance(doc, dict) and doc.get("content"):
                            context_parts.append(doc["content"][:300])

            context = "\n\n".join(context_parts) if context_parts else ""
            system = (
                "You are a helpful AI assistant for Nextera Platform. "
                "Answer questions clearly and concisely based on the provided context."
            )
            user_input = (
                f"Context:\n{context}\n\nQuestion: {interaction['query']}"
                if context
                else interaction["query"]
            )

            ex = Gemma3Example(
                instruction=system,
                input=user_input,
                output=interaction["response"],
            )
            examples.append(asdict(ex))

        if not examples:
            print("  [gemma3] No synthesis examples found in logs — skipping.")
            return 0

        out = os.path.join(self.output_dir, "gemma3_synthesis.jsonl")
        count = save_jsonl(examples, out)
        print(f"  [gemma3] Synthesis dataset: {count} examples → {out}")
        return count

    def prepare(self) -> dict[str, int]:
        os.makedirs(self.output_dir, exist_ok=True)
        return {
            "intent":    self.build_intent_dataset(),
            "synthesis": self.build_synthesis_dataset(),
        }
