"""
Shared helpers for data preparation modules.

Used by data_prep_gemma3.py and data_prep_embeddinggemma.py.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_interactions(filepath: str = "./data/interactions.json") -> list[dict]:
    """Load interaction logs written by SmallLanguageModelAgentOrchestrator."""
    path = Path(filepath)
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def save_jsonl(records: list[dict], output_path: str) -> int:
    """Write records as JSON Lines. Returns record count."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
