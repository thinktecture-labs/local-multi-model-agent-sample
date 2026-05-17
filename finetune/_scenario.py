"""
Shared scenario config loader for finetune scripts.

Reads paths from scenarios/<name>.json so finetune scripts don't need
hardcoded if/else chains. Adding a new tenant = adding a JSON file.
"""

import json
import os
from pathlib import Path


def _load() -> dict:
    name = os.getenv("SCENARIO", "nextera")
    scenarios_dir = Path(__file__).resolve().parents[1] / "scenarios"
    path = scenarios_dir / f"{name}.json"
    with open(path) as f:
        return json.load(f)


_DATA = _load()

SCENARIO_NAME: str = _DATA["name"]
TRAINING_DIR: str = _DATA["paths"]["training_data_dir"]
SUFFIX: str = _DATA["paths"]["training_data_suffix"]
LOGREG_MODEL_DIR: str = _DATA["paths"]["logreg_model_dir"]

# Fine-tuned model GGUFs — scenario-specific filenames
_MODELS = _DATA["models"]
INFERENCE_GGUF_FT: str = _MODELS["inference_gguf_ft"]
FUNCTION_GGUF_FT: str = _MODELS["function_gguf_ft"]
EMBEDDING_GGUF_FT: str = _MODELS["embedding_gguf_ft"]
SYNTHESIS_4B_GGUF_FT: str = _MODELS.get("synthesis_4b_gguf_ft", "")
