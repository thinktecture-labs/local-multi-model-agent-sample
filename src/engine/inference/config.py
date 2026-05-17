"""
Centralized configuration for the multi-model SLM agent.

Network timeouts, generation parameters, and tool limits live here
instead of being scattered across modules. Values with os.getenv()
can be overridden via .env or .env.local.

Scenario switching is multi-tenant: SCENARIO env var selects a JSON
config file from scenarios/<name>.json. Adding a new tenant requires
only a new JSON file — no code changes.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv
    load_dotenv(".env.local", override=True)
    load_dotenv(".env")
except ImportError:
    pass


# ─── Multi-tenant scenario configuration (loaded from JSON) ─────────────────

@dataclass(frozen=True)
class ScenarioConfig:
    """All scenario-specific settings. Loaded from scenarios/<name>.json."""
    name: str
    label: str                  # startup banner / API label
    brand: str                  # UI brand name
    language: str               # "en" | "de" | ...
    # Paths
    db_path: str
    docs_dir_name: str
    training_data_dir: str
    training_data_suffix: str   # Appended to JSONL filenames per scenario (e.g. "")
    demo_images_dir: str
    demo_documents_dir: str
    logreg_model_dir: str
    data_loader_module: str     # e.g. "data.loader"
    chroma_dir: str             # e.g. "./chroma_db"
    # Fine-tuned model GGUFs (scenario-specific)
    inference_gguf_ft: str      # e.g. "models/gemma3-1b-ft-merged/gemma3-1b-ft-nextera-f16.gguf"
    function_gguf_ft: str       # e.g. "models/qwen3.5-4b-toolcalling-ft-merged/qwen3.5-4b-toolcalling-ft-nextera-q4_k_m.gguf"
    embedding_gguf_ft: str      # e.g. "models/embeddinggemma-300m-ft-merged/embeddinggemma-300m-ft-nextera-q8_0.gguf"
    # Prompts
    direct_answer_system_prompt: str
    rag_synthesis_system_prompt: str
    extraction_system_prompt: str
    adversarial_refusal: str
    tool_format_prompt: str
    multi_step_synthesis_prompt: str
    rag_rewrite_prompt: str
    decomposer_fewshot: str        # multi-step query planner fewshot examples
    concretize_examples: str       # numeric substitution fewshot examples
    # SQL tool
    sql_allowed_tables: frozenset[str]
    sql_tool_description: str
    sql_parameter_description: str


def _load_scenario(name: str) -> ScenarioConfig:
    """Load a scenario from scenarios/<name>.json."""
    # Look in project root (up from src/engine/inference/)
    scenarios_dir = Path(__file__).resolve().parents[3] / "scenarios"
    path = scenarios_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario config not found: {path}\n"
            f"Available: {[p.stem for p in scenarios_dir.glob('*.json')]}"
        )
    with open(path) as f:
        data = json.load(f)
    paths = data["paths"]
    models = data["models"]
    prompts = data["prompts"]
    sql = data["sql"]
    return ScenarioConfig(
        name=data["name"],
        label=data["label"],
        brand=data["brand"],
        language=data["language"],
        db_path=paths["db"],
        docs_dir_name=paths["docs_dir_name"],
        training_data_dir=paths["training_data_dir"],
        training_data_suffix=paths["training_data_suffix"],
        demo_images_dir=paths["demo_images_dir"],
        demo_documents_dir=paths["demo_documents_dir"],
        logreg_model_dir=paths["logreg_model_dir"],
        data_loader_module=paths["data_loader_module"],
        chroma_dir=paths.get("chroma_dir", "./chroma_db"),
        inference_gguf_ft=models["inference_gguf_ft"],
        function_gguf_ft=models["function_gguf_ft"],
        embedding_gguf_ft=models["embedding_gguf_ft"],
        direct_answer_system_prompt=prompts["direct_answer_system"],
        rag_synthesis_system_prompt=prompts["rag_synthesis_system"],
        extraction_system_prompt=prompts["extraction_system"],
        adversarial_refusal=prompts["adversarial_refusal"],
        tool_format_prompt=prompts.get("tool_format_prompt", "Turn this tool result into a clear, helpful answer for the user.\n\nUser's question: {query}\nTool used: {tool_name}\nRaw result:\n{result_str}\n\nWrite a concise, human-readable answer:"),
        multi_step_synthesis_prompt=prompts.get("multi_step_synthesis_prompt", "Combine these tool results into a clear, helpful answer.\n\nUser's question: {query}\n\nResults:\n{results_str}\n\nWrite a concise answer that integrates all results:"),
        rag_rewrite_prompt=prompts.get("rag_rewrite_prompt", "Rewrite this query as a short, dense keyword phrase for semantic document search. Output ONLY the rewritten phrase. No explanation, no quotes, no punctuation at the end.\n\nOriginal: {query}\nRewritten:"),
        decomposer_fewshot=prompts.get("decomposer_fewshot", ""),
        concretize_examples=prompts.get("concretize_examples", ""),
        sql_allowed_tables=frozenset(sql["allowed_tables"]),
        sql_tool_description=sql["tool_description"],
        sql_parameter_description=sql["parameter_description"],
    )


SCENARIO = os.getenv("SCENARIO", "nextera")
SCENARIO_CONFIG = _load_scenario(SCENARIO)

# Flat exports for backward compatibility (used across 20+ modules)
DB_PATH = SCENARIO_CONFIG.db_path
DOCS_DIR_NAME = SCENARIO_CONFIG.docs_dir_name
TRAINING_DATA_DIR = SCENARIO_CONFIG.training_data_dir
DEMO_IMAGES_DIR = SCENARIO_CONFIG.demo_images_dir
DEMO_DOCUMENTS_DIR = SCENARIO_CONFIG.demo_documents_dir

# ─── Network ─────────────────────────────────────────────────────────────────
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "30.0"))

# ─── Pipeline deadline (caps total wall-time across all chained LLM calls) ──
PIPELINE_TIMEOUT = float(os.getenv("PIPELINE_TIMEOUT", "60.0"))

# ─── Interaction log (bounded circular buffer) ──────────────────────────────
INTERACTION_LOG_MAX_SIZE = int(os.getenv("INTERACTION_LOG_MAX_SIZE", "1000"))

# ─── Concurrency (per-model request cap to avoid saturating llama-server) ─────
MODEL_CONCURRENCY_LIMIT = int(os.getenv("MODEL_CONCURRENCY_LIMIT", "4"))

# ─── Circuit breaker (per-model fail-fast for crashed servers) ───────────────
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
CIRCUIT_BREAKER_RECOVERY = float(os.getenv("CIRCUIT_BREAKER_RECOVERY", "30.0"))

# ─── Intent classification (gemma3-ft generative fallback path only;
#     primary path is LogReg on embeddinggemma — see
#     src/engine/agent/intent_classifier_logreg.py) ──────────────────────────
CLASSIFY_TEMPERATURE = 0.0
CLASSIFY_MAX_TOKENS = 20

# ─── Query rewriting (gemma3-ft) ──────────────────────────────────────────────
REWRITE_TEMPERATURE = 0.1
REWRITE_MAX_TOKENS = 40

# ─── RAG synthesis (gemma3-ft) ────────────────────────────────────────────────
RAG_SYNTHESIS_TEMPERATURE = 0.1
RAG_SYNTHESIS_MAX_TOKENS = int(os.getenv("RAG_SYNTHESIS_MAX_TOKENS", "600"))

# ─── Tool response formatting (gemma3-ft) ─────────────────────────────────────
TOOL_FORMAT_TEMPERATURE = float(os.getenv("TOOL_FORMAT_TEMPERATURE", "0.2"))
TOOL_FORMAT_MAX_TOKENS = int(os.getenv("TOOL_FORMAT_MAX_TOKENS", "800"))
# Maximum chars of tool result JSON passed to the format-response LLM.
# Prevents context overflow when SQL returns many rows (e.g. 50 rows × 400 chars = 20KB).
# 6000 chars ≈ 2000 tokens, leaving ample room for prompt + completion within 8192 ctx.
TOOL_FORMAT_MAX_RESULT_CHARS = int(os.getenv("TOOL_FORMAT_MAX_RESULT_CHARS", "6000"))

# ─── Direct answer (gemma3-ft) ────────────────────────────────────────────────
DIRECT_ANSWER_TEMPERATURE = float(os.getenv("DIRECT_ANSWER_TEMPERATURE", "0.3"))
DIRECT_ANSWER_MAX_TOKENS = int(os.getenv("DIRECT_ANSWER_MAX_TOKENS", "200"))

# ─── Vision / image analysis (gemma3-4B) ─────────────────────────────────────
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "400"))

# ─── Qwen 3.5 function calling (recommended by Qwen team) ─────────────────
# These are applied when FUNCTION_MODEL contains "qwen".
# Source: https://huggingface.co/Qwen/Qwen3.5-4B (official sampling params)
# Non-thinking instruct mode: temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
# For tool calling we use temp=0.0 (greedy) since accuracy > creativity.
# Thinking is disabled server-side via --chat-template-kwargs + --reasoning-budget 0.
QWEN_FUNCTION_TEMPERATURE = float(os.getenv("QWEN_FUNCTION_TEMPERATURE", "0.0"))
QWEN_FUNCTION_TOP_P = float(os.getenv("QWEN_FUNCTION_TOP_P", "0.95"))
QWEN_FUNCTION_TOP_K = int(os.getenv("QWEN_FUNCTION_TOP_K", "20"))
QWEN_FUNCTION_PRESENCE_PENALTY = float(os.getenv("QWEN_FUNCTION_PRESENCE_PENALTY", "0.0"))

# ─── Multi-step tool use (gemma3-ft planner) ────────────────────────────────
DECOMPOSE_TEMPERATURE = 0.0
DECOMPOSE_MAX_TOKENS = 150
CONCRETIZE_TEMPERATURE = 0.0
CONCRETIZE_MAX_TOKENS = 80
MULTI_STEP_SYNTHESIS_TEMPERATURE = 0.2
MULTI_STEP_SYNTHESIS_MAX_TOKENS = int(os.getenv("MULTI_STEP_SYNTHESIS_MAX_TOKENS", "600"))

# ─── Tool limits ──────────────────────────────────────────────────────────────
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "200"))
VECTOR_SEARCH_MAX_K = int(os.getenv("VECTOR_SEARCH_MAX_K", "15"))
# Number of documents retrieved from the vector store (before synthesis window).
# Used by the RAG handler for curated KB queries. Kept at 7 for tight focus.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "7"))
# Number of retrieved docs passed to the synthesis model (synthesis window ≤ RAG_TOP_K).
# Used by the RAG handler for curated KB queries. Kept at 5 for 4B model quality.
RAG_CONTEXT_DOCS = int(os.getenv("RAG_CONTEXT_DOCS", "5"))
# Document chat uses higher values — uploaded docs have many small semantic chunks
# and aggregation queries (e.g. "how many sessions does speaker X have?") need broader retrieval.
DOC_CHAT_TOP_K = int(os.getenv("DOC_CHAT_TOP_K", "30"))
DOC_CHAT_CONTEXT_DOCS = int(os.getenv("DOC_CHAT_CONTEXT_DOCS", "20"))
# Minimum cosine similarity score for uploaded document chunks to be merged into
# general RAG results. Prevents weakly-matching uploads (e.g. a conference agenda)
# from polluting domain-specific queries. Curated KB chunks are always included.
UPLOAD_MERGE_MIN_SCORE = float(os.getenv("UPLOAD_MERGE_MIN_SCORE", "0.85"))
# Minimum seconds remaining on the pipeline deadline before abandoning a multi-step tool call.
MULTI_STEP_DEADLINE_BUFFER = float(os.getenv("MULTI_STEP_DEADLINE_BUFFER", "5.0"))

# ─── Semantic chunking (chonkie) ─────────────────────────────────────────────
# When true and embeddinggemma is reachable, uploaded documents are chunked
# using chonkie's SemanticChunker (embedding-based topic boundaries with
# Savitzky-Golay smoothing). Falls back to fixed-size chunking on failure.
SEMANTIC_CHUNKING_ENABLED = os.getenv("SEMANTIC_CHUNKING_ENABLED", "true").lower() == "true"
SEMANTIC_CHUNKING_THRESHOLD = float(os.getenv("SEMANTIC_CHUNKING_THRESHOLD", "0.7"))
SEMANTIC_CHUNKING_MAX_TOKENS = int(os.getenv("SEMANTIC_CHUNKING_MAX_TOKENS", "256"))
SEMANTIC_CHUNKING_MIN_SENTENCES = int(os.getenv("SEMANTIC_CHUNKING_MIN_SENTENCES", "1"))

# ─── Voice (Whisper STT + Piper TTS) ────────────────────────────────────────
WHISPER_PORT = int(os.getenv("WHISPER_PORT", "9097"))
WHISPER_URL = f"http://localhost:{WHISPER_PORT}"
PIPER_VOICE_EN = os.getenv("PIPER_VOICE_EN", "en_GB-alan-medium")
PIPER_VOICE_DE = os.getenv("PIPER_VOICE_DE", "de_DE-thorsten-high")
PIPER_VOICES_DIR = os.getenv("PIPER_VOICES_DIR", "models/piper")
AUDIO_CACHE_TTL = float(os.getenv("AUDIO_CACHE_TTL", "120.0"))
AUDIO_CACHE_MAX_ENTRIES = int(os.getenv("AUDIO_CACHE_MAX_ENTRIES", "50"))

# ─── OCR (GLM-OCR via llama-server — optional, upload-time only) ─────────────
OCR_PORT = int(os.getenv("OCR_PORT", "9098"))
OCR_URL = f"http://localhost:{OCR_PORT}/v1"
OCR_MODEL = os.getenv("OCR_MODEL", "glm-ocr")
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "4096"))
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "60.0"))

# ─── Qwen comparison server (dedicated port for three-path demo) ────────────
QWEN_PORT = int(os.getenv("QWEN_PORT", "9100"))
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-35b-a3b")

# ─── Prompt cache — n_keep (protect system prompt from context-window eviction) ─
# n_keep = number of tokens at the START of the prompt that llama-server must
# never evict when the context window fills up during a long conversation.
# Set to (system_message_tokens + chat_template_overhead) so the cached system
# prompt stays warm across turns. Values include ~8 Gemma3 chat-template tokens.
# Derivation: tokenise the prompt string in prompts.py + 8 template tokens, round up.
N_KEEP_DIRECT_ANSWER = int(os.getenv("N_KEEP_DIRECT_ANSWER", "20"))   # ~12 prompt + 8
N_KEEP_RAG_SYNTHESIS = int(os.getenv("N_KEEP_RAG_SYNTHESIS", "75"))   # ~65 prompt + 8 + margin
N_KEEP_VISION        = int(os.getenv("N_KEEP_VISION",        "35"))   # ~26 prompt + 8 + margin

# ─── Cloud cost comparison (GPT-5.4 default pricing per 1M tokens) ─────────
CLOUD_INPUT_COST_PER_1M = float(os.getenv("CLOUD_INPUT_COST_PER_1M", "2.50"))
CLOUD_OUTPUT_COST_PER_1M = float(os.getenv("CLOUD_OUTPUT_COST_PER_1M", "15.00"))

# ─── Cloud comparison (optional — set OPENAI_API_KEY to enable) ─────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_COMPARE_MODEL = os.getenv("OPENAI_COMPARE_MODEL", "gpt-5.4")
CLOUD_COMPARISON_ENABLED = bool(OPENAI_API_KEY)

# ─── CORS (comma-separated origins, default localhost only) ──────────────────
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]
