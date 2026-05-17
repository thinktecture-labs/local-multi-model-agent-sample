"""
Fine-Tuning Package — improve the production model stack on your domain data.

Training pipelines (one per model in the production stack):

  finetune/data_prep.py                — Convert interaction logs → training datasets
  finetune/train_gemma3.py             — LoRA fine-tune Gemma3-1B (intent + classifier features)
  finetune/train_gemma3_4b.py          — LoRA fine-tune Gemma3-4B (RAG synthesis)
  finetune/train_qwen35_toolcalling.py — LoRA fine-tune Qwen3.5-4B (tool calling)
  finetune/train_embeddinggemma.py     — Contrastive fine-tune EmbeddingGemma (retrieval)
  training/train_intent_logreg.py      — Logistic-regression intent classifier on EmbeddingGemma vectors
  finetune/pipeline.py                 — Orchestrate the full fine-tuning workflow

Prerequisites (install fine-tuning extras):
  pip install -r requirements-finetune.txt

Typical workflow:
  1. Run the demo to collect interaction logs  →  data/interactions.json
  2. python -m finetune.pipeline --export-data   # prepare training datasets
  3. python -m finetune.pipeline --train-all     # train the stack
  4. Models are saved to ./models/ and converted to GGUF for llama-server
"""
