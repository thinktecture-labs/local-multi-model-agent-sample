"""
Engine — the core runtime for the multi-model SLM agent.

Subpackages:
  engine.agent        — orchestration, intent classification, query decomposition
  engine.inference    — LLM client, configuration
  engine.scaffolding  — deterministic compensators (expression/SQL builders, confidence)
  engine.knowledge    — RAG pipeline (vector store, document processing)
  engine.tools        — pluggable tool framework
"""
