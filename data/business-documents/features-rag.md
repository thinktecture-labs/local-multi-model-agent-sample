---
title: RAG Pipeline Features
category: features
---
Nextera ships with a production-ready RAG pipeline out of the box. Features include: automatic document chunking (configurable overlap), hybrid search (dense + sparse BM25), re-ranking with cross-encoders, query rewriting for improved retrieval, multi-document synthesis, citation tracking with source attribution, and streaming responses. Supported document types: PDF, DOCX, TXT, Markdown, HTML, CSV. The vector store is backed by ChromaDB or Weaviate (configurable).
