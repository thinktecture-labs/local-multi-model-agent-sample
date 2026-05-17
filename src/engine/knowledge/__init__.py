"""
Knowledge — RAG pipeline for document storage and retrieval.

Re-exports the most commonly used names so downstream code can do:
  from src.engine.knowledge import VectorStore, Document
"""

from .vector_store import VectorStore, Document
from .ocr_client import OCRClient

__all__ = [
    "VectorStore",
    "Document",
    "OCRClient",
]
