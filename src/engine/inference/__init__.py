"""
Inference — LLM client and configuration.

Re-exports the most commonly used names so downstream code can do:
  from src.engine.inference import SmallLanguageModelClient, SmallLanguageModelRole
"""

from .client import SmallLanguageModelClient, SmallLanguageModelRole, LLMResponse, StreamChunk, CircuitBreaker, CircuitOpenError

__all__ = [
    "SmallLanguageModelClient",
    "SmallLanguageModelRole",
    "LLMResponse",
    "StreamChunk",
    "CircuitBreaker",
    "CircuitOpenError",
]
