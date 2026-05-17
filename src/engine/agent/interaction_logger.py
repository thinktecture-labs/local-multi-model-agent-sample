"""
Interaction logging for fine-tuning data collection.

Thread-safe logging of all agent interactions, with export to JSON
for downstream data preparation and fine-tuning pipelines.

Uses a bounded circular buffer (collections.deque) to prevent unbounded
memory growth. Oldest entries are evicted when the buffer is full.
Max size is configurable via INTERACTION_LOG_MAX_SIZE (default 1000).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime

from .types import ExecutionStep, Intent
from ..inference.config import INTERACTION_LOG_MAX_SIZE

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Thread-safe interaction logger with bounded circular buffer."""

    def __init__(self, max_size: int | None = None) -> None:
        cap = max_size if max_size is not None else INTERACTION_LOG_MAX_SIZE
        self._log: deque[dict] = deque(maxlen=cap)
        self._lock = threading.Lock()
        self._total_tokens: int = 0
        self._total_logged: int = 0   # monotonic — includes evicted entries
        self._eviction_count: int = 0

    def log(
        self,
        query: str,
        intent: Intent,
        response: str,
        steps: list[ExecutionStep],
    ) -> None:
        """Persist an interaction for fine-tuning data collection."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query":     query,
            "intent":    intent.value,
            "response":  response,
            "steps":     [
                {"action": s.action, "model": s.model, "details": s.details,
                 "tokens_used": s.tokens_used}
                for s in steps
            ],
            "models_used": list({s.model for s in steps}),
            "total_tokens": sum(s.tokens_used for s in steps),
        }
        with self._lock:
            if len(self._log) == self._log.maxlen:
                self._eviction_count += 1
            self._log.append(entry)
            self._total_tokens += sum(s.tokens_used for s in steps)
            self._total_logged += 1

    def export(self, filepath: str) -> int:
        """Export interaction logs as JSON. Returns the number exported.

        Logs a WARNING if entries were evicted before export — the exported
        file will contain fewer interactions than were actually logged.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with self._lock:
            snapshot = list(self._log)
            evicted = self._eviction_count
        if evicted > 0:
            logger.warning(
                "InteractionLogger: exporting %d entries but %d were evicted "
                "(buffer full). Increase INTERACTION_LOG_MAX_SIZE or export "
                "more frequently to avoid data loss.",
                len(snapshot),
                evicted,
            )
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        return len(snapshot)

    @property
    def interaction_count(self) -> int:
        """Number of interactions currently in the buffer (bounded by max_size)."""
        with self._lock:
            return len(self._log)

    @property
    def total_interactions_logged(self) -> int:
        """Total interactions ever logged, including evicted entries."""
        with self._lock:
            return self._total_logged

    @property
    def eviction_count(self) -> int:
        """Number of entries silently dropped due to buffer overflow."""
        with self._lock:
            return self._eviction_count

    @property
    def total_tokens_generated(self) -> int:
        """Total tokens consumed across all interactions."""
        return self._total_tokens
