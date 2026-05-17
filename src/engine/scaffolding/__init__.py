"""
Scaffolding — deterministic compensators for SLM limitations.

These modules handle precision-critical tasks (math, SQL, confidence scoring)
that small models can't yet do reliably. Each has a clear retirement path
as model capabilities scale up.
"""
