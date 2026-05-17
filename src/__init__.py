"""
Multi-Model Local AI Agent with Small Language Models
=====================================================
A production-grade AI agent built entirely on local, private models.

Source layout:
  src/engine/     — core agent runtime (agent, inference, scaffolding, knowledge, tools)
  src/server/     — FastAPI HTTP layer
  src/clients/    — consumer-facing apps (Observatory UI, iOS, WebGPU)

Architecture: Private | Fast | Zero cost after setup | Domain-customizable
"""
