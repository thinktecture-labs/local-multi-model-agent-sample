---
title: Deployment Options
category: deployment
---
Nextera can be deployed in multiple configurations: (1) Single-node Docker — runs on any Linux, macOS, or Windows machine with 16 GB+ RAM. (2) Kubernetes — Helm chart provided, supports auto-scaling. (3) Air-gapped — no internet connection required (Enterprise only). (4) Hybrid — inference on local GPU, orchestration on Kubernetes. Hardware requirements: minimum 8 GB RAM for small models (<7B), recommended 32 GB + NVIDIA GPU with 16 GB VRAM for production. AMD ROCm and Apple Silicon (Metal) are also supported.
