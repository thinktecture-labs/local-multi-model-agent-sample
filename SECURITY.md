# Security

## Status: Conference Talk Demo

This repository accompanies a conference keynote on local on-device AI. It is
a **demo**, not a production system. Several design decisions traded security
hardening for narrative clarity on stage. Read this document before deploying
anything from this repo to a network where untrusted users can interact with
it.

---

## What's in scope (defended by design)

- **Local-only execution by default.** All models run on the developer's
  machine via `llama-server`. The default `network_mode` blocks the
  cloud-comparison endpoints and prevents accidental egress.
- **Read-only SQL.** The `sql_query` tool whitelists `SELECT` statements and
  blocks `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/...` via regex pre-filter
  before the query reaches SQLite. See `src/engine/tools/sql_query.py`.
- **Code-execution sandbox.** The `calculator` tool uses `simpleeval` with
  no builtins, no imports, and a whitelisted operator set — a malicious
  expression cannot exec arbitrary Python.
- **Prompt-injection pre-filter.** A multi-layer defence in
  `src/engine/agent/intent_classifier.py`:
  - ~30 regex patterns for known injection phrasings (English + German)
  - Gibberish detector (character entropy)
  - Non-ASCII normaliser
  - LogReg-confidence threshold (0.60) — low-confidence queries route to a
    canned refusal message instead of through the agent
- **Network mode toggle.** A single switch in the Observatory UI (and
  `network_mode` env var) gates the cloud-comparison endpoint. With it off,
  no model call can reach an external API.

---

## What's out of scope (do NOT deploy as-is in these contexts)

| Concern | Status | Why |
|---|---|---|
| **Multi-tenant exposure** | ❌ | No per-user auth, no rate limiting, no tenant isolation. The agent is built for a single demo user, not a public API. |
| **Adversarial robustness** | ⚠️ Demo-grade | The pre-filter catches ~93% of canned adversarial queries; it is not pen-tested against a motivated attacker. |
| **SQL injection** | ⚠️ Demo-grade | The regex defence stops `DROP TABLE`-shaped attacks but is not a substitute for parameterised queries against a production DB. |
| **PII / regulated data** | ❌ | No PII redaction, no audit logging, no encryption at rest beyond what your OS provides. The demo data (`Nextera`) is fully synthetic. |
| **Network exposure helpers** | ❌ | The bundled scripts to expose the local server (ngrok / Tailscale) were removed for the public release. If you re-add them, put a proper auth layer in front. |
| **Supply chain pinning** | ⚠️ Snapshot | Dependencies are pinned at the talk's state. The npm package-locks are clean *as of this commit* but will drift; re-audit before any operational deployment. |
| **Fine-tuned model safety** | ⚠️ | The fine-tunes were trained on a synthetic scenario. They have no jailbreak resistance or alignment guardrails beyond what the base Gemma/Qwen models ship with. |
| **`model.joblib` trust** | ⚠️ Snapshot | `models/intent-logreg/model.joblib` is loaded via `joblib.load` (Python pickle), which executes arbitrary code from a malicious file. The committed artefact is produced locally by `training.train_intent_logreg` and treated like source code. **Do not** swap in a `model.joblib` from a PR / fork / download. A future hardening step would migrate the classifier to ONNX or a JSON-coefficients format. |
| **Long-running operation** | ❌ | No telemetry pipeline, no SLO monitoring, no graceful degradation tested beyond what the demo exercises. |

---

## Reporting security issues

**Public, non-sensitive:** open an issue on the GitHub repo.

**Sensitive (please don't disclose publicly):** email
[christian.weyer@thinktecture.com](mailto:christian.weyer@thinktecture.com).

There is **no security SLA**. Severe issues will be acknowledged and may be
documented; the repository is treated as a finished talk rather than an
ongoing project. If you're building on top of this code, you own the
hardening required for your deployment environment.

---

## Threat model summary

The intended attacker is a **curious-but-not-determined demo participant**:
someone who notices the SQL pill or the airplane-mode toggle and tries an
adversarial query during the live demo. The defences listed above handle
that audience comfortably.

The intended attacker is **not** a motivated red team. If your deployment
environment includes that audience, this repository is a starting reference
for the architectural patterns, not the deployable artefact.
