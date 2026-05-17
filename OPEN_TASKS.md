# Open Tasks

Snapshot of work-in-flight, scheduled follow-ups, and deferred items. Updated 2026-05-17 (post-public-release).

This file is the single source of truth for "what's not done yet on this branch / repo". Update it as items land or new items surface.

---

## Active / next-up

### ~~Repo split (post-keynote)~~ — DONE 2026-05-17
- [x] Naming decided: kept private repo as `local-multi-model-agent-slm` (source of truth); created public mirror as [`local-multi-model-agent-sample`](https://github.com/thinktecture-labs/local-multi-model-agent-sample).
- [x] Public repo created (2026-05-16 22:16:55Z).
- [x] Exclusion list confirmed: `slides/` (slidev source) + `docs/operations/` (per-host upgrade + tuning runbooks specific to the maintainer's ASUS Ascent GX10 and Minisforum MS-S1 MAX — fork users on different hardware would mis-apply them) excluded. `presentations/` kept (rendered keynote PDF intentionally public). `RETRAIN_NOTES.md` included (methodology notes, public-OK). `REVIEW_*.md` never tracked so never leaked.
- [x] Single-commit-on-public rule applied via `git archive | tar -x` + orphan `git init -b main` + force-push. Current public HEAD: [`36b05ec`](https://github.com/thinktecture-labs/local-multi-model-agent-sample/commit/36b05ec) — one commit, 392 files, MIT license, `vendor/llama.cpp` submodule pinned at `cf21cdf36ceb456b5312aa7d9058da297f3bf574`.
- [x] First publication run completed (user-driven, three rebuild iterations to fix license alignment + vendor submodule).
- [x] Codified the archive → orphan-init → submodule-add → force-push flow as [`scripts/publish-public.sh`](scripts/publish-public.sh) (2026-05-17). Future syncs: `bash scripts/publish-public.sh` (or `--dry-run` to preview, `--yes` for non-interactive). Auto-discovers the `vendor/llama.cpp` pin from `main`'s tree, verifies licensee picks up MIT, warns on vendor-pin / license mismatch.

See [memory: project-repo-split-post-keynote](.claude/projects/-Users-christianweyer-sources-local-multi-model-agent-slm/memory/project_repo_split_post_keynote.md) for the full plan and [memory: main-branch-single-commit](.claude/projects/-Users-christianweyer-sources-local-multi-model-agent-slm/memory/project_main_branch_single_commit.md) for the squash rule.

### ~~Merge `chore/public-release-cleanup` → `main`~~ — DONE 2026-05-17
- [x] Squash-merged via `git merge --squash` into `main` as commit [`aa51126`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/aa51126) "Public release: SDD 2026 London keynote companion repo" (296 commits → 1 squash, 651 files, +117638 / -57313). License alignment landed separately on top as [`d682bb2`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/d682bb2). 21 Dependabot alerts auto-closed; 13 remain on `main` for the underlying deps.
- [x] Full test/eval/benchmark pass completed on MBP 2026-05-16 before merge (unit 926/1, integration 164/164, e2e 63/63, vitest 189/189, eval matrix 9 result files, benchmarks p50 845ms / OCR / prompt-cache all green).

### ~~Slidev visual verification of `slides_v1.3.md`~~ — DONE 2026-05-16
- [x] Slide 13 PDF saga resolved (commits `dafda8f` → `8ce84d6` → `c26bc50`). Final keynote PDF committed to `presentations/`. User confirmed visual + export.

### ~~vendor/llama.cpp pin bump cf21cdf36 → b9196~~ — DONE 2026-05-17 (later same day)
- [x] `vendor/llama.cpp` submodule bumped from `cf21cdf36` (March 2026, build ~8384) to `b9196` (May 17, 2026 tagged release, ~700 builds of upstream fixes + perf). Commit [`bef86db`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/bef86db).
- [x] `scripts/build_llama.sh` hardened: (a) added Vulkan SDK detection branch for AMD Linux machines (`d8f9a98`), (b) replaced `cmake --build | tail -5` SIGPIPE-prone pipe with a proper log-file pattern (`7b239af`), (c) converted `CMAKE_ARGS` from space-string to bash array to fix shell-quoting of the aarch64+CUDA `-march` flags (`9a0e134`). The earlier string form silently broke DGX builds: cmake saw literal `'` around `-mtune=native` and configured a CPU-only fallback.
- [x] Validation evidence on all four backends (matching the Q4_K_M ship matrix):

  | Machine | Backend | bench median Q4_K_M → b9196 | RAG strict | extraction | pytest e2e |
  |---|---|---|---|---|---|
  | MBP | Metal | 800 → 688 ms (-14%) | 54/80 → 54/80 (0) | 100% / 100% | 63/63 in 5:43 |
  | RTX | CUDA Blackwell | 457 → 329 ms (**-28%**) | 53/80 → 56/80 (**+3**) | 100% / 100% | 63/63 in 3:02 |
  | DGX | CUDA GB10 | 1814 → 1657 ms (-9%) | 55/80 → 56/80 (+1) | 100% / 100% | 63/63 in 12:38 |
  | Strix | Vulkan/RDNA 3.5 | 3088 → 1455 ms (**-53%**) | 56/80 → 57/80 (+1) | 100% / 100% | 63/63 in 10:53 |

- [x] Fleet leaderboard re-ordered: Strix Halo now beats DGX Spark on overall median (1455 ms vs 1657 ms). The newer llama.cpp Vulkan kernels improve Qwen3.5 tool-routing throughput on RDNA 3.5 enough to flip the ranking — the kyuz0 toolbox A/B verdict on Strix is now decisively "host Vulkan/RADV wins" with the b9196 binary.
- [x] Strix-specific dep added: `spirv-headers-devel` is required for the Vulkan build path on Fedora 43 (upstream PR #22009 added a `spirv/unified1/spirv.hpp` include). Documented in `docs/operations/STRIX_HALO_UPGRADE_AND_TUNE.md`.
- [x] DGX-only quirk: an orphan `python3 -m multiprocessing.spawn` listener on `:8000` from a crashed prior eval blocked `start_app.sh`'s reclaim check (which only matches `uvicorn|src.server`). Documented as a follow-up to harden the reclaim regex.
- [x] Refresh: `docs/benchmarks/FINE_TUNING_INSIGHTS.md` four-machines table + per-step breakdown, `docs/benchmarks/benchmark_visualization.html/.png`, `README.md` performance section + Qwen toolcalling latency note — all carry the b9196 numbers.

### ~~Q4_K_M synthesis production switch~~ — DONE 2026-05-17
- [x] `scenarios/nextera.json:25` flipped: `synthesis_4b_gguf_ft` F16 → Q4_K_M (commit [`c1bce0e`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/c1bce0e)). Distribution scripts updated (`build_llama.sh` builds `llama-quantize`; `convert_gemma3_4b_to_gguf.sh` emits both F16+Q4_K_M; `upload_ft_to_hf.sh` adds Q4_K_M entry alongside F16; `download_ft_models.sh` default-fetches Q4_K_M) in commit [`e563ff9`](https://github.com/thinktecture-labs/local-multi-model-agent-slm/commit/e563ff9).
- [x] Validation evidence on all four backends (MBP Metal, RTX CUDA-Blackwell, DGX GB10 CUDA, Strix Vulkan):
  - response_quality (46q all-checks-pass): **93.5%** identical across the fleet, identical 3 tool_use failures (phrasing/currency)
  - extraction (29 fields / 5 cases): **100%** on every machine
  - pytest tests/e2e/: **63/63** on every machine
  - rag_groundtruth (80q strict): 67.5–70.0% across the fleet (3-query spread = Chroma tiebreaker noise, not Q4_K_M)
  - MBP same-machine F16-vs-Q4_K_M A/B: 55/80 vs 54/80 = **1-query phrasing noise**, zero semantic regression
- [x] Realized perf gains: RAG p50 -25% to -57%, image-query p50 -32% to -61% across the fleet. Means (overall bench) dropped 28% (RTX) / 39% (MBP) / 51% (DGX) / 40% (Strix) — far fewer slow-tail outliers. Updated `FINE_TUNING_INSIGHTS.md` four-machines table, `benchmark_visualization.html/.png`, README performance section.
- [x] HF distribution: `thinktecture/gemma3-4b-ft-nextera-q4_k_m` uploaded (private, flip to public after model card attached). F16 repo `thinktecture/gemma3-4b-ft-nextera-f16` remains for A/B comparison.
- [x] Rollback: F16 GGUF stays on disk on all 4 machines. Path: sed nextera.json:25 back to f16, restart :9093.

---

## RAG ground-truth — improvement plan

Tracked in [#10](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/10). Today's honest baselines (commit `b0f68da`):

| Machine | Strict | Coverage<100% | Hallucinated numbers |
|---|---|---|---|
| RTX | 75.0% | 25.0% | 7.5% |
| MBP | 67.5% | 32.5% | 7.5% |

Five-vector plan with Phase 1 (eval-set audit + `retrieval_recall` sub-metric) → Phase 2 (Qwen FT for RAG synthesis) → Phase 3 (optional). See the issue for the full design.

---

## Tracked in GitHub Issues

- [#9](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/9) — Re-bench DGX Spark + Strix Halo (Ryzen AI Max+ 395) for the keynote four-machines chart. Asterisks already in `slides_v1.3.md`; remove once data lands.
- [#10](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/10) — Improve RAG ground-truth faithfulness score (current honest baseline: 75% RTX / 67.5% MBP). Phased improvement plan; target ≥85% RTX / ≥80% MBP.

---

## Review backlog — [`REVIEW_2026-05-15_202209.md`](REVIEW_2026-05-15_202209.md)

5 of 16 items shipped in `6fdbd07` (H2 + H4 + H6 + M3 + M4). 3 of the remaining 11 shipped in `bb5a50c` (H3 + H1 + M2). The remaining 8 are now tracked as GH issues so they survive any future repo split:

- [#11](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/11) — H5 · request_id propagation through logs end-to-end
- [#12](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/12) — M1 · Unify `data_prep.py` + `data_prep_qwen35_toolcalling.py` API
- [#13](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/13) — M5 · Refactor `_strip_thinking` into a state machine
- [#14](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/14) — M6 · Structured JSON logging mode
- [#15](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/15) — L1 · Add a second public scenario
- [#16](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/16) — L2 · Surface p99 latency in eval output
- [#17](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/17) — L3 · Split `README.md` into a docs site
- [#18](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/18) — L4 · Generate `CHANGELOG.md` from commit history

None are pre-public blockers per SECURITY.md framing.

---

## Review backlog — [`REVIEW_2026-05-16_121403.md`](REVIEW_2026-05-16_121403.md)

13 items (N1–N13) from the second architecture review. N1–N5 shipped as the pre-public bundle in `c718d52` (joblib trust note, MAX_TOOL_STEPS cleanup, voice canned-response gate, FT_INSIGHTS date refresh; N1 was already done). N6–N12 are tracked as GH issues so they survive the repo split. N13 is a comment on #13.

- [#19](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/19) — N6 · Cloud-comparison drift (`compare_query` + `CloudOrchestrator` duplicate per-intent prompt logic)
- [#20](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/20) — N7 · Migrate intent LogReg from joblib/pickle to ONNX or JSON-coefficients
- [#21](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/21) — N8 · Pull magic numbers into config (query/extraction caps, tool rounds, upload size)
- [#22](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/22) — N9 · Publish Qwen recommended-sampling tool-routing number alongside greedy
- [#23](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/23) — N10 · Re-measure embeddinggemma MRR on production-size corpus (~120 chunks, not 26)
- [#24](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/24) — N11 · Add SYNTHESIS role to drop the `-vision` suffix dance
- [#25](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/25) — N12 · Replace 8-factor confidence heuristic with logprob-based scoring
- N13 → [#13 comment](https://github.com/thinktecture-labs/local-multi-model-agent-slm/issues/13#issuecomment-4468311077) — `_strip_thinking` state-machine edge cases (leading `<`, partial split across chunks, nested blocks)

Cloud-comparison faithfulness scoring, held-out adversarial set, single-worker uvicorn, rate limiting, upload PII redaction — all explicitly out of scope per SECURITY.md and noted as such in the review.

---

## Demo / public-release polish — shipped today

- [x] Demo-question pills audited end-to-end against the live stack — 11/13 worked correctly, 2 issues triaged.
- [x] L6 chip ("Median NRR for B2B SaaS companies") dropped from `scenarios/nextera.json` in `88daae8` — the underlying Pavilion B2B SaaS benchmarks PDF was removed in `2163a33` (redistribution risk), so the chip would hallucinate for fork users without it.
- [x] Observatory suggestion chips renamed in `24dd27d` — dropped cryptic `L1/L2/L4/L5`, `FT·…`, `Cloud·…` prefixes; chips now describe what they exercise (`Help`, `Customer count`, `ARR calc (2-step)`, `Pricing (RAG)`, `Customer count (FT)`, `Q3 revenue`, `Deal calc`, `Pricing (vs cloud)`, `SLA + cloud escalation`).
- [x] Show Mode sample-query chips renamed in `36d920a` — dropped `D0 Q1`/`D0 Q2`/`D1` demo-flow prefixes (`Top customer`, `2023 revenue`, `Q3 revenue`). Bookend narrative still reads in the labels themselves.
- [x] Cloud SLA escalation verified working as designed — in hybrid + multi-models mode, `confidence_assessment` step emits `should_escalate=True` (factor: `strong_uncertainty -0.35` from `"do not provide"` phrase), EscalationBanner UI fires correctly. No fix needed.
- [x] Stale React pill-count assertion fixed in `600b6c4` — `HealthPills` count was bumped to 10 by the LogReg pill addition in `5ea0dc4`, but three test assertions still expected 9. vitest now 189/189 green.

---

## Bugs / drift surfaced today

- [x] RTX `.env` pointed `FUNCTION_GGUF` at retired FunctionGemma 270M; base Qwen GGUF was missing from disk. Fixed live by downloading from `unsloth/Qwen3.5-4B-GGUF` + updating `.env` + restart in `--all` mode. Backup at `.env.bak.<ts>` on RTX.
- [x] MBP llama-server :9095 had a deleted `qwen3.5-4b-toolcalling-ft-nextera-f16.gguf` mmap-ed; intermittent compute errors. Fixed by killing + restarting (scripts already point at the renamed `-q4_k_m.gguf`).

---

## Out of scope / deferred indefinitely

- Slidev preview, PDF export — owned by you.
- Multi-worker FastAPI / Redis-backed state — explicitly out of scope per [SECURITY.md](SECURITY.md).
- Rate limiting on `/escalate` — same reason.
- React frontend (`src/clients/observatory-react/`) — not audited in this session.
- iOS / WebGPU clients — not audited.

---

## How to update this file

Add new items as `- [ ] short description (~effort, link)`. Mark items done with `- [x]` and leave the line for ~one cycle so commit history shows what landed. Stale items (3+ months untouched) get removed.
