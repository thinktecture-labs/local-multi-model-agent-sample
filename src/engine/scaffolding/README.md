# Scaffolding

This directory holds **deterministic compensators** that sit alongside the
SLMs — pieces of code that solve precision-critical tasks the small models
shouldn't be relied on for.

## Active modules

- `confidence_router.py` — score the agent's intermediate confidence and decide whether to surface a hybrid-routing escalation prompt. Active in production.

## Retired modules

Earlier iterations of the agent shipped two larger pattern-matching
pre-routers — `expression_builder.py` (NL → calculator expressions) and
`sql_builder.py` (NL → parameterised SQL) — to compensate for the
limitations of small base tool-calling models. They were retired once
the Qwen3.5-4B fine-tune (99.4% single-step / 97.5% multi-step chain,
v9 post-2026-05-15 retrain on corrected training data, fully deterministic)
eliminated the need. The follow-up `ExpressionResolver` / `SQLResolver`
protocols (with `NullExpressionResolver` / `NullSQLResolver` Null impls)
that briefly survived as DI hooks have also been removed —
`ToolUseHandler` now delegates 100% of tool routing to Qwen FT.

The git history (search for `expression_builder.py`, `sql_builder.py`,
or `NullExpressionResolver`) preserves the full evolution.
