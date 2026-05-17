# Contributing

## Status: Conference talk demo — pinned to the talk's state

This repository accompanies a conference keynote on local on-device AI. It is
published as a **reference for the architectural patterns shown on stage**
— not as a maintained library. See [`README.md`](README.md),
[`SECURITY.md`](SECURITY.md), and [`NOTICE.md`](NOTICE.md) for the framing.

That shapes what we can and can't accept from outside the original team:

---

## What's welcome

- **Bug reports** filed as GitHub issues, especially:
  - A documented feature doesn't work as described
  - Setup steps fail on a platform listed in [`README.md`](README.md)
  - A factual error in the docs
  - A security concern that isn't already noted in
    [`SECURITY.md`](SECURITY.md) (please disclose privately first — see
    that file)
- **Typo / link / doc-clarity PRs** — small, scoped, and self-explanatory
- **Forks for your own scenarios.** The agent is intentionally
  scenario-pluggable via `scenarios/<name>.json`. The reference scenario
  here (`nextera`) is fictional; everything you need to author a new one
  lives in that JSON plus a per-scenario data loader. See
  [`docs/architecture/DOMAIN_INDEPENDENCE_PLAN.md`](docs/architecture/DOMAIN_INDEPENDENCE_PLAN.md).

## What we won't accept

- **Feature PRs** beyond bug fixes and doc polish. The code is frozen to
  match what was shown on stage; new features would diverge the artifact
  from the talk.
- **Architectural reworks**, even well-motivated ones. The architectural
  choices are deliberate teaching exhibits — even the ones that look
  outdated. See `docs/architecture/` for the rationale.
- **Dependency bumps** unless they're security advisories tracked by
  Dependabot. The locked versions are what was running on stage.

## Filing a useful issue

Include:

- What you ran (exact command, including flags)
- Platform: macOS / Linux distro, Apple Silicon / x86_64 / NVIDIA / AMD,
  Python version, llama.cpp commit (`git -C vendor/llama.cpp rev-parse HEAD`)
- What you expected, what you got
- Logs (relevant excerpt, not the full file)

## Security issues

See [`SECURITY.md`](SECURITY.md) — particularly the "What's out of scope"
section. For exploitable vulnerabilities in the agent's defences (intent
filter, SQL whitelist, calculator sandbox), email the maintainer instead
of opening a public issue first.

## Code of conduct

Be civil. Disagree about the technical content all you like; don't make
it personal.

---

## For maintainers (post-public-release)

This section is mostly a reminder to ourselves:

- Don't merge new features. If something genuinely belongs upstream, the
  right home is a follow-up repo or a new branch, not `main`.
- Dependabot security PRs: review, run the unit suite locally, merge if
  green. Major version bumps deserve their own branch.
- Tag the talk's state once and don't move it. Subsequent fixes go to
  `main` with the tag preserved as the canonical talk artifact.
