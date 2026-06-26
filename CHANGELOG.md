# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-26

### Added

- `sloth train` / `sloth eval` / `sloth export` verbs for Spark-friendly LoRA/QLoRA adapter fine-tuning of Qwen models (issue #6)
- Dependency-free `sloth/tune/` core: JSONL dataset validation (chat + task schemas), TOML run-config loader with Spark-friendly defaults, training-metadata writer (model/method/dataset sha256+lines/hyperparams/timestamp), and a model-scope guard that refuses out-of-scope large-dense full fine-tuning
- Lazy LoRA/QLoRA trainer (`sloth/tune/_trainer.py`) that imports unsloth/torch only inside its run function, with a GPU-free `--dry-run` plan path
- `/finetune` skill driving the validate -> train -> eval -> export loop non-interactively
- explain catalog entries for the three new verbs

### Changed

- unsloth + torch are now BASE runtime dependencies — `uv tool install unsloth-cli` brings the full tuning stack — retiring the prior zero-runtime-dependency rule; introspection verbs stay torch-free and fast via lazy imports, and the afi rubric gate stays green
- CLAUDE.md and README now document the base-dependency policy, the lazy-import discipline, and the fine-tune vs memory/RAG vs retrieval boundary
- markdownlint excludes `.venv` (vendored package READMEs now present via base deps) and devague-generated specs/plans

### Fixed

- `validate_dataset` now fails fast with `CliError` (exit 1) on an empty or blank-only dataset instead of returning `[]` and letting `train` proceed to model load (qodo review)
- `sloth export` refuses an adapter directory that lacks the canonical PEFT files (`adapter_config.json`, `adapter_model.safetensors`) instead of reporting success with an empty file list (qodo review)
- `dataset_digest` counts JSONL **records** (non-blank lines) rather than `b"\n"` bytes, so `training_metadata.json` `line_count` is correct for files without a trailing newline and ignores blank separator lines (qodo review)
- `load_config` validates hyperparameter **types and ranges** (rejecting strings, booleans-for-ints, and out-of-range values) before constructing `RunConfig`, surfacing actionable `CliError`s instead of failing deep in the ML stack (qodo review)
- `_trainer._run_real` validates/loads the dataset **before** the expensive model load, honoring the "validate before spending GPU" contract (qodo review)
- `sloth train --help` now states the LoRA/QLoRA-only scope and full-fine-tuning refusal up front (qodo review)
- Hardened the model parameter-count regex (`sloth/tune/scope.py`) with possessive quantifiers to remove a polynomial-backtracking (ReDoS) risk flagged by Sonar (S5852); renamed `_Backend` fields to snake_case (S116), reduced `_validate_chat_record` cognitive complexity (S3776), and switched a type hint to the `X | Y` union form (S6546)

## [0.3.1] - 2026-06-26

### Changed

- CLAUDE.md: documented issue #6 fine-tuning design constraints (LoRA/QLoRA adapter scope, train/eval/export verbs, dataset schemas, role-specific adapters, fine-tune vs. RAG boundary) and the optional-extra/subprocess rule for the GPU stack.
- CLAUDE.md: added an AgentCulture sibling-ecosystem map (agentfront/steward/guildmaster/devague/devex/agtag/lobes/colleague/culture/daria) and noted the teken→agentfront rename.

## [0.3.0] - 2026-06-24

### Added

- **Memory-discipline "Conventions and workflow" section in `CLAUDE.md`** — a
  per-task *recall-before / remember-after* convention (scope localized to this
  repo's nick) so the vendored `remember` / `recall` skills are actually used,
  not just present: `/recall` before non-trivial work to build on prior
  decisions instead of re-deriving them, and `/remember` when a non-obvious
  decision, constraint, fix-and-why, or hard-won gotcha surfaces. The section
  documents this repo's memory as **in-repo and public** — records resolve to
  `<repo-root>/.eidetic/memory` (committed, team- and mesh-shared). Inserted
  idempotently (skipped if already present), slotted under an existing
  "Conventions and workflow" heading when one exists, else appended.

### Changed

- **Refreshed the `remember` + `recall` wrappers from eidetic-cli 0.10.0**
  (cite-don't-import) — picks up eidetic's **project-local store default**: the
  files backend now resolves per record by visibility — PUBLIC records inside a
  git repo go to `<repo-root>/.eidetic/memory` (committed, team-shared), PRIVATE
  records (or any record outside a repo) go to `$HOME/.eidetic/memory` (never
  committed), an explicit `EIDETIC_DATA_DIR` still wins, and recall reads both
  stores and merges. Also carries the 0.9.3 hardening (interactive-stdin guard,
  `help` as a search term, SIGPIPE-safe suffix parsing). **Recipe policy
  override (the wrappers here are NOT byte-verbatim):** the injected default
  visibility is flipped from eidetic's `private` to **`public`**, so a plain
  `/remember` lands the note in `./.eidetic/memory` in this repo, kept as part
  of the repo — pass `--visibility private` to route a record to `$HOME`
  instead. `remember` drives `eidetic remember` (idempotent upsert of one JSON
  record or an NDJSON batch on stdin); `recall` drives `eidetic recall` with
  four search modes (exact / approximate / keyword / hybrid). Each `SKILL.md` is
  localized only in the illustrative `--scope <nick>` examples (Provenance keeps
  "First-party to eidetic-cli"). Runtime dep: the `eidetic` CLI on PATH (else a
  local eidetic-cli checkout with `uv`) — **`eidetic >= 0.10.0`** for the
  in-repo routing; on an older CLI the public records still work but are stored
  in `$HOME/.eidetic/memory` instead of in-repo. Propagated by rollout-cli's
  `eidetic-memory` recipe.

## [0.2.0] - 2026-06-23

### Added

- **Vendored the `remember` + `recall` memory skills from eidetic-cli**
  (cite-don't-import) — the write/read halves of eidetic's shared
  `~/.eidetic/memory` surface, so this agent (Claude and its colleague backend)
  can persist facts across sessions and recall them later, sharing one store.
  `remember` drives `eidetic remember` (idempotent upsert of one JSON record or
  an NDJSON batch on stdin, dedup by id + content hash); `recall` drives
  `eidetic recall` with four search modes — exact / approximate / keyword /
  hybrid — each hit carrying text, full provenance metadata, a relevance score,
  and a freshness signal. The `.sh` wrappers are byte-verbatim from eidetic-cli
  (their first-party origin); each `SKILL.md` is localized only in the
  illustrative `--scope <nick>` examples (Provenance keeps "First-party to
  eidetic-cli"). Both default to this agent's PRIVATE scope, reading the suffix
  from `culture.yaml`. Runtime dep: the `eidetic` CLI on PATH (else a local
  eidetic-cli checkout with `uv`). Propagated by rollout-cli's `eidetic-memory`
  recipe.

## [0.1.4] - 2026-05-31

### Changed

- Re-initialized CLAUDE.md from the bootstrap seed into a full runtime prompt via /init: documents the `sloth` (package/console-script/import) vs `unsloth-cli` (dist/argparse-prog) naming split, the four CLI contracts (dispatch + CliError, stdout/stderr output split, exit-code policy, explain catalog), the register() seam for adding verbs/noun groups, the agent-first rubric gate rules, and the merge-gating conventions (version-bump-every-PR, SonarCloud gate, cite-don't-import skills, zero runtime deps).

### Fixed

- Added a `("sloth",)` alias to the `explain` catalog so `explain sloth` resolves
  to the root entry. The agent-first rubric's `explain_self` check runs
  `explain <console-script-name>` (the script is `sloth`, not `unsloth-cli`), so
  the alias is load-bearing — without it the `lint` job's rubric gate fails.
- Fixed the README Quickstart, which told users to run `uv run unsloth-cli
  whoami/learn` — commands that fail because the only installed console script is
  `sloth`. The examples now use `uv run sloth <verb>`, with a note on the
  `sloth` (script) vs `unsloth-cli` (dist / argparse prog) split.

## [0.1.3] - 2026-05-31

### Changed

- Expanded the clone-and-rename instructions in `CLAUDE.md`: added `README.md` to
  the rename targets and a portable `git grep` discovery command so a cloner can
  find every occurrence of the template name (hard-coded in ~100 places across the
  package, including the CLI command files and `_ISSUES_URL` in
  `sloth/cli/__init__.py`) rather than renaming by hand.
- Synced `README.md`'s "Make it your own" checklist with `CLAUDE.md`: it now lists
  `README.md` itself as a rename target and points to `CLAUDE.md`'s discovery
  command as the authoritative procedure, so the two onboarding checklists no
  longer drift.

## [0.1.2] - 2026-05-30

### Changed

- Renamed the PR-lifecycle CLI references `agex` / `agex-cli` to `devex` (same
  tool, new name) across `CLAUDE.md`, `docs/skill-sources.md`, `.gitignore`, and
  the vendored `cicd`, `assign-to-workforce`, and `communicate` skills — the
  `cicd` scripts now invoke `devex pr`.
- Logged the vendored-skill in-place patch as a local divergence in
  `docs/skill-sources.md`; the matching canonical rename is tracked upstream for
  guildmaster in
  [agentculture/guildmaster#48](https://github.com/agentculture/guildmaster/issues/48)
  so a future re-sync reconciles cleanly.
- Aligned the documented `devex` version floor to `>=0.21` across the vendored
  `cicd` `SKILL.md` and `workflow.sh` install hint (were `>=0.1`), matching
  `docs/skill-sources.md` and the `await`-era feature set; flagged upstream on
  guildmaster#48.

### Fixed

- SonarCloud now reports code coverage — added `relative_files = true` to
  `[tool.coverage.run]` so `coverage.xml` emits repo-relative paths that map to
  `sonar.sources=sloth` (absolute / `.venv` paths were dropped
  as unmappable). Mirrors the sibling `convertible` setup.

## [0.1.1] - 2026-05-26

### Changed

- **CI gates on the SonarCloud quality gate**
  ([issue #3](https://github.com/agentculture/unsloth-cli/issues/3)) —
  added `sonar.qualitygate.wait=true` to `sonar-project.properties` so a failing
  gate fails the `test` job when `SONAR_TOKEN` is set. Token-less repos and fork
  PRs remain green (the scan step is guarded by `if: env.SONAR_TOKEN != ''`).

## [0.1.0] - 2026-05-26

### Added

- **Onboarded into the AgentCulture mesh** ([issue #1](https://github.com/agentculture/unsloth-cli/issues/1)).
- **Agent-first CLI** cited from teken's (`afi-cli`) `python-cli` reference
  (`teken cli cite`) — verbs `whoami`, `learn`, `explain`, `overview`, `doctor`,
  and the `cli` noun group. Runtime is self-contained (`dependencies = []`);
  `teken>=0.8` is a dev dependency only. Passes the seven-bundle agent-first
  rubric (`teken cli doctor . --strict`). `doctor` checks the agent-identity
  invariants (prompt-file-present, backend-consistency, skills-present).
- **Mesh identity**: `culture.yaml` (`suffix: unsloth-cli`,
  `backend: claude`) and the matching `CLAUDE.md` prompt file.
- **Canonical guildmaster skill kit** (11 skills) vendored under
  `.claude/skills/` (cite-don't-import): `agent-config`, `assign-to-workforce`,
  `cicd`, `communicate`, `doc-test-alignment`, `pypi-maintainer`, `run-tests`,
  `sonarclaude`, `spec-to-plan`, `think`, `version-bump`. Every `SKILL.md`
  carries `type: command` (load-bearing for the culture/claude backend);
  `cicd` / `communicate` consumer-identifying prose adapted, all script bodies
  verbatim. Provenance in `docs/skill-sources.md`. Three skills (`think`,
  `spec-to-plan`, `assign-to-workforce`) originate in `devague`, re-broadcast
  via guildmaster.
- **Build + deploy baseline**: `pyproject.toml` (hatchling), `tests/` (pytest,
  xdist, coverage), `.github/workflows/{tests,publish}.yml` (CI rubric/lint gate,
  PyPI Trusted Publishing), `.flake8`, `.markdownlint-cli2.yaml`,
  `sonar-project.properties`, and `.claude/skills.local.yaml.example`.

### Changed

### Fixed
