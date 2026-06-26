# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**unsloth-cli** is an AgentCulture mesh agent whose intended domain is *an
agent + CLI that simplifies fine-tuning with Unsloth* (adding complementary
actions so an agent can fine-tune models more easily).

**The repo is moving from scaffold to product.** It carries a working
agent-first CLI of *introspection* verbs (`whoami`, `learn`, `explain`,
`overview`, `doctor`, `cli`) cited from teken's `python-cli` reference, a mesh
identity, the vendored skill kit, and the CI/deploy baseline. The Unsloth
fine-tuning verbs (`train`, `eval`, `export`) are now **designed and being
built** on top of that base (issue #6; the converged spec + plan live under
`docs/specs/` and `docs/plans/`). The "Adding a verb or noun group" section
below is the seam they go through, and the **forward work** section captures
their decided design.

## Naming: `sloth` vs `unsloth-cli` (read this first)

Three different names, used in three different places — do not conflate them:

| Name | Where it applies |
|------|------------------|
| `unsloth-cli` | The **dist / PyPI name** (`pyproject.toml` `[project].name`), the argparse `prog`, and all user-facing text in `learn` / `explain` / `overview` output. |
| `sloth` | The **package directory**, the **import name** (`from sloth ...`), and the **installed console-script** (`[project.scripts] sloth = "sloth.cli:main"`). |

Consequence: the real invocations are **`uv run sloth <verb>`** or
**`uv run python -m sloth <verb>`**. `uv run unsloth-cli <verb>` **fails**
(no such script) — the help/`explain` text only *prints* `unsloth-cli` because
that's the argparse `prog`. When you read `unsloth-cli whoami` in CLI output,
run `sloth whoami`.

## Commands

```bash
uv sync                                       # install deps + the package (editable)
uv run pytest -n auto                         # full suite, parallel (pytest-xdist)
uv run pytest tests/test_cli.py::test_whoami_text -v   # a single test
uv run pytest --cov=sloth --cov-report=term   # with coverage (CI gate: fail_under=60)

# Lint suite (CI `lint` job runs all of these — keep them green before a PR):
uv run black --check sloth tests
uv run isort --check-only sloth tests
uv run flake8 sloth tests
uv run bandit -c pyproject.toml -r sloth
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken"

uv run teken cli doctor . --strict            # the agent-first rubric gate (see below)

uv run sloth whoami                           # smoke-run the CLI
```

`black` and `isort` (profile=black) both use **line-length 100** — match it.
`teken` is a **dev-only** dependency. The runtime has **no heavy deps**
(`dependencies = []`); the GPU stack (unsloth + torch) is provided exclusively by the
NGC container — see "GPU stack: NGC container, not a pip dep" under "Conventions that
gate merges". What keeps the introspection verbs fast is the **lazy-import discipline**:
never import torch/unsloth at module top level (still required even inside the
container context).

## Architecture

The introspection CLI is a thin, **pure-stdlib** argparse core built around four
stable contracts — understanding these four is enough to add features safely.
The heavy ML stack (unsloth/torch) is **not a runtime dependency** — it runs inside
the NGC container that the fine-tuning verbs orchestrate. Handlers never import
torch/unsloth at module top level; they lazy-import inside the function body once the
container context is established. This keeps the introspection core import-light and
installable on every architecture.

### 1. Dispatch + error propagation (`sloth/cli/__init__.py`)

`main(argv)` → `_build_parser()` registers every verb → `_dispatch(args)` calls
`args.func(args)`. Handlers either return `None`/`int` (exit code) **or raise
`CliError`**. Two things are deliberate:

- **No traceback ever leaks.** `_dispatch` catches `CliError` (formats + returns
  its `code`) and wraps *any* other exception into a `CliError` pointing at the
  issues URL. A handler should signal failure by raising `CliError`, not by
  printing and returning non-zero.
- **Argparse errors honor the contract too.** `_CliArgumentParser` (the
  `parser_class` for the root parser *and* all subparsers) overrides `.error()`
  to emit the structured `error:`/`hint:` format and exit `1` (not argparse's
  default stderr + exit `2`). Because parse-time errors happen before
  `args.json` exists, `main()` pre-scans raw argv for `--json` and stashes it on
  the class-level `_CliArgumentParser._json_hint` so even parse errors render as
  JSON when asked.

### 2. Output contract (`sloth/cli/_output.py`)

**Strict stream split, never mixed:** results → stdout (`emit_result`), errors →
stderr (`emit_error`), human diagnostics/progress → stderr (`emit_diagnostic`).
Agents parse stdout, so nothing else may land there. `--json` routes the *same*
payloads as structured JSON to the *same* streams. In text mode, errors render
as two lines — `error: <message>` then `hint: <remediation>` — and the `hint:`
prefix is **required by the rubric**, so always supply a remediation on
`CliError`.

### 3. Error + exit-code policy (`sloth/cli/_errors.py`)

`CliError(code, message, remediation)` is the only failure type. Exit codes are
centralized: `0` success, `1` user-input error, `2` environment/setup error,
`3+` reserved. This policy is also printed in `learn`/`explain` output — keep
the three sources (this file, `_errors.py`, the catalog text) in agreement.

### 4. The explain catalog (`sloth/explain/`)

`explain <path>...` is a **global** verb (not nested under a noun) that resolves
a path *tuple* against `catalog.py::ENTRIES` (verbatim markdown keyed by
command-path tuples; `()`, `("unsloth-cli",)`, and `("sloth",)` all map to the
root). Unknown paths raise `CliError`. **Every registered verb/noun must have a
catalog entry** — `tests/test_cli.py::test_every_catalog_path_resolves` walks
`known_paths()` and the rubric checks coverage. The `("sloth",)` alias is
load-bearing: the rubric's `explain_self` check runs `explain <console-script>`,
i.e. `explain sloth`, so dropping it turns the lint gate red.

### Adding a verb or noun group (the main extension seam)

Each command is a module under `sloth/cli/_commands/` exposing
`register(sub)`. To add one:

1. Create `sloth/cli/_commands/<name>.py` with a `cmd_<name>(args)` handler
   (return `int`/`None`, raise `CliError` on failure, support `--json` via
   `getattr(args, "json", False)`) and a `register(sub)` that adds the subparser
   and `set_defaults(func=...)`.
2. Wire it into `_build_parser()` in `sloth/cli/__init__.py` (there's a marked
   "Register your own noun groups here" spot).
3. Add an `ENTRIES` catalog entry for its path in `sloth/explain/catalog.py`.
4. If it's a **noun with action sub-verbs**, it must also expose
   `<noun> overview` — the rubric's `overview_cli_noun_exists` check enforces
   this. The `cli` noun (`sloth/cli/_commands/cli.py`) is the worked example:
   it exists *only* to satisfy that rule and reuses `overview.py`'s shared
   `cli_sections()`/`emit_overview()` helpers. When building a noun subparser,
   pass `parser_class=type(p)` so its parse errors route through the structured
   contract (see `cli.py`).
5. **Descriptive verbs must not hard-fail on a stray path.** `overview` accepts
   an ignored optional `target` positional so `overview <bogus>` still exits 0
   (rubric contract; see `test_overview_graceful_on_bad_path`).

### Identity resolution (`whoami.py`, reused by `doctor` + `overview`)

`whoami` parses the agent's **own** `culture.yaml` — found by walking *up from
`__file__`*, not from the caller's CWD — with a hand-rolled scalar parser (no
YAML dependency, so the introspection path stays import-light). In a wheel
install no `culture.yaml` ships, so it falls back to literal defaults and
`doctor` reports a single info check and exits 0. `doctor` reuses this to verify
the `steward doctor` invariants: **prompt-file-present** + **backend-consistency**
(`claude`→`CLAUDE.md`, `acp`→`AGENTS.md`, `gemini`→`GEMINI.md`) plus a
**skills-present** check, emitting the rubric-shaped
`{healthy, checks: [{id, passed, severity, message, remediation}]}`.

## The forward work — Unsloth fine-tuning verbs (issue #6)

The repo's reason to exist is a Spark-friendly **adapter-tuning** workflow for
Qwen models — *not* full fine-tuning.
[Issue #6](https://github.com/agentculture/unsloth-cli/issues/6) is the source;
the converged spec + plan under `docs/specs/` and `docs/plans/` are the design of
record (this section summarizes them). Build it through the "Adding a verb or
noun group" seam above. The settled, load-bearing decisions:

- **Scope is LoRA / QLoRA adapters**, small models first (Qwen 3.x 4B/9B) then
  larger local adapters (Qwen 3.6 27B dense, Qwen coder variants). The CLI must
  *explicitly warn* that full fine-tuning of large dense models is out of scope
  and refuse or downgrade to adapter-only (the `sloth/tune/scope.py` guard).
- **Three flat verbs — `train`, `eval`, `export`.** They are **global** verbs
  (siblings of `whoami`/`explain`, *not* a `tune` noun group), which is
  rubric-legal because global verbs don't require an `overview` sub-verb. `train`
  validates the dataset → scope-guards the target → runs a small LoRA/QLoRA job
  (or `--dry-run`s the resolved plan) → writes training metadata next to the
  adapter output; `eval` runs an adapter against a small **local, offline** eval
  suite; `export` turns an adapter into a standard PEFT/safetensors layout (so
  `lobes` can serve it and `colleague` can run it as a backend).
- **TOML run-configs** parsed read-only with stdlib `tomllib` (the 3.12+ floor
  has it) drive repeatable runs; omitted fields fall back to documented
  Spark-friendly defaults, and the same config + dataset reproduces the same run.
- **A dependency-free core under `sloth/tune/`.** `datasets.py` (JSONL schema
  validation), `config.py` (the TOML loader + defaults), `metadata.py` (the
  run-metadata writer), and `scope.py` (the adapter-OK vs out-of-scope guard) are
  **pure stdlib and import no torch** — so dataset/config/scope validation
  happens *before* any GPU spend. Only `_trainer.py` touches the ML stack, and it
  **lazy-imports unsloth/torch inside its run function** (raising
  `CliError(code=2)` with an install hint when the stack is unavailable). This is
  the same seam as any other command, just with a heavy-import-isolating core.
- **Two dataset schemas, validated *before* spending GPU**: a **chat** format
  (`{"messages":[{role,content}, ...]}`) and a **task** format
  (`{"task","input","expected_output"}`).
- **A `/finetune` skill** (`.claude/skills/finetune/`) drives the full loop —
  validate dataset → `sloth train` → `sloth eval` → `sloth export` — wrapping the
  flat CLI verbs non-interactively (forwarding `--json`, surfacing
  `error:`/`hint:`).
- **Role-specific adapters, not one mixed blob** — e.g. `culture-contract-lora`,
  `agentculture-cli-teacher-lora`, `repo-maintainer-lora`, `issue-writer-lora`,
  `tool-router-lora`, `agent-first-coach-lora`.
- **The fine-tune vs. retrieval boundary is a design rule, not a footnote.**
  Fine-tuning stores *stable behavior/reflexes* (CLI-contract discipline,
  AgentCulture/CULTURE.DEV terminology, agent-first habits, issue-writing format,
  teacher behavior for `learn`). Memory/RAG stores *changing facts* (project
  state, secrets, user-specific memory). The README must explain this split.

The dependency question is **settled, not open**: torch + unsloth are **not**
runtime dependencies (`dependencies = []`); the GPU stack runs inside the NGC
container (`nvcr.io/nvidia/pytorch:25.11-py3`) that the fine-tuning verbs
orchestrate — see "GPU stack: NGC container, not a pip dep" under "Conventions that
gate merges". The lazy-import discipline (never import torch/unsloth at module top
level) still applies inside the container. These verbs connect to siblings: `lobes`
serves the resulting adapters locally and `colleague` runs them as model backends
(see the ecosystem map below).

## The agent-first rubric gate (must stay green)

CI runs `uv run teken cli doctor . --strict` (the "afi rubric gate"). It is a
hard gate. Beyond the per-verb rules already noted, the ones that bite:

- **`learn`** must be ≥200 chars and mention purpose, the command map, exit
  codes, `--json`, and `explain`. (Renaming the "Commands" header can break the
  marker check — keep the literal cues.)
- Every verb supports `--json` and routes errors through `error:`/`hint:`.

When you add domain verbs, run the gate locally before pushing — it catches
missing catalog entries and missing `overview` nouns that the unit tests might
not.

## Conventions that gate merges

- **Version-bump every PR.** The `version-check` CI job fails any PR whose
  `pyproject.toml` version equals `main`'s — even docs/config/CI-only changes.
  Use the `version-bump` skill (updates `pyproject.toml` + prepends a
  Keep-a-Changelog entry to `CHANGELOG.md`). It posts a sticky PR comment on
  failure.
- **SonarCloud quality gate blocks CI** when `SONAR_TOKEN` is set
  (`sonar.qualitygate.wait=true`); token-less repos and fork PRs skip the scan
  and stay green. Coverage uses `relative_files = true` so `coverage.xml` paths
  map to `sonar.sources=sloth`. **If you add a `[tool.coverage.run] omit`, mirror
  it into `sonar.coverage.exclusions`** — an omitted file is still indexed by
  Sonar and scored as 0% new-code coverage otherwise.
- **PR lifecycle via the `cicd` skill** (delegates to `devex pr`): open, read
  status, reply to review threads, and `await` (gates on the Sonar gate +
  unresolved threads). Do **not** self-merge — open the PR and await human
  merge. Replies auto-sign `- unsloth-cli (Claude)`.
- **GPU stack: NGC container, not a pip dep.** torch + unsloth are **not** runtime
  dependencies — `[project].dependencies` is empty for the GPU stack.
  `uv tool install unsloth-cli` installs only the pure-stdlib introspection CLI, which
  works on every arch including aarch64 / DGX Spark GB10. The GPU stack
  (`nvcr.io/nvidia/pytorch:25.11-py3`) is provided by NVIDIA's official NGC container,
  which the fine-tuning verbs (`train`, `eval`, `export`) orchestrate automatically.
  The in-container dep layer is installed with uv (never pip) into a
  **`--system-site-packages` venv** (so it inherits the container's nv torch; a bare
  `uv pip install --system` fails on the NGC image — PEP-668 as root, root-owned
  site-packages under `--user`). The pins are **validated against NGC 25.11's torch
  2.10** (`transformers==4.57.1 peft==0.18.0 trl==0.24.0 datasets==4.3.0 hf_transfer`,
  then `--no-deps unsloth unsloth_zoo bitsandbytes`; the venv-pulled torch is then
  uninstalled so the nv torch shows through). **Do not float these** — `peft>=0.19`
  hard-requires `torchao>0.16`, which needs `torch>=2.11` the container lacks. The
  exact recipe + Spark gotchas (UMA OOM → `PYTORCH_ALLOC_CONF=expandable_segments`,
  HF-cache mount, the version-matrix deadlock) live in
  [`docs/dgx-spark.md`](docs/dgx-spark.md); measured runs in
  [`docs/benchmarks.md`](docs/benchmarks.md).
  **Why a container:** the previous design listed torch + unsloth as base deps; on
  aarch64 `uv sync` resolved to `torch==2.10.0+cpu` (the CPU-only wheel) and training
  aborted with `"cannot find any torch accelerator"`. Container orchestration removes
  the wheel-resolution problem entirely.
  **Lazy-import discipline still required:** even inside the container context, never
  import torch/unsloth at module top level. Handlers lazy-import inside the function
  body; `sloth/tune/` core modules (`datasets`, `config`, `metadata`, `scope`) stay
  pure-stdlib. The afi rubric gate checks the CLI contract — a top-level heavy import
  that slowed introspection still turns it red.

## Vendored skills (cite-don't-import)

`.claude/skills/` holds 11 skills vendored **verbatim** from `guildmaster` (the
skills supplier; `think`/`spec-to-plan`/`assign-to-workforce` originate in
`devague` and are re-broadcast). Provenance and the re-sync procedure live in
`docs/skill-sources.md`. Rules: edit script bodies **upstream**, not here; every
`SKILL.md` must carry `type: command` (load-bearing — the culture/claude backend
silently skips a `SKILL.md` without it); skills are excluded from Sonar analysis
and markdownlint. The one standing local divergence is the `agex`→`devex` rename
patched in place (tracked upstream on guildmaster#48).

## Conventions and workflow

**Memory discipline — recall before, remember after.** This repo keeps its
eidetic memory **in-repo and public**: records resolve to
`<repo-root>/.eidetic/memory` — committed, and shared with the team and mesh
peers (the `claude` and `colleague` backends both read the same
`unsloth-cli` scope), so memory travels with the repo, not a private
home-dir store. Make it a per-task habit:

- **`/recall` before you start.** Search the store for the area you're about
  to touch — prior decisions, gotchas, "have we done this before?" — so you
  build on what's already known instead of re-deriving it. Do this before
  non-trivial tasks, not just when asked.
- **`/remember` when something worth keeping surfaces.** A non-obvious
  decision and its rationale, a constraint, a fix and *why* it was needed, a
  gotcha that cost time, a fact the next session would otherwise re-learn.
  Capture it as it happens, not at the end when it's faded.

A plain `/remember` lands the note in `./.eidetic/memory` in this repo — no
flag needed (the wrappers here default to `--visibility public`; in-repo
routing needs `eidetic >= 0.10.0`, older CLIs keep records in `$HOME`). Keep
something out of the committed store only by passing `--visibility private`
(routes to `$HOME/.eidetic/memory`, never committed); `/recall` reads both
stores and merges. Don't store what the repo already records (code structure,
git history, what's already in this file or `CHANGELOG.md`) — store what you'd
have to re-derive. These are the `recall`/`remember` skills (`.claude/skills/`),
backed by the `eidetic` store.

## AgentCulture sibling ecosystem

unsloth-cli is one **sibling** in the [AgentCulture](https://github.com/agentculture)
mesh: it wears the shape `steward` defines and pulls its tooling/skills from the
other siblings. When a task points outside this repo, route it to the owner —
and use the `communicate` skill (issues via `agtag`, mesh messages) to reach
them. Most siblings are checked out next to this repo under `../`.

| Sibling | Owns | Touches this repo as |
|---------|------|----------------------|
| **agentfront** (was `teken`, was `afi-cli`) | The agent-first runtime + the rubric that `… cli doctor` enforces | The cited CLI source + the dev-only rubric gate. **Renamed:** the tool is now `agentfront`, but this repo still pins `teken>=0.8` (the deprecated alias) and CI runs `teken cli doctor`. Either name works today; expect the pin to migrate to `agentfront`. |
| **steward** | Agent *alignment* — the sibling-pattern baseline + `steward doctor` | `doctor` here reproduces steward's invariants; `docs/steward/steward-suggestions.md` is its generated report (current finding: add the `ask-colleague` skill). |
| **guildmaster** | The skills *supplier/manager* | Source of the 11 vendored `.claude/skills/` (provenance in `docs/skill-sources.md`). |
| **devague** | The think → spec-to-plan → assign-to-workforce planning chain | True origin of those three skills (re-broadcast via guildmaster). |
| **devex** | The PR-lifecycle CLI (`devex pr`) | What the `cicd` skill delegates to; needs `devex>=0.21` on PATH. |
| **agtag** | Issue I/O (`agtag issue`) | What the `communicate` skill wraps; needs `agtag>=0.1` on PATH. |
| **lobes** (`lobes-cli`) | Runs / assesses / switches the local OpenAI-compatible vLLM model the mesh consumes | Will serve the models the planned fine-tune verbs produce. |
| **colleague** | A swappable coder-agent harness — one runtime, many model backends | Supplies the recommended-but-missing `ask-colleague` skill (per steward's report); a downstream consumer of trained adapters. |
| **culture / daria** | The IRC agent mesh / the awareness agent | Where this agent's identity (`culture.yaml`) is declared and run. |
