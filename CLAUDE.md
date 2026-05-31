# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**unsloth-cli** is an AgentCulture mesh agent whose intended domain is *an
agent + CLI that simplifies fine-tuning with Unsloth* (adding complementary
actions so an agent can fine-tune models more easily).

**Today the repo is the scaffold, not the product.** It carries a working
agent-first CLI of *introspection* verbs (`whoami`, `learn`, `explain`,
`overview`, `doctor`, `cli`) cited from teken's `python-cli` reference, a mesh
identity, the vendored skill kit, and the CI/deploy baseline — but **no Unsloth
fine-tuning verbs exist yet.** Building those is the main forward work, and the
"Adding a verb or noun group" section below is the seam to do it through.

## Naming: `sloth` vs `unsloth-cli` (read this first)

Three different names, used in three different places — do not conflate them:

| Name | Where it applies |
|------|------------------|
| `unsloth-cli` | The **dist / PyPI name** (`pyproject.toml` `[project].name`), the argparse `prog`, and all user-facing text in `learn` / `explain` / `overview` output. |
| `sloth` | The **package directory**, the **import name** (`from sloth ...`), and the **installed console-script** (`[project.scripts] sloth = "sloth.cli:main"`). |

Consequence: the real invocations are **`uv run sloth <verb>`** or
**`uv run python -m sloth <verb>`**. `uv run unsloth-cli <verb>` **fails**
(no such script) — the README's `uv run unsloth-cli …` examples are wrong; the
help/`explain` text only *prints* `unsloth-cli` because that's the argparse
`prog`. When you read `unsloth-cli whoami` in docs/output, run `sloth whoami`.

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
`teken` is a **dev-only** dependency; the runtime has `dependencies = []` and
must stay that way (see "Zero runtime dependencies").

## Architecture

The package is a thin, dependency-free argparse CLI built around four stable
contracts. Understanding these four is enough to add features safely.

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
YAML dependency, to keep runtime deps empty). In a wheel install no
`culture.yaml` ships, so it falls back to literal defaults and `doctor` reports
a single info check and exits 0. `doctor` reuses this to verify the
`steward doctor` invariants: **prompt-file-present** + **backend-consistency**
(`claude`→`CLAUDE.md`, `acp`→`AGENTS.md`, `gemini`→`GEMINI.md`) plus a
**skills-present** check, emitting the rubric-shaped
`{healthy, checks: [{id, passed, severity, message, remediation}]}`.

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
- **Zero runtime dependencies.** `dependencies = []` is load-bearing (the CLI
  is pure stdlib). Anything heavier — including Unsloth/torch when the domain
  verbs arrive — should be an **optional extra** or invoked as an external
  subprocess (resolved via `shutil.which`), not a hard runtime import, so the
  introspection CLI keeps working on a machine without the ML stack.

## Vendored skills (cite-don't-import)

`.claude/skills/` holds 11 skills vendored **verbatim** from `guildmaster` (the
skills supplier; `think`/`spec-to-plan`/`assign-to-workforce` originate in
`devague` and are re-broadcast). Provenance and the re-sync procedure live in
`docs/skill-sources.md`. Rules: edit script bodies **upstream**, not here; every
`SKILL.md` must carry `type: command` (load-bearing — the culture/claude backend
silently skips a `SKILL.md` without it); skills are excluded from Sonar analysis
and markdownlint. The one standing local divergence is the `agex`→`devex` rename
patched in place (tracked upstream on guildmaster#48).
