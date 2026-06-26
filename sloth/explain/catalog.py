"""Markdown catalog for ``unsloth-cli explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("unsloth-cli",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# unsloth-cli

A clonable template for AgentCulture mesh agents. It carries an agent-first CLI
(cited from the teken `python-cli` reference), a mesh identity (`culture.yaml` +
`CLAUDE.md`), the canonical guildmaster skill kit under `.claude/skills/`, and a
buildable/deployable package baseline. Clone it, rename the package, edit
`culture.yaml`, and you have a new agent.

## Verbs

- `unsloth-cli whoami` — identity probe from `culture.yaml`.
- `unsloth-cli learn` — structured self-teaching prompt.
- `unsloth-cli explain <path>` — markdown docs for any noun/verb.
- `unsloth-cli overview` — descriptive snapshot of the agent.
- `unsloth-cli doctor` — check the agent-identity invariants.
- `unsloth-cli cli overview` — describe the CLI surface.
- `unsloth-cli train` — validate a dataset and run/plan a LoRA/QLoRA adapter job.
- `unsloth-cli eval` — score an adapter against a local task-schema eval suite.
- `unsloth-cli export` — export an adapter to a PEFT/safetensors layout.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `unsloth-cli explain whoami`
- `unsloth-cli explain doctor`
"""

_WHOAMI = """\
# unsloth-cli whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    unsloth-cli whoami
    unsloth-cli whoami --json
"""

_LEARN = """\
# unsloth-cli learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    unsloth-cli learn
    unsloth-cli learn --json
"""

_EXPLAIN = """\
# unsloth-cli explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    unsloth-cli explain unsloth-cli
    unsloth-cli explain whoami
    unsloth-cli explain --json <path>
"""

_OVERVIEW = """\
# unsloth-cli overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    unsloth-cli overview
    unsloth-cli overview --json
"""

_DOCTOR = """\
# unsloth-cli doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`claude` → `CLAUDE.md`), plus a
skills-present check. Exits 1 when unhealthy.

## Usage

    unsloth-cli doctor
    unsloth-cli doctor --json
"""

_CLI = """\
# unsloth-cli cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    unsloth-cli cli overview
    unsloth-cli cli overview --json
"""

_TRAIN = """\
# unsloth-cli train

Validate a dataset and run (or plan) a small LoRA/QLoRA adapter job for a Qwen
model. The flow is: load the run config → validate the dataset *before any GPU
work* → scope-guard the (model, method) request → dry-run the plan or train the
adapter and write `training_metadata.json` next to the adapter output.

The dataset schema is inferred from the first record: `chat`
(`{"messages": [{role, content}, ...]}`) or `task`
(`{"task", "input", "expected_output"}`). An out-of-scope request — e.g. full
fine-tuning of a large dense model — is warned about explicitly on stderr and
then refused; scope is LoRA/QLoRA adapters on small models first. Torch/unsloth
are imported lazily inside the trainer, so `--dry-run` never loads the ML stack.

## Usage

    unsloth-cli train --config run.toml
    unsloth-cli train --config run.toml --dry-run
    unsloth-cli train --config run.toml --json

## Key flags

- `--config TOML` (required) — run config: model, dataset, output, and method.
- `--dry-run` — validate and resolve the plan without importing torch or training.
- `--json` — emit the resolved plan / result as structured JSON to stdout.

## Exit codes

- `0` success
- `1` user-input error (missing/invalid config, malformed dataset, out-of-scope request)
- `2` environment / setup error
"""

_EVAL = """\
# unsloth-cli eval

Run a trained LoRA/QLoRA adapter against a local task-schema eval suite and
report exact-match scoring. All inference is local and offline
(`local_files_only=True`); the heavy ML stack is imported lazily inside the
inference backend, so importing the verb stays torch-free.

The suite is a JSONL file whose records conform to the **task** schema
(`{"task", "input", "expected_output"}`), validated before inference. Each
record's prediction is compared to its `expected_output` for an exact match, and
a summary (`total`, `exact_match`, `exact_match_pct`) plus per-record results is
emitted.

## Usage

    unsloth-cli eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    unsloth-cli eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl --json

## Key flags

- `--adapter DIR` (required) — adapter directory produced by `unsloth-cli train`.
- `--suite PATH` (required) — task-schema JSONL eval suite.
- `--json` — emit the scored summary and per-record results as structured JSON.

## Exit codes

- `0` success
- `1` user-input error (missing adapter dir, missing/malformed suite)
- `2` environment / setup error (ML stack not installed)
"""

_EXPORT = """\
# unsloth-cli export

Export a trained adapter to the canonical PEFT/safetensors layout that `lobes`
can serve and `colleague` can run:

    <output>/
      adapter_config.json
      adapter_model.safetensors

This is a pure stdlib file-system operation — no torch or ML runtime is loaded.
When `--output` is omitted (or resolves to the adapter directory itself), the
adapter is normalised in place. Only the `safetensors` format is supported today.

## Usage

    unsloth-cli export --adapter adapters/qwen3-4b-qlora
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --output exported/qwen3-4b
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --json

## Key flags

- `--adapter DIR` (required) — adapter directory to export.
- `--format FMT` — output format (default: `safetensors`).
- `--output DIR` — output directory (default: normalise in place inside `--adapter`).
- `--json` — emit the export result (output dir, format, files) as structured JSON.

## Exit codes

- `0` success
- `1` user-input error (missing adapter dir, unsupported format)
- `2` environment / setup error
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("unsloth-cli",): _ROOT,
    # The console-script / package name is `sloth` (the dist name is
    # `unsloth-cli`). The agent-first rubric's `explain_self` check runs
    # `explain <script-name>`, i.e. `explain sloth`, so alias it to the root.
    ("sloth",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
    ("train",): _TRAIN,
    ("eval",): _EVAL,
    ("export",): _EXPORT,
}
