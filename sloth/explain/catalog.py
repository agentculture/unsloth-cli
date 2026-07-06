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
- `unsloth-cli validate` — validate a JSONL dataset file standalone.
- `unsloth-cli config init` — write a starting `run.toml` with validated defaults.
- `unsloth-cli train` — validate a dataset and run/plan a LoRA/QLoRA adapter job.
- `unsloth-cli eval` — score an adapter against a local task-schema eval suite.
- `unsloth-cli export` — export an adapter to a PEFT/safetensors layout.
- `unsloth-cli runs list` / `runs show <run_id>` — enumerate/inspect past runs
  from the run registry (`<runs-root>/runs.jsonl`) — no directory walking.
- `unsloth-cli summarize <run_id|dir>` — one JSON summary of a past run
  (training_metadata.json + trainer_state.json).
- `unsloth-cli compare <a> <b>` — side-by-side config deltas + summaries for
  two past runs.

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

_VALIDATE = """\
# unsloth-cli validate

Validate a JSONL dataset file standalone — without loading a `run.toml` or
running `sloth train`. Calls the *same*
`sloth.tune.datasets.validate_dataset` function that `sloth train` uses
internally, so the accepted rules never drift between the two verbs.

The dataset schema is inferred from the first record when `--schema` is
omitted: `chat` (`{"messages": [{role, content}, ...]}`) or `task`
(`{"task", "input", "expected_output"}`); the auto-detected schema is echoed
to stderr as a diagnostic. This module is pure stdlib — no torch/unsloth
import, so it stays usable on a machine with no GPU stack installed.

## Usage

    unsloth-cli validate --dataset data/train.jsonl
    unsloth-cli validate --dataset data/train.jsonl --schema task
    unsloth-cli validate --dataset data/train.jsonl --json

## Key flags

- `--dataset PATH` (required) — path to the JSONL dataset file.
- `--schema {chat,task}` — schema to validate against (default: auto-detect
  from the first record).
- `--json` — emit `{valid, schema, line_count}` as structured JSON to stdout.

## Exit codes

- `0` success — dataset is valid.
- `1` user-input error — missing file, invalid JSON, or a record that fails
  schema validation (the validator's own `CliError` propagates verbatim).
- `2` environment error — the dataset file exists but cannot be opened
  (e.g. a permission error).
"""

_CONFIG_INIT = """\
# unsloth-cli config init

Write a starting `run.toml` for `sloth train`, with the `[hyperparameters]`
section populated from the same documented defaults `sloth.tune.config`
uses (`DEFAULT_LORA_R`, `DEFAULT_LEARNING_RATE`, etc.) — the generated file
always round-trips through `sloth.tune.config.load_config` validation.
Refuses to overwrite an existing file unless `--force` is passed.

## Usage

    unsloth-cli config init --model unsloth/Qwen3-4B \\
        --dataset data/train.jsonl --output adapters/out
    unsloth-cli config init --model unsloth/Qwen3-4B \\
        --dataset data/train.jsonl --output adapters/out --method lora
    unsloth-cli config init --model unsloth/Qwen3-4B \\
        --dataset data/train.jsonl --output adapters/out --force --json

## Key flags

- `--model ID` (required) — model identifier (e.g. `unsloth/Qwen3-4B`).
- `--dataset PATH` (required) — path to the JSONL dataset file.
- `--output DIR` (required) — output directory for the adapter.
- `--method {lora,qlora}` — adapter method (default: `qlora`).
- `--force` — overwrite an existing `run.toml`.
- `--path PATH` — override the written config path (default: `<output>/run.toml`).
- `--json` — emit the write result as structured JSON.

## Exit codes

- `0` success — config written.
- `1` user-input error — the target file already exists and `--force` was
  not passed.
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


_RUNS = """\
# unsloth-cli runs

Noun group over the **run registry**: `sloth train` appends one line per real
(non-dry-run) attempt to `<runs-root>/runs.jsonl`, where **runs-root is the
parent directory of the run's `output` dir**. `runs list`/`runs show` read it
back so an agent can enumerate and inspect past runs without walking
directories.

Registry line shape: `{run_id, config_hash, output_dir, model, method,
dataset{sha256,line_count}, started, finished, status}`. `status` only ever
transitions `"running" -> "ok" | "failed"` — there is no pid tracking in v1,
so a killed `train` leaves `status: "running"` forever, reported honestly
as-is (never silently reclassified as "stale"/"incomplete").

## Usage

    unsloth-cli runs list
    unsloth-cli runs list --runs-root adapters/ --json
    unsloth-cli runs show <run_id>
    unsloth-cli runs show <run_id> --json
    unsloth-cli runs overview

## Key flags

- `--runs-root DIR` — directory containing `runs.jsonl` (default: the current
  directory).
- `--json` — emit structured JSON.

## Exit codes

- `0` success — including an honest empty list when no registry exists yet.
- `1` user-input error — `runs show` with an unknown `run_id`.
"""

_SUMMARIZE = """\
# unsloth-cli summarize

Join a past run's `training_metadata.json` with the final/best loss and step
count from its newest `checkpoint-N/trainer_state.json` into one JSON summary.
`<target>` is either a `run_id` (looked up in the registry — see
`unsloth-cli runs`) or a literal output directory; an existing directory takes
precedence when a string happens to be both.

Both halves are optional: a run with no checkpoint yet, or a missing
`training_metadata.json`, still summarizes — the gap is recorded in a `notes`
list rather than raised as an error.

## Usage

    unsloth-cli summarize <run_id>
    unsloth-cli summarize adapters/qwen3-4b-qlora
    unsloth-cli summarize <run_id> --runs-root adapters/ --json

## Key flags

- `--runs-root DIR` — directory containing `runs.jsonl`, used to resolve a
  `run_id` (default: the current directory). Ignored when `<target>` is
  already an existing directory.
- `--json` — emit the summary as structured JSON.

## Exit codes

- `0` success
- `1` user-input error — `<target>` resolves to neither a `run_id` nor an
  existing directory.
"""

_COMPARE = """\
# unsloth-cli compare

Side-by-side comparison of two past runs: the `training_metadata.json`
hyperparameter/config keys that differ between `<a>` and `<b>`, plus each
run's full `unsloth-cli summarize` summary. Each of `<a>`/`<b>` resolves the
same way `summarize`'s `<target>` does (a `run_id` or a literal output
directory).

## Usage

    unsloth-cli compare <run_id_a> <run_id_b>
    unsloth-cli compare adapters/exp-1 adapters/exp-2 --json

## Key flags

- `--runs-root DIR` — directory containing `runs.jsonl`, used to resolve a
  `run_id` on either side (default: the current directory).
- `--json` — emit `{a, b, deltas}` as structured JSON.

## Exit codes

- `0` success
- `1` user-input error — either `<a>` or `<b>` resolves to neither a `run_id`
  nor an existing directory.
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
    ("validate",): _VALIDATE,
    ("config",): _CONFIG_INIT,
    ("config", "init"): _CONFIG_INIT,
    ("train",): _TRAIN,
    ("eval",): _EVAL,
    ("export",): _EXPORT,
    ("runs",): _RUNS,
    ("runs", "list"): _RUNS,
    ("runs", "show"): _RUNS,
    ("runs", "overview"): _RUNS,
    ("summarize",): _SUMMARIZE,
    ("compare",): _COMPARE,
}
