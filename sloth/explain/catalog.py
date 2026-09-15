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
- `unsloth-cli validate` — validate a JSONL dataset file, or an eval suite
  (file or directory), standalone.
- `unsloth-cli config init` — write a starting `run.toml` with validated defaults.
- `unsloth-cli train` — validate a dataset and run/plan a LoRA/QLoRA adapter job.
- `unsloth-cli eval` — score an adapter against a local task-schema eval suite
  (a file or a directory of them).
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

Validate a JSONL dataset file or eval suite standalone — without loading a
`run.toml`, running `sloth train`, or launching a container. Exactly one of
`--dataset` / `--suite` is required (both/neither exits `1` with a `hint:`,
the same pattern `sloth eval`'s `--adapter`/`--model` uses):

- `--dataset PATH` calls the *same* `sloth.tune.datasets.validate_dataset`
  function that `sloth train` uses internally, so the accepted rules never
  drift between the two verbs. The schema is inferred from the first record
  when `--schema` is omitted: `chat` (`{"messages": [{role, content}, ...]}`)
  or `task` (`{"task", "input", "expected_output"}`); the auto-detected schema
  is echoed to stderr as a diagnostic.
- `--suite PATH` calls the *same* `sloth.tune.datasets.validate_suite`
  function that `sloth eval` uses for its pre-launch check, so a suite that
  passes here is guaranteed to pass `sloth eval`'s host-side validation too.
  `PATH` may be a single `.jsonl` file or a directory — a directory is
  expanded to its sorted `*.jsonl` children and every one is validated
  **always against the task schema** (eval suites are task-schema only, the
  same rule `sloth eval` enforces), reporting a per-file record count plus
  the aggregate total. Passing an explicit `--schema` other than `task`
  together with `--suite` exits `1` with a `hint:` — it would otherwise
  report a suite as valid against a schema `sloth eval` will not accept.

This module is pure stdlib — no torch/unsloth import, so it stays usable on a
machine with no GPU stack installed.

## Usage

    unsloth-cli validate --dataset data/train.jsonl
    unsloth-cli validate --dataset data/train.jsonl --schema task
    unsloth-cli validate --dataset data/train.jsonl --json
    unsloth-cli validate --suite data/eval.jsonl
    unsloth-cli validate --suite examples/eval/ --json

## Key flags

- `--dataset PATH` — path to a JSONL training dataset file. Mutually
  exclusive with `--suite`.
- `--suite PATH` — path to a task-schema JSONL eval suite, or a directory of
  them (every `*.jsonl` child is validated). Mutually exclusive with
  `--dataset`.
- `--schema {chat,task}` — schema to validate against. With `--dataset`:
  default auto-detect from the first record. With `--suite`: always `task`
  (eval suites are task-schema only); passing `--schema chat` (or any value
  other than `task`) together with `--suite` exits `1` with a `hint:`.
- `--json` — emit the result as structured JSON: `{valid, schema, line_count}`
  for `--dataset`, or `{valid, schema, files, total_records}` for `--suite`
  (`files` is a list of `{path, line_count}`, one per resolved file).

## Exit codes

- `0` success — dataset/suite is valid.
- `1` user-input error — missing file, invalid JSON, a record that fails
  schema validation (the validator's own `CliError` propagates verbatim,
  naming the file and line for a `--suite` directory), an empty `--suite`
  directory, an explicit non-`task` `--schema` passed with `--suite`, or
  both/neither of `--dataset`/`--suite` passed.
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

The generated file omits `target_modules` (optional; a list of module names, a
single regex string, or a known preset such as `"preset:lfm2"`). Add it under
`[hyperparameters]` to extend which modules the adapter touches.

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

`[hyperparameters]` may set `target_modules` — a list of module names, a single
regex string, or a known preset such as `"preset:lfm2"` (unknown presets fail at
load time, before any GPU spend). `preset:lfm2` adapts all 92 LoRA-able modules
of LFM2.5-1.2B (24 attention, 20 short-conv, 48 feed-forward); Unsloth's default
adapts only q/k/v on the 6 attention layers.

## Exit codes

- `0` success
- `1` user-input error (missing/invalid config, malformed dataset, out-of-scope request)
- `2` environment / setup error
"""

_EVAL = """\
# unsloth-cli eval

Score a trained LoRA/QLoRA adapter — or a merged / quantized model directory —
against a local task-schema eval suite, reporting exact-match scoring. All
inference is local and offline (`local_files_only=True`); the heavy ML stack is
imported lazily inside the inference backend, so importing the verb stays
torch-free.

The suite is one or more JSONL files whose records conform to the **task**
schema (`{"task", "input", "expected_output"}`). `--suite` accepts a single file
or a directory — a directory is expanded to its sorted `*.jsonl` children — and
is repeatable (`--suite a.jsonl --suite dir/`). Every resolved file is validated
against the task schema **before any container is launched**: a malformed file
anywhere in a `--suite` directory exits `1` naming the file and line, and costs
no docker/GPU spend. `--batch-size` is validated on the host too — a value
below `1` exits `1` before the suite is even checked. Each record's prediction
is compared to its `expected_output` for an exact match and scored with a
stdlib token-level F1 (`f1`, no external scoring dependency). The result is an
aggregate summary (`total`, `exact_match`, `exact_match_pct`, `f1`) at the top
level, plus a per-file `files` list (the same fields, scoped to each resolved
suite file) and per-record `results`. `eval.json` (the summary plus
`suite_paths` / `target` / `written_at`) is written into the directory that
was evaluated: the adapter directory for `--adapter`, or the export directory
for `--model` — the parent directory when a single `.gguf` file is passed.

## Two targets: `--adapter` or `--model`

Exactly one is required — passing both, or neither, exits `1` with a `hint:`.

- `--adapter DIR` scores a LoRA adapter by loading its base model and wrapping it
  with the adapter weights.
- `--model DIR` scores a directory produced by `unsloth-cli export`: a merged
  (bf16/4-bit) checkpoint, an AWQ or NVFP4 compressed-tensors checkpoint (both
  auto-detected by transformers from `config.json`), or a GGUF — which is scored
  with llama.cpp's `llama-completion` from the mounted llama.cpp cache.

A `--model` run reports the same score fields plus `model_dir`, `quant_method`
and `quant_format` (read from `config.json`'s `quantization_config`; both `null`
for a plain bf16 merged directory or a GGUF).

## Usage

    unsloth-cli eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    unsloth-cli eval --model exports/qwen3-4b-awq --suite examples/eval/
    unsloth-cli eval --model exports/qwen3-4b-gguf --suite data/eval.jsonl --quant q4_k_m
    unsloth-cli eval --adapter adapters/qwen3-4b-qlora --suite examples/eval/ \\
        --batch-size 16 --json

## Key flags

- `--adapter DIR` — adapter directory produced by `unsloth-cli train`.
- `--model DIR` — merged / quantized / GGUF directory produced by
  `unsloth-cli export` (mutually exclusive with `--adapter`).
- `--suite PATH` (required, repeatable) — a task-schema JSONL eval suite file,
  or a directory of them (every `*.jsonl` child is scored, sorted). Pass
  `--suite` more than once to combine several files/directories.
- `--quant NAME` — when `--model` holds several GGUF files, the quant tag to
  score (case-insensitive, e.g. `q4_k_m`); ignored for a single-GGUF or
  non-GGUF `--model` directory, and for `--adapter`.
- `--batch-size N` — generation batch size for the eval loop (default: `8`);
  must be `>= 1` (`1` is the explicit unbatched mode) or the run exits `1`
  before any suite validation or container launch.
- `--json` — emit the scored summary and per-record results as structured JSON.

## Exit codes

- `0` success
- `1` user-input error (both/neither target, missing adapter or model dir,
  a `--batch-size` below `1`, a missing `--suite` path, an empty `--suite`
  directory, or a malformed record anywhere in the suite — named by file and
  line, before any container launch)
- `2` environment / setup error (ML stack not installed, llama.cpp missing, OOM)
"""

_EXPORT = """\
# unsloth-cli export

Export a trained adapter to a servable model layout. Six formats, two lanes:

- `safetensors` (default) — pure stdlib, **no container**. Copies/normalises the
  adapter into the canonical PEFT layout that `lobes` can serve and `colleague`
  can run:

      <output>/
        adapter_config.json
        adapter_model.safetensors
        tokenizer.json / tokenizer_config.json / special_tokens_map.json /
        vocab.json / merges.txt / tokenizer.model   # copied when present

  Nothing is loaded or converted — unsloth/PEFT already write these files in
  safetensors format during training, so this lane only reorganises and
  validates file-system artefacts. No torch import, no container launch.

- `merged-16bit`, `merged-4bit`, `gguf`, `awq`, `nvfp4` — **host→container**.
  These genuinely need the ML stack (base-model load, merge, ggml conversion,
  llm-compressor one-shot quantization), so the host side validates everything
  cheaply and hands off to the NGC container (same pattern as `sloth train` /
  `sloth eval`). Safety rails: **no-clobber** (a non-empty `--output` is
  refused unless `--force`), **atomic output** (the container writes to
  `<output>.partial` and the host renames it to `<output>` only after a clean
  exit, so a killed/OOM run never leaves a half-written model directory), and
  a **fail-closed disk check** (the estimated artifact size is compared
  against free space under `--output` before any GPU spend; `--dry-run`
  reports both numbers instead of failing).

When `--output` is omitted (or resolves to the adapter directory itself), the
adapter is normalised in place.

## Usage

    unsloth-cli export --adapter adapters/qwen3-4b-qlora
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --output exported/qwen3-4b
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --format gguf \\
        --quant q4_k_m --output exported/qwen3-4b-gguf
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --format awq \\
        --calib data/calib.jsonl --output exported/qwen3-4b-awq
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --format nvfp4 \\
        --output exported/qwen3-4b-nvfp4
    unsloth-cli export --adapter adapters/qwen3-4b-qlora --dry-run --json

## Key flags

- `--adapter DIR` (required) — adapter directory to export.
- `--format FMT` — one of `safetensors`, `merged-16bit`, `merged-4bit`, `gguf`,
  `awq`, `nvfp4` (default: `safetensors`).
- `--output DIR` — output directory (default: normalise in place inside `--adapter`).
- `--quant LIST` — comma-separated ggml quantizations for `--format gguf`
  (e.g. `q4_k_m,q8_0`); ignored (with a diagnostic) for other formats.
- `--calib PATH` — JSONL calibration data for `--format awq`/`nvfp4` (default:
  the run's training dataset); ignored (with a diagnostic) for other formats.
- `--calib-samples N` — cap the number of calibration samples used for `awq`/`nvfp4`.
- `--base ID` — base model id (default: read from the adapter's
  `adapter_config.json` `base_model_name_or_path`); required for the
  container formats when it cannot be resolved.
- `--force` — overwrite a non-empty `--output` directory (container lane only).
- `--dry-run` — resolve and print the export plan (format, base, disk estimate,
  and the exact docker command for container formats) without exporting or
  launching anything.
- `--keep-intermediate` — keep intermediate artifacts (e.g. the F16 GGUF)
  instead of deleting them.
- `--json` — emit the export result (output dir, format, files) as structured JSON.

## Container vs. no-container

- **No container:** `--format safetensors` only. Pure stdlib, instant, no GPU.
- **Container (NGC, same image as `train`/`eval`):** `merged-16bit`,
  `merged-4bit`, `gguf`, `awq`, `nvfp4`. Host validates cheaply, then launches
  `nvcr.io/nvidia/pytorch:25.11-py3` with identity bind-mounts for the parents
  of the adapter/output/calibration paths, forwarding the same args plus a
  hidden `--in-container` recursion guard.

## Paths given to --adapter / --output / --calib / a local --base are canonicalised and
must sit under the working directory, your home, the HF cache or the temp dir;
extend the allow-list with `SLOTH_ALLOWED_ROOTS=<dir>[:<dir>]`.

Exit codes

- `0` success
- `1` user-input error (missing/incomplete adapter dir, unsupported `--format`,
  bad `--quant`, missing calibration file, non-empty `--output` without
  `--force`)
- `2` environment / setup error (container exits non-zero, insufficient disk
  space under `--output`)
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
