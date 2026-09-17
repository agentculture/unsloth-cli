---
name: finetune
type: command
description: >
  Drive a LoRA/QLoRA adapter training run end-to-end for this repo's fine-tuning
  verbs: validates the dataset and scope guard (GPU-free), resolves the plan via
  `sloth train --dry-run`, then (for a real run) runs the full adapter job
  (`sloth train`), evaluates the adapter against one or more eval suites
  (`sloth eval`, repeatable `--suite`), exports it to a standard PEFT/safetensors
  layout (`sloth export`), and, with `--base`, adds a fifth step comparing the
  adapter against its base model (`sloth compare --base`). The loop stops on
  the first non-zero exit and surfaces the CLI's `error:`/`hint:` output
  verbatim. Supports `--json` throughout for machine-readable output. Use when
  the user says "fine-tune a model", "run the training loop", "train a LoRA
  adapter", "validate my dataset", "dry-run the training plan", or "drive the
  train → eval → export loop end-to-end". Dry-run
  mode (GPU-free) works on any machine; a real run requires the tuning stack
  (which ships with `unsloth-cli` — `uv tool install unsloth-cli`) and a
  CUDA-capable GPU. First-party to unsloth-cli; not vendored from guildmaster.
---

# finetune — drive the validate → train → eval → export loop

`finetune` is the orchestrating skill for **unsloth-cli**'s fine-tuning verbs.
It drives `sloth train`, `sloth eval`, and `sloth export` in the order the
design doc mandates: validate first (no GPU), then train, then eval, then
export — and, when `--base` is given, a fifth step comparing the trained
adapter against its base model with `sloth compare --base`. The loop stops at
the first non-zero exit and surfaces the CLI's `error:`/`hint:` contract
verbatim.

The entry point is `scripts/finetune.sh`. Run it from anywhere; CLI resolution
is portable and resolves in order: `SLOTH_BIN` (if set, used verbatim as the
CLI command) → `uv run --project <dir> sloth` from a walked-up unsloth-cli
checkout → installed `sloth` on `PATH`.

## Modes

### `run` — orchestrated end-to-end loop

```bash
bash .claude/skills/finetune/scripts/finetune.sh run \
    --config <run.toml> \
    --suite <suite.jsonl | dir> [--suite <suite.jsonl | dir> ...] \
    [--dry-run] \
    [--json] \
    [--batch-size <n>] \
    [--perplexity] \
    [--export-format <fmt>] \
    [--quant <list>] \
    [--base <hf-id-or-dir>]
```

The loop runs four steps in order, plus an optional fifth when `--base` is given:

| Step | Command | GPU? |
|------|---------|------|
| 1 — validate + plan | `sloth train --config <c> --dry-run` | no |
| 2 — train | `sloth train --config <c>` | **yes** |
| 3 — eval | `sloth eval --adapter <out> --suite <s1> [--suite <s2> ...] [--batch-size <n>] [--perplexity]` | yes |
| 4 — export | `sloth export --adapter <out> --format <fmt>` | no for `safetensors`; **yes** for `merged-16bit`/`merged-4bit`/`gguf`/`awq`/`nvfp4` |
| 5 — compare (only with `--base`) | `sloth compare --base <ref> <out> --config <c>` | yes (two sequential container runs) |

Step banners print `N/5` when `--base` is given and `N/4` otherwise — step 5
is only printed (and only runs) with `--base`.

`--suite` is **repeatable** — pass it more than once to score several named
eval suites in the same `sloth eval` call (each `--suite` is forwarded
verbatim; a stem collision between two `--suite` entries exits 1, per `sloth
explain eval`). `--batch-size <n>` and `--perplexity` are forwarded to the
eval step (`sloth eval --batch-size <n>` / `sloth eval --perplexity`) only
when given.

Export format is `--export-format` (default `safetensors`), forwarded to
`sloth export --format <fmt>`: `safetensors`, `merged-16bit`, `merged-4bit`,
`gguf`, `awq`, `nvfp4`. `--quant` (comma-separated ggml quant list, e.g.
`q4_k_m,q8_0`) is forwarded to `sloth export --quant <list>` and applies to
`--export-format gguf` only.

`--base <hf-id-or-dir>` adds step 5: an adapter-vs-base-model comparison via
`sloth compare --base <ref> <adapter_dir> --config <run.toml>`, gated on the
run config's `[eval.thresholds]` section (or the built-in baseline when the
config has none — see `sloth explain compare` / `sloth explain config`).

**With `--dry-run`**: only step 1 runs (validate the dataset, scope-guard the
model+method, print the resolved plan). Exits 0 on success. No torch import,
no GPU required. Use this to check a new config before committing GPU time.

**Without `--dry-run`**: step 1 always runs first (the plan JSON is captured to
derive the adapter output dir), then steps 2–4 run in sequence, and step 5 too
when `--base` is given. Any step that exits non-zero stops the loop; the CLI's
`error:` / `hint:` lines are already on stderr.

The adapter output directory is derived from the training plan's `output` field
(captured via `--json` from the dry-run step 1). It matches the `output` key in
the `[run]` section of the TOML — so relative paths resolve against the working
directory where you invoke `finetune.sh`.

### `<verb> [args...]` — thin pass-through

```bash
bash .claude/skills/finetune/scripts/finetune.sh train --config run.toml --dry-run
bash .claude/skills/finetune/scripts/finetune.sh eval  --adapter adapters/my-lora --suite data/eval.jsonl --json
bash .claude/skills/finetune/scripts/finetune.sh export --adapter adapters/my-lora --format safetensors
```

Any first argument that is not `run` or `help` is forwarded verbatim to
`sloth <verb> [args...]`. Use this to drive an individual step or any other
`sloth` verb (`whoami`, `doctor`, `explain`, …) through the same portable CLI
resolution.

### `help` — usage

```bash
bash .claude/skills/finetune/scripts/finetune.sh help
```

## Flags for `run`

| Flag | Required | Description |
|------|----------|-------------|
| `--config <run.toml>` | yes | Path to the TOML describing model, dataset, output, and method. |
| `--suite <suite.jsonl \| dir>` | yes, repeatable | Path to an eval suite (chat/task/instruction/structured/toolcall schema, auto-detected). Pass more than once to score several named suites in one `sloth eval` call. |
| `--dry-run` | no | Run step 1 only (validate + plan, GPU-free). |
| `--json` | no | Forward `--json` to every `sloth` call for machine-readable output. |
| `--batch-size <n>` | no | Forwarded to step 3's `sloth eval --batch-size <n>` (generation batch size). |
| `--perplexity` | no | Forwarded to step 3's `sloth eval --perplexity` (also compute held-out perplexity/loss). |
| `--export-format <fmt>` | no | Format for step 4: `safetensors` (default), `merged-16bit`, `merged-4bit`, `gguf`, `awq`, `nvfp4`. Forwarded to `sloth export --format <fmt>`. |
| `--quant <list>` | no | Comma-separated ggml quantizations for `--export-format gguf` (e.g. `q4_k_m,q8_0`). Forwarded to `sloth export --quant <list>`. |
| `--base <hf-id-or-dir>` | no | Adds step 5: adapter-vs-base-model comparison via `sloth compare --base <ref> <out> --config <c>`. |

## Dataset schemas

Two training-dataset schemas are recognised by `sloth train`:

- **chat** — `{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}`
- **task** — `{"task": "…", "input": "…", "expected_output": "…"}`

Schema is auto-detected from the first non-blank record.

`sloth eval`'s `--suite` accepts five schemas, each auto-detected per record:
**chat**, **task**, **instruction**, **structured**, and **toolcall** — see
`sloth explain eval` for the exact field shape of each.

## Scope guard

`sloth train` enforces the repo's scope rule before any GPU work:

- **In scope**: LoRA / QLoRA adapters for Qwen 3.x 4B/9B and similar small models.
- **Out of scope**: full fine-tuning of large dense models (refused with `CliError`).

A scope warning is printed to stderr as a diagnostic; a hard refusal exits 1 with
`error:` / `hint:`. The dry-run step always checks scope, so violations surface
before GPU time is spent.

## Example — dry-run smoke (no GPU needed)

```bash
# Create a minimal config and dataset, then validate without a GPU:
cat > /tmp/run.toml <<'EOF'
[run]
model   = "unsloth/Qwen3-4B"
method  = "qlora"
dataset = "/tmp/train.jsonl"
output  = "/tmp/adapters/qwen3-4b-qlora"
EOF

printf '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n' \
    > /tmp/train.jsonl
printf '{"task":"greet","input":"hi","expected_output":"hello"}\n' \
    > /tmp/suite.jsonl

bash .claude/skills/finetune/scripts/finetune.sh run \
    --config /tmp/run.toml \
    --suite  /tmp/suite.jsonl \
    --dry-run
```

Expected: prints the resolved plan (model, method, dataset, output, hyperparameters)
and exits 0. No torch, no GPU.

## Example — real run (needs GPU + tuning stack)

```bash
bash .claude/skills/finetune/scripts/finetune.sh run \
    --config data/runs/qwen3-4b-qlora.toml \
    --suite  data/eval/contract-suite.jsonl
```

Runs all four steps, writing the adapter to the `output` dir in the TOML and
emitting a summary of each step to stderr. Add `--export-format gguf --quant
q4_k_m` (or `merged-16bit` / `merged-4bit` / `awq` / `nvfp4`) to export a
deployable format instead of the default `safetensors` — see
[deployment targets](../../../docs/fine-tuning.md#deployment-targets) for which
format each target platform expects.

## Example — repeated `--suite`, and comparing against the base model

```bash
bash .claude/skills/finetune/scripts/finetune.sh run \
    --config data/runs/qwen3-4b-qlora.toml \
    --suite  data/eval/regression.jsonl \
    --suite  data/eval/toolcalls.jsonl \
    --batch-size 16 \
    --perplexity \
    --base   unsloth/Qwen3-4B \
    --json
```

Scores both `regression` and `toolcalls` suites in the eval step (batch size
16, with perplexity), then adds step 5 — `sloth compare --base unsloth/Qwen3-4B
<adapter_dir> --config data/runs/qwen3-4b-qlora.toml` — which fails the whole
run (exit 1) if the adapter regresses past the config's `[eval.thresholds]`
(or the built-in baseline).

## Exit codes

The script propagates the exit code of the first failing `sloth` call verbatim:

| Code | Meaning |
|------|---------|
| 0 | All steps succeeded (or dry-run validated). |
| 1 | User-input error (bad config, malformed dataset, out-of-scope request). |
| 2 | Environment error (tuning stack not installed, file not found). |

## Requirements

- **Dry-run**: stdlib Python 3.11+ (no torch, no GPU). CLI resolution order:
  `SLOTH_BIN` (if set) → `uv run --project <dir> sloth` from a walked-up
  unsloth-cli checkout → installed `sloth` on `PATH`.
- **Real run**: the tuning stack (ships with `unsloth-cli` — `uv tool install
  unsloth-cli`) and a CUDA-capable GPU. See `sloth explain train` for the annotated TOML
  template.

## Provenance

First-party to **unsloth-cli** — this skill drives this repo's own verbs.
It is not vendored from guildmaster (the external skills supplier); do not add it
to `docs/skill-sources.md`.
