---
name: finetune
type: command
description: >
  Drive a LoRA/QLoRA adapter training run end-to-end for this repo's fine-tuning
  verbs: validates the dataset and scope guard (GPU-free), resolves the plan via
  `sloth train --dry-run`, then (for a real run) runs the full adapter job
  (`sloth train`), evaluates the adapter against a task-schema JSONL eval suite
  (`sloth eval`), and exports it to a standard PEFT/safetensors layout (`sloth
  export`). The loop stops on the first non-zero exit and surfaces the CLI's
  `error:`/`hint:` output verbatim. Supports `--json` throughout for
  machine-readable output. Use when the user says "fine-tune a model", "run the
  training loop", "train a LoRA adapter", "validate my dataset", "dry-run the
  training plan", or "drive the train → eval → export loop end-to-end". Dry-run
  mode (GPU-free) works on any machine; a real run requires the tuning stack
  (which ships with `unsloth-cli` — `uv tool install unsloth-cli`) and a
  CUDA-capable GPU. First-party to unsloth-cli; not vendored from guildmaster.
---

# finetune — drive the validate → train → eval → export loop

`finetune` is the orchestrating skill for **unsloth-cli**'s fine-tuning verbs.
It drives three CLI verbs — `sloth train`, `sloth eval`, `sloth export` — in the
order the design doc mandates: validate first (no GPU), then train, then eval,
then export. The loop stops at the first non-zero exit and surfaces the CLI's
`error:`/`hint:` contract verbatim.

The entry point is `scripts/finetune.sh`. Run it from anywhere; CLI resolution
is portable (installed `sloth` on `PATH`, else `uv run sloth` from a checkout).

## Modes

### `run` — orchestrated end-to-end loop

```bash
bash .claude/skills/finetune/scripts/finetune.sh run \
    --config <run.toml> \
    --suite <suite.jsonl> \
    [--dry-run] \
    [--json]
```

The loop runs four steps in order:

| Step | Command | GPU? |
|------|---------|------|
| 1 — validate + plan | `sloth train --config <c> --dry-run` | no |
| 2 — train | `sloth train --config <c>` | **yes** |
| 3 — eval | `sloth eval --adapter <out> --suite <suite>` | yes |
| 4 — export | `sloth export --adapter <out> --format safetensors` | no |

**With `--dry-run`**: only step 1 runs (validate the dataset, scope-guard the
model+method, print the resolved plan). Exits 0 on success. No torch import,
no GPU required. Use this to check a new config before committing GPU time.

**Without `--dry-run`**: step 1 always runs first (the plan JSON is captured to
derive the adapter output dir), then steps 2–4 run in sequence. Any step that
exits non-zero stops the loop; the CLI's `error:` / `hint:` lines are already
on stderr.

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
| `--suite <suite.jsonl>` | yes | Path to a task-schema JSONL eval suite (`{"task","input","expected_output"}`). |
| `--dry-run` | no | Run step 1 only (validate + plan, GPU-free). |
| `--json` | no | Forward `--json` to every `sloth` call for machine-readable output. |

## Dataset schemas

Two schemas are recognised by `sloth train`:

- **chat** — `{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}`
- **task** — `{"task": "…", "input": "…", "expected_output": "…"}`

Schema is auto-detected from the first non-blank record. The eval suite (`--suite`)
must be task-schema JSONL.

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
emitting a summary of each step to stderr.

## Exit codes

The script propagates the exit code of the first failing `sloth` call verbatim:

| Code | Meaning |
|------|---------|
| 0 | All steps succeeded (or dry-run validated). |
| 1 | User-input error (bad config, malformed dataset, out-of-scope request). |
| 2 | Environment error (tuning stack not installed, file not found). |

## Requirements

- **Dry-run**: stdlib Python 3.11+ (no torch, no GPU). `sloth` must be installed
  or the repo must be on PATH with `uv` available.
- **Real run**: the tuning stack (ships with `unsloth-cli` — `uv tool install
  unsloth-cli`) and a CUDA-capable GPU. See `sloth explain train` for the annotated TOML
  template.

## Provenance

First-party to **unsloth-cli** — this skill drives this repo's own verbs.
It is not vendored from guildmaster (the external skills supplier); do not add it
to `docs/skill-sources.md`.
