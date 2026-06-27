---
name: unsloth-cli-guide
type: command
description: >
  Explain how to use unsloth-cli — the agent-first CLI and its fine-tuning verbs.
  Covers the two faces of the tool (GPU-free introspection verbs: whoami / learn /
  explain / overview / doctor / cli; and the LoRA/QLoRA fine-tuning verbs: train /
  eval / export that orchestrate the NGC container), the agent-first output
  contract (results→stdout, errors→stderr, `--json`, exit codes 0/1/2, the
  `error:`/`hint:` shape), the dataset schemas + TOML run-config, and the DGX Spark
  prerequisites. Surfaces the CLI's *own* live teaching output via
  `scripts/guide.sh` (a thin wrapper over `sloth learn` / `overview` / `explain` /
  `cli overview`). Use when the user asks "how do I use unsloth-cli", "how does
  sloth work", "what can this CLI do", "how do I fine-tune with this", "explain the
  CLI", "how do I get started", or is new to the repo. For *running* the
  fine-tuning loop use the sibling `/finetune` skill; this skill TEACHES, it does
  not train. First-party to unsloth-cli; not vendored from guildmaster.
---

# unsloth-cli-guide — how to use unsloth-cli

This skill teaches an agent (or a person) how to drive **unsloth-cli**. It pairs a
written guide (below) with `scripts/guide.sh`, which prints the CLI's *own* live
teaching output so the answer is never stale.

```bash
bash .claude/skills/unsloth-cli-guide/scripts/guide.sh overview   # descriptive snapshot
bash .claude/skills/unsloth-cli-guide/scripts/guide.sh learn      # the self-teaching prompt
bash .claude/skills/unsloth-cli-guide/scripts/guide.sh verbs      # the CLI surface itself
bash .claude/skills/unsloth-cli-guide/scripts/guide.sh explain train   # docs for a verb
bash .claude/skills/unsloth-cli-guide/scripts/guide.sh finetune   # fine-tuning quickstart + train --help
```

Every underlying verb supports `--json`; pass it through (e.g. `… learn --json`).

## What unsloth-cli is

An **agent + CLI that simplifies fine-tuning with Unsloth**. It has two faces:

1. **Introspection verbs** — a pure-stdlib, agent-first CLI that installs and runs
   *everywhere* (no GPU, no heavy deps): `whoami`, `learn`, `explain`, `overview`,
   `doctor`, `cli`.
2. **Fine-tuning verbs** — `train`, `eval`, `export` for **LoRA/QLoRA adapter**
   tuning of Qwen models. The GPU stack (torch + unsloth) is **not** a pip
   dependency; these verbs run the GPU work inside NVIDIA's **NGC PyTorch
   container** and orchestrate it for you.

> **Names:** the dist/PyPI name is `unsloth-cli`, but the installed console script
> and import package are `sloth`. Real invocations are **`sloth <verb>`** or
> **`uv run sloth <verb>`** (or `python -m sloth <verb>`). The help text *prints*
> `unsloth-cli` only because that is the argparse program name.

## The agent-first contract (true for every verb)

- **Results → stdout, errors/diagnostics → stderr — never mixed.** Agents parse
  stdout, so nothing else lands there.
- **`--json`** routes the same payloads as structured JSON to the same streams.
- **Exit codes:** `0` success · `1` user-input error · `2` environment/setup error
  · `3+` reserved.
- **Errors** render as two lines — `error: <message>` then `hint: <remediation>`
  (in JSON mode: `{"code","message","remediation"}` on stderr). There is always a
  `hint:`.

## Introspection verbs (no GPU)

| Verb | What it does |
|------|--------------|
| `sloth whoami` | This agent's nick, version, backend, model (from `culture.yaml`). |
| `sloth learn` | A structured self-teaching prompt (purpose, command map, contract). |
| `sloth explain <path>…` | Markdown docs for any verb/noun path, e.g. `sloth explain train`. |
| `sloth overview` | Read-only descriptive snapshot of the agent. |
| `sloth doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `sloth cli overview` | Describe the CLI surface itself. |

Start with `sloth learn`, then `sloth explain <verb>` for any verb you want to go
deeper on.

## Fine-tuning verbs (LoRA/QLoRA, via the NGC container)

**Scope: adapters only.** LoRA and QLoRA on Qwen 3.x (4B/9B class). Full
fine-tuning of large dense models is out of scope — `sloth train` warns and
refuses rather than attempting it silently.

The loop, and what each verb does:

```bash
sloth train --config run.toml --dry-run   # GPU-free: validate + print the plan and docker command
sloth train --config run.toml             # real LoRA/QLoRA job in the NGC container → adapter + metadata
sloth eval  --adapter DIR --suite suite.jsonl   # run the adapter against a local task-schema suite
sloth export --adapter DIR --output OUT   # standard PEFT/safetensors layout (servable/runnable)
```

The **`/finetune`** skill runs all four steps as one loop and stops on the first
non-zero exit. Use this guide to *understand* the verbs; use `/finetune` to *run*
them.

### Inputs

- **Dataset (JSONL)** — two schemas, validated *before* any GPU spend:
  - **chat:** `{"messages": [{"role","content"}, …]}`
  - **task:** `{"task","input","expected_output"}`
- **Run config (TOML)** — model, method (`lora`/`qlora`), dataset, output, and
  hyperparameters (omitted keys fall back to Spark-friendly defaults).

Runnable examples live in `examples/` (`chat-smoke.jsonl`, `eval-suite.jsonl`,
`qlora-smoke.toml`, `lora-smoke.toml`). Try:
`sloth train --config examples/qlora-smoke.toml --dry-run`.

### Running on a GPU (DGX Spark)

The fine-tuning verbs need Docker + the NVIDIA Container Toolkit and pull the NGC
image (`nvcr.io/nvidia/pytorch:25.11-py3`). Preflight failures exit `2` with a
remediation. The full operator guide — prerequisites, the validated dependency
set, and the Spark/UMA gotchas — is in
[`docs/dgx-spark.md`](../../../docs/dgx-spark.md). Measured runs:
[`docs/benchmarks.md`](../../../docs/benchmarks.md); exact tested matrix:
[`docs/tested.md`](../../../docs/tested.md). The introspection verbs need none of
this.

## Where to go deeper

- `sloth learn` / `sloth explain <path>` — the CLI's own docs (always current).
- [`docs/fine-tuning.md`](../../../docs/fine-tuning.md) — feature/CLI reference.
- [`docs/dgx-spark.md`](../../../docs/dgx-spark.md) — DGX Spark operator guide.
- [`README.md`](../../../README.md) / [`CLAUDE.md`](../../../CLAUDE.md) — overview + contributor conventions.
- The `/finetune` skill — drive the validate → train → eval → export loop.

## Provenance

First-party to **unsloth-cli** (like `/finetune`); not vendored from guildmaster.
It teaches the CLI's surface — it does not modify the repo or train anything.
