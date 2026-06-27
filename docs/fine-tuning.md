# Fine-tuning reference

unsloth-cli adds three flat verbs — **`train`**, **`eval`**, **`export`** — for
Spark-friendly **LoRA / QLoRA adapter** tuning of Qwen models, plus a
[`/finetune`](../.claude/skills/finetune/SKILL.md) skill that drives the full
loop. This page is the feature/CLI reference; for the DGX Spark operator guide see
[`dgx-spark.md`](dgx-spark.md) and for measured results [`benchmarks.md`](benchmarks.md).

The verbs are **global** (siblings of `whoami`/`explain`), not a `tune` noun group.
Every verb supports `--json` and routes failures through `error:` / `hint:` on
stderr with the standard exit codes (`0` ok, `1` user error, `2` environment error).

## Scope — adapters only, by design

**LoRA and QLoRA adapters only.** Full fine-tuning of large dense models is out of
scope: `sloth train` warns explicitly and refuses (or downgrades to adapter-only)
rather than silently attempting it (`sloth/tune/scope.py`). Targets: Qwen 3.x
4B/9B and comparable adapter-class models; larger local adapters (Qwen 3.6 27B
dense, Qwen coder variants) are the stretch goal.

## The verbs

### `sloth train --config run.toml`

`load_config → validate dataset → scope-guard → (dry-run | train)`.

1. Loads the TOML run config.
2. **Validates the dataset before any GPU spend** — a malformed line fails fast
   with the offending line number.
3. Scope-guards the (model, method) request.
4. Either prints the resolved plan + the exact `docker run` command (`--dry-run`,
   GPU-free, works anywhere) or runs the real LoRA/QLoRA job inside the NGC
   container and writes the adapter + `training_metadata.json`.

```bash
sloth train --config examples/qlora-smoke.toml --dry-run   # plan only, no GPU
sloth train --config examples/qlora-smoke.toml             # real run
sloth train --config examples/qlora-smoke.toml --json      # machine-readable
```

### `sloth eval --adapter DIR --suite suite.jsonl`

Loads the base model (read from the adapter's `adapter_config.json`) + the LoRA
adapter, runs each **task-schema** item through `generate()`, and reports
exact-match. Fully local — no network. Returns
`{total, exact_match, exact_match_pct, results}`.

### `sloth export --adapter DIR --output OUT`

Pure stdlib, no GPU/container. Validates the adapter has the canonical PEFT files
and emits a standard `safetensors` layout (adapter weights + tokenizer) that
[lobes](https://github.com/agentculture/lobes-cli) can serve and
[colleague](https://github.com/agentculture/colleague) can run as a backend.

## Dataset schemas

Two JSONL schemas, validated **before** any GPU time; the schema is inferred from
the first record.

**Chat** — instruction-following / conversational behavior:

```json
{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}
```

**Task** — structured input/output:

```json
{"task": "write-issue", "input": "…", "expected_output": "…"}
```

The trainer renders each record into a single training `text` column — chat
records via the model's chat template, task records via a `Task:/Input:/Output:`
prompt (the same shape `sloth eval` scores against). Worked examples:
[`examples/chat-smoke.jsonl`](../examples/chat-smoke.jsonl),
[`examples/eval-suite.jsonl`](../examples/eval-suite.jsonl).

## Run config (TOML) + Spark-friendly defaults

Parsed read-only with stdlib `tomllib`. Omitted keys fall back to documented
Spark-friendly defaults; the same config + dataset reproduces the same run.

```toml
[run]
model   = "unsloth/Qwen3-1.7B"    # base model (Qwen 3.x adapter-class target)
method  = "qlora"                 # "lora" or "qlora" — the only supported methods
dataset = "examples/chat-smoke.jsonl"
output  = "runs/qlora-smoke"

[hyperparameters]
lora_r        = 8     # rank (default 16)
lora_alpha    = 16    # default 16
lora_dropout  = 0.0   # default 0.0
learning_rate = 2e-4  # default 2e-4
max_seq_len   = 1024  # default 2048
batch_size    = 1     # default 2 (Spark-friendly: low VRAM)
grad_accum    = 4     # default 4
max_steps     = 10    # default 60 (smoke; raise for production)
seed          = 3407  # default 3407
load_in_4bit  = true  # default true (required for qlora)
```

Ready-to-run configs: [`examples/qlora-smoke.toml`](../examples/qlora-smoke.toml),
[`examples/lora-smoke.toml`](../examples/lora-smoke.toml).

## The `/finetune` skill

Drives the loop non-interactively: validate dataset → `sloth train` → `sloth eval`
→ `sloth export`, stopping on the first non-zero exit and surfacing the CLI's
`error:`/`hint:` output. Dry-run mode is GPU-free and runs anywhere; a real run
needs the NGC container + a CUDA GPU.

## What belongs in fine-tuning vs. memory / RAG

A design rule, not a footnote — it decides where a capability lives in the mesh.

**Fine-tune** stores *stable behavior and reflexes* — bake into weights:

- CLI-contract discipline (error/hint format, exit-code policy, stream split)
- AgentCulture / CULTURE.DEV terminology and patterns
- Agent-first habits (action verbs, structured `--json`, correct error routing)
- Issue-writing format; teacher behavior for `learn` / `explain`

**Memory / RAG** stores *changing facts* — would go stale in weights:

- Current project state, open issues, branch status
- Secrets, tokens, per-deployment config
- User-specific preferences; anything better served by retrieval

**Decision rule:** *"Would this still be correct six months from now on any
deployment of the mesh?"* Yes → consider fine-tuning. Changes over time / per-user
→ memory / RAG.

## Role-specific adapters

Small, role-specific adapters rather than one mixed blob — e.g.
`culture-contract-lora`, `agentculture-cli-teacher-lora`, `repo-maintainer-lora`,
`tool-router-lora`, `agent-first-coach-lora`.

## Architecture (where to look)

The dependency-free core under `sloth/tune/` is pure stdlib and imports no torch,
so dataset/config/scope validation happens before any GPU spend:

| Module | Responsibility |
|--------|----------------|
| `datasets.py` | JSONL schema validation (chat + task) |
| `config.py` | TOML loader + Spark-friendly defaults + type/range checks |
| `metadata.py` | `training_metadata.json` writer (model/method/dataset sha256+lines/hparams/timestamp) |
| `scope.py` | adapter-OK vs out-of-scope guard |
| `container.py` | NGC `docker run` orchestration (pure stdlib; no torch) |
| `_trainer.py` | the **only** module that imports torch/unsloth/trl — lazily, inside its run functions |
