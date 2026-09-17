# Fine-tuning reference

## Agent path

The machine-readable loop, before any of the human-oriented text below: score
an adapter, then gate it against its base model, both in `--json`.

```bash
sloth eval --adapter runs/qlora-smoke --suite examples/eval/regression.jsonl \
    --suite examples/eval/instruction-following.jsonl --json
```

`sloth eval --json` always reports results **keyed by suite name**:

```text
{"suites": {<name>: {total, exact_match, exact_match_pct, f1, results, ...}}}
```

(verbatim from `sloth explain eval`; a `--model` target additionally carries
`model_dir`, `quant_method`, `quant_format`). Exit codes: `0` success; `1`
user-input error (both/neither target, a missing `--suite` path, a name
collision, a malformed suite record, a train/eval overlap, `--batch-size < 1`);
`2` environment/setup error (ML stack missing, llama.cpp missing, OOM).

Once the adapter has eval results on disk, gate it against its base model:

```bash
sloth compare --base unsloth/Qwen3-4B runs/qlora-smoke --json
```

`sloth compare --base --json` reports:

```text
{a, b, deltas: {suites: {<suite>: {<metric>: {a, b, delta}}}}, thresholds,
 verdict: {passed, failures}}
```

(verbatim from `sloth explain compare`). The `verdict` block is the answer an
agent should branch on: `verdict.passed` is `true`/`false`, and
`verdict.failures` names every failed check (a regression-tagged suite's
`exact_match_pct` drop beyond `regression_drop_pp`, `compliance_pct` below
`compliance_min_pct`, a latency ratio above `latency_max_ratio`, a suite with
fewer than `min_suite_rows` rows, or a `base_load_in_4bit` mismatch). The
report is written to stdout **before** a failing gate's error goes to stderr,
so an agent parsing stdout always gets `thresholds` + `verdict` even on exit
`1`. Exit codes: `0` success (gate passed); `1` user-input error (target
resolution, missing `<b>`/`--base` conflict, no eval results to compare
against, or a failed `--base` threshold check); `2` environment error (the
container could not run — an OOM eval lands here with a `hint:`).

An agent driving the full loop non-interactively should run
`sloth train` → `sloth eval --json` → `sloth compare --base --json` and stop
on the first non-zero exit, surfacing the CLI's `error:`/`hint:` output
verbatim — this is exactly what the [`/finetune`](../.claude/skills/finetune/SKILL.md)
skill automates.

## Overview

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

**Starter suite:** [`examples/eval/`](../examples/eval/) holds the starter eval
suite — three task-schema JSONL files (`cli-contract.jsonl`,
`agentculture-terms.jsonl`, `task-format.jsonl`, ~40+ items total) covering the
CLI output contract, AgentCulture sibling/naming terminology, and short
instruction-following items (rewrite/extract/classify). Each line is
`{"task", "input", "expected_output"}` with a short, one-sentence-or-shorter
`expected_output` so exact-match has a chance. [`examples/eval-suite.jsonl`](../examples/eval-suite.jsonl)
remains the original 4-item smoke suite for quick sanity checks. To add a new
domain, drop another `<domain>.jsonl` file into `examples/eval/` with the same
three-key schema and validate it with
`sloth validate --dataset examples/eval/<domain>.jsonl --schema task`.

### `sloth export --adapter DIR --output OUT`

Exports a trained adapter to one of **six formats** (`--format FMT`, default
`safetensors`). Only `safetensors` is host-only pure stdlib; every other format
launches the NGC container (the same image `train`/`eval` use):

| Format | Lane | What it produces |
|--------|------|-------------------|
| `safetensors` (default) | host, pure stdlib, no GPU | Canonical PEFT adapter layout (`adapter_config.json`, `adapter_model.safetensors`, tokenizer files) — servable by [lobes](https://github.com/agentculture/lobes-cli), runnable by [colleague](https://github.com/agentculture/colleague). |
| `merged-16bit` | container | Base model + adapter merged into a single bf16 checkpoint. |
| `merged-4bit` | container | Base model + adapter merged into a 4-bit checkpoint. |
| `gguf` | container | ggml-quantized GGUF via `--quant` (comma-separated, e.g. `q4_k_m,q8_0`; the full allowlist is in `sloth explain export`). |
| `awq` | container | W4A16 llm-compressor one-shot quantization; takes `--calib`/`--calib-samples` (default calibration: the run's training dataset). |
| `nvfp4` | container | NVFP4 llm-compressor one-shot quantization; same `--calib`/`--calib-samples` as `awq`. |

Container-lane safety rails: **no-clobber** (a non-empty `--output` needs
`--force`), **atomic output** (writes to `<output>.partial`, renamed to
`<output>` only on a clean exit — a killed/OOM run never leaves a
half-written model directory), and a **fail-closed disk check** (estimated
artifact size vs. free space under `--output`, checked before any GPU spend;
`--dry-run` reports both numbers instead of failing). See measured export
sizes in [`benchmarks.md`](benchmarks.md).

```bash
sloth export --adapter runs/qlora-smoke --output runs/qlora-smoke-export                       # safetensors (default), no GPU
sloth export --adapter runs/qlora-smoke --format gguf --quant q4_k_m --output runs/export-gguf  # container
sloth export --adapter runs/qlora-smoke --format awq --output runs/export-awq                   # container
sloth export --adapter runs/qlora-smoke --format nvfp4 --output runs/export-nvfp4                # container
sloth export --adapter runs/qlora-smoke --format gguf --dry-run --json                           # plan only, no GPU
```

Exit codes: `0` success; `1` user-input error (missing/incomplete adapter,
unsupported `--format`, bad `--quant`, non-empty `--output` without `--force`);
`2` environment error (container exits non-zero, insufficient disk space).

### `sloth bench --adapter <dir> --benchmark mmlu`

Where `sloth eval` scores your own local suite, `sloth bench` scores the
**standard public benchmark**: it runs the lm-evaluation-harness (`lm_eval`,
part of the container's benchmark dep layer) inside the NGC container and writes
`<target>/eval/mmlu.json` in the *same* result shape a suite produces — flat
`acc` / `acc_norm` / `exact_match_pct` / `total`, plus `per_subject` and a
`harness` provenance block — so `sloth summarize` and `sloth compare` fold a
benchmark in with no special-casing. `--adapter` benches a LoRA/QLoRA adapter
against the base model recorded in its `training_metadata.json`
(`pretrained=<base>,peft=<dir>`, with `load_in_4bit=True` for a QLoRA run);
`--model` benches a merged/exported directory. If the harness cannot load the
4-bit-base + PEFT-adapter combination, the run exits `2` with a `hint:` naming
the fallback — export merged to 16-bit and bench that directory with `--model`.
`--limit N` shrinks a run to a smoke test, and `--offline` forbids network
access outright (see the warm-cache check in [`dgx-spark.md`](dgx-spark.md)).
For a network-free, MMLU-*style* check there is also
`examples/eval/mmlu-subset.jsonl` — 120 committed, original lettered questions
across 12 subjects, scored through `sloth eval --suite`, which detects an
all-letter answer key and reports `choice_acc_pct` on the letter the model
picked.

```bash
sloth bench --adapter runs/qlora-smoke --benchmark mmlu            # full 5-shot MMLU
sloth bench --adapter runs/qlora-smoke --limit 20 --json           # smoke run
sloth bench --model runs/qlora-smoke-export --offline              # merged export, no network
sloth eval  --adapter runs/qlora-smoke --suite examples/eval/mmlu-subset.jsonl
```

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
target_modules = "preset:lfm2"  # optional — see below
```

### `target_modules`

Optional key under `[hyperparameters]`. It accepts three forms:

- a non-empty list of module names (e.g. `["q_proj", "k_proj"]`);
- a single regex string matched against fully-qualified module paths;
- `"preset:<name>"` — a known preset expanded to its regex at load time.

Unknown presets are rejected by `load_config` at load time, before any GPU spend.
`"preset:lfm2"` resolves to
`model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|conv\.(in|out)_proj|feed_forward\.w[123])`
and adapts all 92 LoRA-able modules of LFM2.5-1.2B (24 attention, 20 short-conv,
48 feed-forward). Unsloth's default adapts only q/k/v on the 6 attention layers,
and a plain name list cannot reach the conv layers — hence the preset.

Ready-to-run configs: [`examples/qlora-smoke.toml`](../examples/qlora-smoke.toml),
[`examples/lora-smoke.toml`](../examples/lora-smoke.toml),
[`examples/lfm2-lora.toml`](../examples/lfm2-lora.toml).

## Deployment targets

Target formats per deployment platform. Coverage is honest — see
[`tested.md`](tested.md): the formats live-tested on hardware so far are QLoRA
bnb-4bit and bf16 LoRA on Qwen3-1.7B on GB10/Spark (2026-06-26), plus LFM2.5-1.2B
`awq`/`nvfp4` **serving** (not training) on Thor (JetPack R38.2.2,
follow-ups #22, 2026-09-15).

| Platform | Target format(s) | Live-tested |
|----------|------------------|-------------|
| Orin | `gguf`, `awq W4A16` | not yet — tracked in [#23](https://github.com/agentculture/unsloth-cli/issues/23) (needs lobes down on the Spark) |
| Thor | `nvfp4`, `awq W4A16` | **2026-09-15**: LFM2.5-1.2B `nvfp4` and `awq` exports load and generate with `vllm/vllm-openai:v0.29.0-aarch64` on JetPack R38.2.2 (follow-ups #22, [`tested.md`](tested.md)) |
| Spark | `nvfp4`; `bf16 + LoRA via lobes hand` | `bf16` LoRA on Qwen3-1.7B (2026-06-26, [`tested.md`](tested.md)) |

Every target format above is produced by `sloth export --format <fmt>` (see
[`sloth export --adapter DIR --output OUT`](#sloth-export---adapter-dir---output-out)
above for the full format table and flags); `merged-16bit` is the `bf16` row's
`--format`. Measured export sizes for these formats are in
[`benchmarks.md`](benchmarks.md).

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

### Allowed paths

Every path `sloth export` receives (`--adapter`, `--output`, `--calib`, a local
`--base`, the dataset recorded in `training_metadata.json`) is canonicalised and
must live under the working directory, your home directory, the Hugging Face
cache or the system temp dir. Anything else exits 1 with a hint. Extend the
allow-list with `SLOTH_ALLOWED_ROOTS=<dir>[:<dir>]`.
