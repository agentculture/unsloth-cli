# Benchmarks & validation evidence

This page records **real, on-hardware** LoRA/QLoRA runs of the `sloth train` /
`sloth eval` / `sloth export` verbs on an NVIDIA **DGX Spark (GB10, Blackwell)**.
It is evidence that the fine-tuning path actually works end-to-end — not a
`--dry-run`, and not a unit test with a fake backend.

> **Status:** validated 2026-06-26. Both LoRA and QLoRA complete real training
> steps with a decreasing loss, write a loadable PEFT adapter + run metadata, and
> the adapter then evaluates and exports.

## Test environment

| Component | Value |
|-----------|-------|
| Hardware | NVIDIA **DGX Spark**, **GB10** (Blackwell), aarch64, 121 GB unified memory |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| Container | `nvcr.io/nvidia/pytorch:25.11-py3` (NGC) |
| torch | `2.10.0a0+…nv25.11` (Blackwell-native, CUDA 13.0) |
| unsloth / unsloth_zoo | 2026.6.9 / 2026.6.7 |
| transformers / peft / trl | **4.57.1 / 0.18.0 / 0.24.0** (the pinned, validated set) |
| torchao / bitsandbytes | 0.14.0+git (container) / 0.49.2 |
| Model | `unsloth/Qwen3-1.7B` (4-bit QLoRA auto-maps to `unsloth/qwen3-1.7b-unsloth-bnb-4bit`) |

**Why a 1.7B model?** The box was busy serving other models (~85 GB of unified
memory held by vLLM servers, ~5 GB free), so a small model was chosen to validate
the *pipeline* without disturbing those workloads. The repo's production target is
Qwen3 **4B / 9B** adapters on a box with free memory; the code path is identical.
See [`dgx-spark.md`](dgx-spark.md) for the memory story.

## Results

Both runs: `batch_size=1`, `grad_accum=4`, `max_seq_len=1024`, `lora_r=8`,
`lora_alpha=16`, `max_steps=10`, `seed=3407`, on the 10-line
[`examples/chat-smoke.jsonl`](../examples/chat-smoke.jsonl).

### `sloth train` — QLoRA (4-bit) · [`examples/qlora-smoke.toml`](../examples/qlora-smoke.toml)

| Metric | Value |
|--------|-------|
| Steps | 10 |
| `train_runtime` | **12.04 s** |
| Throughput | 3.32 samples/s · 0.83 steps/s |
| `train_loss` (mean) | 6.31 |
| Loss curve | 9.46 → 8.71 → 6.73 → 5.58 → 4.54 → **4.25** (decreasing) |
| Peak GPU alloc (model load) | ~1.47 GB |
| Adapter size | `adapter_model.safetensors` = **34.9 MB** |
| Full pipeline wall (warm cache) | ~45 s (container start + dep import + train + save) |

### `sloth train` — LoRA (16-bit) · [`examples/lora-smoke.toml`](../examples/lora-smoke.toml)

| Metric | Value |
|--------|-------|
| Steps | 10 |
| `train_runtime` | **12.09 s** |
| Throughput | 3.31 samples/s · 0.83 steps/s |
| `train_loss` (mean) | 6.63 |
| Loss curve | 9.93 → 9.07 → 7.16 → 5.85 → 4.35 → **4.53** (decreasing) |
| Adapter size | 34.9 MB |
| First-run wall | ~151 s (includes a one-time ~3.4 GB 16-bit base-model download) |

### `sloth eval` — QLoRA adapter · [`examples/eval-suite.jsonl`](../examples/eval-suite.jsonl)

| Metric | Value |
|--------|-------|
| Suite size | 4 task-schema items |
| `exact_match` | 0 / 4 (**0.0 %**) |
| Wall | ~28 s |

A 10-step smoke adapter is not expected to score above zero on exact-match — the
point is that the **eval pipeline runs**: it loads the base model + adapter,
generates a prediction per item, and scores it. Predictions are coherent
continuations (e.g. *"The results go to the data ware…"*), confirming the adapter
loaded and generated.

### `sloth export` — QLoRA adapter

Runs natively on the host (pure stdlib, no GPU/container, ~instant). Produced the
standard PEFT/safetensors layout: `adapter_config.json`,
`adapter_model.safetensors`, and the tokenizer files — ready for
[lobes](https://github.com/agentculture/lobes-cli) to serve or
[colleague](https://github.com/agentculture/colleague) to run as a backend.

## Run metadata (written next to the adapter)

`sloth train` writes `training_metadata.json` alongside the adapter, e.g. for the
QLoRA run:

```json
{
  "model": "unsloth/Qwen3-1.7B",
  "method": "qlora",
  "dataset": { "sha256": "3ee8e7dce344…", "line_count": 10 },
  "hyperparameters": { "lora_r": 8, "lora_alpha": 16, "learning_rate": 0.0002,
                       "max_seq_len": 1024, "batch_size": 1, "grad_accum": 4,
                       "max_steps": 10, "seed": 3407, "load_in_4bit": true },
  "timestamp": "2026-06-26T20:30:14Z"
}
```

The same config + dataset reproduces the same run; the dataset SHA-256 pins the
exact training data.

## Reproducing these

From a repo checkout on the DGX Spark (Docker + NVIDIA Container Toolkit
installed — see [`dgx-spark.md`](dgx-spark.md)):

```bash
uv run sloth train --config examples/qlora-smoke.toml      # QLoRA (4-bit)
uv run sloth train --config examples/lora-smoke.toml       # LoRA  (16-bit)
uv run sloth eval  --adapter runs/qlora-smoke --suite examples/eval-suite.jsonl
uv run sloth export --adapter runs/qlora-smoke --output runs/qlora-smoke-export
```

To scale up to the production target, point `model` at `Qwen/Qwen3.5-4B` (or a
Qwen3 9B), raise `max_steps`, and run on a box with free unified memory.

## What is *not* yet benchmarked

- Production-scale models (Qwen3 4B / 9B) on a free box — needs headroom this run
  didn't have. The orchestration and trainer code are identical; only the model
  size and step count change.
- Throughput at larger `max_seq_len` / `batch_size`.
- Multi-hundred-step convergence and eval accuracy on a real corpus.
