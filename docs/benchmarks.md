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

### `sloth export` — container formats, measured sizes (LFM2.5-1.2B)

> **Status:** validated 2026-09-15, on the same DGX Spark (GB10, Blackwell)
> environment as above. Adapter: `LFM2.5-1.2B-Base` (`preset:lfm2`
> `target_modules`, see [`fine-tuning.md`](fine-tuning.md#target_modules)).
> `gguf` used the prebuilt `llama.cpp` `b10909` (arm64) conversion tool.

| `--format` | Lane | Measured output size |
|------------|------|-----------------------|
| `merged-16bit` (bf16) | container | **2.34 GB** |
| `gguf` (`--quant q4_k_m`) | container | **0.73 GB** |
| `awq` (W4A16) | container | **1.08 GB** |
| `nvfp4` | container | **1.12 GB** |

The `merged-16bit` size (2.34 GB) matches the bf16 base-model size, as
expected — merging LoRA deltas into a bf16 checkpoint does not change its
dtype footprint. The other three formats compress the merged model roughly
2–3x relative to bf16, consistent with 4-bit-class weight quantization. These
are the numbers the [deployment targets](fine-tuning.md#deployment-targets)
table's `gguf`/`awq`/`nvfp4` rows produce in practice — see that table for
which platform each target format maps to.

### `sloth eval` — per-format scores on `examples/eval/` (LFM2.5-1.2B, follow-ups #22)

> **Status:** measured 2026-09-15 on the same DGX Spark, plan
> `lfm2-5-delivery-follow-ups-22` task t10. Suite: the starter directory
> [`examples/eval/`](../examples/eval/) — **46 task-schema items** in three files
> (`agentculture-terms` 16, `cli-contract` 14, `task-format` 16). Adapter:
> `runs/lfm2-lora` (10-step smoke LoRA, `preset:lfm2`); the four export dirs are
> that adapter's `merged-16bit` / `gguf q4_k_m` / `awq` / `nvfp4` outputs from
> the 2026-09-15 batch. All rows `--batch-size 8` unless stated (`max_new_tokens`
> 100, greedy). Wall time includes the container start and dep-layer check.

| Target | exact_match | token-F1 (aggregate) | F1 by file: terms / cli / task | Wall |
|--------|-------------|----------------------|--------------------------------|------|
| adapter (`--adapter`, bf16 base + LoRA) | 0 / 46 | **0.038** | 0.031 / 0.041 / 0.043 | 220 s |
| adapter, `--batch-size 1` | 0 / 46 | 0.060 | 0.036 / 0.068 / 0.078 | 524 s |
| `merged-16bit` | 0 / 46 | 0.036 | 0.033 / 0.031 / 0.044 | 276 s |
| `gguf` Q4_K_M (llama.cpp) | 0 / 46 | 0.092 | 0.037 / 0.114 / 0.127 | 130 s |
| `awq` W4A16 (compressed-tensors) | 0 / 46 | 0.060 | 0.017 / 0.085 / 0.080 | 283 s |
| `nvfp4` (compressed-tensors) | 0 / 46 | 0.040 | 0.032 / 0.034 / 0.052 | 387 s |

What these numbers do and do not say:

- **Exact-match is 0/46 on every target, so exact-match cannot express
  quantization loss for this adapter** — the 10-step smoke LoRA never learned
  the suite's answers (its predictions echo the `Input:` template or wander). The
  spec's honesty condition h15 treats an all-zero row set as a *failed* condition,
  not a pass: the suite is fine, the adapter is the smoke one. A production
  adapter (hundreds of steps on a real corpus) is what makes this table a
  quantization-loss measurement.
- **Token-F1 does move**, and it moves in the wrong direction to be a
  quality signal at this scale: the Q4_K_M and AWQ rows score *higher* than the
  bf16 merge because their noisier generations happen to overlap more
  whitespace tokens with the references. Treat F1 differences of a few
  hundredths as noise here.
- **Batching changes the generations.** The same adapter at batch 8 vs batch 1
  gives F1 0.038 vs 0.060 (both 0/46). Left-padded batched decoding is not
  bit-identical to serial decoding; the batch size is therefore part of every
  row above, and a serial-vs-batched equivalence check on a model with
  non-zero scores is an open plan risk (r8).
- **Batching pays for itself:** 220 s vs 524 s for the same 46 items (2.4x),
  most of the difference being generation time, since container start and
  model load are the same in both.

The four export rows (`merged-16bit` / `gguf` / `awq` / `nvfp4`) each have their
own `runs/lfm2-exports/<fmt>/eval.json`. The two adapter rows (batch 8 and
batch 1) both write to the same `runs/lfm2-lora/eval.json` — it is
**latest-only**, and since batch 1 ran second, that file currently holds the
batch-1 result, not batch 8. **The batch size itself is not recorded inside
`eval.json`** — it lives only in this table and in the invocation that
produced the run; do not infer it from the file. `sloth summarize` and `sloth
compare` render whichever file is present.

## Serving latency + tok/s

Once an adapter is exported (`sloth export`), **serving it is a manual
operator step — unsloth-cli does not start or call lobes.** The hand-off is:
export the adapter to a servable layout here, then a human operator points
[lobes-cli](https://github.com/agentculture/lobes-cli) at it (`lobes deploy`
et al. — see that repo's own docs) and runs its benchmark verb against the
now-serving endpoint.

The exact command, read from `lobes/cli/_commands/benchmark.py`'s
`register()`/`cmd_benchmark()`:

```bash
lobes benchmark --model <served-model-name> --json
```

(`--purpose`, `--input-len`/`--output-len`, and `--runs` override the
workload shape and repetition count; they default to the deployment's
configured `VLLM_PURPOSE` and 2 runs respectively. `--all-lobes` and
`--profile` are separate, unrelated modes of the same verb — not part of this
hand-off.)

`cmd_benchmark`'s single-model path calls `lobes.assess.run_benchmark`, whose
return shape is, verbatim (`lobes/assess.py`, `run_benchmark`, lines ~569-598):

```text
{model, endpoint, max_model_len, purpose, input_len, output_len,
 decode_rates, prefill}
```

`--json` mode merges in one extra top-level key, `host` (`{image,
gpu_memory}`), read from the deployment's compose config rather than from
`run_benchmark` itself. `decode_rates` is the list of per-run decode
throughput samples (tokens/sec); `prefill` is the prefill-latency measurement
dict for the configured `input_len`.

Measured numbers for this repo's exported adapters — batch 1 and batch 8,
per export format — are **not yet collected**; that is the live-run task's
job, not this doc's:

| Export format | `--purpose` / batch | decode tok/s | prefill latency |
|----------------|---------------------|--------------|------------------|
| `merged-16bit` | batch 1 | (to be measured by the live run) | (to be measured by the live run) |
| `merged-16bit` | batch 8 | (to be measured by the live run) | (to be measured by the live run) |
| `gguf` (`q4_k_m`) | batch 1 | (to be measured by the live run) | (to be measured by the live run) |
| `gguf` (`q4_k_m`) | batch 8 | (to be measured by the live run) | (to be measured by the live run) |
| `awq` | batch 1 | (to be measured by the live run) | (to be measured by the live run) |
| `awq` | batch 8 | (to be measured by the live run) | (to be measured by the live run) |
| `nvfp4` | batch 1 | (to be measured by the live run) | (to be measured by the live run) |
| `nvfp4` | batch 8 | (to be measured by the live run) | (to be measured by the live run) |

This table is serving cost, not model quality — it belongs beside the
per-format `sloth eval` scores above, not in place of them: a fast, cheap
format that regresses accuracy is still a regression.

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
- Multi-hundred-step convergence and eval accuracy on a real corpus — the
  per-format table above is measured on a 10-step smoke adapter and scores 0/46
  exact-match everywhere; it becomes a quantization-loss number only with a
  trained adapter.
