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

## Full benchmark suite — fixture adapter (2026-09-17, plan `full-benchmark-suite`)

### `sloth train` — the fixture adapter · [`examples/demo-lora.toml`](../examples/demo-lora.toml)

The adapter every table below is measured on (plan `full-benchmark-suite`, task
t16). LFM2.5-1.2B-Base, LoRA r=16 on `preset:lfm2`, 300 steps on
[`examples/demo-corpus.jsonl`](../examples/demo-corpus.jsonl) (591 chat rows,
generated by `examples/generate_suites.py`) with a 10% seeded holdout
(`[eval] holdout_fraction = 0.1, seed = 3407, eval_steps = 50`). Same box as
above; lobes' vLLM resident (~94 GB of the 121 GB UMA), page cache reclaimed
before the run (the first attempt OOMed at backend init — plan risk r5).

| Metric | Value |
|--------|-------|
| Command | `uv run sloth train --config examples/demo-lora.toml --json` (2026-09-17) |
| Rows | 532 train / 59 holdout (chat rows rendered to task rows on both sides — plan risk r10) |
| Steps · epochs | 300 · 2.26 |
| `train_runtime` | 91.3 s (13.1 samples/s · 3.29 steps/s); 3 min 43 s wall incl. container + layer install |
| Loss | mean train 0.956; last logged step 0.18; **holdout `eval_loss` 2.11** (recorded at steps 50…300 in `loss_history`) |
| Adapter | `runs/demo-lora/` — `adapter_model.safetensors`, `training_metadata.json` with `loss_history` (300 entries), `final_train_loss`, `final_eval_loss`, `holdout` |

### `sloth eval` — every suite family on the fixture adapter (batch 8)

`uv run sloth eval --adapter runs/demo-lora --suite <each file below> --batch-size 8
--perplexity --tool-call-family lfm2 --train-dataset examples/demo-corpus.train.jsonl --json`
(2026-09-17, one invocation, nine named suites, 484 rows, **31 min 13 s** wall;
base precision bf16 LoRA; greedy, `max_new_tokens = 100`). `latency_ms` is batch
wall time / rows; `tokens_per_s` is generated tokens over generate wall time.

| Suite (family) | Rows | Exact % | Token-F1 | Compliance % | Choice acc % | Perplexity | Median latency ms | tok/s |
|---|---|---|---|---|---|---|---|---|
| `agentculture-terms` (target-task) | 16 | 0.00 | 0.0521 | — | — | 16.21 | 2785 | 35.9 |
| `cli-contract` (target-task) | 14 | 0.00 | 0.1255 | — | — | 11.35 | 2679 | 32.3 |
| `task-format` (target-task) | 16 | 0.00 | 0.1198 | — | — | 20.23 | 3047 | 32.8 |
| `regression` (regression) | 116 | 6.03 | 0.3743 | — | — | 32.40 | 2538 | 35.9 |
| `instruction-following` (instruction following) | 56 | 0.00 | 0.0687 | 19.64 | — | 160.47 | 2844 | 35.4 |
| `structured-output` (structured output) | 55 | 3.64 | 0.0364 | 12.73 | — | 251.85 | 2837 | 35.2 |
| `tool-call` (tool call) | 32 | 56.25 | 0.5625 | 0.00 | — | 263.42 | 1905 | 28.3 |
| `mmlu-subset` (MMLU-style subset (letter choice)) | 120 | 8.33 | 0.1085 | — | 23.33 | 51.12 | 3507 | 28.7 |
| `demo-corpus-holdout` (holdout) | 59 | 1.69 | 0.1201 | — | — | 1.51 | 2507 | 36.7 |

**Reading this table honestly — batch 8 is invalid on LFM2.5.** Every
accuracy figure above is an artefact of the batched eval path, not of the
adapter: at batch 8 (left padding) the generations are degenerate — runs of `:`
characters, or `Input: …` echoed until the 100-token budget — on **every**
suite, whereas batch 1 on the same adapter, suites and prompts (next table)
scores 5/14 on `cli-contract` and 9/16 on `task-format`. LFM2.5's short-conv
layers do not tolerate left padding the way an attention-only model does, so the
"batched is 2.4× faster but not bit-identical" caveat from the #22 run (risk r8)
is, for LFM2, "batched is wrong" (plan risk r13). The table is kept because it is
the measurement that exposed the defect, and because it shows what the new
families report on garbage output: instruction-following compliance 19.6 % and
structured-output compliance 12.7 % (constraint / JSON-schema checks that exact
match and F1 alone could never report), tool-call compliance 0 % (the 56 %
"exact" figure on that suite is empty-vs-empty — tool-call rows carry no
reference text, so `compliance_pct` is the metric there), letter accuracy at
chance (23.3 % vs 25 %), and 2.5–3.5 s per row because the echoing generations
never stop early. **All accuracy, perplexity and latency claims for LFM2 below
are batch 1.**

### `sloth eval` — every suite family on the fixture adapter (**batch 1**, the valid LFM2 numbers)

Same command as above with `--batch-size 1` (2026-09-17, one invocation, nine
named suites, 484 rows, **27 min 37 s** wall; bf16 LoRA; greedy,
`max_new_tokens = 100`). Perplexity is a labelled forward pass and therefore
identical to the batch-8 run; every generation-based column differs.

| Suite (family) | Rows | Exact % | Token-F1 | Compliance % | Choice acc % | Perplexity | Median latency ms | tok/s |
|---|---|---|---|---|---|---|---|---|
| `agentculture-terms` (target-task) | 16 | 0.00 | 0.2641 | — | — | 16.21 | 2377 | 6.7 |
| `cli-contract` (target-task) | 14 | 35.71 | 0.6357 | — | — | 11.35 | 2151 | 6.3 |
| `task-format` (target-task) | 16 | 56.25 | 0.6277 | — | — | 20.23 | 1464 | 7.2 |
| `regression` (regression) | 116 | 11.21 | 0.5852 | — | — | 32.40 | 1260 | 9.3 |
| `instruction-following` (instruction following) | 56 | 0.00 | 0.1126 | 46.43 | — | 160.47 | 1891 | 7.6 |
| `structured-output` (structured output) | 55 | 0.00 | 0.0000 | 76.36 | — | 251.85 | 2730 | 9.2 |
| `tool-call` (tool call) | 32 | 0.00 | 0.0000 | 0.00 | — | 263.42 | 1581 | 5.9 |
| `mmlu-subset` (MMLU-style subset (letter choice)) | 120 | 60.00 | 0.6012 | — | 61.67 | 51.12 | 984 | 4.7 |
| `demo-corpus-holdout` (holdout) | 59 | 10.17 | 0.3595 | — | — | 1.51 | 1880 | 7.0 |

**What this measures.** Target-task exact match is now non-zero on two of the
three suites (`cli-contract` 5/14, `task-format` 9/16) and the holdout split
scores 6/59 — the adapter *did* learn the trained behaviour, though
`agentculture-terms` stays at 0/16 exact with token-F1 0.26 because the demo
corpus teaches full-sentence answers ("The distribution name is unsloth-cli.")
where the suite expects the bare term ("unsloth-cli"): F1 is the fairer
target-task metric for this corpus (plan risk r3). Instruction-following
compliance 46.4 % and structured-output compliance 76.4 % are the new
families' verdicts on format/constraint behaviour; tool-call compliance is 0 %
because a *Base* checkpoint never emits LFM2's `<|tool_call_start|>` wrapper
(the suite is meaningful on an Instruct base). The MMLU-style subset scores
61.7 % letter accuracy. Per-row latency at batch 1 is 1.0–2.7 s (5–9 tok/s in
HF eager generation on the Spark with lobes' vLLM resident — a serving figure
belongs to lobes, see below). Whether any of this is *better than the base
model* is the `sloth compare --base` table.

### `sloth bench` — MMLU through lm-evaluation-harness (fixture adapter)

`uv run sloth bench --adapter runs/demo-lora --benchmark mmlu --limit 5 --json`
(2026-09-17, **5 min 22 s** wall incl. container + first-run download of the 57
MMLU subject splits; in-container command as recorded on stderr:
`lm_eval --model hf --model_args pretrained=LiquidAI/LFM2.5-1.2B-Base,peft=<adapter> --tasks mmlu --num_fewshot 5 --limit 5 --output_path <tmp> --log_samples`).

| Metric | Value |
|--------|-------|
| Harness | lm_eval 0.4.13, task `mmlu`, 5-shot, `--limit 5` (= 5 documents **per subject**, 57 subjects) |
| Documents | 285 |
| `acc` | **0.5789** (`acc_norm` is not reported by the MMLU task and is stored as `null`) |
| Per subject | 61 entries in `eval/mmlu.json`; highest `astronomy` 1.0, `high_school_psychology` 1.0, `medical_genetics` 1.0; lowest `elementary_mathematics` 0.0, `moral_scenarios` 0.0, `college_mathematics` 0.2 |
| Adapter load | `peft=<adapter>` on the bf16 base **worked directly** — frame park v4 is resolved for LoRA adapters (a QLoRA / `load_in_4bit=True` adapter is still unmeasured; the documented fallback is to bench the merged-16bit export) |
| Warm-cache check (`--offline`) | second run `uv run sloth bench … --offline --json`: **2 min 23 s** (vs 5 min 22 s cold), zero `Generating … split` lines, lm_eval logs `Using the latest cached version of the dataset … (offline mode is enabled)` for every subject, identical `acc` 0.5789 — the HF-cache mount makes the harness reproducible without network after the first fill (the only downloads on the second run are the uv dep-layer wheels, which are not HF traffic) |
| Result file | `runs/demo-lora/eval/mmlu.json` (schema_version 2, `suite: "mmlu"`, `exact_match_pct = acc × 100` so `summarize` folds it) |

A 5-per-subject smoke is a **noisy** estimate (285 documents); it is a
sanity number for the fixture adapter, not a claim comparable to published
LFM2.5-1.2B MMLU figures, which use the full 14 042-document test set.

### `sloth export` + `sloth eval --model` — quantization loss on a trained adapter (batch 1)

The measurement issue #28 (item 1) asked for: the per-format table re-scored on
an adapter that actually learned something. Three target suites
(`agentculture-terms` / `cli-contract` / `task-format`, 46 rows), batch 1,
greedy, `max_new_tokens = 100`, 2026-09-17, one export + one eval per row; the
adapter row is the same nine-suite batch-1 run as above.

| Target | Weights | Export wall | Eval wall | Exact match (per suite) | Token-F1 (per suite) | Median latency ms (per suite) | Commands |
|---|---|---|---|---|---|---|---|
| adapter (bf16 base + LoRA) | 34 MB adapter | — | part of the 27 min nine-suite run | **14/46** (0/16 / 5/14 / 9/16) | 0.264 / 0.636 / 0.628 | 2377 / 2151 / 1464 | `sloth eval --adapter runs/demo-lora … --batch-size 1` |
| merged-16bit | 2.34 GB | 42 s | 192 s | **14/46** (0/16 / 5/14 / 9/16) | 0.274 / 0.622 / 0.628 | 1815 / 1770 / 1429 | `sloth export --adapter runs/demo-lora --format merged-16bit … ; sloth eval --model <dir> … --batch-size 1` |
| gguf Q4_K_M | 0.73 GB | 55 s | 71 s | **0/46** (0/16 / 0/14 / 0/16) | 0.238 / 0.484 / 0.363 | 1513 / 1466 / 1420 | `sloth export --adapter runs/demo-lora --format gguf --quant q4_k_m … ; sloth eval --model <dir> … --batch-size 1` |
| awq (W4A16, compressed-tensors) | 1.08 GB | 81 s | 188 s | **12/46** (1/16 / 1/14 / 10/16) | 0.375 / 0.374 / 0.723 | 1704 / 1443 / 1432 | `sloth export --adapter runs/demo-lora --format awq --calib examples/chat-smoke.jsonl … ; sloth eval --model <dir> … --batch-size 1` |
| nvfp4 (compressed-tensors) | 1.12 GB | 64 s | 217 s | **9/46** (0/16 / 2/14 / 7/16) | 0.249 / 0.368 / 0.564 | 2463 / 2279 / 1596 | `sloth export --adapter runs/demo-lora --format nvfp4 --calib examples/chat-smoke.jsonl … ; sloth eval --model <dir> … --batch-size 1` |

**Reading it.** merged-16bit reproduces the adapter (14/46, F1 within 0.02) —
the merge is lossless as far as these suites can tell. AWQ W4A16 keeps 12/46 and
even gains on `task-format` (10/16, F1 0.72); NVFP4 keeps 9/46. **GGUF Q4_K_M
drops to 0/46** with F1 0.24 / 0.48 / 0.36 — a far larger loss than the other
4-bit formats, which points at the llama.cpp scoring path (prompt rendering,
stop conditions, the whitespace token approximation) as much as at Q4_K_M
itself (plan risk r14, follow-up). The two `0/46 on every format` caveats from
the #22 run are closed by this table; the GGUF row is the one that now needs a
diagnosis rather than a caveat.

### `sloth compare --base` — base model vs the fixture adapter (batch 1)

`uv run sloth compare --base LiquidAI/LFM2.5-1.2B-Base runs/demo-lora --config examples/demo-lora.toml --json`
(2026-09-17, **1 h 12 min** wall: the adapter re-scored over the nine suites,
then the untouched bf16 base over the same files, two sequential container
runs, batch 1 reused from the adapter's result files). **Provenance caveat:**
the base-side numbers below are taken from the second container run's stdout
JSON as captured on stderr, because the run could not write
`eval-base/<suite>.json` — docker had created that bind-mounted directory as
root (plan risk r15, fixed in the same PR: compare now pre-creates it and
refuses to pass with an empty base side). A verification re-run with the fix
is recorded in the delivery doc if it completed before the PR.

| Suite | Rows | Exact % base → adapter (Δ pp) | Token-F1 base → adapter (Δ) | Compliance % base → adapter | Choice acc % base → adapter (Δ pp) | Median latency ms base → adapter |
|---|---|---|---|---|---|---|
| `agentculture-terms` | 16 | 0.00 → 0.00 (+0.00) | 0.036 → 0.264 (+0.228) | — → — | — → — (—) | 6446 → 2377 |
| `cli-contract` | 14 | 0.00 → 35.71 (+35.71) | 0.068 → 0.636 (+0.568) | — → — | — → — (—) | 6211 → 2151 |
| `task-format` | 16 | 0.00 → 56.25 (+56.25) | 0.099 → 0.628 (+0.529) | — → — | — → — (—) | 6548 → 1464 |
| `demo-corpus-holdout` | 59 | 0.00 → 10.17 (+10.17) | 0.073 → 0.359 (+0.286) | — → — | — → — (—) | 6724 → 1880 |
| `regression` | 116 | 0.00 → 11.21 (+11.21) | 0.055 → 0.585 (+0.531) | — → — | — → — (—) | 6209 → 1260 |
| `instruction-following` | 56 | 0.00 → 0.00 (+0.00) | 0.080 → 0.113 (+0.033) | 23.21 → 46.43 | — → — (—) | 6346 → 1891 |
| `structured-output` | 55 | 0.00 → 0.00 (+0.00) | 0.000 → 0.000 (+0.000) | 20.00 → 76.36 | — → — (—) | 6540 → 2730 |
| `tool-call` | 32 | 0.00 → 0.00 (+0.00) | 0.000 → 0.000 (+0.000) | 0.00 → 0.00 | — → — (—) | 3416 → 1581 |
| `mmlu-subset` | 120 | 0.00 → 60.00 (+60.00) | 0.028 → 0.601 (+0.573) | — → — | 75.83 → 61.67 (-14.16) | 7141 → 984 |

**Verdict against the `[eval.thresholds]` baseline (2 pp regression drop,
95 % compliance, latency ratio 1.10, 100 rows minimum).** Computed by hand from
the rows above because the captured compare run had no base side to gate:
the regression-tagged suite *improves* (0 → 11.2 pp exact, F1 +0.53 — a Base
checkpoint does not follow the `Task:/Input:/Output:` shape at all, the adapter
does), latency improves (the adapter stops early; the base echoes to the
100-token budget, 6–7 s per row), **but the gate still fails**: compliance is
46 % / 76 % / 0 % against a 95 % floor, and seven of the nine suites have fewer
than 100 rows. Recorded as **failing**, not tuned. The success signal's
"target-task exact match > 0 with a positive delta" *is* met (0 → 14/46 over
the three target suites; `agentculture-terms` alone stays 0 → 0 on exact,
+0.23 F1). And the table shows the degradation exact match and F1 alone could
not report: the adapter **loses 14 pp of general knowledge** on the MMLU-style
subset (75.8 % → 61.7 % letter accuracy) — a real regression that the current
gate does not fail, because `regression_drop_pp` is applied to
`exact_match_pct` on regression-tagged suites only (plan risk r16, follow-up).
Base perplexity is not in the table: `compare --base` does not forward
`--perplexity` yet.

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

- Production-scale models (Qwen3 4B / 9B) on a free box — needs headroom the
  fixture runs above did not have (lobes' vLLM resident). The orchestration and
  trainer code are identical; only the model size and step count change.
- Throughput at larger `max_seq_len` / `batch_size` — and, for LFM2, *any*
  batched generation until the eval loop gains a per-family padding guard (r13).
- A QLoRA (`load_in_4bit = true`) adapter through `sloth bench` — the lm_eval
  `peft=` load was proven on a bf16 LoRA only.
- Tool-call compliance on an *Instruct* base (the fixture is a Base checkpoint,
  which never emits LFM2's tool-call wrapper — the 0 % row is expected, not a
  measurement of the adapter).
- The GGUF Q4_K_M scoring path (r14) and the serving tok/s table (lobes, manual
  hand-off — see above).
