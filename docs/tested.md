# Tested configurations — what was actually validated

A precise, honest record of **exactly** what has been run on hardware, so the
coverage is unambiguous: e.g. fine-tuning was validated on **`unsloth/Qwen3-1.7B`**
but **not** on Qwen 4B/9B. Treat this as a living tracker — add a row when you
validate a new model/config; do not claim coverage a real run didn't produce.

Companion pages: [`benchmarks.md`](benchmarks.md) (the numbers),
[`dgx-spark.md`](dgx-spark.md) (how/why), [`fine-tuning.md`](fine-tuning.md) (the
feature reference).

## Common environment (every run below)

| Component | Value |
|-----------|-------|
| Date | 2026-06-26 / 2026-06-27 |
| Hardware | NVIDIA **DGX Spark**, **GB10** (Blackwell), aarch64, 121 GB unified memory |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| Container | `nvcr.io/nvidia/pytorch:25.11-py3` (NGC) |
| torch | `2.10.0a0+…nv25.11` (CUDA 13.0) |
| unsloth / unsloth_zoo | 2026.6.9 / 2026.6.7 |
| transformers / peft / trl | 4.57.1 / 0.18.0 / 0.24.0 |
| torchao / bitsandbytes / datasets | 0.14.0+git / 0.49.2 / 4.3.0 |
| **Model** | **`unsloth/Qwen3-1.7B`** (QLoRA auto-maps to `unsloth/qwen3-1.7b-unsloth-bnb-4bit`) — the *only* model trained |
| Train hyperparameters | `batch_size=1`, `grad_accum=4`, `max_seq_len=1024`, `lora_r=8`, `lora_alpha=16`, `max_steps=10`, `seed=3407` |
| Train dataset | `examples/chat-smoke.jsonl` (10 lines, **chat** schema) |
| Eval suite | `examples/eval-suite.jsonl` (4 items, **task** schema) |

## ✅ Tested — passed

"Shipped host path" = the real `uv run sloth <verb>` (full `container.py`
orchestration: preflight → in-container `--system-site-packages` venv install →
run), the way an end user invokes it.

| Verb | Method / mode | Model | Invocation | Result |
|------|---------------|-------|------------|--------|
| `train --dry-run` | qlora | Qwen3-1.7B | host, GPU-free | ✅ plan + docker command rendered |
| `train` | **QLoRA** (4-bit) | Qwen3-1.7B | shipped host path | ✅ `train_runtime` 9.31 s, loss 9.46→4.25, adapter + metadata, **host-owned** |
| `train` | **LoRA** (16-bit) | Qwen3-1.7B | shipped host path | ✅ `train_runtime` 10.65 s, loss 9.93→4.53, **host-owned** |
| `eval` | QLoRA adapter | Qwen3-1.7B | shipped host path | ✅ 4 items scored, ran end-to-end (exact_match 0/4 — smoke, not an accuracy claim) |
| `export` | QLoRA adapter | — | shipped host path (pure stdlib) | ✅ standard PEFT/safetensors layout |

Each `train`/`eval` was *also* exercised via the in-container trainer code with a
pre-baked dep image during bring-up; the shipped-host-path rows above are the
authoritative ones.

## ✅ Tested — passed (2026-09-15, LFM2.5 + quantized export)

Same box and container as above. Differences from the 2026-06 matrix: the dep
layer now pins `datasets==4.8.5`, `llmcompressor==0.11.0`,
`compressed-tensors==0.16.0` (deviation d3); `unsloth` / `unsloth_zoo` float
(measured 2026.9.4 / 2026.9.3 in the live layer). Model:
**`LiquidAI/LFM2.5-1.2B-Base`**, config `examples/lfm2-lora.toml`
(`method = "lora"`, `lora_r = 16`, `target_modules = "preset:lfm2"`,
`max_steps = 10`, `examples/chat-smoke.jsonl`).

| Verb | Method / mode | Model | Invocation | Result |
|------|---------------|-------|------------|--------|
| `train --dry-run` | lora, **no** `target_modules`, `lora_r = 64` | LFM2.5-1.2B-Base | host, GPU-free | ✅ exit 0; stderr carries exactly two `note:` lines (set `preset:lfm2`; lobes' hand lane caps rank at 32); stdout JSON unchanged |
| `train` | **LoRA** (16-bit), `preset:lfm2` | LFM2.5-1.2B-Base | shipped host path: `uv run sloth train --config examples/lfm2-lora.toml --json` | ✅ 2 m 19 s wall incl. dep-layer install; `adapter_config.json` `target_modules` = the lfm2 regex (attention + short-conv + feed-forward), `r = 16`; `training_metadata.json` carries `dataset.path` and the resolved `target_modules` |
| serve (vLLM) | `--enable-lora`, `max_lora_rank 16` | LFM2.5-1.2B-Base + the adapter above | `vllm/vllm-openai:nightly` (v0.26.1rc1), `LLM.generate(lora_request=…)` | ✅ adapter loads with no unsupported-module warning; vLLM JIT-compiled `_lora_expand_kernel` / `_lora_shrink_kernel` during the adapter call |
| `eval` | LoRA adapter | LFM2.5-1.2B-Base | shipped host path: `uv run sloth eval --adapter runs/lfm2-lora --suite examples/eval-suite.jsonl --json` | ✅ 4 items scored end-to-end (exact_match 0/4 — smoke adapter, not an accuracy claim) |

| `export --format merged-16bit` | LoRA adapter → bf16 merged | LFM2.5-1.2B-Base | shipped host path: `uv run sloth export --adapter runs/lfm2-lora --format merged-16bit --output runs/lfm2-exports/merged16 --json` | ✅ `model.safetensors` 2.34 GB; `export.json` + `runs/lfm2-lora/exports.json` record unsloth 2026.9.4, transformers 4.57.1, peft 0.18.0, llmcompressor 0.11.0, compressed-tensors 0.16.0, llama.cpp b10909 |
| `export --format gguf --quant q4_k_m` | Q4_K_M via prebuilt llama.cpp b10909 (arm64) | LFM2.5-1.2B-Base | shipped host path (first run) | ✅ `LFM2.5-1.2B-Base.Q4_K_M.gguf` 0.73 GB in 1 m 9 s incl. the prebuilt fetch into `~/.cache/unsloth-cli/home/.unsloth/llama.cpp`; no F16 intermediate or merged weights left behind |
| `export --format gguf --quant q4_k_m` | same, second run | LFM2.5-1.2B-Base | shipped host path (cache reuse) | ✅ 48 s, no llama.cpp download (cache mount reused) |
| `export --format awq` | W4A16_ASYM via llm-compressor 0.11.0, `pipeline=basic`, LFM2 per-layer mappings (no v→out pair) | LFM2.5-1.2B-Base | shipped host path; calibration = the run's training dataset (10 rows, resolved from `training_metadata.json`) | ✅ compressed-tensors `pack-quantized`, `model.safetensors` 1.08 GB |
| `export --format nvfp4` | NVFP4 via llm-compressor 0.11.0 (torch-2.11 API shim active) | LFM2.5-1.2B-Base | shipped host path; same calibration | ✅ compressed-tensors `nvfp4-pack-quantized`, `model.safetensors` 1.12 GB, 59 s |
| `export --format merged-4bit` | bnb 4-bit base + merge (`merged_4bit_forced`) | LFM2.5-1.2B-Base | shipped host path | ✅ `model.safetensors` 0.80 GB |
| `eval --model` | quantization-loss check on the same suite | LFM2.5-1.2B-Base merged-16bit / awq / nvfp4 | shipped host path: `uv run sloth eval --model runs/lfm2-exports/<dir> --suite examples/eval-suite.jsonl --json` | ✅ three runs (37 s, 40 s, 64 s): exact_match 0/4 on each, identical to the `--adapter` score (0/4) — the 10-step smoke adapter has nothing to lose; result JSON carries `quant_method`/`quant_format` (`compressed-tensors` `pack-quantized` / `nvfp4-pack-quantized`, null for bf16) |
| serve (vLLM) | quantized dirs | awq + nvfp4 outputs above | `vllm/vllm-openai:nightly` on GB10 | ✅ both load with `quantization=compressed-tensors` auto-detected and generate coherent text (`gpu_memory_utilization 0.12`, `max_model_len 512`) |

| `/finetune` skill `run` | train → eval → export `--export-format gguf --quant q4_k_m --json` | LFM2.5-1.2B-Base | `bash .claude/skills/finetune/scripts/finetune.sh run --config <lfm2 toml> --suite examples/eval-suite.jsonl --export-format gguf --quant q4_k_m --json` (with this checkout's `sloth` first on PATH) | ✅ exit 0 in 5 m 28 s; adapter + `runs/lfm2-skill-gguf/LFM2.5-1.2B-Base.Q4_K_M.gguf`. **Known gap:** stdout carries the 3 JSON results interleaved with the container's banner and trainer progress lines (pre-existing stream-through, plan risk r10) — not stdout-only yet |

Not measured in this batch: QLoRA-trained adapters through the export formats, Qwen3
through awq/nvfp4/gguf, any Jetson-side load, any accuracy metric beyond the 4-item
smoke suite (plan risks r3, r4, r8).

Before-state evidence (deviation record, main at `39a3f93`): the same TOML
dry-run on `main` ignored the `target_modules` key silently — the plan's
`hyperparameters` had no such key.

## ❌ Not tested (explicit gaps)

Do not assume these work just because the 1.7B path does. The code path is often
identical, but they have **not** been run on hardware.

### Models

- **Qwen3 4B / 9B** — the repo's production targets — **not tested.**
  `Qwen/Qwen3.5-4B` is cached on this box but was **not** trained (insufficient
  free unified memory: ~85 GB was held by running vLLM servers, ~5 GB free).
- Larger local adapters (Qwen 3.6 27B dense, Qwen coder variants) — not tested.
- Any model other than `unsloth/Qwen3-1.7B` (2026-06) and `LiquidAI/LFM2.5-1.2B-Base`
  (2026-09-15) — not tested.

### Configs & scale

- Only **`max_steps=10`, `max_seq_len=1024`, `batch_size=1`, `lora_r=8`** on a
  **10-line** dataset. No production-length training, no larger sequence/batch, no
  convergence or eval-accuracy validation (the 0% exact-match is expected for a
  10-step adapter and is **not** an accuracy result).

### Code paths

- **Task-schema *training*** — only the **chat** schema was trained on hardware.
  The task-schema training render path (`Task:/Input:/Output:`) is unit-tested
  only; task schema was used only as the *eval* suite.
- **Scope-guard refusal of a real large-dense full-fine-tune** on hardware —
  unit-tested only (no real out-of-scope model was loaded).
- **Checkpoint / resume**, multi-GPU, and quantization variants beyond bnb-4bit /
  16-bit — not implemented / not tested.

### Platform

- Only **GB10 (Blackwell, aarch64)** + **NGC 25.11 / torch 2.10**. No other GPU,
  arch, driver, or container image was tested. `--gpus all` worked via **CDI**
  (Docker default runtime `runc`); a host requiring the `nvidia` runtime was not
  tested.

## How to extend this matrix

When you validate a new model or config:

1. Run it via the shipped host path, e.g.
   `uv run sloth train --config <your.toml>` (then `eval` / `export`).
2. Record `train_runtime`, final loss, adapter size, and host memory headroom.
3. Add a row to **✅ Tested** and remove the corresponding **❌ Not tested** gap.

To attempt the production target, set `model = "Qwen/Qwen3.5-4B"` and run on a box
with free unified memory (free it with
`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` and by stopping other GPU
processes — see [`dgx-spark.md`](dgx-spark.md)).
