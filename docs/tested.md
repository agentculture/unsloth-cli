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

## ❌ Not tested (explicit gaps)

Do not assume these work just because the 1.7B path does. The code path is often
identical, but they have **not** been run on hardware.

### Models

- **Qwen3 4B / 9B** — the repo's production targets — **not tested.**
  `Qwen/Qwen3.5-4B` is cached on this box but was **not** trained (insufficient
  free unified memory: ~85 GB was held by running vLLM servers, ~5 GB free).
- Larger local adapters (Qwen 3.6 27B dense, Qwen coder variants) — not tested.
- Any non-Qwen / any model other than `unsloth/Qwen3-1.7B` — not tested.

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
