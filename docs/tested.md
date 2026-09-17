# Tested configurations — what was actually validated

A precise, honest record of **exactly** what has been run on hardware, so the
coverage is unambiguous: e.g. fine-tuning was validated on **`unsloth/Qwen3-1.7B`**
but **not** on Qwen 4B/9B. Treat this as a living tracker — add a row when you
validate a new model/config; do not claim coverage a real run didn't produce.

Companion pages: [`benchmarks.md`](benchmarks.md) (the numbers),
[`dgx-spark.md`](dgx-spark.md) (how/why), [`fine-tuning.md`](fine-tuning.md) (the
feature reference).

## Common environment (the Spark runs; every run below unless a row names another device)

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

Rows below inherit this Spark environment except where a row's own columns name
a different device — e.g. the Thor serving row in the follow-ups #22 section.

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

Not measured in the 2026-09-15 LFM2.5 batch (closed by the follow-ups #22 section below): QLoRA-trained adapters through the export formats, Qwen3
through awq/nvfp4/gguf, any Jetson-side load, any accuracy metric beyond the 4-item
smoke suite (plan risks r3, r4, r8).

Before-state evidence (deviation record, main at `39a3f93`): the same TOML
dry-run on `main` ignored the `target_modules` key silently — the plan's
`hyperparameters` had no such key.

## ✅ Tested — passed (2026-09-15, follow-ups #22: stdout purity, QLoRA + Qwen3 through export)

Same box and container as above, on the `feat/follow-ups-22-a` integration branch
(plan `lfm2-5-delivery-follow-ups-22`, task t9). The dep layer now **pins**
`unsloth==2026.9.4 unsloth_zoo==2026.9.3 bitsandbytes==0.50.2` (measured in-container
by t8 with `uv pip list` after the CLI's own install line; bump procedure in
`docs/dgx-spark.md`). Every `--json` run below was captured as
`> out.json 2> err.log` and checked with `wc -l out.json` and
`python -c 'import json,sys; json.load(open("out.json"))'`. Memory note: the box held
lobes' vLLM (~90 GB of the 121 GB UMA); the first `train` attempt died with CUDA OOM at
backend init until page cache was reclaimed (touch + free a 20 GiB anonymous mmap,
no sudo needed).

| Verb | Method / mode | Model | Invocation | Result |
|------|---------------|-------|------------|--------|
| NGC banner probe | no sloth, no GPU | — | `docker run --rm nvcr.io/nvidia/pytorch:25.11-py3 python -c pass > out 2> err` | ✅ `wc -l out` = **36**, `wc -l err` = 0 — the banner is the image entrypoint's and lands on **stdout**, so only host-side capture can strip it (assumption c40 holds) |
| `train --json` | LoRA, `preset:lfm2`, 10 steps | LFM2.5-1.2B-Base | `uv run sloth train --config <lfm2-lora.toml, output runs/lfm2-lora-t9> --json > out.json 2> err.log` | ✅ 144 s; **`wc -l out.json` = 1**, `json.loads` ok (`status: trained`); `wc -l err.log` = 219 carrying the banner, the uv layer install and all 10 trl `{'loss': …}` dicts |
| `eval --adapter --json` | LoRA adapter, 4-item smoke suite | LFM2.5-1.2B-Base | `uv run sloth eval --adapter runs/lfm2-lora --suite examples/eval-suite.jsonl --json > out.json 2> err.log` | ✅ 122 s; **`wc -l out.json` = 1**, `json.loads` ok; `wc -l err.log` = 191; exact_match **0/4**, token-F1 0.087; every `prediction` starts after the prompt (continuation-only scoring live-verified — the model echoes the `Input:` template but the prompt itself is excluded); `runs/lfm2-lora/eval.json` written |
| `export --base <id> --json` | merged-16bit onto a **different** base (staged adapter copy) | LFM2.5-1.2B-Base adapter → `LiquidAI/LFM2.5-1.2B-Instruct` | `uv run sloth export --adapter runs/lfm2-lora --base LiquidAI/LFM2.5-1.2B-Instruct --format merged-16bit --output runs/lfm2-exports/base-override-instruct --json` | ✅ 108 s; **`wc -l out.json` = 1**; `export.json` records `base: LiquidAI/LFM2.5-1.2B-Instruct`; `model.safetensors` 2.34 GB — the staged `_adapter-override` copy loads |
| `export --format merged-16bit` | **QLoRA (4-bit-trained) adapter** → bf16 merged (risk r3) | `unsloth/qwen3-1.7b-unsloth-bnb-4bit` + `runs/qlora-smoke` | `uv run sloth export --adapter runs/qlora-smoke --format merged-16bit --output runs/qlora-smoke-exports/merged16 --json` | ✅ 150 s; `model.safetensors` 3.44 GB, `config.json` has `torch_dtype: bfloat16` and **no** `quantization_config` — Unsloth dequantises then merges implicitly; no exporter change needed |
| `export --format awq --calib examples/chat-smoke.jsonl` | W4A16 via llm-compressor 0.11.0, **default (inferred) mappings** — no LFM2-style custom list | `unsloth/Qwen3-1.7B` + `runs/lora-smoke` (June LoRA) | `uv run sloth export --adapter runs/lora-smoke --format awq --calib examples/chat-smoke.jsonl --output runs/lora-smoke-exports/awq --json` | ✅ 84 s; `model.safetensors` 1.98 GB, `compressed-tensors` `pack-quantized` 4-bit; llm-compressor logged `28 mappings were skipped due to incompatible shapes` (the GQA v_proj→o_proj pair, one per layer) and carried on — **no custom mapping needed** (risk r4 of the shipped plan). `--calib` was required because the June adapter's `training_metadata.json` predates the `dataset.path` field |
| `export --format nvfp4 --calib examples/chat-smoke.jsonl` | NVFP4 via llm-compressor 0.11.0 (torch-2.11 shim active) | same | `… --format nvfp4 --calib examples/chat-smoke.jsonl --output runs/lora-smoke-exports/nvfp4 --json` | ✅ 82 s; `model.safetensors` 2.04 GB, `nvfp4-pack-quantized` |
| `export --format gguf --quant q4_k_m` | Q4_K_M via prebuilt llama.cpp b10909 (cache reuse) | same | `… --format gguf --quant q4_k_m --output runs/lora-smoke-exports/gguf --json` | ✅ 67 s; `Qwen3-1.7B.Q4_K_M.gguf` 1.11 GB; no F16 intermediate left behind |

Every `export --json` row above: **`wc -l out.json` = 1**, `json.loads` ok; the
container's banner, uv layer install and Unsloth/llm-compressor progress all on
stderr (62–186 lines). No new row names a model larger than Qwen3-1.7B / LFM2.5-1.2B.

| `eval --suite examples/eval/` (dir) | adapter baseline, **batch 8 vs batch 1** timing | LFM2.5-1.2B-Base + `runs/lfm2-lora` | `uv run sloth eval --adapter runs/lfm2-lora --suite examples/eval/ --batch-size 8 --json` and the same with `--batch-size 1` | ✅ 46 items in 3 files; **220 s (batch 8) vs 524 s (batch 1)**; exact_match 0/46 both; token-F1 0.038 vs 0.060 — batched decoding is not bit-identical (plan risk r8); per-file + aggregate scores and `runs/lfm2-lora/eval.json` written |
| `eval --model` per export format | merged-16bit / gguf Q4_K_M / awq / nvfp4 on the 46-item suite | LFM2.5-1.2B-Base exports (2026-09-15 batch) | `uv run sloth eval --model runs/lfm2-exports/<fmt> --suite examples/eval/ --batch-size 8 --json` | ✅ four runs (276 s / 130 s / 283 s / 387 s), every one 1 stdout line; exact_match **0/46 on every format** (smoke adapter — not an accuracy claim; h15 treats an all-zero set as a failed condition until a trained adapter exists); token-F1 0.036 / 0.092 / 0.060 / 0.040; full table in `docs/benchmarks.md` |
| serve (vLLM) on **Thor** | `awq` and `nvfp4` LFM2.5 exports loaded off-box | NVIDIA Thor, JetPack R38.2.2 (L4T 6.8.12-tegra, driver 580.00), `vllm/vllm-openai:v0.29.0-aarch64` | `docker run --rm --gpus all -v ~/lfm2-exports:/exports --entrypoint /usr/bin/python3 vllm/vllm-openai:v0.29.0-aarch64 /exports/thor_load.py /exports/<fmt> <fmt>` (`LLM(gpu_memory_utilization=0.12, max_model_len=512, enforce_eager=True)`, one greedy 24-token completion) | ✅ both load with `quantization=compressed-tensors` auto-detected (`pack-quantized` / `nvfp4-pack-quantized`) and generate coherent text; 134 s / 124 s incl. engine init, with lobes' vLLM workers resident on the Thor. The Thor is an operator-local host (`ssh thor`), not referenced by shipped code |

**Before-state of this batch (why the `export` rows needed a fix first):** the first
chain of exports all exited 1 inside the container with
`--adapter path is outside the allowed roots: /home/spark/git/unsloth-cli/runs/…` —
the path sanitizer added to `export.py` in #20's review batch (after that PR's live
rows) also runs in-container, where cwd is `/workspace` and `HOME` is the export
home. Fixed by forwarding `SLOTH_ALLOWED_ROOTS` (the identity-mounted parents) into
the container env (deviation d2). Every real `sloth export` on `main` between #20 and
this fix was broken.

## ✅ Tested — passed (2026-09-17, benchmark dep layer: lm_eval + sacrebleu)

Same box and container as above, on the `feat/full-benchmark-suite` branch (plan
`full-benchmark-suite`, task t12). The dep layer gains a third tuple,
`DEP_LAYER_BENCH_PACKAGES = ("lm_eval==0.4.13", "sacrebleu==2.6.0")`, installed with a
plain `uv pip install` **after** the `--no-deps` layer. Measured by mounting the
already-built dep-layer venv (`~/.cache/unsloth-cli/home/.unsloth-cli-venv`) and
installing the two packages into it, then diffing `uv pip list`.

| Step | Invocation | Result |
|------|------------|--------|
| before | `uv pip list` in the built venv | transformers 4.57.1, peft 0.18.0, trl 0.24.0, datasets 4.8.5, accelerate 1.13.0, numpy 2.3.5, unsloth 2026.9.4, bitsandbytes 0.50.2 |
| install | `uv pip install lm_eval==0.4.13 sacrebleu==2.6.0` (dry-run, then real) | ✅ **45 packages added, 0 changed, 0 removed** — evaluate 0.4.6, scikit-learn 1.9.1, scipy 1.18.1, sqlitedict 2.1.0, rouge-score 0.1.2, portalocker 4.3.2, tabulate 0.10.0, … ; no torch / torchvision / transformers / peft / trl / datasets / numpy line in the diff |
| after | `uv pip list` | transformers 4.57.1, peft 0.18.0, trl 0.24.0, datasets 4.8.5, accelerate 1.13.0, numpy 2.3.5 unchanged; lm-eval 0.4.13, sacrebleu 2.6.0 present |
| import | `python -c "import torch; …"` / transformers, peft, trl, datasets / lm_eval, sacrebleu / unsloth | ✅ `torch 2.10.0a0+b558c986e8.nv25.11 cuda 13.0 True`; `tf 4.57.1 peft 0.18.0 trl 0.24.0 ds 4.8.5`; `lm_eval 0.4.13 sacrebleu 2.6.0`; `import unsloth` still patches (🦥 banner) |

Not exercised here: an actual `lm_eval` MMLU run (task t13) and the sacrebleu scorer
(task t4) — this row only proves the pins coexist with the validated window.

## ✅ Tested — passed (2026-09-17, full benchmark suite: fixture adapter, every suite family, bench, compare, exports)

Same box and container as above, on the `feat/full-benchmark-suite` branch (plan
`full-benchmark-suite`, task t16). lobes' vLLM resident (~94 GB of the 121 GB UMA)
throughout; page cache reclaimed (touch a 16–20 GiB anonymous mmap) before every
GPU step. Every `--json` invocation below produced exactly one stdout line
(`json.loads` ok). Numbers and their reading are in
[`docs/benchmarks.md`](benchmarks.md#full-benchmark-suite--fixture-adapter-2026-09-17-plan-full-benchmark-suite).

| Verb | Mode | Model | Invocation | Result |
|------|------|-------|------------|--------|
| `train --json` | LoRA `preset:lfm2`, 300 steps, `[eval] holdout_fraction=0.1` | LFM2.5-1.2B-Base | `uv run sloth train --config examples/demo-lora.toml --json` | ✅ 3 min 43 s wall (`train_runtime` 91.3 s); first attempt ❌ CUDA OOM at backend init (exit 2 with the memory hint) until the page cache was reclaimed; `training_metadata.json` carries `loss_history` (300 entries), `final_eval_loss` 2.11, `holdout` (532/59 rows, `eval_steps` 50) |
| `eval --adapter --json` (9 suites, batch 8) | `--perplexity --tool-call-family lfm2 --train-dataset …train.jsonl` | same + `runs/demo-lora` | `uv run sloth eval --adapter runs/demo-lora --suite <9 files> --batch-size 8 --perplexity --tool-call-family lfm2 --train-dataset examples/demo-corpus.train.jsonl --json` | ✅ 31 min 13 s; nine `eval/<suite>.json` files; but **batch 8 output is degenerate on LFM2** (0/46 exact everywhere; see r13). First attempt ❌ exit 1: the overlap check flagged the run's own holdout against the full corpus (r12) — passing the train split fixed it |
| `eval --adapter --json` (9 suites, batch 1) | same flags, `--batch-size 1` | same | same with `--batch-size 1` | ✅ 27 min 37 s; target suites 14/46 exact, holdout 6/59, structured compliance 76 %, MMLU-style 61.7 % letters — the valid LFM2 numbers |
| `eval --adapter --results-dir` (4 suites, batch 1) | `--results-dir runs/demo-lora/eval-batch1` | same | `… --suite <3 target> --suite examples/demo-corpus.holdout.jsonl --batch-size 1 --results-dir runs/demo-lora/eval-batch1 --json` | ✅ 5 min 8 s; flat `<results-dir>/<suite>.json` files, batch-8 files untouched |
| `bench --json` | `--benchmark mmlu --limit 5` (5-shot, 285 docs) | same | `uv run sloth bench --adapter runs/demo-lora --benchmark mmlu --limit 5 --json` | ✅ 5 min 22 s incl. first-run download of 57 subject splits; `lm_eval --model hf --model_args pretrained=…,peft=<adapter>` loaded the LoRA directly (park v4 resolved for bf16 LoRA); `eval/mmlu.json` acc 0.5789 |
| `bench --offline --json` | same, warm cache | same | `… --offline --json` | ✅ 2 min 23 s; no split generation, lm_eval logs "Using the latest cached version … (offline mode is enabled)" per subject; identical acc 0.5789 |
| `compare --base --json` (1st) | adapter re-eval + base eval, batch 1 | LFM2.5-1.2B-Base vs `runs/demo-lora` | `uv run sloth compare --base LiquidAI/LFM2.5-1.2B-Base runs/demo-lora --config examples/demo-lora.toml --json` | ❌ exit 1 after 21 min: in-container `eval --model <hf-id>` rejected the repo id ("model directory not found") — the host accepted it, the seam did not. **Fixed** (`run_eval_model` remote-id path) |
| `compare --base --json` (2nd) | same | same | same | ⚠ exit 0 after 1 h 12 min but **vacuous**: docker created the bind-mounted `eval-base/` as root, the container's writer got EACCES on every suite, the base side came back empty and the verdict passed with one-sided deltas (r15). Base numbers recovered from the run's captured stdout JSON. **Fixed** (compare pre-creates the dir and exits 1 on an empty base side); verification run recorded in the delivery doc |
| `export` ×4 + `eval --model` ×4 | merged-16bit / gguf q4_k_m / awq / nvfp4, 3 target suites, batch 1 | `runs/demo-lora` | `uv run sloth export --adapter runs/demo-lora --format <fmt> [--quant q4_k_m] [--calib examples/chat-smoke.jsonl] --output runs/demo-lora-exports/<fmt> --json` then `uv run sloth eval --model runs/demo-lora-exports/<fmt> --suite <3 target> --batch-size 1 --json` | ✅ exports 42 / 55 / 81 / 64 s (2.34 GB / 0.73 GB / 1.08 GB / 1.12 GB); evals 192 / 71 / 188 / 217 s; exact 14 / **0** / 12 / 9 of 46 — the first per-format table with non-zero rows (#28 item 1); the GGUF 0/46 is a follow-up (r14) |
| `validate --suite examples/eval/` | per-file schema detection (deviation d1) | — | `uv run sloth validate --suite examples/eval/` | ✅ 305 records across 7 files, each reported with its detected schema (task / instruction / structured / toolcall) |

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

- Training and eval ran only on **GB10 (Blackwell, aarch64)** + **NGC 25.11 /
  torch 2.10**. `--gpus all` worked via **CDI** (Docker default runtime `runc`);
  a host requiring the `nvidia` runtime was not tested.
- The only *other* device tested is **NVIDIA Thor** (JetPack R38.2.2, L4T
  6.8.12-tegra, driver 580.00, `vllm/vllm-openai:v0.29.0-aarch64`), and only for
  **serving** the LFM2.5 `awq`/`nvfp4` exports (follow-ups #22) — not for
  training or eval. **Orin Nano is still untested** — tracked in
  [#23](https://github.com/agentculture/unsloth-cli/issues/23).

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
