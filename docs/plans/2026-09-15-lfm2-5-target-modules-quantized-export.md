# Build Plan — LFM2.5 `target_modules` + quantized export

slug: `lfm2-5-target-modules-quantized-export` · status: `exported` · from frame: `lfm2-5-target-modules-quantized-export`

> unsloth-cli trains full-coverage LoRA adapters for LFM2.5 (short-conv blocks included, via an explicit `target_modules` key with an lfm2 preset) and exports any adapter as merged-16bit, merged-4bit, quantized GGUF, or AWQ W4A16, so the same adapter runs on a Jetson Orin Nano (llama.cpp), a Jetson Thor or DGX Spark (vLLM), or lobes' hand lane

## Tasks

### t1 — PR1/t1 — config: `target_modules` key (list | regex string | preset:<name>) + lfm2 preset resolution

- instruction: Follow the `_require_int`/`_require_float` idiom at config.py:126-189 and add `_require_target_modules`. Keep the module importing only stdlib. Do NOT try to reach conv layers with a name list — the live probe showed Unsloth folds a list into a tag regex without 'conv' (0 conv modules adapted from an 8-name list); only the regex string form adapts all 92 LFM2.5-1.2B modules. Owned files: sloth/tune/config.py, sloth/tune/presets.py (new), tests/`test_tune_config.py`.
- covers: c2, h28
- acceptance:
  - sloth/tune/config.py: RunConfig.`target_modules` is list\[str\] | str | None (default None); `load_config` accepts a non-empty list of non-empty strings, a single non-empty string, or absence; anything else (wrong type, empty, unknown preset) raises CliError(code=1) whose remediation names the three accepted forms
  - a new pure-stdlib module sloth/tune/presets.py exposes PRESETS = {'lfm2': r'model\.layers\.\d+\.(`self_attn`\.(q|k|v|out)`_proj`|conv\.(in|out)`_proj`|`feed_forward`\.w\[123\])'} and `resolve_target_modules`(value) which maps 'preset:<name>' to its regex and passes lists/regex strings through unchanged
  - tests/`test_tune_config.py` covers: absent key -> None; explicit list round-trips; regex string round-trips; preset:lfm2 resolves to the documented regex; wrong type / empty / preset:nope -> exit 1 with hint; black/isort/flake8 clean

### t2 — PR1/t2 — registry: `config_hash` stays stable when optional fields are unset

- instruction: The hash is over dataclasses.asdict(config) (registry.py:147-158). Filter None-valued top-level keys, nothing else, so every pre-existing run keeps its id. Owned files: sloth/tune/registry.py, tests/`test_runs_registry.py`.
- covers: c30, h20
- acceptance:
  - sloth/tune/registry.py `compute_config_hash` drops keys whose value is None before hashing; tests/`test_runs_registry.py` asserts the hash of examples/lora-smoke.toml equals the literal hex value it has on main today (record it in the test) and that setting `target_modules` changes it

### t3 — PR1/t3 — trainer: pass resolved `target_modules` to `get_peft_model` and surface it in the plan + metadata

- instruction: Only `_trainer.py` changes: the `get_peft_model` call at 270-276 and `_resolved_hyperparameters` at 91-104. Resolve via sloth.tune.presets.`resolve_target_modules`. Owned files: sloth/tune/`_trainer.py`, tests/`test_tune_trainer.py`.
- depends on: t1
- covers: c3, h3
- acceptance:
  - sloth/tune/`_trainer.py` passes `target_modules`=<resolved value> to backend.`fast_language_model`.`get_peft_model` only when the config sets it (kwarg omitted when None so Unsloth's default path is unchanged)
  - `_resolved_hyperparameters` includes '`target_modules`' (resolved regex/list or null), so `run_training`(`dry_run`=True) plan and `training_metadata.json` both carry it with no change to metadata.py
  - tests/`test_tune_trainer.py`: fake backend events\['`get_peft`'\]\[0\]\['`target_modules`'\] equals the resolved value for preset:lfm2 and the kwarg is absent when unset; metadata file contains the same value

### t4 — PR1/t4 — train host preflight: LFM2 hint, rank>32 hand-lane diagnostic, chat-template check

- instruction: Insert between dataset validation and the scope guard (train.py:232-247). The HF cache layout is ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/; read files with json/pathlib only. Owned files: sloth/cli/`_commands`/train.py, tests/`test_cmd_train.py`.
- depends on: t1, t3
- covers: c27, h17, c9, h8
- acceptance:
  - sloth train (dry-run and real) with a model id containing 'lfm2' (case-insensitive) and no `target_modules` emits exactly one `emit_diagnostic` line on stderr recommending `target_modules` = "preset:lfm2" and proceeds; stdout JSON is unchanged and parses
  - when the model id contains 'lfm2' and `lora_r` > 32 a stderr diagnostic says lobes' hand lane caps LoRA rank at 32; never an error
  - for a chat-schema dataset, the host preflight looks for `chat_template` in `tokenizer_config.json` or a `chat_template`.jinja under the local HF cache snapshot for the model; if the model is cached and neither exists it exits 1 with a hint to use the task schema; if the model is not cached yet it emits a diagnostic and proceeds; sloth validate and sloth/tune/datasets.py import graphs are unchanged (tests/`test_packaging_import_light.py` still passes)
  - tests/`test_cmd_train.py` covers all three behaviours with tmp HF-cache fixtures; no transformers import on the host

### t5 — PR1/t5 — docs + catalog + example for `target_modules` and the deployment-target table

- instruction: Docs-only task for PR1; the PR2 docs task (t12) depends on this one to avoid same-file merges on catalog.py/README/docs/fine-tuning.md. Owned files: docs/fine-tuning.md, README.md, examples/lfm2-lora.toml, sloth/explain/catalog.py (train/config entries only).
- depends on: t1
- covers: c20, h12
- acceptance:
  - docs/fine-tuning.md documents `target_modules` (list / regex / preset:lfm2) in the \[hyperparameters\] block and adds a 'Deployment targets' table with rows Orin (gguf, awq W4A16), Thor (nvfp4), Spark (nvfp4, bf16 + LoRA via lobes hand), each naming the format live-tested for it and linking docs/tested.md
  - examples/lfm2-lora.toml exists (LiquidAI/LFM2.5-1.2B-Base, method lora, `lora_r` 16, `target_modules` = "preset:lfm2") and sloth train --config examples/lfm2-lora.toml --dry-run exits 0 in tests
  - catalog.py `_TRAIN` and `_CONFIG_INIT` mention `target_modules` and preset:lfm2; README's config section mentions the key; uv run teken cli doctor . --strict stays green; markdownlint clean

### t6 — PR1/t6 — live validation on the Spark: LFM2.5-1.2B-Base LoRA with preset:lfm2, served by vLLM, recorded in docs/tested.md

- instruction: If Unsloth import dies with CUDA out-of-memory at import while lobes' vLLM holds most of the UMA, reclaim page cache first (touch-and-free a 16 GiB mmap works without sudo) — see memory record unsloth-cli-quant-export-livetest-2026-09-15. Use the version-bump skill and open PR 1 via the cicd skill after this task. Owned files: docs/tested.md (PR1 rows), CHANGELOG.md, pyproject.toml.
- depends on: t2, t3, t4, t5
- covers: c21, h13
- acceptance:
  - before the change (main): sloth train --config examples/lfm2-lora.toml --dry-run shows no `target_modules` in the plan (record the command + output in the PR description) — the before-state
  - after: a real sloth train run of examples/lfm2-lora.toml completes on the Spark; the adapter's `adapter_config.json` `target_modules` equals the lfm2 regex; the adapter loads in vllm/vllm-openai with --enable-lora (LoRARequest generation succeeds); docs/tested.md gains the train row with the exact invocation and the LFM2 hint behaviour is shown

### t7 — PR2/t7 — export CLI host side: new formats, flags, validation, dry-run, no-clobber, atomic output, disk estimate, container routing

- instruction: Read eval.py first — it is the routing template. Estimate bytes from the base model's config.json parameter count if cached (2 bytes/param merged; x2 for gguf's F16 intermediate; 0.6 bytes/param awq/nvfp4), else from the parameter count parsed from the model id (sloth/tune/scope.py scanner style). Do not implement the in-container work here — call sloth.tune.`_exporter`.`run_export` (t9) from the --in-container branch. Owned files: sloth/cli/`_commands`/export.py, tests/`test_cmd_export.py`.
- covers: c5, h4, c8, h7, c33, h23, c36, h26, c26, h16
- acceptance:
  - `SUPPORTED_FORMATS` = {safetensors, merged-16bit, merged-4bit, gguf, awq, nvfp4}; --format safetensors keeps today's pure-stdlib behaviour byte-for-byte and the module never imports sloth.tune.container at module level (existing container-not-called tests scoped to safetensors, not deleted)
  - new flags: --quant (comma list, validated against the ggml allowlist `q4_k_m` `q5_k_m` `q8_0` f16 `q2_k` `q3_k_m` `q4_0` `q4_1` `q5_0` `q6_k`), --calib PATH, --calib-samples N, --base ID (default: `adapter_config.json` `base_model_name_or_path`), --force, --dry-run, hidden --in-container; every invalid value exits 1 with a hint before any container.launch call
  - --dry-run (all formats) exits 0 with no container launch and a plan containing format, quant, base, output, `estimated_bytes`, `free_bytes`, `docker_command`; the real run exits 2 with a hint when `free_bytes` < `estimated_bytes` (test with a fake statvfs)
  - a non-empty --output without --force exits 1 with a hint; the container writes into <output>.partial and the host renames to <output> only on exit 0 (a fake launcher returning 137 leaves <output> absent and .partial present)
  - host->container routing mirrors eval.py:122-146 (identity mounts of adapter/output/calib parents, checkout=`_repo_root`(), --in-container recursion guard) and passes `UNSLOTH_LLAMA_TAG`/HOME through container.launch's new env parameter; tests/`test_cmd_export.py` covers every bullet; uv run teken cli doctor . --strict green

### t8 — PR2/t8 — container: llama.cpp cache mount, explicit HOME, env passthrough, llm-compressor pin, memory preflight

- instruction: Unsloth resolves its llama.cpp dir from Path.home()/.unsloth/llama.cpp (`unsloth_zoo`/`llama_cpp.py`:132-138) so HOME must be explicit and the mount target must match; prebuilt install is skipped on the second run when the dir is populated (measured: 48s first, 28s cached). Owned files: sloth/tune/container.py, tests/`test_tune_container.py`, docs/dgx-spark.md.
- covers: c7, h6, c35, h25
- acceptance:
  - container.launch/`build_command` accept env=\[(k,v)...\] and extra mounts; export runs set HOME=/workspace/.home (or a documented constant) and bind-mount a host-owned cache dir (default ~/.cache/unsloth-cli/llama.cpp, override via `SLOTH_LLAMA_CPP_CACHE`) at <HOME>/.unsloth/llama.cpp; `UNSLOTH_LLAMA_TAG` is set to a pinned constant (the prebuilt tag validated live, b10909, or newer once re-validated)
  - `DEP_LAYER_PACKAGES` gains llmcompressor==0.11.0 and compressed-tensors==0.16.0 (the pin matrix comment explains the torch-2.11 API shim and that 0.10.0.3 runs NVFP4 natively but breaks AWQ on LFM2); docs/dgx-spark.md pin table updated
  - preflight() reads /proc/meminfo on Linux and emits a stderr hint when MemFree < 4 GiB naming the Unsloth-import OOM and the page-cache reclaim workaround; it never blocks; tests/`test_tune_container.py` covers env passthrough, the mount args, the tag env, and the meminfo hint with a fake meminfo

### t9 — PR2/t9 — `_exporter.py`: lazy in-container seam for merged/gguf (Unsloth) and awq/nvfp4 (llm-compressor), calibration, cleanup, export.json

- instruction: Copy the seam discipline of `_trainer.py`:145-173 (`_load_backend`) so tests can monkeypatch one loader. Import unsloth before trl/transformers/peft. The fx-tracing failure ('NoneType has no attribute `get_mask_sizes`') is why pipeline='basic' is mandatory for LFM2; the GQA shape error (512 vs 2048) is why the `v_proj`->`out_proj` pair is dropped. Reference: the session probe script recorded in memory unsloth-cli-quant-export-livetest-2026-09-15. Owned files: sloth/tune/`_exporter.py` (new), tests/`test_tune_exporter.py` (new), tests/`test_lazy_import.py`, tests/`test_packaging_import_light.py`.
- covers: c6, h5, c24, h1, c29, h18, c31, h21, c32, h22, c34, h24
- acceptance:
  - new sloth/tune/`_exporter.py` with `run_export`(plan) that lazy-imports unsloth/torch/llmcompressor inside the function (tests/`test_lazy_import.py` + `test_packaging_import_light.py` extended to it); missing stack -> CliError(code=2) with the NGC hint; CUDA OOM -> CliError(code=2) with the memory hint
  - merged-16bit / merged-4bit call model.`save_pretrained_merged`(dir, tokenizer, `save_method`='`merged_16bit`'|'`merged_4bit`'); gguf calls model.`save_pretrained_gguf`(dir, tokenizer, `quantization_method`=\[...\]) then moves the \*.gguf files from Unsloth's <dir>`_gguf` suffix dir into the requested dir and deletes the F16 intermediate unless `keep_intermediate`; a fake backend test asserts each call's kwargs and the final file layout
  - awq: merged-16bit first, then llmcompressor.oneshot(pipeline='basic', recipe=\[AWQModifier(mappings=..., `duo_scaling`='both'), QuantizationModifier(scheme='`W4A16_ASYM`', targets=\['Linear'\], ignore=\['`lm_head`'\])\]); for `model_type` lfm2 the mappings are built per layer from config.`layer_types` (`operator_norm` -> `self_attn` q/k/v on `full_attention` layers, `operator_norm` -> conv.`in_proj` on conv layers, `ffn_norm` -> `feed_forward` w1/w3, `feed_forward`.w3 -> w2, and NO `v_proj`->`out_proj` pair); other architectures use llm-compressor defaults; nvfp4: QuantizationModifier(scheme='NVFP4', targets=\['Linear'\], ignore=\['`lm_head`'\]) with the same basic pipeline; `save_pretrained`(`save_compressed`=True)
  - calibration: default = the run's dataset rendered exactly as `_trainer`.`_format_records` renders it (import the same helper), --calib overrides, --calib-samples caps; fewer than 64 samples -> one stderr diagnostic; the count and source are recorded
  - the torch.accelerator.`get_memory_info` shim is one function applied only when the attribute is missing AND `compressed_tensors`.`__version__` == '0.16.0', maps to torch.cuda.`mem_get_info`, and two tests prove applied/skipped
  - export.json is written next to the artifacts and appended (list) to <adapter>/exports.json with: format, quant, base, adapter, files{name:bytes}, calibration{source,count}, versions{unsloth,`unsloth_zoo`,transformers,peft,llmcompressor,`compressed_tensors`,`llama_cpp_tag`}, timestamp; tests/`test_tune_exporter.py` covers every bullet with fakes, no GPU

### t10 — PR2/t10 — surface exports in runs / summarize / compare

- instruction: Mirror `find_latest_checkpoint` / `read_trainer_state` (summary.py:100-149); summarize.py's renderer is hand-listed (23-47) so add an explicit block; compare.py delta block at 37-56. Owned files: sloth/tune/summary.py, sloth/cli/`_commands`/summarize.py, sloth/cli/`_commands`/compare.py, sloth/cli/`_commands`/runs.py, their tests.
- depends on: t9
- covers: c22, h14
- acceptance:
  - sloth/tune/summary.py `build_summary` discovers <`output_dir`>/exports.json and any export.json under the run dir and returns an 'exports' list; summarize text mode prints one line per export (format, quant, files, bytes); compare shows export presence/format deltas; runs show renders the exports key; tests for each renderer

### t11 — PR2/t11 — eval --model DIR for merged / quantized outputs

- instruction: Add `run_eval_model` next to `run_eval` in `_trainer.py` (or in `_exporter.py` to keep `_trainer.py` untouched after t3 — prefer `_exporter.py`). Owned files: sloth/cli/`_commands`/eval.py, sloth/tune/`_exporter.py` (eval helper only), sloth/explain/catalog.py (`_EVAL` entry), tests/`test_cmd_eval.py`.
- depends on: t3, t9
- covers: c39, h27
- acceptance:
  - sloth eval accepts --model DIR (mutually exclusive with --adapter: both or neither -> exit 1 + hint); host side routes to the container like --adapter; in-container, `run_eval_model` loads a merged/awq/nvfp4 dir via transformers (compressed-tensors auto-detected) or, when the dir holds a .gguf, scores via llama-completion from the mounted llama.cpp cache
  - result JSON has the same score fields as adapter eval plus `model_dir`, `quant_method`, `quant_format` (from config.json `quantization_config`, null for bf16/gguf); catalog `_EVAL` updated; tests/`test_cmd_eval.py` covers the exclusivity error, host routing, and a fake-backend --model run

### t12 — PR2/t12 — docs, catalog, README, benchmarks and the /finetune skill for the export formats

- instruction: Depends on t5 so catalog.py/README/docs/fine-tuning.md are edited once per PR, not concurrently. Owned files: sloth/explain/catalog.py (`_EXPORT`), README.md, docs/fine-tuning.md (export section), docs/benchmarks.md, .claude/skills/finetune/SKILL.md, .claude/skills/finetune/scripts/finetune.sh.
- depends on: t5, t7
- covers: c14, h9
- acceptance:
  - catalog.py `_EXPORT` enumerates all six formats, every new flag, the container/no-container split and the exit codes; README.md:86 row and docs/fine-tuning.md export section list the formats and the target table links them; docs/benchmarks.md gains measured sizes (bf16 2.34 GB, gguf q4 0.73 GB, awq 1.08 GB, nvfp4 1.12 GB for LFM2.5-1.2B)
  - .claude/skills/finetune/SKILL.md run-mode table and scripts/finetune.sh accept --export-format FMT and --quant LIST (default safetensors) and forward them to sloth export; grep 'safetensors' across README, docs/fine-tuning.md, catalog `_EXPORT` and finetune.sh shows the format list; markdownlint and teken cli doctor . --strict green

### t13 — PR2/t13 — live validation on the Spark: every export format + eval --model scores, recorded in docs/tested.md; PR 2 opened

- instruction: Run sequentially, the box is memory-tight with lobes' vLLM resident; keep `gpu_memory_utilization` low (0.12 worked) for the vLLM load checks. Owned files: docs/tested.md (PR2 rows), CHANGELOG.md, pyproject.toml.
- depends on: t6, t7, t8, t9, t10, t11, t12
- covers: c1, h10, c25, h11, c23, h15
- acceptance:
  - on the LFM2.5-1.2B-Base adapter from t6: sloth export --format merged-16bit, gguf --quant `q4_k_m`, awq, nvfp4 (and merged-4bit) each exit 0 on the Spark; merged loads with transformers, gguf runs with llama-completion, awq and nvfp4 load in vllm/vllm-openai; a second gguf export reuses the cached llama.cpp (no download in the log)
  - sloth eval --model on merged-16bit, awq and nvfp4 (and --adapter) against examples/eval-suite.jsonl produces four scores recorded side by side in docs/tested.md; each tested.md row carries the exact invocation; any format that fails live is listed under Not tested, never claimed
  - the /finetune skill run mode completes train -> eval -> export --export-format gguf with --json and every step's JSON is on stdout only; version bumped; PR 2 opened via the cicd skill; CI test job >= 60% coverage and the rubric gate green

## Risks

- [unknown_nonblocking] unsloth / `unsloth_zoo` / bitsandbytes are installed unpinned (--no-deps) in the container; the live probes ran on 2026.9.4 while docs cite 2026.6.9 — decide whether t8 pins them (recommended: pin to the version validated in t6/t13) (task t8)
- [follow_up] The torch.accelerator.`get_memory_info` shim exists only because NGC 25.11 ships torch 2.10; when the image moves to torch>=2.11 the shim test (t9) must flip to 'skipped' and the shim be deleted (task t9)
- [unknown_nonblocking] QLoRA-trained (4-bit) adapters were not merged in the probes; Unsloth's dequantize-then-merge path is unmeasured until t13 runs a qlora example (task t13)
- [unknown_nonblocking] Qwen3 through awq/nvfp4/gguf is unmeasured (all probes were LFM2.5); llm-compressor default mappings should fit Qwen3 but the GQA v->o pair may need the same skip (task t13)
- [unknown_nonblocking] Two concurrent gguf exports sharing the mounted llama.cpp cache could race on Unsloth's prebuilt install (no lock); single-operator today (task t8)
- [unknown_nonblocking] Disk estimate uses a parameter-count heuristic; a wrong count under-estimates and the fail-closed check passes when it should not — t13 compares `estimated_bytes` to measured sizes and tightens the constants (task t7)
