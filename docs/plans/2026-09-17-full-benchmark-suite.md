# Build Plan — full benchmark suite

slug: `full-benchmark-suite` · status: `exported` · from frame: `full-benchmark-suite`

> unsloth-cli ships a full benchmark suite: sloth eval scores target-task accuracy, holdout generalization, a regression set against the base model, instruction following, structured-output/tool-call compliance, held-out perplexity/loss, per-item latency + tok/s, and standard benchmarks (MMLU), each suite recorded side by side and summarized/compared across runs

## Tasks

### t1 — metrics core: open metrics dict, `schema_version`, suite-keyed result files

- instruction: Keep the legacy `EVAL_JSON_NAME` constant exported for readers (t5 handles legacy reads). Suite name = the suite file/dir stem, sanitised to \[a-z0-9-\]. Aggregate keys: `exact_match_pct`, f1 stay; add any numeric metric generically as mean. Tests go in tests/`test_tune_metrics.py` (new file).
- covers: c2, h2, c3, h3, c33, h28
- acceptance:
  - sloth/tune/metrics.py: `score_records` returns an open metrics dict per row and summarize/aggregate fold any numeric metric key without naming it; a fake metric registered in a test appears in the aggregate
  - `write_eval_json`(directory, suite, payload, `batch_size`=...) writes eval/<suite>.json carrying `schema_version`=2, suite, `batch_size`, target, `written_at`, `base_load_in_4bit`; a second call with another suite leaves the first file intact
  - metrics.py passes `test_metrics_module_imports_only_stdlib` unchanged

### t2 — datasets core: seeded holdout split, cross-schema overlap check, new suite schemas + JSON-Schema-subset validation

- instruction: Do not touch metrics.py or `_trainer.py`. Keep `detect_schema` backwards compatible: existing chat/task files still detect as before. Document each new schema in the module docstring; t14 copies that text into the catalog.
- covers: c5, h5, c28, h22, c6, c32, h27
- acceptance:
  - sloth/tune/datasets.py: `split_holdout`(path, fraction, seed) returns identical partitions on repeated calls with the same seed and writes <stem>.train.jsonl/<stem>.holdout.jsonl; chat rows split into task rows (prompt = all but final assistant turn rendered as text, `expected_output` = final assistant content) so the holdout is scorable
  - `overlap_check`(`train_path`, `suite_paths`) normalises chat and task rows to (prompt, `expected_output`) and returns file:line of every duplicate; sloth-level callers raise CliError(code=1) with a hint naming them
  - `validate_suite` accepts three new schemas: instruction (task keys + constraints\[\] of {`max_words`|`min_words`|`must_contain`|`must_not_contain`|`must_refuse`|`json_only`}), structured ({task,input,`json_schema`} with the documented subset type/required/properties/enum/items/additionalProperties; any other keyword -> validation error naming it) and toolcall ({task,input,`expected_tool_call`:{name,arguments}})
  - tests/`test_tune_datasets.py` covers all three criteria; module stays stdlib-only

### t3 — config: \[eval\] section (`holdout_fraction`, seed, `eval_steps`, perplexity) and \[eval.thresholds\] baseline

- instruction: Follow the existing \[hyperparameters\] parsing style. The thresholds baseline values come from frame decision c36 and are the single source the docs cite. Keep tomllib-only.
- covers: c23
- acceptance:
  - sloth/tune/config.py: `load_config` reads \[eval\] with defaults `holdout_fraction`=0.0, `eval_steps`=0 (off), perplexity=false, and \[eval.thresholds\] with defaults `regression_drop_pp`=2.0, `compliance_min_pct`=95.0, `latency_max_ratio`=1.10, `min_suite_rows`=100; unknown keys raise the existing config error with a hint
  - RunConfig exposes eval and thresholds as frozen dataclasses; tests/`test_tune_config.py` covers defaults, overrides and rejection; sloth config init template documents the keys

### t4 — scorers module: instruction constraints, JSON-schema-subset checker, per-family tool-call parser registry, lazy GLEU/BLEU

- instruction: Parsers: qwen3 = <`tool_call`>{json}</`tool_call`> blocks; lfm2 = the LFM2.5 tool-call token wrapper (read the tokenizer chat template in the HF cache for the exact tokens; cite it in the docstring). Family detection helper: `family_for_model`(`model_id`) by substring, overridable via \[eval\] `tool_call_family`.
- covers: c6, h6, c10, h10, c32
- acceptance:
  - sloth/tune/scorers.py (new, stdlib at top level): `score_constraints`(prediction, constraints) -> {passed: bool, failed: \[names\]}; `check_json_subset`(prediction, schema) -> {valid: bool, errors: \[...\]} for the documented subset; `parse_tool_call`(prediction, family) via a registry with at least qwen3 and lfm2 parsers, unknown family raises ValueError listing supported families
  - gleu(prediction, expected) and bleu(...) import sacrebleu inside the function and raise CliError(code=2, 'install sacrebleu in the container dep layer') when absent; tests monkeypatch sys.modules to prove both paths
  - tests/`test_tune_scorers.py` (new) has one fixture per parser family; tests/`test_lazy_import.py` stays green

### t5 — summary + summarize: read every eval/<suite>.json plus legacy eval.json, open metric pass-through, per-suite text render

- instruction: This is the compatibility seam for frame park v1: additive nesting, never drop the flat keys. Do not edit compare.py here (t10 owns it).
- depends on: t1
- covers: c3, h3, c2, h2, c33, h28, c18, h16, c21, h19, c20, h18
- acceptance:
  - sloth/tune/summary.py: `read_eval` returns {`suite_name`: payload} from eval/\*.json and maps a flat legacy eval.json to suite 'legacy'; `build_summary`\['eval'\] = {'suites': {...}, plus top-level `exact_match_pct`/f1 taken from the 'target' suite when present else the first suite} so existing readers of summary\['eval'\]\['f1'\] keep working
  - a fixture run dir copied from a run at a072b88 (flat eval.json only) summarises with unchanged `exact_match_pct`/f1 (covers h19); a mixed dir reports both
  - sloth summarize text output prints one block per suite with every numeric metric key, `batch_size` and base precision; tests/`test_tune_summary.py` + tests/`test_cmd_summarize.py` updated; runs.jsonl RunRecord fields unchanged (tests/`test_runs_registry.py`)

### t6 — eval loop: latency + token counts, chat-schema rendering, perplexity forward pass, base precision recorded (adapter and --model paths)

- instruction: Keep the shared-plumbing contract between `_trainer.py` and `_exporter.py` (comment at `_trainer.py`:344-348). No change to `_run_real` here (t7 owns training-time eval). Perplexity on the --model path is only for transformers-loadable exports; GGUF returns a CliError(code=1) with a hint.
- depends on: t1, t4
- covers: c8, h8, h26, c4, c28, h22, c31
- acceptance:
  - sloth/tune/`_trainer.py` `_generate_predictions` returns per-row `generated_tokens` and `latency_ms` = batch wall time / rows measured with time.`perf_counter` around model.generate; `run_eval` and `_exporter`.`run_eval_model` both write them, plus suite-level `median_latency_ms` and `tokens_per_s`, plus `batch_size` and `base_load_in_4bit` into the result
  - `run_eval` scores instruction/structured/toolcall/chat suites by delegating to sloth.tune.scorers and `eval_prompt` renders chat rows via the tokenizer chat template with `add_generation_prompt`
  - `run_perplexity`(model, tokenizer, suite) returns exp(mean token NLL) from a labelled forward pass, never from generate; fake-backend tests in tests/`test_tune_trainer.py` and tests/`test_cmd_eval.py` cover every criterion

### t7 — training-time eval: `eval_dataset` + `eval_steps` into SFTTrainer, loss history into `training_metadata.json`

- instruction: Serialised after t6 because both touch `_trainer.py`. Read `log_history` from trainer.state, not TrainOutput, so eval losses are included.
- depends on: t6, t2, t3
- covers: c4, h4
- acceptance:
  - sloth/tune/`_trainer.py` `_run_real`: when config.eval.`holdout_fraction` > 0 the dataset is split via datasets.`split_holdout` and SFTTrainer receives `eval_dataset` with `eval_strategy`='steps' and `eval_steps`=config.eval.`eval_steps`; fake-backend test asserts the kwargs
  - sloth/tune/metadata.py records `loss_history` (step, `train_loss`, `eval_loss`) from trainer.state.`log_history` and `final_train_loss`; holdout split path + seed recorded so the run is reproducible

### t8 — eval CLI: repeated --suite by name, --perplexity, --batch-size in results, overlap refusal before container launch, tool-call family flag

- instruction: Suite name derives from the path stem; collisions (two suites with one stem) exit 1 with a hint. Keep --in-container recursion guard untouched.
- depends on: t1, t2, t4
- covers: c3, c5, h5, c6, c20, h18
- acceptance:
  - sloth eval --adapter X --suite A --suite B writes eval/A.json and eval/B.json in one invocation; --json output lists suites with their metrics; text output prints one block per suite
  - sloth eval --train-dataset <path> (or the dataset recorded in `training_metadata.json` when present) runs datasets.`overlap_check` and exits 1 with a file:line hint before any docker invocation (fake launcher asserts zero calls)
  - --perplexity and --tool-call-family flags are forwarded in-container; --batch-size is recorded in every result; tests/`test_cmd_eval.py` covers all criteria

### t9 — compare --base: sequential base and adapter container runs, per-suite deltas, thresholds from config, precision guard, exit codes

- instruction: Base eval of a raw HF id needs no export: run the --model path with a no-adapter load inside the container; write its results under <adapter>/eval-base/<suite>.json. Thresholds default to the c36 baseline when no --config is given.
- depends on: t5, t8
- covers: c7, h7, c30, h24, c31, h25, c23, h21
- acceptance:
  - sloth compare --base <hf-id-or-dir> <adapter-dir> \[--config run.toml\] evaluates the base on every suite the adapter has via two separate container invocations (fake launcher records two calls), then emits deltas per suite and per metric
  - exit 1 with hint when a regression-tagged suite drops more than thresholds.`regression_drop_pp`, when compliance < `compliance_min_pct`, when median latency ratio > `latency_max_ratio`, or when a suite has fewer rows than `min_suite_rows`; the applied thresholds are echoed in --json
  - exit 1 with hint when `base_load_in_4bit` differs between the two result sets; simulated OOM on the second run exits 2 with the memory hint and leaves the first result on disk; tests/`test_cmd_compare.py` covers all criteria

### t10 — suites + demo corpus: regression, instruction-following, structured-output, tool-call suites and the demo training corpus

- instruction: Derive the demo corpus from this repo's own docs (CLAUDE.md, docs/fine-tuning.md, explain catalog) and the AgentCulture terminology the suites test; paraphrase, never copy eval rows. Regression rows should be generic (arithmetic, short QA, formatting) so a base model scores well.
- depends on: t2
- acceptance:
  - examples/eval/regression.jsonl (>= 100 general-capability task rows), instruction-following.jsonl (>= 50 constraint rows incl. `must_refuse`), structured-output.jsonl (>= 50 `json_schema` rows), tool-call.jsonl (>= 30 rows) all pass sloth validate
  - examples/demo-corpus.jsonl (chat schema, >= 500 rows) whose rows teach the answers in examples/eval/agentculture-terms.jsonl, cli-contract.jsonl and task-format.jsonl without duplicating any eval row (`overlap_check` returns none); provenance and licence stated in examples/README.md

### t11 — external dataset option: train/eval accept a Hugging Face Hub dataset id or a path outside the repo

- instruction: Keep sloth/tune/datasets.py stdlib: the HF load happens in `_trainer.py` behind the lazy backend; datasets.py only validates rendered rows. `training_metadata.json` records the hf id + revision instead of a sha256 for hub datasets.
- depends on: t3
- acceptance:
  - run-config dataset = 'hf:<org>/<name>\[:split\]' resolves inside the container through the mounted HF cache via datasets.`load_dataset` and is rendered to the chat or task schema by a documented column mapping in \[run.`dataset_map`\]; a local path outside the repo is mounted read-only
  - sloth validate on an hf: dataset validates the mapped first N rows without a container launch when the dataset is already cached, else exits 2 with a hint; tests/`test_cmd_validate.py` + tests/`test_cmd_train.py` cover it with the fake launcher

### t12 — container dep layer: add `lm_eval` + sacrebleu pins, live-validate against NGC 25.11, record in docs/tested.md + docs/dgx-spark.md

- instruction: Run the real install inside docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 on this Spark before pinning (frame lapse l1: the host-side resolve is not evidence; frame assumption c29 is what you are verifying). If `lm_eval` drags a conflicting transformers/torch, install it --no-deps and list its runtime deps explicitly like unsloth.
- covers: c9, h9, c14, h14, c15, h15
- acceptance:
  - sloth/tune/container.py: a new `DEP_LAYER_BENCH_PACKAGES` tuple (`lm_eval`==<measured>, sacrebleu==<measured>) is installed in `_inner_script` after `DEP_LAYER_PACKAGES`; tests/`test_tune_container.py` asserts the install line contains it
  - docs/tested.md gains a dated row with the uv pip list output showing transformers 4.57.1, peft 0.18.0, trl 0.24.0, datasets 4.8.5, nv torch 2.10 still present after the bench layer; docs/dgx-spark.md pin-bump section names the new tuple
  - pyproject.toml \[project\].dependencies is still \[\]; tests/`test_packaging_import_light.py` green

### t13 — bench verb: MMLU via `lm_eval` in the container plus the vendored MMLU-subset suite

- instruction: Register in sloth/cli/`__init__.py` at the marked spot; catalog entry in sloth/explain/catalog.py plus the `_ROOT` verb list. Resolve frame park v4 here: try `lm_eval`'s hf backend with peft= + `load_in_4bit` on the QLoRA fixture; if it fails, bench the merged-16bit export and state that in the docs.
- depends on: t12, t8
- covers: c9, h9, c11
- acceptance:
  - sloth bench --adapter|--model <dir> --benchmark mmlu \[--limit N\] runs `lm_eval` inside the container and writes eval/mmlu.json in the same result shape (metrics dict incl. acc, `acc_norm`, per-subject) so summarize/compare pick it up; fake launcher test asserts the in-container command
  - examples/eval/mmlu-subset.jsonl (>= 100 rows, licence noted in examples/README.md) scores through sloth eval --suite with letter-choice extraction; sloth explain bench exists and teken cli doctor . --strict stays green
  - a second bench run with a warm HF cache makes no network download (documented check in docs/dgx-spark.md)

### t14 — finetune skill loop, catalog, rubric gate and CPU-only test pass

- instruction: SKILL.md step table gains the compare step and the suite list; keep the loop stop-on-first-nonzero behaviour.
- depends on: t8, t9, t13
- covers: c11, h11, c13, h13
- acceptance:
  - .claude/skills/finetune/scripts/finetune.sh run accepts repeated --suite and forwards --batch-size, --perplexity and --base (compare step added as step 5); tests/`test_finetune_skill_script.py` covers repeated --suite and the compare step
  - sloth explain eval / compare / bench / config document every new flag, schema and threshold key; uv run teken cli doctor . --strict passes
  - uv run pytest -n auto on a CPU-only box passes with the gpu tests skipped and coverage >= 60; only tests under tests/`test_gpu_smoke.py` are gpu-marked

### t15 — docs: fine-tuning.md agent-first eval path, README fine-tune vs retrieval boundary, serving hand-off to lobes

- instruction: Cite lobes/assess.py `run_benchmark` output keys verbatim (model, endpoint, `decode_rates`, prefill). Do not invent numbers: the live-run task fills the measured tables.
- depends on: t14
- covers: c19, h17
- acceptance:
  - docs/fine-tuning.md shows the agent path first: sloth eval --json -> sloth compare --base --json verdict, with the JSON shape and exit codes, before any human-oriented text
  - docs/benchmarks.md gains a 'Serving latency + tok/s' section that cites the exact lobes benchmark command and its output shape and states the hand-off is a manual operator step; README explains which suite catches which regression

### t16 — live run on the Spark: train the fixture adapter, score every suite and MMLU, base-vs-adapter compare, fill docs/benchmarks.md, delivery doc, close #28 items 1-2

- instruction: Free UMA first (lobes vLLM resident: see eidetic record unsloth-cli-followups-22-livetest-gotchas item 3). Run serially: train, eval all suites at batch 8 and batch 1, bench mmlu, compare --base. If a threshold fails, record it failing; do not tune thresholds to pass.
- depends on: t7, t10, t11, t15
- covers: c1, h1, c12, h12, c22, h20, c23, h21
- acceptance:
  - a >= 300-step LoRA on examples/demo-corpus.jsonl (LFM2.5-1.2B via examples/lfm2-lora.toml or Qwen3-1.7B) exists under a documented run dir with `training_metadata.json` incl. `loss_history`; sloth compare --base reports target-task exact match > 0 with a positive delta
  - docs/benchmarks.md has one measured table per family (target-task, holdout, regression, instruction-following, structured-output, tool-call, perplexity, latency/tok-s at batch 1 and 8, MMLU) each row stating model, steps, batch size, base precision, date and the command; the per-format quantization-loss table is re-scored with at least one non-zero exact-match row; the batched-vs-serial outcome is stated
  - the delivery doc under docs/deliveries/ shows at least one degradation the new suites caught that exact/F1 missed (h20), reports every missed c23 threshold as failing, and issue #28 items 1-2 are referenced as closed by it

## Risks

- [unknown_nonblocking] demo corpus quality: if the >= 300-step adapter still scores 0 on target-task rows the success signal fails and the corpus (not the metrics) must be reworked; no second corpus is planned (task t10)
- [unknown_nonblocking] summary\['eval'\] additive nesting (t5) is the chosen resolution of frame park v1; any agent that iterates summary\['eval'\] keys generically will see the new 'suites' key (task t5)
- [unknown_nonblocking] t6 and t7 both edit `_trainer.py` and are serialised by dependency; t8 and t10 both edit CLI tests fixtures — verify disjoint files at fan-out
- [unknown_nonblocking] `lm_eval` 0.4.13's hf backend may not load a bitsandbytes 4-bit base + peft adapter; fallback is benching the merged-16bit export (frame park v4) (task t13)
- [unknown_nonblocking] UMA headroom for the live run: lobes vLLM resident (~90 of 121 GB) OOMs a 1.2B train; the live run needs the box freed or the page-cache trick, and its numbers depend on what else is resident (task t16)
