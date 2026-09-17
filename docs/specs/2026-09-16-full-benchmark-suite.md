# full benchmark suite

> unsloth-cli ships a full benchmark suite: sloth eval scores target-task accuracy, holdout generalization, a regression set against the base model, instruction following, structured-output/tool-call compliance, held-out perplexity/loss, per-item latency + tok/s, and standard benchmarks (MMLU), each suite recorded side by side and summarized/compared across runs
> instruction: run the whole loop on one adapter: sloth eval --adapter <dir> --suite examples/eval/ --suite <holdout> --suite <regression> --suite <ifeval-style> --suite <structured>; then sloth summarize <run> --json shows every suite side by side and sloth compare <base> <adapter> --json shows per-suite deltas

## Audience

- agents and operators driving the /finetune loop (Claude/colleague backends) who need a machine-readable verdict on whether an adapter improved its target behaviour without degrading the base model, plus the lobes/colleague siblings that decide whether to serve it
  - instruction: check README/docs/fine-tuning.md name agents + operators as the eval readers and that --json is the primary consumer path

## Before → After

- Before: sloth eval scores exactly one metric family (exact match + token F1) on task-schema suites, overwrites a single eval.json per directory, computes no loss/perplexity/latency, cannot score instruction-following or structured output, and has no base-vs-adapter regression check (sloth/tune/metrics.py:106-118, summary.py:131-141, compare.py:57, `_trainer.py`:291-310)
  - instruction: git show main:sloth/tune/metrics.py | grep -c 'f1' ; the only metric keys are `exact_match` and f1
- After: sloth eval scores one adapter across several named suites in one invocation, each retained under its own name with batch size and metrics, sloth summarize renders all of them, and sloth compare reports per-suite deltas between the base model and the adapter with a pass/fail verdict
  - instruction: after one train+eval run, ls <adapter>/eval/ shows one file per suite and sloth compare --json contains a per-suite deltas object

## Why it matters

- a fine-tune that raises target-task accuracy while silently degrading base capabilities, breaking JSON output or halving tok/s is worse than no fine-tune; today the repo cannot detect any of those, and its only accuracy table is 0/46 on a smoke adapter (docs/benchmarks.md:193-201)

## Requirements

- the eval scoring pipeline gains an open metrics dict: today sloth/tune/metrics.py:106-118 `score_records` hard-codes `exact_match` + f1, summary.py:131-141 `_eval_summary` reduces to `exact_match_pct`/f1/`file_count`, compare.py:57 `_EVAL_KEYS`=(`exact_match_pct`, f1) and eval.py:306-312 `_render_text` print only those, so every new metric kind (accuracy, perplexity, BLEU, compliance rate, latency) must pass through all four without per-key edits
  - instruction: unit test: register a fake metric in `score_records`, run summarize + compare on a fixture run, assert the key appears in both JSON payloads and text output
  - honesty: adding a metric key to `score_records` surfaces it in eval.json, summarize --json, compare --json and the text renderers with no edit to summary.py, compare.py or eval.py
- eval results become suite-keyed and retained side by side: metrics.py:38 `EVAL_JSON_NAME`='eval.json' is one flat file per target dir overwritten on every run (metrics.py:178-184 docstring), summary.py:107-128 `read_eval` reads only that name, and no suite name or `batch_size` field exists (only `suite_paths`/target/`written_at`) — a suite id, batch size and per-suite file (e.g. eval/<suite>.json) are required so target-task, holdout, regression, instruction-following and compliance suites coexist on one adapter (extends issue #28 item 2)
  - instruction: test: eval twice with suite A then suite B into one tmp adapter dir; assert eval/A.json and eval/B.json both exist and summarize --json has both keys; assert `batch_size` is present
  - honesty: two sloth eval runs with different --suite (or --batch-size) against the same adapter leave both result files on disk, each carrying suite name, `batch_size`, target and `written_at`, and summarize lists both
- held-out loss / perplexity: `_trainer.py`:291-310 builds only `train_dataset` and SFTConfig/SFTTrainer receive no `eval_dataset`, `eval_strategy` or `eval_steps`; trainer.train()'s TrainOutput is discarded and metadata.py:114-124 records no loss; the eval path (`run_eval` / `run_eval_model`) only generates text — a forward-pass loss/perplexity over a held-out JSONL and a loss history in `training_metadata.json` are required
  - instruction: fake-backend unit test asserts SFTTrainer receives `eval_dataset` + `eval_strategy`; gpu-marked smoke asserts perplexity is finite and lower for the adapter than the base on the training-like suite
  - honesty: a training run with an \[eval\] holdout configured records eval loss at `eval_steps` in `training_metadata.json`, and sloth eval --perplexity <suite> returns exp(mean token NLL) computed by a forward pass with labels, never by generation
- holdout split + train/eval overlap check live in the pure-stdlib core: sloth/tune/datasets.py has validators (chat 103-143, task 146-184), `detect_schema` (192-201), `validate_dataset`/`validate_suite`, but no split function, no seed, and no cross-file dedup; a seeded holdout split and a hash-based exact-duplicate check between the training file and every eval suite run before the container launches (same GPU-free gate as schema validation)
  - instruction: tests in tests/`test_tune_datasets.py`: split twice with seed 3407 and assert identical partitions; eval with an overlapping row exits 1 with a hint naming file:line
  - honesty: the holdout split is deterministic for a given seed (same rows every run) and sloth eval refuses (exit 1, error:/hint:) when any suite row exactly duplicates a training row, before any container launch
- instruction-following and structured-output/tool-call suites need new record schemas and stdlib scorers: datasets.py:34 `TASK_KEYS`={task,input,`expected_output`} is string-in/string-out only, with no field for format constraints, refusal expectation, JSON schema or expected tool call; eval.py:172 validates suites with schema='task' hard-coded and chat-schema suites are rejected — new suite kinds (e.g. constraints\[\], `json_schema`, `expected_tool_call`) with json.loads/regex/schema-shape scorers are required, still pure stdlib in metrics.py
  - instruction: tests: a constraints\[\] row passes/fails as expected; a `json_schema` row rejects malformed JSON and a missing required key; metrics.py still passes `test_metrics_module_imports_only_stdlib`
  - honesty: instruction-following suites carry machine-checkable constraints (max words, required/forbidden substrings, must-refuse, JSON-only) and structured-output suites carry a JSON schema or expected tool call; scoring is pure stdlib and reports a per-suite compliance rate
- regression set = one fixed capability suite scored against both the untouched base model and the adapter, with sloth compare gaining a base-vs-adapter mode and a pass/fail threshold: today compare.py:108-137 is run-vs-run only, a base model never evaluated has no eval.json so `_eval_deltas` returns {} silently (compare.py:75-76), and eval --model requires an exported directory (eval.py:97-147 `_resolve_target`) so evaluating the raw HF base needs a no-op export first
  - instruction: test with the fake backend: base scores 80%, adapter 70%, threshold 5pp -> exit 1 with hint; 78% -> exit 0
  - honesty: sloth compare --base <hf-model-or-dir> <adapter> evaluates the untouched base on every suite the adapter has, emits per-suite deltas, and exits non-zero when a regression suite drops below the configured threshold
- latency + tok/s: the offline eval path records per-item wall clock and generated-token counts in `_trainer.py`:447-488 `_generate_predictions` (greedy, `max_new_tokens`=100 at :351, batch 8 at :354; outputs\[row\] and `max_new_tokens` already in scope, no `perf_counter` anywhere) and writes them into the suite result; the serving-side tok/s + TTFT figure is obtained by calling lobes-cli's `run_benchmark` (lobes/assess.py:569-598, output {model,endpoint,`decode_rates`\[\],prefill{}}) against the exported model, not reimplemented
  - instruction: assert `latency_ms` > 0 and `generated_tokens` > 0 on fake-backend rows; docs/benchmarks.md serving section cites the lobes benchmark command and output
  - honesty: every eval result row carries `generated_tokens` and `latency_ms` measured around model.generate, the suite summary carries median latency and aggregate tok/s, and the serving figure comes from lobes benchmark (documented hand-off), not from code in this repo
  - honesty: `latency_ms` is defined per batch-wall-time / rows and every result row carries the `batch_size` it was measured under; the benchmarks doc reports serial (batch 1) latency as the per-item figure and batch-8 throughput as tok/s, never mixing the two
- standard benchmarks (MMLU) run inside the NGC container via a harness added to container.py's `DEP_LAYER` tuples (97-107 / 119-123) with a live-measured exact pin per docs/dgx-spark.md:88-131; the container has default bridge networking (no --network flag in container.py:410-460) and mounts the host HF cache at /opt/hf-cache (container.py:148-151), so the MMLU dataset is fetched once and reused; zero references to `lm_eval`/MMLU/IFEval/sacrebleu exist in unsloth-cli or lobes-cli today
  - instruction: follow docs/dgx-spark.md pin-bump procedure; record the uv pip list line; run twice and compare wall time / network
  - honesty: MMLU runs inside the NGC container with the harness pinned to a version measured live on nvcr.io/nvidia/pytorch:25.11-py3, recorded in docs/tested.md, and a second run with a warm HF cache completes without re-downloading the dataset
- BLEU/GLEU (or any third-party scorer) is lazy-imported inside a function in a separate module, never in sloth/tune/metrics.py: tests/`test_lazy_import.py`:174-198 `test_metrics_module_imports_only_stdlib` AST-checks that every top-level import in metrics.py is stdlib, and `test_packaging_import_light.py`:78-92 repeats it at package level; the sanctioned pattern is `_trainer`.`_load_backend` / `_exporter`.`_load_eval_backend`
  - instruction: tests/`test_lazy_import.py` stays green; add a test that the scorer without sacrebleu installed raises CliError code 2
  - honesty: no non-stdlib import appears at module top level in sloth/tune/metrics.py or any sloth/tune core module; BLEU-family scorers import their library inside the function and raise CliError(code=2) with an install hint when it is absent
- the /finetune loop and the explain catalog follow the new suites: .claude/skills/finetune/scripts/finetune.sh:217 issues one hard-coded 'eval --adapter --suite --json' call with a single string --suite (120-126) and forwards no --batch-size/--model; sloth/explain/catalog.py:273+ `_EVAL` and the `_ROOT` verb list (21-45) are hand-written, so every new verb (e.g. bench) or eval flag needs catalog text and a register() line in sloth/cli/`__init__.py`:93-113
  - instruction: tests/`test_finetune_skill_script.py` covers repeated --suite; run uv run teken cli doctor . --strict
  - honesty: finetune.sh run accepts repeated --suite and forwards --batch-size; explain eval (and any new verb) documents every new flag; teken cli doctor . --strict stays green
- docs/benchmarks.md gains one measured section per benchmark family; today it reports train samples/s + steps/s (43, 56, not token-normalised), per-format exact/F1 (106-156) and lists MMLU/BLEU/perplexity nowhere; its 'What is not yet benchmarked' section (193-201) already admits the per-format table is 0/46 on a 10-step smoke adapter
  - instruction: review the doc after the live run; every table cites the command that produced it
  - honesty: docs/benchmarks.md has one measured table per family (target-task, holdout, regression, instruction-following, structured-output, perplexity, latency/tok-s, MMLU), each row stating model, adapter steps, batch size and date
- tests reuse the existing GPU-free fakes: tests/`test_tune_trainer.py`:706-784 `_FakeTokenizer`/`_FakeEvalModel`/`_install_fake_ml` for the adapter path, tests/`test_cmd_eval.py`:928-1002 `_fake_backend`/`_EvalBackend` for the --model path, and tests/`test_gpu_smoke.py`:36-65,144 @pytest.mark.gpu + subprocess CUDA probe for any real-GPU benchmark test
  - instruction: uv run pytest -n auto on a CPU box passes with the gpu tests skipped; coverage gate `fail_under`=60 holds
  - honesty: all new metric code is covered without a GPU via `_install_fake_ml` / `_fake_backend`, and the only GPU-gated tests are the live smoke rows
- the holdout split must produce scorable rows: every shipped training set is chat-schema (examples/lora-smoke.toml, qlora-smoke.toml, lfm2-lora.toml all set dataset = examples/chat-smoke.jsonl) while eval scores task rows only, so a split of a chat corpus yields rows sloth eval cannot score today — either chat-schema eval (render all but the last assistant turn, score against it) ships with c5, or the split emits task rows from the final assistant turn; the overlap check in c5 must likewise normalise across schemas (chat final-assistant content vs `expected_output`)
  - instruction: test: split examples/chat-smoke.jsonl 80/20 with seed 3407 and run sloth eval --suite <holdout> on the fake backend; it scores, not exits 1
  - honesty: sloth eval on a holdout split from a chat-schema training file scores every row (no schema rejection), and the overlap check catches a chat row whose final assistant content equals a task row's `expected_output`
- base-vs-adapter compare never holds both models resident: on the Spark's unified memory a 1.2B LoRA train already OOMs at backend init while lobes vLLM holds ~90 of 121 GB (eidetic record unsloth-cli-followups-22-livetest-gotchas), so sloth compare --base runs the base eval and the adapter eval as two sequential container invocations and an OOM in either maps to CliError(code=2) with the memory hint that `_trainer.py` already emits
  - instruction: fake-launcher test asserts two docker run invocations for compare --base; OOM path reuses the existing code-2 mapping
  - honesty: sloth compare --base issues two separate container runs (visible in the fake launcher call list) and a simulated CUDA OOM in the second exits 2 with the memory hint, leaving the first run's result file intact
- regression and perplexity deltas compare like with like: for a QLoRA adapter (`load_in_4bit`, config.py) the base run uses the same 4-bit quantised base, and for a LoRA adapter the bf16 base, and every result file records the base precision — otherwise dequantisation noise is counted as adapter regression
  - instruction: result file carries `base_load_in_4bit`; test that compare refuses (exit 1, hint) to diff two results with different base precision
  - honesty: sloth compare exits 1 with a hint when the base and adapter result files disagree on base precision, and docs/benchmarks.md states the precision on every delta row
- results carry `schema_version` and summarize/compare keep reading the legacy flat eval.json written by runs before this change, so older adapters on disk stay summarisable and comparable; a run directory holding both a legacy eval.json and eval/<suite>.json files reports the legacy file as suite 'legacy'
  - instruction: fixture run dir with only a flat eval.json: summarize --json still returns its `exact_match_pct`/f1; mixed dir returns both
  - honesty: sloth summarize and sloth compare on a run directory produced at a072b88 (flat eval.json only) exit 0 and report its metrics unchanged

## Honesty conditions

- every benchmark family in the announcement is measured by a real run on the DGX Spark and its numbers appear in docs/benchmarks.md; no family is announced on the strength of unit tests alone
- pyproject.toml \[project\].dependencies remains \[\] and uv tool install unsloth-cli on aarch64 without a GPU still succeeds
- uv pip list inside the container after the dep layer still shows transformers 4.57.1, peft 0.18.0, trl 0.24.0, datasets 4.8.5 and the container torch 2.10
- runs.jsonl records gain no eval fields; sloth runs show still reads only the registry and exports index
- the /finetune skill and colleague can consume the verdict from sloth compare --json alone (no text parsing), and docs/fine-tuning.md shows the agent path first
- one sloth eval invocation with several --suite values writes one result file per suite under the adapter dir, and sloth summarize --json lists every suite with its `batch_size` and metrics
- the before-state is reproducible from main at a072b88: eval.json holds only `exact_match`/f1 keys and a second eval overwrites the first
- the delivery doc shows at least one case where the new suites catch a degradation (regression, compliance or latency) that exact match + F1 alone would have missed
- every threshold in the success signal is checked by a real run on the #28 adapter and the measured values, with batch size, are recorded in docs/benchmarks.md; a missed threshold is reported as failing, not softened
- inside the container, adding `lm_eval`==0.4.13 and sacrebleu==2.6.0 to the dep layer leaves transformers 4.57.1 / peft 0.18.0 / trl 0.24.0 / datasets 4.8.5 and the nv torch 2.10 in place (uv pip list diff is additive only)
- sloth validate rejects a structured-output row whose schema uses a keyword outside the documented subset, and the subset is listed in explain eval

## Success signals

- on a multi-hundred-step adapter, sloth compare base adapter --json reports target-task exact match > 0 with a positive delta, regression-suite delta >= -2 percentage points, structured-output compliance >= 95%, per-item median latency within 10% of base, and held-out perplexity lower than the base model's; all numbers stated with batch size and reproduced in docs/benchmarks.md
  - instruction: run the documented command on the #28 adapter (or the fixture adapter) and diff the JSON against the thresholds in this claim

## Scope / boundaries

- \[project\].dependencies stays empty (pyproject.toml:16; CLAUDE.md 'GPU stack: NGC container, not a pip dep'): every benchmark library (lm-eval harness, sacrebleu, evaluate) lives only in container.py's in-container dep tuples, never as a host dependency
  - instruction: grep dependencies pyproject.toml; tests/`test_packaging_import_light.py` green
- the pinned window transformers==4.57.1 / peft==0.18.0 / trl==0.24.0 / datasets==4.8.5 is not floated (docs/dgx-spark.md:66-86 peft>=0.19 → torchao>0.16 → torch>=2.11 deadlock on NGC 25.11 torch 2.10); any harness whose resolver conflicts with it is installed --no-deps like unsloth/bitsandbytes, and its pin is measured live in the container, not guessed from PyPI
  - instruction: run the install line in docker and diff against docs/tested.md
- runs.jsonl (registry.py:99-139) stays a training-run index (`run_id`, `config_hash`, dataset digest, timing, status) — benchmark results live in the run's output directory beside `training_metadata.json`, discovered by summarize/compare, not copied into the registry line
  - instruction: tests/`test_runs_registry.py` RunRecord field list unchanged
- the stdlib structured-output checker implements a documented JSON Schema subset (type, required, properties, enum, items, additionalProperties) — not full Draft 2020-12; rows using unsupported keywords are rejected at validate time with a hint, never silently passed
  - instruction: test: a suite row using an unsupported keyword (e.g. $ref) fails sloth validate with exit 1 and a hint naming the keyword

## Non-goals

- no serving tok/s or TTFT benchmark is reimplemented in unsloth-cli; lobes-cli owns it (lobes benchmark / lobes assess, lobes/assess.py:569-598 `run_benchmark`, `_tool_probe` :288) — unsloth-cli at most calls it or documents the hand-off
- colleague's scripts/`bench_dual.py` is not reused: it times agent-loop wall clock and states 'QUALITY is NOT graded here' (`bench_dual.py`:19-21) — a different domain from model-output quality

## Assumptions

- lm-eval 0.4.13 resolves inside the pinned window on a host uv pip compile with constraints transformers==4.57.1 peft==0.18.0 trl==0.24.0 datasets==4.8.5 torch==2.10.0: it pins datasets==4.8.5, pulls evaluate==0.4.6, scikit-learn, sqlitedict (68 packages) and does not itself require transformers; sacrebleu 2.6.0 resolves to 7 packages (lxml, numpy, regex, portalocker, tabulate, colorama) — indicative only, the in-container install against the NGC torch build is the real test
  - instruction: run the exact install line in docker run --rm nvcr.io/nvidia/pytorch:25.11-py3 and uv pip list; record in docs/tested.md

## Scope exploration

- `s1` — `sloth/tune/metrics.py + summary.py + compare.py + eval.py (the exact/F1 reducers)`: one metric family only; `exact_match` + token-F1 are hard-coded in metrics.`score_records` (106-118), summary.`_eval_summary` (131-141), compare.`_EVAL_KEYS` (57) and eval.`_render_text` (306-312); no registry or --metric flag; a new metric today means editing all four
  - seeds: `c2`
- `s2` — `metrics.write_eval_json / summary.read_eval (eval.json storage convention)`: latest-only overwrite of a single eval.json per directory; no suite id, no batch size recorded; summarize/compare can only see one suite per run — issue #28 item 2 already asks for `batch_size` in eval.json
  - seeds: `c3`
- `s3` — `sloth/tune/_trainer.py _run_real (252-339) + sloth/tune/metadata.py`: no `eval_dataset` wired into SFTTrainer, no `eval_steps`, TrainOutput discarded, no loss in `training_metadata.json`; loss/perplexity is computed nowhere today
  - seeds: `c4`
- `s4` — `sloth/tune/datasets.py`: no split, no holdout fraction, no overlap detection between train rows and eval rows; train and eval share the task record shape so a (task,input,`expected_output`) hash check is straightforward and stdlib
  - seeds: `c5`
- `s5` — `sloth/tune/datasets.py TASK_KEYS + eval.py:172 schema='task' + _trainer.py:491-493 eval_prompt`: task schema is the only eval schema; chat-schema eval rows are not rendered or scored (`eval_prompt` and `score_records` hard-code task/input/`expected_output`); no constraint/schema/tool-call fields exist
  - seeds: `c6`
- `s6` — `sloth/cli/_commands/compare.py + eval.py _resolve_target`: compare diffs two runs' own eval.json only; no notion of 'the base this adapter started from'; base-model eval needs an exported dir; no threshold gate
  - seeds: `c7`
- `s7` — `sloth/tune/_trainer.py _generate_predictions (447-488) + lobes-cli lobes/assess.py run_benchmark (569-598)`: no timing or token counting in the eval generation loop; lobes already measures decode tok/s + prefill TTFT against a vLLM endpoint and lobes assess probes `tool_calls` shape live over HTTP — serving benchmark belongs to lobes, offline per-item latency belongs here
  - seeds: `c8`
- `s8` — `sloth/tune/container.py DEP_LAYER_PACKAGES / DEP_LAYER_NODEPS_PACKAGES + docker run args`: pins are module tuples with no config/env extension hook; container has outbound network and the HF cache mount; no offline flag (`HF_HUB_OFFLINE`) exists; adding a harness is a tuple edit + live pin validation
  - seeds: `c9`
- `s9` — `tests/test_lazy_import.py:174-198 + tests/test_packaging_import_light.py:78-92`: a top-level 'import sacrebleu' in metrics.py fails the stdlib-only AST test; heavy scorers must sit behind a lazy `_load_`\*() seam in another module
  - seeds: `c10`
- `s10` — `.claude/skills/finetune/scripts/finetune.sh + sloth/explain/catalog.py + sloth/cli/__init__.py`: loop runs exactly one eval suite and stops on first non-zero exit; catalog is hand-maintained (no auto-doc); registration seam is the marked spot at `__init__.py`:112-113
  - seeds: `c11`
- `s11` — `docs/benchmarks.md`: measures train throughput, export sizes, per-format exact/F1 with batch axis; no standard benchmark, no perplexity, no tok/s; caveat section says accuracy numbers are unvalidated (0/46)
  - seeds: `c12`
- `s12` — `tests/test_tune_trainer.py + tests/test_cmd_eval.py + tests/test_gpu_smoke.py`: fake-ML seams exist for both eval paths and a gpu marker for live runs; new metric code is unit-testable without a GPU by reusing them
  - seeds: `c13`
- `s13` — `pyproject.toml:16 + CLAUDE.md GPU-stack convention`: dependencies = \[\] is a merge-gating invariant; benchmark deps go in the container layer
  - seeds: `c14`
- `s14` — `docs/dgx-spark.md:46-131 (pin-bump procedure + torchao deadlock)`: new benchmark deps must satisfy the same live-validated pin discipline; a dep that pulls peft/transformers/torch recreates the deadlock
  - seeds: `c15`
- `s15` — `lobes-cli lobes/assess.py + lobes/cli/_commands/benchmark.py`: mature decode tok/s + prefill latency + live `tool_calls` probe against a vLLM endpoint exists; nothing offline/JSONL-based there
  - seeds: `c16`
- `s16` — `colleague scripts/bench_dual.py`: agent-loop wall-clock only, no model-quality metrics; not reusable
  - seeds: `c17`
- `s17` — `sloth/tune/registry.py + sloth/cli/_commands/runs.py`: registry holds no eval data; runs list/show never read eval.json; eval results reach agents only via summarize/compare
  - seeds: `c18`
- `s18` — `challenge pass / hidden-dependency lens: examples/*.toml dataset keys + examples/eval/*.jsonl + datasets.py schemas`: training data is chat-schema, eval suites are task-schema; a holdout split is unscoreable and the overlap check is schema-blind unless normalised
  - seeds: `c28`
- `s19` — `challenge pass / hidden-dependency lens: examples/chat-smoke.jsonl vs examples/eval/ (overlap probe)`: 0 overlapping rows; 1 row of the legacy examples/eval-suite.jsonl duplicates cli-contract.jsonl; the fixture adapter needs a corpus that does not exist yet
  - seeds: `q4` (question, resolved)
- `s20` — `challenge pass / cheap-probe lens: uv pip compile lm_eval + sacrebleu against the pin constraints (scratch)`: both resolve; harness adds evaluate + scikit-learn + sqlitedict; no transformers/torch pulled by the harness itself; host-side resolution, not the NGC build
  - seeds: `c29`
- `s21` — `challenge pass / failure-mode lens: container.py docker run assembly + eidetic UMA OOM record`: no code path loads two models today; a naive base+adapter compare would; sequential runs + existing OOM mapping contain it
  - seeds: `c30`
- `s22` — `challenge pass / unstated-assumption lens: sloth/tune/config.py load_in_4bit + the QLoRA path in _trainer.py`: the 'base model' of a QLoRA adapter is the 4-bit base; the spec said 'untouched base' without fixing precision
  - seeds: `c31`
- `s23` — `challenge pass / unstated-assumption lens: _trainer.py _generate_predictions batching (447-488) + issue #28 r8`: per-item latency inside a batch is not measurable; defined as batch wall time / rows with `batch_size` recorded
  - seeds: `c8`
- `s24` — `challenge pass / adjacent-systems lens: lobes/assess.py _tool_probe + model chat templates (Qwen3, LFM2.5)`: lobes checks OpenAI `tool_calls` shape over HTTP; offline scoring of raw generations must pick a wire format; the spec left it open
  - seeds: `q5` (question, resolved)
- `s25` — `challenge pass / security-and-correctness lens: stdlib-only scoring in metrics.py (no jsonschema library allowed)`: full JSON Schema is not implementable in stdlib within scope; an explicit subset with fail-closed validation bounds it
  - seeds: `c32`
- `s26` — `challenge pass / migration-and-reversibility lens: summary.read_eval (107-128) + existing run dirs on disk`: no versioning today; older run dirs would become unreadable or silently empty; backward read is the spec-side containment
  - seeds: `c33`
- `s27` — `challenge pass / missing-counter-evidence lens: examples/eval/ sizes (16+16+14 rows) vs the c23 thresholds`: thresholds are tighter than the sampling noise of the shipped suites; a size or confidence rule is missing
  - seeds: `q6` (question, resolved)
- `s28` — `challenge pass / overlooked-actors lens: lobes serving hand-off (docs/benchmarks.md serving section, lobes benchmark)`: the serving tok/s figure needs the exported model served by lobes first — a manual operator step the docs must state; no code finding
  - seeds: `c16`
- `s29` — `challenge pass / observability lens: in-container result separation (previous frame's park on stdout sentinel vs result file) + container.py mounted output dir`: new suite result files travel the same mounted-dir path as eval.json today; latency is measured in-container and written into the same file; clean pass, residual risk only if the stdout-sentinel option is chosen for results
  - seeds: `c3`
- `s30` — `challenge pass / security lens: instruction-following must-refuse rows + the --in-container recursion guard (eval.py:494-499)`: refusal scoring by substring/regex is brittle but stdlib; no new privilege or network surface added by scoring; the recursion guard is unchanged; clean pass

## Decisions

- GLEU in the request means the BLEU-style sentence-level scorer (GLEU/BLEU on generated vs expected text), not the GLUE task suite; it is delivered as a lazy-imported scorer (c10) beside exact match and F1, not as a benchmark suite
  - instruction: the scorer reports a gleu (and bleu) key per row and per suite; sacrebleu is imported inside the function; `test_metrics_module_imports_only_stdlib` stays green
- MMLU ships two ways: lm-evaluation-harness pinned in the container dep layer for numbers comparable to published MMLU, and a small vendored MMLU-subset JSONL under examples/eval/ scored by the existing suite path as a fast, dependency-free smoke
  - instruction: harness: docs/benchmarks.md MMLU row cites the `lm_eval` command + pin; subset: examples/eval/mmlu-subset.jsonl scores via sloth eval --suite with letter-choice extraction
- issue #28 items 1 and 2 are folded into this frame: this work trains the multi-hundred-step adapter (LFM2.5-1.2B via examples/lfm2-lora.toml or Qwen3-1.7B) that all benchmark tables are measured on, re-scores the per-format quantization-loss table with it, records `batch_size` in eval results, and settles the batched-vs-serial question; #28 item 3 (spec provenance nits) stays on the follow-ups frame
  - instruction: docs/benchmarks.md per-format table has at least one non-zero exact-match row; eval results carry `batch_size`; the batched-vs-serial outcome is stated in docs/benchmarks.md
- four new suite files ship under examples/eval/: regression (fixed general-capability rows scored on base and adapter alike), instruction-following (constraint rows), structured-output (JSON-schema / tool-call rows) — these three hand-authored and small — plus an MMLU-subset JSONL; target-task, holdout, perplexity, latency and GLEU reuse the existing task rows or the training split and need no new file
  - instruction: ls examples/eval/ shows regression.jsonl, instruction-following.jsonl, structured-output.jsonl, mmlu-subset.jsonl; each validates with sloth validate and scores with sloth eval --suite
- the training corpus is chosen per fine-tuning task: the repo ships a demo dataset (a wiki-derived corpus is an acceptable source) that the fixture adapter and benchmark tables are measured on, and train/eval accept an external dataset option (a Hugging Face Hub dataset id or a local path outside the repo) for real tasks
  - instruction: sloth train --dataset <hf-id-or-path> resolves external corpora through the mounted HF cache; examples/ ships the demo corpus with its provenance
- structured-output/tool-call scoring parses each model family's native tool-call syntax through a per-family parser registry keyed by model family (detected from the model id / chat template); an unknown family is rejected at validate time with a hint listing supported families, never scored with a guess
  - instruction: parsers live in a stdlib module with one test fixture per family; adding a family is one parser + one fixture
- pass/fail thresholds and the minimum suite size are configurable per run through an \[eval.thresholds\] section of the TOML run-config, with a documented default baseline (regression drop 2 pp, structured-output compliance 95%, latency within 10% of base, minimum suite size to be set in the baseline) that operators adjust; sloth compare reports which baseline it applied
  - instruction: compare --json carries the thresholds it applied; a TOML override changes the verdict in a test; explain config lists the keys and defaults

## Hard questions

- which tool-call wire format does the structured-output scorer parse: OpenAI-style JSON extracted from the response text (model-agnostic, what lobes assess checks), or each model's chat-template-native syntax (Qwen <`tool_call`> JSON vs LFM2.5 tool-call tokens), which is model-specific and needs a per-family parser? (resolved: each model family gets its own tool-call parser (Qwen3 <`tool_call`> JSON, LFM2.5 tool-call tokens, ...); scoring is template-native, not OpenAI-JSON-only)
- MMLU delivery: lm-evaluation-harness inside the container (a new heavy dep needing a live-validated pin and Hub access on first run) or a small vendored MMLU-subset JSONL scored by the existing exact-match path with letter-choice extraction (no new dep, but not comparable to published numbers)? (resolved: both: lm-evaluation-harness in the container for published-comparable MMLU, plus a vendored MMLU-subset JSONL scored by the existing path as a GPU-cheap smoke)
- the request says 'GLEU': is that GLEU (the sentence-level BLEU variant, needs sacrebleu/nltk lazy-imported) or GLUE (the classification task suite, which would ride the same harness as MMLU)? (resolved: GLEU = the BLEU-style sentence-level scorer (not the GLUE task suite); lands as a lazy-imported scorer beside exact/F1)
- does this work depend on issue #28 item 1 (a multi-hundred-step adapter so scores are non-zero) landing first, or do the new suites bring their own fixtures so the metrics can be proven to discriminate independently of that adapter? (resolved: fold issue #28 items 1 and 2 into this frame: train the multi-hundred-step adapter here and use it as the fixture; `batch_size` in eval.json lands via c3)
- the success signal's 2 percentage-point regression threshold equals one row on a 46-row suite: what minimum suite size (e.g. >= 100 rows) or confidence rule do the regression and target-task suites need before a delta counts as pass/fail rather than noise? (resolved: thresholds are configurable: a documented baseline (e.g. 2 pp regression drop, 95% compliance, 10% latency) that an \[eval.thresholds\] TOML section can adjust per run; minimum suite size is part of the same baseline)
- no training corpus in the repo covers the eval suites: a probe found zero overlap between examples/chat-smoke.jsonl (10 rows) and the 46 rows under examples/eval/, so the #28 adapter cannot learn the eval answers from anything shipped — which corpus trains it: an authored synthetic corpus generated from the AgentCulture docs/CLI contract that the eval suites test, or an external dataset? (resolved: corpus is task-dependent: ship a demo dataset (and/or a wiki-derived one) for the fixture adapter, and add an external-dataset option so a real fine-tuning task can bring its own corpus)

## Open parks

- [unknown_nonblocking] eval.json shape change: nesting results per suite breaks agents that read summary\['eval'\]\['f1'\] as a flat float today (summarize.py:45-50, compare.py:60-81); additive nesting vs a versioned break is a /think decision
- [unknown_nonblocking] no offline mode exists (no `HF_HUB_OFFLINE`/--network none in container.py); whether benchmark runs must be reproducible without network after the first cache fill is undecided
- [unknown_nonblocking] batched (8) vs serial (1) decoding is not bit-identical (F1 0.038 vs 0.060, issue #28 risk r8); any latency/tok-s and accuracy number must state its batch size until that is resolved
- [unknown_nonblocking] whether lm-eval 0.4.13's hf backend loads a QLoRA adapter the way sloth does (peft= model arg plus bitsandbytes 4-bit base) or needs the merged-16bit export as its input; decides whether MMLU is scored on the adapter or on the export
- [unknown_nonblocking] two concurrent sloth eval runs of the same suite on the same adapter dir would race on eval/<suite>.json; single-writer today, no locking; acceptable until the mesh runs evals in parallel
