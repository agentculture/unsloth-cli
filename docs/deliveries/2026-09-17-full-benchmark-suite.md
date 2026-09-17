# Delivery Summary — full benchmark suite

plan: `full-benchmark-suite` · run: `complete` (with recorded failures) · date: `2026-09-17`
baseline: `devague summary` skeleton (plan `full-benchmark-suite`, 16 tasks, 6 waves)

## Intent

> unsloth-cli ships a full benchmark suite: sloth eval scores target-task
> accuracy, holdout generalization, a regression set against the base model,
> instruction following, structured-output/tool-call compliance, held-out
> perplexity/loss, per-item latency + tok/s, and standard benchmarks (MMLU),
> each suite recorded side by side and summarized/compared across runs

The frame (`docs/specs/2026-09-16-full-benchmark-suite.md`, 17 scope entries,
a rigorous `/challenge` pass) folded issue #28 items 1 and 2 into this work by
user decision `c26`. The plan (`docs/plans/2026-09-17-full-benchmark-suite.md`)
was executed via `/assign-to-workforce` on the integration branch
`feat/full-benchmark-suite`: 15 code/doc tasks fanned out to task agents in
isolated worktrees (sonnet for well-scoped modules, opus for the trainer,
compare, bench and corpus tasks), t12 and t16 done by the main agent because
they need the GPU box. The run moves the package from 0.8.2 to 0.9.0.

## Planned Work

Quoted from the `devague summary` skeleton:

- `t1` — metrics core: open metrics dict, `schema_version`, suite-keyed result files
- `t2` — datasets core: seeded holdout split, cross-schema overlap check, new suite schemas + JSON-Schema-subset validation
- `t3` — config: `[eval]` section (`holdout_fraction`, seed, `eval_steps`, perplexity) and `[eval.thresholds]` baseline
- `t4` — scorers module: instruction constraints, JSON-schema-subset checker, per-family tool-call parser registry, lazy GLEU/BLEU
- `t5` — summary + summarize: read every `eval/<suite>.json` plus legacy `eval.json`, open metric pass-through, per-suite text render
- `t6` — eval loop: latency + token counts, chat-schema rendering, perplexity forward pass, base precision recorded (adapter and `--model` paths)
- `t7` — training-time eval: `eval_dataset` + `eval_steps` into SFTTrainer, loss history into `training_metadata.json`
- `t8` — eval CLI: repeated `--suite` by name, `--perplexity`, `--batch-size` in results, overlap refusal before container launch, tool-call family flag
- `t9` — compare `--base`: sequential base and adapter container runs, per-suite deltas, thresholds from config, precision guard, exit codes
- `t10` — suites + demo corpus: regression, instruction-following, structured-output, tool-call suites and the demo training corpus
- `t11` — external dataset option: train/eval accept a Hugging Face Hub dataset id or a path outside the repo
- `t12` — container dep layer: add `lm_eval` + sacrebleu pins, live-validate against NGC 25.11, record in `docs/tested.md` + `docs/dgx-spark.md`
- `t13` — bench verb: MMLU via `lm_eval` in the container plus the vendored MMLU-subset suite
- `t14` — finetune skill loop, catalog, rubric gate and CPU-only test pass
- `t15` — docs: fine-tuning.md agent-first eval path, README fine-tune vs retrieval boundary, serving hand-off to lobes
- `t16` — live run on the Spark: train the fixture adapter, score every suite and MMLU, base-vs-adapter compare, fill `docs/benchmarks.md`, delivery doc, close #28 items 1-2

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `metrics.score_records(extra_metrics=)`, generic numeric fold in `summarize`/`aggregate`, suite-keyed `write_eval_json(dir, suite, payload, batch_size=…)` → `eval/<suite>.json` (schema_version 2); legacy shape kept (+28 tests) |
| `t2` | delivered | `split_holdout`, `overlap_check`, three new schemas (`instruction`, `structured`, `toolcall`) with a fail-closed JSON-Schema subset (+36 tests); the `schema` key it returns is the *source* schema (r10) |
| `t3` | delivered | `EvalConfig` / `ThresholdsConfig` (c36 baseline), `[eval]`/`[eval.thresholds]` parsing, config hash unchanged for configs without `[eval]` (+42 tests) |
| `t4` | delivered | `sloth/tune/scorers.py`: constraints, JSON-subset checker, `qwen3` + `lfm2` tool-call parsers (tokens read from the cached chat templates), `score_tool_call`, lazy `gleu`/`bleu` — sacrebleu has no GLEU, so Google-GLEU is computed in-process (r6) (+61 tests) |
| `t5` | delivered | `read_eval` → `{suite: payload}` incl. `legacy`; `summary["eval"]` gains `suites` and keeps the flat keys (frame park v1 resolved additively) (+8 tests) |
| `t6` | delivered | per-row `generated_tokens`/`latency_ms`, suite-level `median_latency_ms`/`tokens_per_s`/`compliance_pct`/`perplexity`, chat rendering, `run_perplexity` (labelled forward pass), `base_load_in_4bit` (+24 tests) |
| `t7` | delivered | holdout split before the model load, `eval_dataset`/`eval_strategy`/`eval_steps` into SFTTrainer, `loss_history` + `final_*_loss` + `holdout` in metadata; `hf:` datasets skip the split with a diagnostic (+14 tests) |
| `t8` | delivered | repeated named `--suite`, `--perplexity`, `--tool-call-family`, `--train-dataset` overlap refusal (zero container calls), `{"suites": {…}}` JSON (+22 tests); main-agent integration fix: split the seam's per-file entries by suite instead of duplicating the aggregate |
| `t9` | delivered | `compare --base` (two sequential runs, per-suite generic deltas, thresholds echo, verdict, exit matrix), `--results-dir` on eval, HF-id `--model` on the host (+29 tests); two live-found defects fixed by the main agent (in-container HF-id rejection; root-owned `eval-base/` → vacuous pass) |
| `t10` | delivered | `regression.jsonl` (116), `instruction-following.jsonl` (56, 12 `must_refuse`), `structured-output.jsonl` (55), `tool-call.jsonl` (32), `demo-corpus.jsonl` (591 chat rows), `examples/generate_suites.py`, `examples/README.md` (+11 tests); triggered deviation `d1` |
| `t11` | partial | `hf:<org>/<name>[:split]` datasets with `[run.dataset_map]`, cache-only `sloth validate`, hf metadata shape (+53 tests); **the read-only mount for external local paths was not implemented** (needs `container.py`, r9) |
| `t12` | delivered | `DEP_LAYER_BENCH_PACKAGES = (lm_eval==0.4.13, sacrebleu==2.6.0)` installed after the `--no-deps` layer; live-measured in-container: 45 packages added, none of torch/transformers/peft/trl/datasets changed; rows in `docs/tested.md`, bump procedure in `docs/dgx-spark.md` (+4 tests) |
| `t13` | delivered | `sloth bench` + `sloth/tune/_bench.py` (lazy `lm_eval`), `eval/mmlu.json` in the shared result shape, `examples/eval/mmlu-subset.jsonl` (120 **MMLU-style** rows, authored — `cais/mmlu` was not cached), `extract_choice_letter` + `choice_acc_pct` (+61 tests); main-agent fix: the `--model` path now scores letter-choice suites too |
| `t14` | delivered | `finetune.sh run`: repeated `--suite`, `--batch-size`, `--perplexity`, `--base` (step 5 compare); catalog covers every new flag and `[eval]`/`[eval.thresholds]`/`[run.dataset_map]` key; 1053 tests, coverage 93.5 %, only `test_gpu_smoke.py` gpu-marked (+4 tests) |
| `t15` | delivered | `docs/fine-tuning.md` agent path first (eval `--json` → compare `--base --json` verdict, exit codes), `docs/benchmarks.md` serving section citing `lobes benchmark` and `run_benchmark`'s output keys, README "which suite catches which regression" |
| `t16` | delivered with recorded failures | fixture adapter (`examples/demo-lora.toml`, 300 steps, 91 s train, holdout `eval_loss` 2.11); nine suites at batch 8 **and** batch 1; MMLU bench cold and `--offline`; base-vs-adapter compare (base numbers recovered from the captured run; verification re-run below); per-format quantization-loss table with non-zero rows (closes #28 item 1); `batch_size` recorded in every result and the batched-vs-serial question answered (closes #28 item 2). The c23 success signal is **partially met and partially failing** — see Delivery Claims |

## Mid-work Decisions

- `d1` (approved) — `sloth validate --suite` detects each file's schema instead
  of forcing `task`; `datasets.detect_schema` became the single five-way
  detector and the trainer/eval CLI delegate to it. Reason: t10 put the
  new-schema suites under `examples/eval/` per its acceptance criteria and no
  task owned `validate.py`, so a documented command regressed on the branch.
- Live-run scheduling: the fixture training was started while waves 4–5 (docs)
  were still running, because it depends on no docs task; all measurements were
  taken on the final merged code except the batch-8 eval, which predates the
  d1 follow-through refactor (no scoring code changed by it).
- Batch 8 → batch 1: after the batch-8 run showed degenerate LFM2 output, the
  remaining measurement chain was stopped and re-run at batch 1 (r13). No
  threshold was changed.
- Base numbers for the compare table are taken from the second compare run's
  captured container stdout because that run could not write `eval-base/`
  (r15); the fix landed in the same PR and a verification run was queued.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t10` (`d1`) | new-schema suites under `examples/eval/` broke the task-only `validate --suite`; no task owned the fix | `needs-follow-up` → resolved in-PR |
| `t11` | read-only mount needs `container.py`, off-limits to the task | `needs-follow-up` (r9) |
| `t5`/`t8`/`t9`/`t13` integration | concurrent tasks agreed on shapes only through their briefs; four seams needed main-agent fixes after merge (per-suite split, letter-choice on `--model`, remote HF id in the seam, root-owned results dir) | `acceptable` — all caught by the live run or the TDD gate, all fixed with tests |
| `t16` | batch-8 numbers invalid on LFM2; compare's base side vacuous on first success | `needs-follow-up` (r13, r15, r16) — measured, documented, not smoothed |

## Evidence

- tests: `uv run pytest -n auto -q` — **1059 passed, 1 skipped** (gpu smoke, no CUDA in the host interpreter); coverage 93 % (gate 60)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r sloth` — clean; `uv run teken cli doctor . --strict` — healthy; `markdownlint-cli2` on every touched doc — 0 errors
- commits: `281626e` (spec + plan) .. `ab237b3` (compare fix), 40 commits on `feat/full-benchmark-suite`, one `merge:` per task with tests run before and after each merge
- live rows: `docs/tested.md` (2026-09-17 section), numbers in `docs/benchmarks.md` "Full benchmark suite — fixture adapter"
- frame/plan state: `.devague/frames/full-benchmark-suite.json`, `.devague/plans/full-benchmark-suite.json` (risks r1–r16, lapses l1–l5, deviation d1)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| Every suite family in the announcement is measured by a real Spark run (h1) | high | `docs/benchmarks.md` fixture-adapter section: train, nine suites at batch 8 and 1, bench cold + offline, compare, four export formats — every table cites its command |
| Target-task exact match > 0 with a positive delta over the base (c23) | high, **met** | batch 1: 0/46 (base) → 14/46 (adapter); `cli-contract` 5/14, `task-format` 9/16; `agentculture-terms` stays 0/16 exact (+0.23 F1) because the corpus teaches full-sentence answers (r3) |
| Regression-suite delta ≥ −2 pp (c23) | high, **met on the gated metric, failing on the ungated one** | `regression.jsonl` exact 0 → 11.2 pp, F1 +0.53; but MMLU-style letter accuracy **75.8 % → 61.7 %** (−14 pp), which the gate does not cover (r16) |
| Structured-output compliance ≥ 95 % (c23) | high, **failing** | 76.4 % (base 20 %); instruction-following 46.4 % (base 23 %); tool-call 0 % on a Base checkpoint (expected) |
| Median latency within 10 % of base (c23) | high, **met** | adapter 1.0–2.7 s per row vs base 6.2–7.1 s (base echoes to the token budget) — batch 1 |
| Held-out perplexity lower than the base's (c23) | **unverified** | adapter holdout perplexity 1.51 measured; `compare --base` does not forward `--perplexity`, so no base perplexity exists (follow-up) |
| Minimum suite size (thresholds baseline 100 rows) | high, **failing** | 7 of 9 suites have < 100 rows (the three original target suites have 14–16) |
| The new suites catch a degradation exact/F1 alone would miss (h20) | high | the 14 pp MMLU-style drop; the batch-8 degeneration visible as 2.5–3.5 s/row latency and 0 compliance; structured/instruction compliance as a number exact match cannot give |
| Batched vs serial (#28 item 2) | high | not "≤ X different": **batch 8 is invalid on LFM2** (r13); `batch_size` is recorded in every result file |
| Quantization loss on a trained adapter (#28 item 1) | high | merged-16bit 14/46 (lossless), AWQ 12/46, NVFP4 9/46, GGUF Q4_K_M 0/46 (r14) |
| MMLU through lm-evaluation-harness in the container (c9/h9) | high | acc 0.5789 (5-shot, 285 docs), PEFT adapter loaded directly (park v4 resolved for bf16 LoRA), offline re-run identical with no download |
| Every threshold checked and a missed one reported as failing (h21) | high | this table; nothing tuned |

Lapse ledger evidence (filed the moment each was reported; adjudication is the
user's):

| Lapse | Code | What |
|-------|------|------|
| `l1` (approved) | `assumption-for-measurement` | host-side `uv pip compile` stood in for the in-container dep probe; superseded by t12's live install |
| `l2` | `grader-unverified` | t2's first split test was true by construction (read after overwrite); caught by its own failure |
| `l3` | `grader-unverified` | t9 wrote the implementation before its tests |
| `l4` | `control-absent` | t13 generated the MMLU-style subset before checking its answer distribution (61/120 keyed "B") |
| `l5` | `grader-unverified` | t13's first lazy-import guard used `ast.walk` and its first scorer test used the internal constraint shape |

## Remaining Work

- Verification `compare --base` run with the r15 fix — queued after the offline
  bench; its outcome is appended below when it completes.
- r13: per-family padding guard in `_generate_predictions` (default batch 1 for
  `lfm2`, or right-padding with a position-id fix); until then every LFM2 number
  is batch 1.
- r16: gate `choice_acc_pct`/`compliance_pct` on regression-tagged suites and
  tag `mmlu-subset` by default; forward `--perplexity` through `compare --base`.
- r14: diagnose the GGUF Q4_K_M 0/46 (llama.cpp prompt path vs the quantisation).
- r12: the overlap check should use `holdout.train_path` from metadata when the
  run split its own holdout.
- r10: keep chat rows on the train side of `split_holdout` (only render the
  holdout side) so a chat corpus keeps its chat template under `[eval]`.
- r9: read-only mount for external dataset paths (`container.py`).
- r3: a corpus whose answers match the suites' terse style, or suites that
  accept the corpus's sentence style — `agentculture-terms` exact stays 0
  either way until one moves.
- Frame park v4 (QLoRA through `sloth bench`) and v2 (offline mode) remain
  open parks; v4 is resolved for bf16 LoRA only.
- Tool-call compliance on an Instruct base; production-scale models on a free
  box (unchanged from the #22 caveats).

- unsloth-cli (Claude)
