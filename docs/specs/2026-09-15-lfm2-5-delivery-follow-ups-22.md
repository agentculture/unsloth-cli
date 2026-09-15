# lfm2-5 delivery follow-ups (#22)

> unsloth-cli closes the nine follow-ups left open by the LFM2.5 `target_modules` + quantized export delivery (issue #22): JSON-only stdout on real train/eval/export runs, a /finetune resolver that runs the checkout it lives in, QLoRA and Qwen3 measured through every export format, a Jetson-side load, a real quantization-loss number, pinned unsloth versions with a documented bump path, and the torch shim scheduled for deletion
> instruction: gh issue view 22 --comments after the last PR merges

## Audience

- Agents and operators who drive sloth train/eval/export and the /finetune skill non-interactively and parse stdout as JSON; secondarily lobes/colleague as consumers of exported adapters, and the Jetson deployment targets named in docs/fine-tuning.md
  - instruction: read docs/specs/<slug>.md audience + requirements

## Before → After

- Before: After #21, sloth train/eval/export --json stdout carried 3 JSON lines among 125 non-JSON lines on a real run (docs/tested.md:73); the /finetune skill ran another worktree's sloth from PATH (risk r9); every quantized export was judged by one coherent completion and a 0/4 smoke suite (lapse l1); QLoRA adapters, Qwen3, and any Jetson load were unmeasured (docs/tested.md:75-77); unsloth floated between 2026.6.9 and 2026.9.4 (risk r1)
  - instruction: grep each number in those files
- After: A real /finetune run on the Spark yields exactly three JSON lines on stdout and everything else on stderr; the skill runs the checkout it lives in; docs/tested.md has rows for a QLoRA adapter and for Qwen3-1.7B through merged-16bit/gguf/awq/nvfp4, one Jetson-side load, a side-by-side eval score per format, and pinned unsloth versions with a bump path; issue #22 closes with every item either done or explicitly deferred
  - instruction: gh issue view 22 --comments; the tested.md stdout row

## Why it matters

- The CLI's whole value to an agent is a parseable stdout; while train/eval/export --json leak container output, every downstream skill has to grep for JSON, and while quantization loss is unmeasured, no adapter can honestly be handed to lobes or a Jetson as 'good enough'
  - instruction: read README; read benchmarks table; devague lapse list shows l1 superseded or annotated

## Requirements

- sloth train/eval/export --json emit only the in-container result on host stdout on a real run — container.launch routes the container's whole stdout to the host's stderr and captures the result via a JSON-aware path; today sloth/tune/container.py:475-481 `_stream`() is a bare subprocess.run with inherited stdio and train.py:448 / eval.py:200 / export.py:901 consume only the exit code
  - instruction: Implement in container.launch(): capture stdout, tee every line to sys.stderr, keep the last line that json.loads as the result (or read a result file from the mounted output dir — see park v5), return it to the caller; train.py/eval.py/export.py emit that result via `emit_result`
  - honesty: With the fix in place, docker's stdout is never connected to the host's stdout: a tests/`test_tune_container.py` test asserts that launch() passes stdout=PIPE (or a file) to subprocess and that only the parsed in-container result is written via `emit_result`
- README.md:41-43 and :95 promise 'results go to stdout, errors/diagnostics to stderr (never mixed)' for every verb unconditionally; until item 1 ships, docs/tested.md:73 is the only place that says train/eval/export --json is not stdout-only on a real run — the README promise becomes true (or is caveated) in the same change
  - instruction: Ship the README change in the same PR as the container fix
  - honesty: README.md's stdout/stderr promise at :41-43 and :95 is either true on a real run (verified by the c4 row) or carries an explicit caveat line naming the gap; never both silent and false
- .claude/skills/finetune/SKILL.md:28-29 and :178-179 document 'installed sloth on PATH, else uv run sloth from a checkout' — the docs change in the same PR as the resolver so the stated order matches the shipped order
  - instruction: Edit SKILL.md:28-29 and :176-181 in the resolver PR
  - honesty: SKILL.md's stated resolution order matches `resolve_sloth`'s code order line for line, and markdownlint passes on SKILL.md
- QLoRA-trained adapters are measured through the export formats on the Spark: examples/qlora-smoke.toml → sloth export --format merged-16bit first (the cheapest probe of Unsloth's dequantise-then-merge path), then gguf/awq/nvfp4 only if merged-16bit holds; results become docs/tested.md rows and resolve risk r3
  - instruction: uv run sloth train --config examples/qlora-smoke.toml then uv run sloth export --adapter <run> --format merged-16bit; if it passes, also try gguf --quant `q4_k_m`
  - honesty: docs/tested.md gets a row for examples/qlora-smoke.toml → sloth export --format merged-16bit stating pass or fail with the error text; a fail is recorded, not hidden
- Qwen3 is measured through awq, nvfp4 and gguf on the Spark — one run each on unsloth/Qwen3-1.7B from the existing lora-smoke/qlora-smoke adapters — and the outcomes become docs/tested.md rows, resolving risk r4
  - instruction: Use the existing lora-smoke adapter; run awq first since it is the mapping-sensitive one
  - honesty: docs/tested.md gets one row each for unsloth/Qwen3-1.7B → awq, nvfp4, gguf, each stating pass/fail and, for awq, whether a custom mapping was needed
- At least one exported artifact is loaded on a Jetson device and recorded as a Live-tested cell in the docs/fine-tuning.md:157-161 deployment table and a docs/tested.md row: awq/nvfp4 on the Thor via vLLM in this batch; gguf on the Orin Nano via llama.cpp/Ollama in a separate session (issue #23)
  - instruction: Blocked on q2; if no device, resolve by deferring (park) and say so in the issue
  - honesty: The Live-tested cell for at least one of the Orin / Thor rows in docs/fine-tuning.md:157-161 names a date and a device instead of 'not yet', backed by a docs/tested.md row with the load command
- Quantization loss becomes a number: sloth eval --model is run on merged-16bit, gguf, awq and nvfp4 outputs of the same adapter against the examples/eval/ starter suite (>= 40 task-schema items across three domain files, scored per file and in aggregate on exact-match and token-level F1), and the side-by-side scores land in docs/benchmarks.md; today examples/eval-suite.jsonl is 4 items with exact-match only
  - instruction: Blocked on q3 for suite source and metric; then run sloth eval --model per export dir
  - honesty: docs/benchmarks.md has a table with one row per export format (merged-16bit, gguf, awq, nvfp4) and the adapter baseline, all scored on the same suite of >= 20 items, and at least one row is not 0
- sloth eval gains a --quant <name> selector so --model <dir> holding several GGUFs picks the matching file instead of exiting 1; today sloth/tune/`_exporter.py`:850-865 `_find_gguf` raises 'holds several GGUF files' with the hint to pass --model <dir>/<name>.gguf
  - instruction: Attach in `_resolve_target` / `_find_gguf`; catalog entry for eval must document --quant
  - honesty: sloth eval --model <dir> --quant `q4_k_m` on a dir holding `q4_k_m` and `q8_0` GGUFs scores the `q4_k_m` file; --quant naming a missing quant exits 1 with a hint listing the present ones; a unit test covers both
- sloth/tune/container.py:107-111 `DEP_LAYER_NODEPS_PACKAGES` pins unsloth, `unsloth_zoo` and bitsandbytes to the versions docs/tested.md:51-52 measured live on 2026-09-15 (unsloth 2026.9.4, `unsloth_zoo` 2026.9.3; bitsandbytes at whatever that run installed), tests/`test_tune_container.py`:762-778 asserts the pins reach the install line, and docs/dgx-spark.md:52-63 gains a 'bump path' paragraph next to the version matrix
  - instruction: Read the live versions first (park v3), then pin, then re-run one export to prove the pinned layer resolves
  - honesty: The pinned unsloth / `unsloth_zoo` / bitsandbytes versions equal the versions printed by uv pip list inside the container on the run that produced the new docs/tested.md rows
- Two #20-review fixes get one live run each on the Spark and a docs/tested.md row: sloth eval --adapter runs/lfm2-lora --suite examples/eval-suite.jsonl (does continuation-only scoring move the smoke score off 0/4?) and one sloth export --base <id> (does the staged `_adapter`-override copy load?)
  - instruction: Cheapest items; run first on the Spark to confirm the box works before the longer probes
  - honesty: docs/tested.md gets one row for sloth eval --adapter runs/lfm2-lora --suite examples/eval-suite.jsonl with the `exact_match` count, and one row for sloth export --base <id> stating whether the staged copy loaded
- The continuation-only unit tests are tightened alongside the live check: the fake tokenizer's decode() honours its tokens argument so the `prompt_len` slice is asserted, not just the result shape
  - instruction: uv run pytest tests/`test_tune_trainer.py` -k continuation
  - honesty: The fake tokenizer in tests/`test_tune_trainer.py` TestRunEval decodes its actual token slice, and a test asserts that a prediction never begins with the prompt text
- sloth eval --suite accepts either one JSONL file or a directory; for a directory every \*.jsonl is scored and the result carries per-file and aggregate {total, `exact_match`, `exact_match_pct`, f1} plus the per-item results; a token-level F1 is computed in pure stdlib (whitespace tokens, normalized) next to exact-match for both --adapter and --model
  - instruction: tests/`test_cmd_eval.py` + tests/`test_tune_trainer.py`
  - honesty: sloth eval --suite examples/eval/ on the fake backend returns one entry per file plus an aggregate, each with `exact_match` and f1 fields; an F1 unit test scores ('a b c','a b d') as 0.667 to three decimals
- `run_eval` and `run_eval_model` generate in batches (padding-side left, batch size from config or a --batch-size flag defaulting to 8) so a wide suite does not scale one prompt per forward pass; continuation-only slicing stays correct per row under padding
  - instruction: tests/`test_tune_trainer.py`; one live timing row
  - honesty: A fake-backend test feeds 3 prompts of different lengths in one batch and asserts each prediction excludes its own prompt; docs/tested.md records wall-clock for the 40-item suite batched vs the old per-prompt path
- If the container exits 0 but no line of its captured stdout parses as JSON (or, in text mode, no result marker is found), container.launch raises CliError(code=2) whose message carries the last 20 captured lines — a missing result is never reported as success
  - instruction: unit test
  - honesty: A monkeypatched subprocess yielding 40 non-JSON lines and exit 0 makes launch() raise CliError(code=2) whose message contains the 40th line
- The capture path streams the container's stdout line by line to the host's stderr as it arrives (subprocess.Popen + iteration, not run() with `capture_output`), so operators keep live progress; the existing exit-code mapping (1 → user, 2/137/other → environment, 127 → docker missing) and the OSError branch at container.py:475-481 are preserved unchanged
  - instruction: run the existing tests + one streaming test
  - honesty: tests/`test_tune_container.py`'s exit-code mapping tests pass unmodified after the change, and a new test asserts stderr receives the first container line before the process ends
- Suite validation happens before GPU spend, matching the dataset rule: sloth eval validates every \*.jsonl in a --suite directory host-side (schema task, non-empty) before launching the container, and sloth validate gains --suite <file|dir> reusing the same function; the /finetune skill's --suite help, .claude/skills/finetune/SKILL.md, the explain catalog's eval entry (--suite dir, --quant, --batch-size) and .claude/skills/unsloth-cli-guide/SKILL.md are updated in the same PR, and uv run teken cli doctor . --strict stays green
  - instruction: tests/`test_cmd_eval.py` + tests/`test_cmd_validate.py` + the rubric gate in CI
  - honesty: sloth eval --suite <dir with one malformed file> exits 1 naming the file and line before any container launch (fake launcher asserts it was never called); sloth explain eval mentions --quant and --batch-size; teken cli doctor --strict passes

## Honesty conditions

- Issue #22 is closed with a comment mapping each of its nine items to a docs/tested.md row, a merged PR, or a named deferral issue (#23 for the Orin Nano load); no item is left unmentioned
- No handler under sloth/cli/`_commands`/ gains its own subprocess or stdout-parsing code for the container; grep -l 'subprocess' sloth/cli/`_commands`/{train,eval,export}.py stays empty
- docs/tested.md records the stdout-line count of the /finetune --json loop on the Spark after the fix, with the command used
- The old frame lfm2-5-target-modules-quantized-export is not edited; h14's satisfaction is recorded as evidence on this frame's plan (devague evidence) pointing at the docs/tested.md row
- Running the skill from inside a checkout that is NOT the PATH-installed one executes that checkout's sloth: verified by a test that installs a fake sloth on PATH, sets the override (or checkout-first mode), and asserts the checkout's uv run --project form is what the stub does not receive
- All existing tests in tests/`test_finetune_skill_script.py` still pass unchanged in their assertions (only the fixture may change), and the fixture pins resolution explicitly rather than relying on PATH order
- Exporter code that reads `training_metadata.json` is added only if the c10 probe fails; if it passes, git log shows no exporter change for r3
- No model larger than 1.7B (Qwen) or 1.2B (LFM2.5) appears in new docs/tested.md rows of this batch
- The follow-up PRs touch no file under a lobes or jetson-containers checkout and add no new serve lane to sloth
- transformers, peft, trl, datasets, llmcompressor and compressed-tensors pins in `DEP_LAYER_PACKAGES` are byte-identical before and after the pin PR
- `_apply_memory_info_shim` and its two tests are present and unchanged at the end of this batch; a follow-up risk on the plan names the NGC torch >= 2.11 trigger
- No file under .github/workflows/ changes in this batch except for reasons unrelated to GPU execution
- Risks r5, r6, r7 remain listed and unresolved on the lfm2-5-target-modules-quantized-export plan at the end of this batch, or are resolved only with a decision text pointing at a separate issue
- The exported spec's audience section names agents parsing stdout first; every requirement's check is expressible as a command an agent can run, not a visual judgement
- At close, gh issue view 22 shows a closing comment whose nine lines each point at a docs/tested.md row, a merged PR, or issue #23; the /finetune --json loop on the Spark produced 3 stdout lines
- The before-state numbers (3 of 125 stdout lines, 0/4 smoke score, unsloth 2026.6.9 vs 2026.9.4) are quoted from docs/tested.md:73, docs/deliveries/2026-09-15-...md:102 and the plan's r1 text, not restated from memory
- README.md's stdout contract at :41-43 is true on a real run at close, verified by the c4 tested.md row, and docs/benchmarks.md carries a per-format score so the 'no accuracy loss' claim is no longer capped at low by lapse l1
- The measurement is the literal command in c29's instruction, run once per verb on the Spark, with wc -l output pasted into the docs/tested.md row
- The six-row count is taken from a new dated heading in docs/tested.md added by this batch's PRs, and the benchmarks table has >= 5 rows (4 formats + adapter baseline) on a suite whose item count is stated and >= 40
- grep -c '==' on `DEP_LAYER_NODEPS_PACKAGES` in sloth/tune/container.py returns 3; uv run pytest tests/`test_tune_container.py` -k pin passes; grep -i 'bump' docs/dgx-spark.md matches a heading
- docker run --rm <ngc image> python -c 'pass' prints a non-empty banner to stdout on the Spark (recorded as one line in docs/tested.md)
- A test asserts `emit_result`(`json_mode`=True) output contains exactly one newline for a nested dict
- Every file under examples/eval/ passes sloth validate --schema task (or the suite validator from the next claim) with 0 errors
- A fake-backend test with `pad_token`=None and `eos_token` set batches 3 prompts; one with both None runs batch size 1 and emits a diagnostic line on stderr
- The pin PR's diff to sloth/tune/container.py is confined to `DEP_LAYER_NODEPS_PACKAGES`
- shellcheck reports no SC2086/SC2046-style word-splitting warning on the `SLOTH_BIN` line, and a test with `SLOTH_BIN`='' falls through to the checkout branch
- grep -ri 'ssh thor' sloth tests .claude docs README.md returns only docs/tested.md lines

## Success signals

- On a real sloth train --json, sloth eval --json and sloth export --json run on the Spark, stdout parses as exactly 1 JSON document each (0 non-JSON lines), measured by piping stdout through python -c 'import sys,json; json.loads(sys.stdin.read())' and recorded in docs/tested.md
  - instruction: three rows, each with the wc -l line
- docs/tested.md gains >= 6 new rows: qlora-smoke → merged-16bit, Qwen3-1.7B → awq, nvfp4, gguf, one Jetson-side load, one live eval --adapter on runs/lfm2-lora; and docs/benchmarks.md carries an exact-match (or replacement metric) score per export format on a suite of >= 20 items
  - instruction: count rows under the heading; check the suite size line above the table
- sloth/tune/container.py `DEP_LAYER_NODEPS_PACKAGES` has 3 '==' pins and tests/`test_tune_container.py` asserts each reaches the install line; docs/dgx-spark.md has a section whose heading contains 'bump'
  - instruction: run the three commands

## Scope / boundaries

- The stdout fix lands in container.py's launch path once, for all three verbs, not per verb: train.py:448, eval.py:200 and export.py:901 all call container.launch() and ignore its stdout today, so a capture path added in launch() covers them without touching handler logic
  - instruction: Keep the change inside sloth/tune/container.py plus one call-site line per verb
- Unit tests cannot prove stdout purity: tests/`test_tune_container.py`:538-630 monkeypatch `_stream` to a canned exit code and .github/workflows/tests.yml has no GPU/docker job (non-goal c12 of the shipped spec) — the honest gate for h14 is a manual Spark run of the /finetune loop counting non-JSON stdout lines, recorded in docs/tested.md
  - instruction: Run bash .claude/skills/finetune/scripts/finetune.sh run --config examples/lfm2-lora.toml --suite examples/eval-suite.jsonl --json > out 2> err on the Spark and record wc -l out
- tests/`test_finetune_skill_script.py`:62-89 stubs a fake 'sloth' on PATH and every test asserts on the argv that stub records; flipping to checkout-first would bypass the stub (the tests run from inside the real checkout), so the test fixture must gain a way to pin resolution (an env override such as `SLOTH_BIN` honoured before both branches) rather than the tests being deleted
  - instruction: Prefer an env override honoured before both branches; set it in `stub_env`
- No new model family is added to the tested set beyond Qwen3-1.7B and LFM2.5-1.2B-Base; docs/tested.md:90-95 records Qwen3 4B/9B as not tested for memory reasons on the shared Spark, and that stays true in this batch
  - instruction: grep the new rows
- The transformers==4.57.1 / peft==0.18.0 / trl==0.24.0 / torchao-0.14 deadlock documented in CLAUDE.md:253-255 and docs/dgx-spark.md:66-72 is untouched — pinning unsloth is additive to that matrix, never a float of peft or torchao
  - instruction: git diff on container.py:93-103 is empty
- Risks r5 (concurrent gguf exports racing on the llama.cpp cache), r6 (disk-estimate heuristic) and r7 (four unverified call-signature inferences) are not in issue #22 and stay open on the shipped plan; this frame covers exactly the nine issue items
  - instruction: devague plan show
- `emit_result` (sloth/cli/`_output.py`:17-22) writes JSON with json.dump and no indent — one line per result — and that stays true, because the host capture identifies the in-container result as a single JSON line; any future pretty-printing must go through the same capture contract
  - instruction: tests/`test_cli.py`
- Pinning unsloth/`unsloth_zoo`/bitsandbytes adds no venv-invalidation or fingerprint logic: container.py:251-261 already recreates the venv and reruns uv pip install unconditionally, so a pin change or a revert takes effect on the next run with no cleanup step
  - instruction: git diff --stat
- `SLOTH_BIN` is exec'd as a command word (SLOTH=($`SLOTH_BIN`)), never eval'd or sourced, and an unset or empty value is ignored; the skill trusts it exactly as much as PATH
  - instruction: shellcheck .claude/skills/finetune/scripts/finetune.sh; tests/`test_finetune_skill_script.py`
- The Thor host ('ssh thor') is an operator-local test target: no shipped code, test, skill, or doc instruction depends on it; docs/tested.md rows record what was run there, and the serve image used on Thor is written into the row
  - instruction: run the grep after the Thor rows land

## Non-goals

- No lobes-cli change and no Jetson-repo integration code: the shipped spec's non-goal c10 stands — the Jetson item is a validation run that produces docs rows, not a new serve lane or a jetson-containers package
  - instruction: git diff --stat is confined to this repo
- The `_apply_memory_info_shim` in sloth/tune/`_exporter.py`:215-241 is not deleted in this batch: its guard is version-exact (`compressed_tensors` == '0.16.0' and torch.accelerator lacking `get_memory_info`) and NGC 25.11 still ships torch 2.10, so deletion is gated on the container image moving to torch >= 2.11 (risk r2), at which point the call site at :576, the two tests at tests/`test_tune_exporter.py`:521-558 and the module docstring go together
  - instruction: git diff on `_exporter.py`:215-241 is empty; devague plan risk list has the r2 successor
- No new CI job and no GPU runner (shipped non-goal c12 stands): every 'measure on the Spark' item above is a manual run recorded in docs/tested.md, and the unit suite keeps its fake backends
  - instruction: git diff --stat .github

## Assumptions

- Honesty condition h14 on the shipped frame ('each step's JSON lands on stdout only') is a confirmed condition whose delivery doc line 76 records it as unmet; this frame's success signal for item 1 is that h14's check passes on a re-run, and the old frame is not re-opened
  - instruction: git diff on .devague/frames/lfm2-5-target-modules-quantized-export.json is empty
- The /finetune skill's `resolve_sloth` (.claude/skills/finetune/scripts/finetune.sh:18-41) resolves in the order: `SLOTH_BIN` env override → 'uv run --project <dir> sloth' for the unsloth-cli checkout found by walking up from `BASH_SOURCE` → 'sloth' on PATH; an installed-tool invocation with no checkout above the script therefore still uses PATH
  - instruction: Wait for q1's answer before implementing; whichever policy wins, add a resolution-order test for each branch
- `training_metadata.json` already records hyperparameters.`load_in_4bit` (sloth/tune/`_trainer.py`:104 → metadata.py:113-121) but export.py:205-216 reads only the dataset key and `_exporter.py` never opens the file — if the r3 probe shows the implicit path fails for 4-bit adapters, the exporter reads that key and loads the base quantised; if it passes, no code change
  - instruction: Conditional on c10; do not pre-build
- The NGC image's entrypoint banner is printed by the container itself before sloth runs, so silencing trl/transformers progress in-container (`disable_tqdm`, logging to stderr) cannot by itself make host stdout JSON-only; the host-side capture is necessary, and in-container progress suppression is an optional noise reduction on stderr, not a substitute
  - instruction: one-line probe on the Spark
- The LFM2.5 training data (examples/chat-smoke.jsonl, per examples/lfm2-lora.toml:17) is chat schema while eval suites are task schema; the starter suite under examples/eval/ is authored directly in task schema, deriving items from chat rows by taking the final assistant turn as `expected_output` and the preceding user turn as input — no converter verb is added
  - instruction: run the validator over the directory
- Batched eval requires a pad token and left padding; `run_eval` sets tokenizer.`padding_side`='left' and, when tokenizer.`pad_token` is None, uses `eos_token` as pad — and if neither exists it falls back to batch size 1 with a diagnostic on stderr rather than failing
  - instruction: tests/`test_tune_trainer.py`
- The pinned bitsandbytes version is one that installed on aarch64 inside NGC 25.11 in the 2026-09-15 run (park v3 reads it from the container), so pinning cannot introduce a wheel-resolution failure the floating install did not already pass
  - instruction: the pinned value equals uv pip list output from that run
- sloth eval --quant matches the GGUF filename case-insensitively on the quant token (Unsloth writes e.g. \*-`Q4_K_M`.gguf while --quant takes `q4_k_m`), and export's --quant list is the source of the accepted names
  - instruction: tests/`test_cmd_eval.py` with mixed-case filenames

## Scope exploration

- `s1` — `sloth/tune/container.py:457-481 (_run_quiet, _stream) + launch() 631-720`: launch() runs docker via `_stream`() = subprocess.run(cmd, check=False) with inherited stdio; the only result the host reads is the exit code. There is no JSON-aware capture, no sentinel line, no result file, so the NGC banner, uv bootstrap and trainer progress land on host stdout alongside the in-container JSON (issue #22 item 1, risk r10).
  - seeds: `c2`
- `s2` — `sloth/cli/_commands/train.py:447-448, eval.py:200, export.py:901`: All three verbs call container.launch(...) and use only the return code (train ignores it, export checks truthiness) — no handler parses container output, so one capture path in launch() serves all three.
  - seeds: `c3`
- `s3` — `tests/test_tune_container.py:538-630 + .github/workflows/tests.yml:1-131`: Container tests only assert exit-code→CliError mapping via a monkeypatched `_stream`; CI runs test/lint/version-check on ubuntu-latest with no docker or GPU. Stdout-purity verification is manual on the Spark.
  - seeds: `c4`
- `s4` — `README.md:41-43, :95 vs docs/tested.md:73`: README states the stdout/stderr split unconditionally; tested.md:73 records the known gap ('3 JSON results interleaved with the container's banner and trainer progress lines', risk r10). The two must agree after the fix.
  - seeds: `c5`
- `s5` — `.devague/frames/lfm2-5-target-modules-quantized-export.json c22/h14 + docs/deliveries/2026-09-15-...md:76`: h14 text: 'each step's JSON lands on stdout only'; status confirmed (as a condition), delivery doc: 'honesty condition h14 unmet; risk r10'. The follow-up frame inherits h14 as its item-1 success bar.
  - seeds: `c6`
- `s6` — `.claude/skills/finetune/scripts/finetune.sh:18-41 resolve_sloth`: Order today: (1) command -v sloth on PATH, used unconditionally; (2) only if absent, walk up from `BASH_SOURCE` for pyproject.toml with name = "unsloth-cli" and run uv run --project. No env-var override exists (no `SLOTH_BIN`). Issue #22 item 2 / risk r9: on the Spark, PATH held an editable install of another worktree, so the run-mode validation exercised the wrong branch.
  - seeds: `c7`
- `s7` — `tests/test_finetune_skill_script.py:62-89 (stub_env fixture, _read_calls)`: Fixture prepends a tmp bin dir holding a fake sloth to PATH; the three forwarding tests and the help test all read calls.jsonl written by that stub. The uv-run fallback branch has no test at all. A checkout-first change without an override would make every test hit the real sloth.
  - seeds: `c8`
- `s8` — `.claude/skills/finetune/SKILL.md:28-29, :176-181`: SKILL.md states PATH-first, uv-fallback, and the installed-tool scenario via uv tool install unsloth-cli. It names no override. Any resolver change must update these lines.
  - seeds: `c9`
- `s9` — `sloth/tune/_exporter.py:387-409 (_load_adapter, merged formats) + examples/qlora-smoke.toml`: The exporter has no QLoRA detection: `load_in_4bit` is set only from the requested format (plan\['format'\] == 'merged-4bit', line 407), never from the adapter's training metadata; a 4-bit-trained adapter exported as merged-16bit relies entirely on Unsloth's implicit dequantise+merge. qlora-smoke.toml sets method = qlora, `load_in_4bit` = true.
  - seeds: `c10`
- `s10` — `sloth/tune/_trainer.py:104, sloth/tune/metadata.py:113-121, sloth/cli/_commands/export.py:205-216`: `load_in_4bit` is written into `training_metadata.json` under hyperparameters but no export code path reads it. Whether that matters is unmeasured until the r3 probe runs.
  - seeds: `c11`
- `s11` — `sloth/tune/_exporter.py:521-530 (_build_recipe AWQ mapping selection)`: Custom AWQ mappings are built only when config.`model_type` == 'lfm2' and `layer_types` is set; every other architecture (Qwen3 included) gets llm-compressor's default AWQMapping table with `duo_scaling`='both'. Whether Qwen3's GQA needs the `v_proj`→`o_proj` pair dropped, as LFM2 did, is unmeasured.
  - seeds: `c12`
- `s12` — `docs/tested.md:83-119 (Not tested section)`: Only unsloth/Qwen3-1.7B (2026-06) and LiquidAI/LFM2.5-1.2B-Base (2026-09-15) have rows; Qwen3 4B/9B were attempted but the box lacked memory. The follow-ups add format coverage, not model coverage.
  - seeds: `c13`
- `s13` — `docs/fine-tuning.md:151-161 deployment-target table`: Orin → gguf / awq W4A16 'not yet — an explicit gap in tested.md'; Thor → nvfp4 'not yet'; Spark → bf16 LoRA tested 2026-06-26. The table also predates the 2026-09-15 LFM2.5 rows. Ollama is mentioned nowhere in the docs; llama.cpp only as the Spark-side gguf conversion tool (dgx-spark.md:128-144).
  - seeds: `c14`
- `s14` — `shipped frame non-goal c10 + scope entry s43 (challenge pass)`: c10: 'No lobes-cli changes and no Jetson-repo integration ... gguf export targets llama.cpp/Ollama on Jetson ... none of which reference unsloth-cli today'; s43: 'Not run: QLoRA merge, Qwen3 paths, any quality metric on the quantized outputs, Jetson-side load.'
  - seeds: `c15`
- `s15` — `examples/eval-suite.jsonl + sloth/tune/_trainer.py:467-472 (eval result shape)`: The only suite is 4 task-schema items; eval emits total / `exact_match` / `exact_match_pct` / per-item results, stdout only, no file. Lapse l1 (approved) caps the 'no accuracy loss' claim at low: judged by one coherent completion and 0/4 on every variant.
  - seeds: `c16`
- `s16` — `sloth/tune/_exporter.py:850-865 _find_gguf + sloth/cli/_commands/eval.py:123-124`: Multi-GGUF dirs already fail closed (exit 1, remediation names the exact-file form); eval.py accepts a single .gguf path as --model. A --quant selector attaches in `_resolve_target`/`_find_gguf` before the raise. Risk r8's 'first \*.gguf alphabetically' wording is stale — the selector-less fail-closed behaviour shipped.
  - seeds: `c17`
- `s17` — `sloth/tune/container.py:93-111, :229-230 (DEP_LAYER_PACKAGES / DEP_LAYER_NODEPS_PACKAGES)`: transformers/peft/trl/datasets/llmcompressor/compressed-tensors are pinned; unsloth, `unsloth_zoo`, bitsandbytes are an unpinned --no-deps tuple installed by uv on every run (no venv fingerprint, uv pip install reruns unconditionally at :253/:261, so a pin change takes effect without invalidation logic).
  - seeds: `c18`
- `s18` — `docs/dgx-spark.md:52-72 + CLAUDE.md:253-255 + docs/tested.md:21, :51-52`: dgx-spark.md's matrix lists unsloth as 'unpinned, --no-deps' and cites 2026.6.9 only inside the deadlock explanation; CLAUDE.md names no unsloth version; tested.md is the only source of the measured versions (2026.6.9/2026.6.7 June, 2026.9.4/2026.9.3 September). Neither doc has a bump path.
  - seeds: `c18`
- `s19` — `CLAUDE.md:253-255 'Do not float these'`: The merge-gating convention pins the resolved dep layer against NGC 25.11 torch 2.10; peft>=0.19 needs torchao>0.16 needs torch>=2.11. The unsloth pin joins that matrix; it does not reopen it.
  - seeds: `c19`
- `s20` — `sloth/tune/_exporter.py:215-241, :576 + tests/test_tune_exporter.py:521-558`: Shim maps torch.accelerator.`get_memory_info` → torch.cuda.`mem_get_info` only when `compressed_tensors` is exactly 0.16.0 and the attribute is missing; applied unconditionally before oneshot for awq/nvfp4. Tests cover applied and two skipped cases. Nothing outside these two files references it.
  - seeds: `c20`
- `s21` — `sloth/tune/_trainer.py:446-450 + sloth/tune/_exporter.py:996-1018 (continuation-only scoring)`: Both eval seams slice outputs\[0\]\[`prompt_len`:\] before decode. Unit coverage is fake-backend only, and the fake tokenizer's decode() at tests/`test_tune_trainer.py`:735-736 ignores its tokens argument, so the slice itself is never asserted — the tests prove load order and result shape, not the fix.
  - seeds: `c21`
- `s22` — `sloth/tune/_exporter.py:348-384 (_adapter_for_base, _drop_override_staging) + tests/test_tune_exporter.py:640-655`: --base stages output/`_adapter`-override with symlinked adapter files and a rewritten `base_model_name_or_path`, cleaned up after export unless `keep_intermediate`; exercised only through the fake `_load_backend` fixture, and tests/`test_cmd_export.py`:496-522 covers argv only.
  - seeds: `c21`
- `s23` — `tests/test_tune_trainer.py:664-799 (TestRunEval) + tests/test_cmd_eval.py:815-868`: Fake decode() returns a constant 'cba' regardless of input; assertions cover load sequence and result keys only.
  - seeds: `c22`
- `s24` — `.github/workflows/tests.yml + shipped frame non-goal c12`: CI: test, lint, version-check on ubuntu-latest only. c12: 'No new CI job and no GPU runner ... real LFM2.5 + gguf runs are manual on the Spark.'
  - seeds: `c23`
- `s25` — `devague plan show --plan lfm2-5-target-modules-quantized-export (Risks r1-r10)`: All ten risks are open. Issue #22 maps to r10, r9, r3, r4, (Jetson: s43/c10), r8+l1, r1, r2, and the two review fixes; r5/r6/r7 have no issue item.
  - seeds: `c24`
- `s26` — `.claude/skills/finetune/scripts/finetune.sh:18-41 (policy decision)`: Three viable resolver policies; the test fixture's PATH stub constrains checkout-first unless an override is added. User decision, not derivable from the code.
  - seeds: `q1` (question, resolved)
- `s27` — `docs/fine-tuning.md:157-161 (Orin / Thor rows, 'not yet')`: The table names both devices as intended targets; nothing in the repo records which device the operator has. Blocking for item 5 only.
  - seeds: `q2` (question, resolved)
- `s28` — `examples/eval-suite.jsonl (4 items) + sloth/tune/_trainer.py:467-472 (exact-match only)`: No larger suite exists and eval reports exact-match only; a 0/N result on every variant cannot express quantization loss. Both the suite source and the metric are user decisions.
  - seeds: `q3` (question, resolved)
- `s29` — `examples/eval-suite.jsonl + sloth/cli/_commands/eval.py:308-337 (--suite single file today)`: q3 decided the suite becomes a directory of domain files with a starter set of about 40 items; the CLI accepts exactly one file today (required=True).
  - seeds: `c35`, `c36`
- `s30` — `challenge pass / failure-mode lens: sloth/tune/container.py:690-720 exit-code mapping`: Today success is 'exit code 0' with no result check; once the host parses a result, a silent-success path opens when the container prints nothing parseable. Seeded a fail-closed requirement.
  - seeds: `c37`
- `s31` — `challenge pass / observability lens: sloth/tune/container.py:475-481 _stream + launch()`: A naive `capture_output`=True would buffer a 10-minute training run silently and could deadlock on a full pipe; the mapping of 137 to an OOM hint must survive. Seeded a line-streamed tee requirement.
  - seeds: `c38`
- `s32` — `challenge pass / unstated-assumptions lens: NGC entrypoint banner vs trl progress (docs/tested.md:73)`: tested.md attributes the noise to both the container banner and trainer progress; only the host can strip the banner. Seeded the assumption that the fix is host-side by necessity.
  - seeds: `c40`
- `s33` — `challenge pass / adjacent-systems lens: sloth/cli/_output.py:17-27 emit_result`: json.dump without indent → single line, so last-JSON-line detection is sound today; recorded as a boundary so it cannot drift.
  - seeds: `c41`
- `s34` — `challenge pass / adjacent-systems lens: examples/lfm2-lora.toml:17 + examples/chat-smoke.jsonl + examples/eval-suite.jsonl`: q3's 'held-out rows of the training data' assumed a schema match that does not exist: training rows are {messages:\[...\]}, suites are {task,input,`expected_output`}. Seeded an assumption fixing the derivation rule.
  - seeds: `c42`
- `s35` — `challenge pass / adjacent-systems lens: sloth/cli/_commands/validate.py:1-16, sloth/explain/catalog.py:247-285, .claude/skills/finetune/scripts/finetune.sh:62, SKILL.md`: validate is dataset-only; the catalog and skill help say 'suite.jsonl' (single file) and know nothing of --quant or --batch-size; the rubric gate checks catalog coverage. Seeded a requirement bundling these.
  - seeds: `c43`
- `s36` — `challenge pass / failure-mode lens: sloth/tune/_trainer.py:443 and sloth/tune/_exporter.py:993 (single-prompt tokenizer calls, no pad handling)`: Both eval paths tokenize one prompt at a time with no `padding_side` or `pad_token` handling; batching cannot be dropped in without it. Seeded the fallback assumption.
  - seeds: `c44`
- `s37` — `challenge pass / reversibility lens: sloth/tune/container.py:251-261 (uv venv + uv pip install every run)`: Rollback of a bad pin is a one-line revert because nothing is cached by version; recorded as a boundary so nobody adds invalidation logic 'for safety'.
  - seeds: `c45`
- `s38` — `challenge pass / adjacent-systems lens: aarch64 wheel availability for the --no-deps tuple`: unsloth/`unsloth_zoo` are pure-python; bitsandbytes ships arch-specific wheels. Pinning to the live-installed version is what makes the pin safe on Spark.
  - seeds: `c46`
- `s39` — `challenge pass / security lens: .claude/skills/finetune/scripts/finetune.sh resolve_sloth (SLOTH_BIN)`: An env-provided executable is the same trust level as PATH lookup; the only hazard is word-splitting/eval, ruled out by the boundary. Clean otherwise.
  - seeds: `c47`
- `s40` — `challenge pass / overlooked-actors lens: docs/tested.md:61,71 (vLLM serve rows) vs the Thor target`: The Spark serve checks used a nightly vLLM image; nothing records a Thor-side image. Parked.
- `s41` — `challenge pass / lifecycle lens: sloth/tune/_exporter.py:850-865 _find_gguf + export --quant naming`: The selector's match rule was unstated; Unsloth's uppercase file suffix vs the CLI's lowercase quant names would make a naive equality miss. Seeded the assumption.
  - seeds: `c48`
- `s42` — `challenge pass / observability lens: sloth/tune/_trainer.py:467-472 + sloth/cli/_commands/compare.py:87-93`: eval writes nothing to disk; compare/summarize read only `training_metadata.json`. The per-format score table in c30 would be assembled by hand. Raised as a hard question on c35 (needs a user decision).
- `s43` — `challenge pass / concurrency lens: sloth/tune/container.py llama.cpp cache mount + VENV_DIR`: Nothing in this batch adds shared mutable state; the existing single-operator race (risk r5) stays out of scope per c24. Clean pass.
- `s44` — `challenge pass / migration lens: examples/eval-suite.jsonl + --suite single-file callers`: c35 keeps the single-file form, so no caller migrates; the 4-item file stays as the smoke suite. Clean pass.
- `s45` — `challenge pass / cheap-probes lens: read-only greps this session`: Probed: docker run argv has no -t/-i flags (no TTY to lose on capture); --json is forwarded into the container argv (train.py:332-334); `emit_result` is single-line JSON; validate.py is dataset-only; `run_eval` has no pad handling; finetune.sh derives `export_output` from `adapter_dir` (line 226) and parses no JSON. Not probed: the NGC banner's exact stream (assumption C above), Thor availability.

## Decisions

- Jetson-side load covers both devices: Thor in this batch, Orin Nano via llama.cpp/Ollama tracked in issue #23 because it needs lobes taken down on the Spark
  - instruction: c14's tested-row condition is satisfied by either device in this batch; the Orin row is owed by #23
- The /finetune resolver order is `SLOTH_BIN` override → own checkout (uv run --project) → PATH sloth; the test fixture sets `SLOTH_BIN` to its stub
  - instruction: tests/`test_cmd_eval.py` + tests/`test_tune_trainer.py`
- Eval grows for width: --suite accepts a directory of task-schema JSONL files (per-file + aggregate scores), a ~40-item starter suite ships under examples/eval/ in three domain files, exact-match is joined by a stdlib token-level F1, and `run_eval` batches generation
  - instruction: tests/`test_tune_trainer.py`; one live timing row
- sloth eval writes its full result JSON to eval.json inside the evaluated directory (run dir for --adapter, export dir for --model) in addition to stdout; sloth summarize and sloth compare read it to render per-format scores
  - instruction: write eval.json next to `training_metadata.json` / export.json; summarize.py and compare.py gain an eval block; a re-run overwrites eval.json

## Hard questions

- Which resolver policy do you want for /finetune: (a) checkout-first — prefer 'uv run --project <own repo> sloth' whenever the script sits inside an unsloth-cli checkout; (b) keep PATH-first and add an explicit `SLOTH_BIN` env override plus a documented PATH-precedence note in SKILL.md; or (c) checkout-first AND the env override (the override is what keeps tests/`test_finetune_skill_script.py`'s PATH stub working)? (resolved: Checkout-first plus an explicit `SLOTH_BIN` override: `resolve_sloth` honours `SLOTH_BIN` first, then a walked-up unsloth-cli checkout via uv run --project, then PATH.)
- Which Jetson device is actually reachable for the device-side load: an Orin (gguf via llama.cpp or Ollama), a Thor (awq/nvfp4 via vLLM), both, or neither yet — and if neither, does the Jetson item stay parked in this frame or get deferred to a later one? (resolved: Both devices. Thor (awq/nvfp4 via vLLM) runs with the rest of this batch; the Orin Nano gguf load needs lobes down on the Spark and is split into issue #23 for a separate session.)
- Where does the real eval suite come from: author a larger task-schema JSONL in examples/ (which domain, how many items), reuse held-out rows of an existing training dataset, or bring an external suite — and is exact-match the metric you want quantization loss reported in, or should eval grow a token-level score first? (resolved: Starter suite in-repo (~40 task-schema items under examples/eval/ in three files: CLI contract, AgentCulture terminology, task-format following; held-out rows of the LFM2.5 training data are the first source); --suite accepts a directory and scores every \*.jsonl with per-file and aggregate results; metrics are exact-match plus a stdlib token-level F1; `run_eval` batches generation.)
- Should sloth eval also write its result JSON into the run or export directory (e.g. eval.json next to `training_metadata.json` / export.json) so sloth summarize and sloth compare can render per-format scores, instead of the benchmarks table being hand-assembled from stdout? (resolved: Yes: sloth eval writes its result JSON into the evaluated directory (the run dir for --adapter, the export dir for --model) so summarize and compare can render per-format scores.)

## Open parks

- [unknown_nonblocking] Whether Unsloth's `from_pretrained` + `save_pretrained_merged` dequantises a 4-bit-trained adapter correctly when the base is loaded in 16-bit (`load_in_4bit`=False) is unmeasured; the qlora-smoke → merged-16bit probe answers it before any exporter code changes (risk r3)
- [unknown_nonblocking] Whether Qwen3's GQA needs the `v_proj`→`o_proj` AWQ pair dropped the way LFM2 did (sloth/tune/`_exporter.py`:476-511) is unmeasured; llm-compressor defaults are the first attempt, a custom mapping only if the run fails (risk r4)
- [unknown_nonblocking] The bitsandbytes version installed on the 2026-09-15 run is not recorded anywhere (docs/tested.md:23 and docs/benchmarks.md:22 record 0.49.2 for the June batch only); the pin PR must read it from the container before writing it
- [unknown_nonblocking] When NVIDIA's NGC pytorch image moves to torch >= 2.11 is outside this repo's control; the shim deletion (risk r2) has no date and is tracked, not scheduled
- [unknown_nonblocking] How the in-container result is separated from the stream: a sentinel-delimited last line parsed from captured stdout, or a result file written under the mounted output dir and read by the host — both satisfy c2; choose during /think or plan, after checking whether trl's progress dicts can be redirected to stderr in-container at all

## Resolved vagueness

- [unknown_nonblocking] Where the Thor is, whether vllm/vllm-openai:nightly (the image tested.md:61 used on the Spark) runs on it, and whether that image supports LFM2 with awq/nvfp4 on Thor's compute capability is unrecorded; the Thor row may need a different serve image — resolved: Thor is reachable from the Spark as 'ssh thor' for this operator only; it is used for the live Thor rows and never referenced by shipped code, tests, or skills.
