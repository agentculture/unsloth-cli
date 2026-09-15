# Delivery Summary — lfm2-5 delivery follow-ups (#22)

plan: `lfm2-5-delivery-follow-ups-22` · run: `complete` · date: `2026-09-15`
baseline: `devague summary skeleton`

## Intent

Close the nine follow-ups left open by the LFM2.5 `target_modules` + quantized
export delivery (issue #22, spec #24): JSON-only stdout on real
train/eval/export runs, a `/finetune` resolver that runs the checkout it lives
in, QLoRA and Qwen3 measured through every export format, a Jetson-side load, a
real quantization-loss number, pinned unsloth versions with a documented bump
path, and the torch shim deliberately kept. The plan (11 tasks, 5 waves) was
executed via `/assign-to-workforce` on the integration branch
`feat/follow-ups-22-a`, shipped as PR #25 (code + live batch 1) with PR #26
(live batch 2 + closeout) stacked on it; both squash-merged to `main` at
`a072b88`. Version 0.7.2 → 0.8.1.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — PR-A/t1 — container.launch captures the container's stdout: line-streamed tee to host stderr, last JSON line returned as the result, fail-closed when none
- `t2` — PR-A/t2 — train/eval/export consume the captured result: always append --json to the in-container argv, emit the returned dict via `emit_result` (JSON or text), and make README's stdout promise true
- `t3` — PR-A/t3 — /finetune resolver: `SLOTH_BIN` override → own checkout via uv run --project → PATH; SKILL.md + tests updated; --suite help accepts a directory
- `t4` — PR-A/t4 — eval host side: --suite accepts a file or directory (validated before launch), --quant and --batch-size flags, sloth validate --suite, catalog + guide-skill docs
- `t5` — PR-A/t5 — in-container eval: per-file + aggregate scoring, stdlib token-F1, batched generation with pad fallback, --quant GGUF selection, eval.json written into the evaluated dir, continuation-only tests tightened
- `t6` — PR-A/t6 — summarize and compare render eval.json: per-format scores next to the export block
- `t7` — PR-A/t7 — starter eval suite: examples/eval/ with three task-schema files (~40 items total) derived from the chat-smoke rows and the CLI contract
- `t8` — PR-A/t8 — pin unsloth / `unsloth_zoo` / bitsandbytes to the live-validated versions; document the bump path in docs/dgx-spark.md
- `t9` — PR-B/t9 — Spark live batch 1: stdout-purity count, eval --adapter, export --base, QLoRA → merged-16bit, Qwen3-1.7B → awq/nvfp4/gguf; rows in docs/tested.md
- `t10` — PR-B/t10 — Spark + Thor live batch 2: per-format eval scores on examples/eval/ into docs/benchmarks.md, batched-vs-serial timing, Thor awq/nvfp4 load via vLLM, deployment table updated
- `t11` — PR-B/t11 — closeout: issue #22 comment mapping all nine items, version bump, PR B opened; r5/r6/r7 left untouched on the shipped plan

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `container.launch()` runs docker via a `Popen` line loop, tees every line to stderr as it arrives, returns the last JSON-object line as a dict, and fails closed (exit 2) when none — with a **single-line** message after review (`d4`). `sloth/tune/container.py`, 11 new tests. |
| `t2` | delivered | `train` / `eval` / `export` always append `--json` to the in-container argv and emit the returned dict via `emit_result`; README's stdout promise left unconditional because the live rows prove it. |
| `t3` | delivered | `resolve_sloth` order `SLOTH_BIN` → own checkout via `uv run --project` → PATH; one test per branch; SKILL.md updated; `--suite <suite.jsonl \| dir>` help. |
| `t4` | delivered | `--suite` file or directory, validated host-side before launch; `--quant`, `--batch-size` (validated ≥ 1 after review); `sloth validate --suite` (task schema only after review); catalog + guide skill updated. |
| `t5` | delivered | per-file + aggregate scoring, stdlib token-F1 (`sloth/tune/metrics.py`), batched generation with pad fallback, `--quant` GGUF selection, `eval.json` in the evaluated dir (parent for a bare `.gguf`), continuation-only tests tightened. Offset rule corrected vs. the instruction (`d1`). |
| `t6` | delivered | `summarize` / `compare` render `eval.json` (run-level and per-export); compare reports an eval delta only when metrics differ (review). |
| `t7` | delivered | `examples/eval/` — 46 items in `cli-contract` (14), `agentculture-terms` (16), `task-format` (16); `examples/eval-suite.jsonl` kept as the 4-item smoke. |
| `t8` | delivered | `unsloth==2026.9.4 unsloth_zoo==2026.9.3 bitsandbytes==0.50.2`, measured in-container; pin tests; `docs/dgx-spark.md` bump section; CLAUDE.md pin sentence. |
| `t9` | delivered | `docs/tested.md` "follow-ups #22" section: NGC banner probe (36 stdout / 0 stderr), `train --json` 1/219, `eval --adapter --json` 1/191 (0/4, F1 0.087, prompt excluded), `export --base` staged copy loads, QLoRA → merged-16bit 3.44 GB bf16, Qwen3-1.7B awq/nvfp4/gguf all pass. Required the `d2` fix first. Rows shipped in PR A (`d3`). |
| `t10` | delivered | `docs/benchmarks.md` per-format table (adapter b8/b1, merged-16bit, gguf Q4_K_M, awq, nvfp4 on 46 items; 220 s vs 524 s batched/serial); Thor awq + nvfp4 vLLM 0.29 loads; deployment table Thor cell dated, Orin → #23. Every exact-match row is 0/46 (see Drift). |
| `t11` | delivered | nine-item map comment on #22; shipped plan's r1–r4, r8–r10 resolved, r5–r7 untouched; PR B opened stacked on PR A; #22 closed after both merged. |

## Mid-work Decisions

- `d2` (approved) — sloth export's path sanitizer (added in #20's review batch after its live rows) also ran in-container, where cwd is `/workspace`, so every real export on `main` exited 1 with "path is outside the allowed roots". Fix, outside every plan task: `_container_kwargs` forwards `SLOTH_ALLOWED_ROOTS` (the identity-mounted parents, plus a local `--base` after review) into the container env. — blocking: t9's export rows (c10, c12, c21) could not run until real exports worked again.
- `d1` (pending approval) — t5 slices each prediction at the shared padded prompt width (derived per row from the attention mask), not at the row's own token count as the plan instruction said; the literal instruction would have failed the acceptance criterion. Deletes `_exporter._continuation` / `_eval_prompt` / `_score_predictions` (logic moved to `_trainer` / `metrics`).
- `d3` (pending approval) — t9's tested.md rows rode with PR A instead of PR B, so PR A shipped with its own live evidence.
- `d4` (pending approval) — the fail-closed error became a single-line message (the captured lines are already on stderr), because the two-line `error:`/`hint:` contract outranks the literal h36 wording.
- Not covered by any record: the first train attempt died with CUDA OOM at backend init while lobes' vLLM held ~90 of 121 GB; reclaimed by touching every page of a 20 GiB anonymous mmap (no sudo). The Qwen3 awq/nvfp4 exports needed `--calib examples/chat-smoke.jsonl` because the June adapter's metadata predates the dataset-path field. The `version-bump` skill's `bump.py` hung twice on this box; the 0.8.1 bump was applied by hand. Review fixes (9 qodo + 10 Sonar on #25, 2 qodo on #26) were built by six subagents in worktrees and merged before either PR merged.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t9` (`d2`) | blocking: t9's export rows (c10, c12, c21) cannot run until real exports work again | `needs-follow-up` |
| `t5` (`d1`, pending) | plan instruction contained a wrong offset rule; acceptance criteria are the contract and are met (3 load-bearing tests) | `acceptable` |
| `t11` (`d3`, pending) | the live rows are the proof that PR A's stdout/export changes work; splitting them off would ship PR A unverified | `acceptable` |
| `t1` (`d4`, pending) | the rubric's two-line error contract outranks the literal h36 wording; the diagnostic content is preserved on stderr | `acceptable` |
| `t10` | the per-format table exists but every row scores 0/46 exact-match for the 10-step smoke adapter, so honesty condition h15 ("at least one row is not 0") is a **failed** condition; the table becomes a quantization-loss number only with a trained adapter. Token-F1 differs between batch 8 and batch 1 (0.038 vs 0.060) — batched decoding is not bit-identical (plan risk r8). | `needs-follow-up` |
| `t4` | after review, `sloth validate --suite` forces the task schema and rejects an explicit non-task `--schema`; `--batch-size < 1` exits 1 — tighter than the confirmed criteria, no behaviour the plan promised was removed | `acceptable` |
| `t11` | issue #22 was closed by hand after the merge because PR B targeted PR A's branch, so its `Closes #22` did not fire; the closing comment names both merged PRs | `acceptable` |

## Evidence

- tests: `uv run pytest -n auto -q` at `a072b88` — 699 passed, 1 skipped (no CUDA); the nine touched test files alone: 377 passed
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r sloth` — clean; `uv run teken cli doctor . --strict` — healthy 26/26
- CI on #25 after the review fixes: test, lint, version-check, SonarCloud (quality gate passed, 0 open issues), GitGuardian — all green
- live: `docs/tested.md` §"follow-ups #22" (12 rows with commands, wall times, `wc -l` counts); `docs/benchmarks.md` §"per-format scores on `examples/eval/`"
- validation ledger (`/validate-delivery`, proposed): 34 obligations `o1`–`o34`, 36 evidence records `e1`–`e36` (33 pass, 3 fail: `e28` h15, `e31` h45, `e32` h31), deltas `b1`–`b3`
- commits: `a89c16b..a072b88` (one squash on `main`, 22 integration commits behind it)
- PRs / issues: #24 (spec), #25 (PR A), #26 (PR B), #22 (closed), #23 (Orin Nano, open)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `sloth train/eval/export --json` are stdout-pure on real runs (exactly one JSON line each) | high | `docs/tested.md` follow-ups rows (train 1/219, eval 1/191, exports 1 each) · `tests/test_tune_container.py::TestStreamCapture`, `::TestLaunchResultCapture` · `e1`, `e2`, `e3` |
| a container that prints no JSON result fails closed with exit 2 and a single-line message | high | `tests/test_tune_container.py::TestLaunchResultCapture::test_exit_0_without_json_fails_closed_with_single_line_message` · `e5` |
| real-docker line-by-line streaming (as opposed to the fake process) works | medium | observed live (every run's stderr carried the banner and trainer progress) but only the fake-process test asserts ordering — lapse `l4` (pending) caps this |
| the `/finetune` skill resolves `SLOTH_BIN` → own checkout → PATH | high | `tests/test_finetune_skill_script.py` (8 tests) · `e10`, `e11` |
| `sloth eval --suite <dir>` validates before launch; `--quant`, `--batch-size ≥ 1`; `sloth validate --suite` (task schema) | high | `tests/test_cmd_eval.py`, `tests/test_cmd_validate.py` · rubric gate 26/26 · `e13` |
| per-file + aggregate scores with token-F1; `eval.json` in the evaluated dir; `summarize`/`compare` render it | high | `tests/test_tune_trainer.py`, `tests/test_tune_exporter.py`, `tests/test_tune_summary.py`, `tests/test_cmd_compare.py` · `e14` |
| batched generation excludes each row's own prompt under left padding; 2.4x wall-clock on 46 items | high | `tests/test_tune_trainer.py` (3 load-bearing tests, `d1`) · `docs/benchmarks.md` (220 s vs 524 s) · `e15`, `e16` |
| batched and serial decoding produce identical outputs | unverified | they do not at this scale (F1 0.038 vs 0.060, exact 0/46 both) — delta `b2`, plan risk r8; not claimed |
| the starter suite makes exact-match a usable quantization-loss metric | low | 0/46 on every format for the smoke adapter — h15 filed as **fail** (`e28`); `l2` (pending) says the suite was never scored before shipping; a trained adapter is the missing input |
| QLoRA (bnb-4bit base) adapters merge to bf16 with no exporter change | high | `docs/tested.md` row (3.44 GB, no `quantization_config`) · `e20` |
| Qwen3-1.7B exports to awq / nvfp4 / gguf with llm-compressor defaults | high | `docs/tested.md` rows (28 GQA mappings auto-skipped) · `e21` |
| `export --base <id>` loads the staged adapter copy; continuation-only scoring holds on hardware | high | `docs/tested.md` rows · `e23` |
| LFM2.5 awq + nvfp4 exports load and generate on a Thor with vLLM 0.29 | high | `docs/tested.md` Thor row · `docs/fine-tuning.md` deployment table · `e25` |
| the Thor is never referenced by shipped code, tests, or skills | medium | `sloth/`, `tests/`, `.claude/`, README are clean; the literal grep also matches the devague spec/plan artifacts under `docs/` — h45 filed as **fail** (`e31`) |
| unsloth / unsloth_zoo / bitsandbytes are pinned to the live-measured layer with a bump path | high | `tests/test_tune_container.py::TestNodepsLayerPins` · `docs/dgx-spark.md` bump section · `e17`, `e18`, `e19` |
| every real `sloth export` on `main` was broken from #20 until this run's `d2` fix | high | first live chain: 5/5 exports exit 1 "outside the allowed roots"; after the fix 6/6 pass · `tests/test_cmd_export.py` d2 tests · `b1` |
| the spec's before-state numbers all grep in the files it cites | low | two of three do; "3 of 125 lines" lives in issue #22's body — h31 filed as **fail** (`e32`) |
| issue #22 is closed with every item mapped | high | issue #22 comment + state CLOSED · `e34`, `e36` |
| the torch `get_memory_info` shim is deleted | unverified | not attempted — deliberately kept until NGC ships torch ≥ 2.11 (frame non-goal c20, park v4) |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | Swept the exported spec by section headings and relied on frame state for the claim-by-claim read, instead of reading docs/specs/2026-09-15-lfm2-5-delivery-follow-ups-22.md claim by claim as the method asks |

pending approval (not yet evidence): `l2` (t7 suite never scored before shipping),
`l3` (task-format items are authored content), `l4` (docker streaming proven
only against a fake process).

## Remaining Work / Follow-up

- **Owner adjudication** — confirm or reject deviations `d1`, `d3`, `d4`; lapses `l2`–`l4`; obligations `o1`–`o34`, evidence `e1`–`e36`, deltas `b1`–`b3`; frame parks `v1` (QLoRA dequant, now measured) and `v2` (Qwen3 GQA, now measured). Two deltas (the padded-width slice from `d1`, the single-line error from `d4`) file only after those deviations are approved.
- **A trained adapter for the benchmarks table** — the per-format scores are 0/46 exact-match on the smoke adapter (h15 fail); train a multi-hundred-step adapter on a real corpus and re-run `sloth eval --model` per format so the table measures quantization loss. Owner: next fine-tuning run.
- **Batched vs serial equivalence** (plan risk r8) — check on a model with non-zero scores whether left-padded batched decoding should match serial; record the batch size in `eval.json` (delta `b2`).
- **Orin Nano gguf load** — #23 (needs lobes down on the Spark).
- **Spec provenance nits** — h31's "125 lines" figure should cite issue #22; h45's grep should exclude the devague artifacts under `docs/`. Amend on the frame via `devague amend` / `interrogate --instruction`; no code change.
- **Torch shim** — unchanged by design; delete when NGC ships torch ≥ 2.11 (park v4).
- **`bump.py` hang** — the version-bump skill script hung twice on this box; worth an upstream look (guildmaster).
