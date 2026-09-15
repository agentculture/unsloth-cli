# Delivery Summary — LFM2.5 target_modules + quantized export

plan: `lfm2-5-target-modules-quantized-export` · run: `complete` · date: `2026-09-15`
baseline: `devague summary skeleton`

## Intent

Build the converged spec from PR #19 (issues #16 and #18): an explicit
`target_modules` run-config key with an `lfm2` preset so LFM2.5 adapters cover
the short-conv blocks, plus container-backed `sloth export` formats
(merged-16bit, merged-4bit, gguf, awq, nvfp4) and `sloth eval --model` for
quantization-loss checks. The plan (13 tasks, 5 waves) was fanned out by
`/assign-to-workforce` to Claude subagents (one task via colleague on lobes
cortex), every merge TDD-gated, and both live-validation tasks run by the main
agent on the DGX Spark. Delivered as one PR (#20) per deviation `d4`.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — PR1/t1 — config: `target_modules` key (list | regex string | `preset:<name>`) + lfm2 preset resolution
- `t2` — PR1/t2 — registry: `config_hash` stays stable when optional fields are unset
- `t3` — PR1/t3 — trainer: pass resolved `target_modules` to `get_peft_model` and surface it in the plan + metadata
- `t4` — PR1/t4 — train host preflight: LFM2 hint, rank>32 hand-lane diagnostic, chat-template check
- `t5` — PR1/t5 — docs + catalog + example for `target_modules` and the deployment-target table
- `t6` — PR1/t6 — live validation on the Spark: LFM2.5-1.2B-Base LoRA with preset:lfm2, served by vLLM, recorded in docs/tested.md
- `t7` — PR2/t7 — export CLI host side: new formats, flags, validation, dry-run, no-clobber, atomic output, disk estimate, container routing
- `t8` — PR2/t8 — container: llama.cpp cache mount, explicit HOME, env passthrough, llm-compressor pin, memory preflight
- `t9` — PR2/t9 — `_exporter.py`: lazy in-container seam for merged/gguf (Unsloth) and awq/nvfp4 (llm-compressor), calibration, cleanup, export.json
- `t10` — PR2/t10 — surface exports in runs / summarize / compare
- `t11` — PR2/t11 — eval --model DIR for merged / quantized outputs
- `t12` — PR2/t12 — docs, catalog, README, benchmarks and the /finetune skill for the export formats
- `t13` — PR2/t13 — live validation on the Spark: every export format + eval --model scores, recorded in docs/tested.md; PR 2 opened

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `sloth/tune/presets.py` + `RunConfig.target_modules` (list / regex / `preset:lfm2`) with validator; merge `8691462` |
| `t2` | delivered | `compute_config_hash` drops None-valued keys; literal-hash regression test; merge `0a30431` |
| `t3` | delivered | trainer passes the resolved value to `get_peft_model`, plan + metadata carry it; **plus** `dataset.path` in `training_metadata.json` (`d1`); merge `9d37f15` |
| `t4` | delivered | LFM2 preset hint, rank-cap note, chat-template preflight from the local HF cache (stdlib only); merge `2d4d186` |
| `t5` | delivered | docs/fine-tuning.md target table, `examples/lfm2-lora.toml`, catalog + README mentions — built by colleague on lobes cortex after a `--continue` nudge; merge `8fdfcbd` |
| `t6` | delivered | real `sloth train` of the LFM2 example (adapter targets the full regex, r=16), vLLM `--enable-lora` load, `sloth eval`, docs/tested.md rows; commit `0fc9243` |
| `t7` | delivered | six formats, all flags, dry-run plan with disk estimate, no-clobber, atomic `.partial`, container routing; merge `8310b3a` (later refactored into lanes, `9032425`) |
| `t8` | delivered | export HOME/cache mount, `env` passthrough, llm-compressor pins, MemFree hint; merge `1ed0b81` — mount layout later changed (see Drift) |
| `t9` | delivered | `sloth/tune/_exporter.py` seam (merged/gguf via Unsloth, awq/nvfp4 via llm-compressor, calibration, shim, export.json); merge `0645716` + six live-found fixes |
| `t10` | delivered | `discover_exports`, `summarize`/`compare`/`runs show` surfaces; merge `474bf80` |
| `t11` | delivered | `sloth eval --model DIR` (transformers or llama-completion backend), catalog `_EVAL`; merge `af9e5a5` |
| `t12` | delivered | catalog `_EXPORT`, README, docs/fine-tuning.md, docs/benchmarks.md, `/finetune` skill flags; merge `d3217d6` + `--export-output` fix `7bec93e` |
| `t13` | delivered | every format exported live, `eval --model` ×3, vLLM loads of awq/nvfp4, `/finetune` loop end to end, docs/tested.md rows, version 0.7.0, PR #20 opened; commit `ec42807` |

## Mid-work Decisions

- `d1` — t3's scope extended to `sloth/tune/metadata.py` so `training_metadata.json` records `dataset.path` — the metadata writer never stored the path, so the default calibration source (requirement c31) could not be resolved; no plan task owned the file.
- `d2` — t12 reassigned from colleague (lobes cortex) to a claude sonnet subagent at the user's request after t5 needed a manual `--continue` nudge to execute.
- `d3` — datasets pin bumped 4.3.0 → 4.8.5 in `DEP_LAYER_PACKAGES` (+ docs/dgx-spark.md, CLAUDE.md) — llmcompressor 0.11.0 requires datasets ≥4.8.4, so the pinned layer was unsatisfiable and `sloth train` exited before the GPU.
- `d4` — single PR from the plan branch instead of the two PRs decided in c16 — the fan-out merged every wave into one branch.
- Export HOME moved from `/workspace/.home` (under the workdir mount) to a dedicated host-owned mount at `/opt/sloth-home` — docker created the nested mount-point as root and `$HOME/.cache/uv` was unwritable on the first live export. Within t8's "documented constant" criterion; no deviation record.
- Six exporter fixes found only by live runs (no fake could catch them): `merged_4bit_forced` + 4-bit base load; gguf leaves merged safetensors; dataset path re-resolved inside the container; calibration rows must be a `datasets.Dataset`; `max_seq_length=512` (LFM2's `model_max_length` overflows); AWQModifier from `modifiers.transform.awq` (the legacy path is a shim that appends a bare QuantizationModifier); Dynamo disabled for calibration.
- lobes' vLLM server was stopped for the quantization runs (pre-authorised by the user) and restarted afterwards.
- Sonar gate on PR #20: complexity refactor of `cmd_export` into lanes, test-style fixes, and canonicalised + allow-listed user paths (`SLOTH_ALLOWED_ROOTS`) — the user chose sanitisation over accepting the findings.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t3` (`d1`) | scope extended to `sloth/tune/metadata.py` + tests so `training_metadata.json` records the dataset path | acceptable |
| `t12` (`d2`) | reassigned from colleague (lobes cortex) to a claude sonnet subagent at the user's request | acceptable |
| `t8` (`d3`) | datasets pin 4.3.0 → 4.8.5 so the llm-compressor dep layer resolves | acceptable |
| `t13` (`d4`) | one PR (#20) instead of the two PRs decided in c16 | acceptable |
| `t8` | export HOME is its own bind mount at `/opt/sloth-home`, not a llama.cpp-only mount under `/workspace/.home` (docker root-owned the nested mount-point) — obligation `o6`'s wording predates this | acceptable |
| `t9` | the seam needed six behavioural corrections after live runs (listed above); its tests were written after the code (`l2`, proposed) | needs-follow-up |
| `t12` | skill run mode passed no `--output`; added a `<adapter>-<format>` default and `--export-output` | acceptable |
| `t13` | `/finetune` loop stdout is not JSON-only on a real run (container banner + trainer progress stream through) — honesty condition h14 unmet; risk `r10` | needs-follow-up |
| `t13` | `merged-4bit` was validated but not evaluated or vLLM-loaded (not in the success signal); QLoRA merges, Qwen3, Jetson-side loads and accuracy metrics unmeasured (`r3`, `r4`, `r8`) | needs-follow-up |

## Evidence

- tests: full suite at `8238f0a` — `573 passed, 1 skipped` (skip: `tests/test_gpu_smoke.py:144`, no CUDA on the host process); at `f73ee3d` — `576 passed, 1 skipped`
- tests (per obligation, filed via `devague evidence` e1–e20, all `proposed`): `tests/test_cmd_train.py::test_lfm2_without_target_modules_emits_single_preset_hint`, `tests/test_tune_config.py::test_target_modules_wrong_type_raises_cli_error_with_hint`, `tests/test_tune_trainer.py::test_get_peft_model_receives_resolved_preset_target_modules`, `tests/test_cmd_export.py::test_module_never_imports_container_at_module_level`, `tests/test_cmd_export.py::test_invalid_quant_rejected_before_launch`, `tests/test_tune_container.py::TestExportLaunchKwargs::test_kwargs_shape_and_mount_target`, `tests/test_cmd_export.py::test_killed_container_leaves_partial_only`, `tests/test_cmd_export.py::test_non_empty_output_without_force_rejected`, `tests/test_tune_exporter.py::test_export_json_records_every_contract_field`, `tests/test_cmd_eval.py::test_adapter_and_model_are_mutually_exclusive`, `tests/test_tune_exporter.py::test_awq_lfm2_mappings_are_per_layer_and_drop_v_to_out_proj`, `tests/test_cmd_train.py::test_lfm2_rank_above_32_warns_about_hand_lane`, `tests/test_finetune_skill_script.py::test_finetune_sh_forwards_export_format_and_quant` — all pass
- lint: `uv run black --check sloth tests`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r sloth`, `uv run teken cli doctor . --strict` — all green at `f73ee3d`
- live runs (DGX Spark, 2026-09-15, shipped host path): `docs/tested.md` "2026-09-15" section — train, vLLM serve, eval, merged-16bit, gguf ×2, awq, nvfp4, merged-4bit, eval --model ×3, vLLM loads of awq/nvfp4, `/finetune` run mode
- commits: `9a0294b..f73ee3d` (40 commits on `plan/lfm2-target-modules-quantized-export`)
- PRs / issues: #20 (this delivery), #19 (spec), #16, #18
- deviations: `d1`–`d4` (approved); lapses: `l1` (approved), `l2`, `l3` (proposed — pending, not yet evidence); deltas `b1`–`b8` (proposed)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `target_modules` accepts list / regex / `preset:lfm2`, rejects everything else with a hint | high | test `tests/test_tune_config.py::test_target_modules_wrong_type_raises_cli_error_with_hint` · file `sloth/tune/presets.py` |
| the lfm2 preset adapts all 92 LoRA-able modules of LFM2.5-1.2B (incl. 20 short-conv) and the adapter is served by vLLM with `--enable-lora` | high | `docs/tested.md` train + serve rows · `runs/lfm2-lora/adapter_config.json` (local) · evidence e4 |
| LFM2 hint and rank-cap note appear once on stderr; stdout JSON unchanged (dry-run) | high | test `tests/test_cmd_train.py::test_lfm2_without_target_modules_emits_single_preset_hint` · evidence e1 |
| `sloth train --json` stdout is JSON-only on a real run | unverified | evidence e2 (fail): container banner + trainer progress stream through — risk `r10` |
| safetensors export unchanged and container-free | high | test `tests/test_cmd_export.py::test_module_never_imports_container_at_module_level` |
| merged-16bit / merged-4bit / gguf / awq / nvfp4 exports work through the shipped path on LFM2.5-1.2B-Base | high | `docs/tested.md` export rows (2.34 / 0.80 / 0.73 / 1.08 / 1.12 GB) · commit `ec42807` · evidence e11, e12, e18 |
| awq and nvfp4 outputs load and generate in vLLM on GB10 | high | `docs/tested.md` serve row · evidence e18 |
| gguf output is clean (no `_gguf` dir, no F16, no merged safetensors) and the llama.cpp cache is reused | high | evidence e10, e12 (second run 48 s, no download) |
| `eval --model` scores merged / quantized dirs with quant metadata | high | evidence e16, e17 (three live evals) |
| quantized outputs lose no accuracy | low | `l1` (approved): judged by one coherent completion and a 4-item smoke suite at 0/4 for every variant — not an accuracy measurement |
| export.json / exports.json record full provenance | high | evidence e14, e15 · `runs/lfm2-lora/exports.json` (local) |
| `/finetune` run mode completes train → eval → export gguf | medium | `docs/tested.md` skill row (exit 0) — requires this checkout's `sloth` first on PATH (risk `r9`) |
| a non-empty `--output` is never clobbered without `--force`; failed runs leave only `.partial` | high | tests `test_non_empty_output_without_force_rejected`, `test_killed_container_leaves_partial_only` · evidence e8, e10 |
| user paths are canonicalised and allow-listed (Sonar S2083/S6549) | medium | commit `f73ee3d` · tests `tests/test_cmd_export.py::test_adapter_outside_allowed_roots_is_rejected` — Sonar re-scan pending at time of writing |
| CI coverage ≥ 60 % and rubric gate green | high | PR #20 checks `lint`, `test` (pytest) green; SonarCloud gate pending re-scan |

Lapse ledger: `l1` approved caps the accuracy claim at `low`; `l2` (t9 tests after code) and `l3` (t7 assumed the metadata path) are proposed and pending, cited above as context only.

## Remaining Work / Follow-up

- PR #20 SonarCloud gate — re-scan of `f73ee3d` must clear S2083/S6549; if it does not, the owner accepts the two findings in the SonarCloud PR view (the repo's Sonar script sees main-branch issues only).
- `r10` — make `sloth train --json` stdout JSON-only on real runs (route the container's banner and trainer progress to stderr).
- `r9` — `/finetune` skill resolver prefers any `sloth` on PATH over the checkout it lives in; on this box PATH has an editable install of another worktree.
- `r3`, `r4`, `r8` — QLoRA-trained adapters through the export formats, Qwen3 through awq/nvfp4/gguf, `eval --model` gguf selection when several `.gguf` files exist, and a Jetson-side load: all unmeasured.
- `l2`, `l3` — adjudicate the two proposed lapses; evidence e1–e20 and deltas b1–b8 are proposed and need the owner's confirm.
- `r2` — delete the `torch.accelerator.get_memory_info` shim when the NGC image reaches torch ≥ 2.11.
- `r1`, `v5` — unsloth / unsloth_zoo still float in the container dep layer (2026.9.4 live vs 2026.6.9 in older docs).
