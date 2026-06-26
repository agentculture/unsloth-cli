# unsloth-cli ships Spark-friendly LoRA/QLoRA adapter fine-tuning for Qwen models

> unsloth-cli ships Spark-friendly LoRA/QLoRA adapter fine-tuning for Qwen
> models — flat `sloth train` / `sloth eval` / `sloth export` verbs that validate
> JSONL datasets before spending GPU, run small reproducible config-driven
> (TOML) adapter jobs, and write training metadata next to the adapter — plus a
> `/finetune` skill that drives the loop. The unsloth/torch stack ships with the
> install (`uv tool install unsloth-cli`); the introspection verbs stay fast via
> lazy imports.

<!-- Note: the headline above was reconciled post-export to match confirmed
     decisions c9 (flat verbs, not a 'tune' noun group) and c10 (unsloth/torch is
     a BASE dependency, not an optional extra). The original devague announcement
     string predated those decisions. -->

> _Provenance: converged devague frame
> `.devague/unsloth-cli-ships-spark-friendly-lora-qlora-adapte.json`; built from
> [issue #6](https://github.com/agentculture/unsloth-cli/issues/6)._

## Audience

- AgentCulture mesh agents and their operators (Spark first) doing local, Jetson/Spark-class adapter tuning of Qwen models — plus the downstream siblings that consume the result: lobes (serves the adapter) and colleague (runs it as a model backend).

## Before → After

- Before: Today unsloth-cli is the scaffold only: introspection verbs (whoami/learn/explain/overview/doctor/cli) but zero fine-tuning. There is no reproducible, agent-friendly way to train a role-specific adapter — you'd hand-roll a torch/unsloth script with no dataset validation, no run metadata, and none of the CLI's error/exit-code contract.
- After: An agent or operator goes from a JSONL dataset to a validated, reproducible LoRA/QLoRA adapter — training metadata written next to the output and a config file for repeatable runs — locally on a Spark, driven by the flat CLI verbs (sloth train / sloth eval / sloth export) or the controlling skill. The full unsloth/torch stack ships with the install via 'uv tool install unsloth-cli'.

## Why it matters

- Fine-tuning is how the mesh stores STABLE behavior and reflexes (CLI-contract discipline, AgentCulture/CULTURE.DEV terminology, agent-first habits, issue-writing format, teacher behavior) directly into the models its siblings run — while memory/RAG keeps the CHANGING facts. This adapter-tuning loop is the repo's actual reason to exist; today it's only scaffold.

## Honesty conditions

- Running an introspection verb (e.g. sloth whoami) does NOT trigger a torch import — top-level import of the sloth package stays torch-free via lazy imports — so introspection startup is fast and 'teken cli doctor . --strict' stays green even with torch installed as a base dep.
- The verbs and the /finetune skill are usable by an agent non-interactively (every verb supports --json and routes errors via error:/hint:), and the produced adapter uses a standard PEFT/safetensors layout that lobes can serve and colleague can run as a backend.
- The README draws an explicit, actionable line between what belongs in a fine-tune (stable behavior/reflexes) and what belongs in memory/RAG (changing facts) — concrete enough that a contributor can decide where a given capability goes.
- train/eval/export are genuinely net-new verbs (no fine-tuning path exists in the repo today), so the scaffold-only baseline is accurate and this is not re-implementing an existing flow.
- The eval verb runs an adapter against a small LOCAL suite with no network, and the afi rubric gate stays green: every new verb/noun has a catalog entry, supports --json, routes errors through error:/hint:, and any action-noun exposes 'overview'.
- Re-running the same config file + dataset reproduces the same training setup, and a metadata file written next to the adapter records model, method, dataset hash/size, hyperparameters, and a timestamp.
- Pointing train at a large dense full-fine-tune target emits an explicit out-of-scope warning (and refuses or clearly downgrades to adapter-only) rather than silently attempting it.

## Success signals

- Issue #6's acceptance criteria all pass: validate JSONL before training, run a small Qwen LoRA/QLoRA job end-to-end, config files drive repeatable runs, training metadata written next to the adapter, an eval command against a small local suite, documented Spark-friendly defaults, an explicit out-of-scope warning for large dense full-FT, and a README section on the fine-tune vs memory/RAG vs retrieval split. AND the afi rubric gate (teken cli doctor --strict) stays green.

## Scope / boundaries

- NOT full fine-tuning of large dense models — the CLI must explicitly WARN this is out of scope (and refuse/downgrade to adapter-only). NOT a serving/inference server (that's lobes). NOT memory/RAG or retrieval (separate concern; the README must explain the fine-tune vs RAG vs retrieval split). NOT distributed/multi-GPU cluster training. (Note: 'no hard torch dependency' is deliberately NO LONGER a boundary — see decision c10; unsloth/torch is a base dep.)

## Non-goals

- The first implementation is intentionally minimal: reproducibility and agent-friendliness matter more than breadth of model/method coverage. We are not chasing every Unsloth-supported model or PEFT method in v1.

## Decisions

- Verb topology: FLAT top-level verbs — sloth train / sloth eval / sloth export — sitting alongside the introspection verbs (not a 'tune' noun group). Rubric-legal because global verbs don't require an 'overview' sub-verb.
- Dependency policy REVERSED for this feature: unsloth (and its torch stack) is a BASE runtime dependency in pyproject [project].dependencies — 'uv tool install unsloth-cli' brings the full tuning stack. Product-owner decision: uv is the default installer and bundling everything beats an optional extra. CONSEQUENCE: the prior load-bearing 'dependencies = []' / zero-runtime-deps rule is retired; CLAUDE.md (Zero runtime dependencies, Architecture, forward-work) and any lint/CI assumptions must be rewritten to match.
- Even though unsloth is a base dep, handlers LAZY-import torch/unsloth inside the function body (never at module top-level), so introspection verbs (whoami/learn/explain/doctor/overview/cli) keep fast startup and the afi rubric gate (teken cli doctor --strict) stays green — the gate checks the CLI contract, not dependency count.
- Run-config format: TOML, parsed with stdlib tomllib (read-only, available on the >=3.12 floor). Human-friendly, supports comments, matches pyproject.toml. Configs are inputs so read-only is fine.
- Controlling skill is named /finetune; it drives the full loop — validate dataset -> sloth train -> sloth eval -> sloth export — wrapping the flat CLI verbs.

## Hard questions

- Does 'validate JSONL before spending GPU' live in the zero-dep core (always available) while only the actual training step is gated behind the extra? If validation needed torch, the 'works without the ML stack' promise breaks.
- risk: Because unsloth/torch is now a BASE dependency, 'uv tool install unsloth-cli' will FAIL entirely on a machine where the torch/unsloth wheels don't resolve (e.g. an unsupported arch). The introspection CLI is no longer guaranteed to install everywhere — that property is traded away by this decision. Spark (ARM/Blackwell) wheel availability must be verified.
- risk: Unsloth/torch versions move fast and are GPU/arch-specific (Spark = ARM/Blackwell). A pinned ML extra may not resolve on every machine; the subprocess-vs-extra choice affects how badly this bites.
