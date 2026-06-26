# Build Plan — unsloth-cli ships Spark-friendly LoRA/QLoRA adapter fine-tuning for Qwen models

slug: `unsloth-cli-ships-spark-friendly-lora-qlora-adapte` · status: `exported` · from frame: `unsloth-cli-ships-spark-friendly-lora-qlora-adapte`

> Flat `sloth train` / `sloth eval` / `sloth export` verbs that validate JSONL
> datasets before spending GPU, run small reproducible TOML-config-driven adapter
> jobs, and write training metadata next to the adapter — plus a `/finetune`
> skill that drives the loop. The unsloth/torch stack ships with the install
> (`uv tool install unsloth-cli`); introspection verbs stay fast via lazy imports.

<!-- Headline reconciled post-export to match confirmed decisions c9 (flat verbs,
     not a 'tune' noun group) and c10 (unsloth/torch is a BASE dependency, not an
     optional extra). The devague announcement string predates those decisions.

     Dependency waves (from `devague plan waves`):
       wave 0: t1 t2 t3 t4 t5 t6 t7 t8   (disjoint files, no torch)
       wave 1: t9 t11 t12
       wave 2: t10
       wave 3: t13
       wave 4: t14
     Same-wave tasks touch disjoint files — safe to fan out to parallel agents. -->

## Build order (dependency waves)

| Wave | Tasks | Touches |
|------|-------|---------|
| 0 | t1 t2 t3 t4 t5 t6 t7 t8 | `sloth/tune/{datasets,config,metadata,scope}.py`, `tests/test_lazy_import.py`, `pyproject.toml`, `README.md`, `CLAUDE.md` |
| 1 | t9 t11 t12 | `sloth/tune/_trainer.py`, `sloth/cli/_commands/{eval,export}.py` |
| 2 | t10 | `sloth/cli/_commands/train.py` |
| 3 | t13 | `sloth/cli/__init__.py`, `sloth/explain/catalog.py` |
| 4 | t14 | `.claude/skills/finetune/` |

## Tasks

### t1 — Dataset validation module (sloth/tune/datasets.py): validate JSONL chat + task schemas, pure stdlib, no torch

- covers: c7, c12
- acceptance:
  - Valid chat JSONL ({messages:[{role,content}]}) and task JSONL ({task,input,expected_output}) pass; malformed lines (bad JSON, missing/extra keys, wrong types) raise CliError carrying the offending line number + a hint
  - Validation runs with no torch installed; a test asserts 'torch' absent from sys.modules after running it

### t2 — TOML run-config loader (sloth/tune/config.py) via stdlib tomllib with Spark-friendly defaults

- covers: c12, h6, c7
- acceptance:
  - A valid run.toml (model, method, dataset, output, hyperparams) loads into a typed config; missing required keys raise CliError with hint; a method other than lora/qlora is rejected
  - Omitted optional fields fall back to documented Spark-friendly defaults; the same file loads identically on repeat (deterministic)

### t3 — Training-metadata writer (sloth/tune/metadata.py): write run metadata next to the adapter output

- covers: h6, c12, c7
- acceptance:
  - After a (mocked) run, a metadata file beside the adapter records model, method, dataset sha256 + line count, hyperparameters, and an ISO-8601 timestamp; re-reading round-trips every field

### t4 — Model-scope guard (sloth/tune/scope.py): classify adapter-OK vs out-of-scope large-dense full-FT

- covers: c13, h7
- acceptance:
  - A LoRA/QLoRA target on a supported small/medium Qwen returns OK; a large-dense full-fine-tune request returns an explicit out-of-scope warning and refuses or downgrades to adapter-only — asserted by test

### t5 — Lazy-import discipline guard test (tests/test_lazy_import.py)

- covers: h5
- acceptance:
  - Importing the sloth package and running 'sloth whoami' leaves 'torch' and 'unsloth' absent from sys.modules; the test fails if any imported module pulls torch at top level

### t6 — Add unsloth + torch to [project].dependencies (base) in pyproject.toml

- covers: c12, c7
- acceptance:
  - pyproject [project].dependencies lists unsloth (with torch as required); 'uv lock' resolves; introspection verbs still run and the lazy-import test stays green

### t7 — README: fine-tune vs memory/RAG vs retrieval split, Spark-friendly defaults, out-of-scope warning

- covers: c4, h9, c7
- acceptance:
  - README has a section drawing an actionable line between fine-tune (stable reflexes) and memory/RAG (changing facts), documents Spark-friendly defaults, and states full-FT of large dense models is out of scope

### t8 — CLAUDE.md: retire the dependencies=[] rule; document train/eval/export + base-dep + lazy-import policy

- covers: c5, c13
- acceptance:
  - CLAUDE.md no longer asserts 'dependencies = []' as load-bearing; it documents unsloth/torch as a base dep, the lazy-import discipline, and the three new verbs via the 'Adding a verb or noun group' seam

### t9 — Lazy LoRA/QLoRA trainer adapter (sloth/tune/_trainer.py): lazy-import unsloth/torch inside the run function

- depends on: t2, t3, t4
- covers: c7, c12
- acceptance:
  - torch/unsloth import only inside the run function; when unsloth is absent the function raises CliError(code=2) with an install hint ('uv tool install unsloth-cli'); a dry-run path returns the resolved plan without importing torch
  - Module has no top-level torch/unsloth import — the lazy-import guard test (t5) stays green

### t10 — 'sloth train' verb (sloth/cli/_commands/train.py): validate -> scope-guard -> (dry-run|train) -> write metadata

- depends on: t1, t2, t3, t4, t9
- covers: c7, c12, c13, h7
- acceptance:
  - 'sloth train --config run.toml --dry-run' validates the dataset, prints the resolved plan in text and --json, exits 0 without importing torch; an invalid dataset exits 1 with error:/hint:
  - 'sloth train' at a large-dense full-FT target emits the out-of-scope warning (t4) and refuses; a real run writes metadata (t3) next to the adapter

### t11 — 'sloth eval' verb (sloth/cli/_commands/eval.py): run an adapter against a small LOCAL suite, no network

- depends on: t1, t4
- covers: h3, c7
- acceptance:
  - 'sloth eval --adapter <dir> --suite <file>' runs offline (test asserts no socket/network access) and emits results in text and --json; a missing adapter/suite exits 1 with error:/hint:

### t12 — 'sloth export' verb (sloth/cli/_commands/export.py): adapter -> safetensors

- depends on: t3, t4
- covers: h8, c7
- acceptance:
  - 'sloth export --adapter <dir> --format safetensors' writes a standard PEFT/safetensors adapter layout (lobes-servable / colleague-runnable); an unsupported --format exits 1 with error:/hint:; --json reports the output path

### t13 — Wire train/eval/export into _build_parser + add explain catalog entries

- depends on: t10, t11, t12
- covers: c1, h3, h8, h10, c5
- acceptance:
  - _build_parser registers train/eval/export; explain resolves ('train',)/('eval',)/('export',) so test_every_catalog_path_resolves stays green; each verb supports --json and routes errors via error:/hint:
  - 'uv run teken cli doctor . --strict' exits 0 (afi rubric gate green) with the three verbs present

### t14 — /finetune skill (.claude/skills/finetune/SKILL.md + scripts/finetune.sh): drive validate -> train -> eval -> export

- depends on: t13
- covers: h8, c1, c12, c2
- acceptance:
  - SKILL.md carries 'type: command'; the script drives the four steps non-interactively, forwards --json, and surfaces error:/hint: from the CLI; a sample run produces an adapter + metadata
