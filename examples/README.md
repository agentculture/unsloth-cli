# Examples — run configs, eval suites, demo corpus

Everything in this directory is sample data: ready-to-run TOML run configs, the
eval suites under [`eval/`](eval/), and the demo training corpus
[`demo-corpus.jsonl`](demo-corpus.jsonl). Nothing here is imported by the
package at runtime.

## Eval suites (`eval/`)

| File | Rows | Schema | What it measures |
|------|------|--------|------------------|
| `agentculture-terms.jsonl` | 16 | `task` | AgentCulture naming and sibling-ownership terminology |
| `cli-contract.jsonl` | 14 | `task` | The agent-first output contract (streams, `error:`/`hint:`, exit codes, `--json`) |
| `task-format.jsonl` | 16 | `task` | The task-format conventions (rewrite / extract / classify) |
| `regression.jsonl` | 116 | `task` | General capability — arithmetic, unit conversion, capitals, string formatting, sorting, date arithmetic |
| `instruction-following.jsonl` | 56 | `instruction` | Constraint following: `must_refuse` (12 rows), `max_words`, `min_words`, `must_contain`, `must_not_contain`, `json_only` |
| `structured-output.jsonl` | 55 | `structured` | JSON output against a `json_schema` (flat, `enum`, `items`, nested, `additionalProperties: false`) |
| `tool-call.jsonl` | 32 | `toolcall` | Emitting `expected_tool_call` for a stated tool and its parameters |
| `mmlu-subset.jsonl` | 120 | `task` | MMLU-**style** lettered multiple choice across 12 subjects (original questions — *not* MMLU rows) |

`regression.jsonl` is deliberately generic: a *base* model should already score
well on it. It is the guard that catches an adapter eroding general ability
that has nothing to do with the behavior being trained.

`sloth validate` accepts the `chat` and `task` schemas today, so
`regression.jsonl` validates directly:

```bash
uv run sloth validate --dataset examples/eval/regression.jsonl --schema task
```

The `instruction`, `structured` and `toolcall` suites are validated through
`sloth.tune.datasets.validate_suite(path, schema=...)` — see
`tests/test_examples_suites.py`. Because they are not task-schema, pointing
`--suite` at the whole `examples/eval/` directory validates every `*.jsonl`
child against the task schema and therefore rejects them; pass the task-schema
files individually until the CLI grows per-schema suite handling.

## MMLU-style subset (`eval/mmlu-subset.jsonl`)

**This is not MMLU.** Not one row comes from `cais/mmlu`, `hails/mmlu_no_train`,
or any other third-party dataset: all 120 questions and every distractor were
written for this repository, and the file carries the repository's own licence
(MIT, see [`../LICENSE`](../LICENSE)). It is an MMLU-*style* suite — the same
lettered multiple-choice shape, spread over 12 subjects (astronomy, biology,
chemistry, physics, world history, geography, mathematics, computer science,
economics, psychology, logic, nutrition) — that exists so the letter-choice
scoring path can be exercised **offline, with no download**, on a machine that
has never fetched MMLU.

For the real, quotable MMLU number, run the benchmark verb instead:

```bash
uv run sloth bench --adapter runs/qlora-smoke --benchmark mmlu
```

which runs the lm-evaluation-harness inside the NGC container (see
[`../docs/dgx-spark.md`](../docs/dgx-spark.md)).

Each row is task-schema, with the options rendered into `input` and a bare
letter as `expected_output`:

```json
{"task": "mmlu-style multiple choice (astronomy)",
 "input": "A light-year is a unit of what?\nA. Mass\nB. Time\nC. Distance\nD. Brightness\nAnswer with the letter.",
 "expected_output": "C"}
```

Because *every* `expected_output` is a single `A`-`D` letter, `sloth eval`
switches the file into **letter-choice scoring**: the model's answer letter is
extracted (tolerating `"B"`, `"B."`, `"(B)"`, `"Answer: B"`) and reported as
`choice_match` per row and `choice_acc_pct` for the suite, alongside the
unchanged `exact_match` numbers.

```bash
uv run sloth eval --adapter runs/qlora-smoke --suite examples/eval/mmlu-subset.jsonl
```

The answer key is exactly balanced — 30 rows per letter — so always guessing a
fixed letter scores chance (25%), not 50%. Regenerate the committed file with:

```bash
uv run python examples/generate_mmlu_subset.py
```

Like `generate_suites.py` it is pure stdlib and fully deterministic;
`tests/test_examples_suites.py` asserts both the determinism and the balance.

## Demo corpus (`demo-corpus.jsonl`)

591 `chat`-schema rows (optional `system`, then `user`, then `assistant`) that
teach the answers the three hand-authored suites score:
`agentculture-terms.jsonl`, `cli-contract.jsonl` and `task-format.jsonl`. It is
a worked demonstration of the fine-tune-versus-retrieval boundary — it teaches
*stable behavior and terminology*, not changing project facts.

Every row is paraphrased. No eval row is reproduced: the test suite asserts
`overlap_check(demo-corpus, [all seven eval suites]) == []`, so nothing the
corpus trains on is also scored.

## Provenance and licence

The four generated eval suites and the demo corpus are produced by
[`generate_suites.py`](generate_suites.py), a pure-stdlib, fully deterministic
script — no randomness, no clock reads, no network. Its content is written from
this repository's own documentation (`CLAUDE.md`, `docs/fine-tuning.md` and the
`sloth/explain/` catalog) plus generic, non-proprietary general-knowledge items
(arithmetic, capital cities, unit conversions, dates). No third-party dataset,
benchmark or scraped corpus is included.

Regenerate the committed files with:

```bash
uv run python examples/generate_suites.py
```

Running it twice produces byte-identical output; `tests/test_examples_suites.py`
asserts that.

Licence: the same as the repository — MIT, see [`../LICENSE`](../LICENSE).
