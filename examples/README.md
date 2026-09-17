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
