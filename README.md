# unsloth-cli

Agent + CLI that simplifies fine-tuning with Unsloth, adding complementary actions so an agent can fine-tune models more easily.

## What you get

- **An agent-first CLI** cited from [teken](https://github.com/agentculture/teken)
  (`afi-cli`) — the runtime package has no third-party dependencies.
- **A mesh identity** — `culture.yaml` (`suffix` + `backend`) and the matching
  prompt file (`CLAUDE.md` for `backend: claude`).
- **The canonical guildmaster skill kit** (11 skills) under `.claude/skills/`,
  vendored cite-don't-import. See [`docs/skill-sources.md`](docs/skill-sources.md).
- **A build + deploy baseline** — pytest, lint, the agent-first rubric gate, and
  PyPI Trusted Publishing wired into GitHub Actions.

## Quickstart

```bash
uv sync
uv run pytest -n auto                 # run the test suite
uv run sloth whoami                   # identity from culture.yaml
uv run sloth learn                    # self-teaching prompt (add --json)
uv run teken cli doctor . --strict    # the agent-first rubric gate CI runs
```

The installed console script is `sloth` (the dist name is `unsloth-cli`); run
`sloth <verb>` or `python -m sloth <verb>`. The CLI prints `unsloth-cli` in its
help/`explain` text because that is the argparse program name.

## CLI

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path. |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `cli overview` | Describe the CLI surface itself. |

Every command supports `--json`. Results go to stdout, errors/diagnostics to
stderr (never mixed). Exit codes: `0` success, `1` user error, `2` environment
error, `3+` reserved.

## Fine-tuning

unsloth-cli ships three flat verbs for LoRA/QLoRA adapter tuning of Qwen models,
plus a `/finetune` skill that drives the full loop. torch + unsloth are **not**
installed as pip dependencies — they run inside an NGC Docker container that the
fine-tuning verbs orchestrate. The introspection verbs (`whoami`, `learn`, `explain`,
etc.) install and start everywhere, with no GPU stack required.

### Out of scope

**Full fine-tuning of large dense models is not supported.** The CLI targets
LoRA and QLoRA adapters on small-to-medium Qwen models (Qwen 3.x 4B / 9B and
comparable adapter-class targets). Pointing `sloth train` at a large dense
full-fine-tune target emits an explicit warning and refuses or downgrades to
adapter-only — it does not attempt the job silently.

### Commands

| Verb | What it does |
|------|--------------|
| `sloth train` | Validate JSONL dataset → run LoRA/QLoRA adapter job → write run metadata |
| `sloth eval` | Run an adapter against a small local eval suite (no network) |
| `sloth export` | Convert an adapter to safetensors (servable by lobes, runnable by colleague) |

The `/finetune` skill drives the full loop non-interactively:
validate dataset → `sloth train` → `sloth eval` → `sloth export`.

Every verb supports `--json` and routes errors through `error:` / `hint:` on stderr.

### DGX Spark / NGC container

The `train`, `eval`, and `export` verbs execute inside NVIDIA's official PyTorch NGC
container (`nvcr.io/nvidia/pytorch:25.11-py3`), which ships a Blackwell-ready torch
build. The verbs bind-mount the repo checkout into the container and install the
fine-tuning dep layer with uv (never pip):

```bash
# In-container dep layer (installed automatically by sloth train / eval / export)
uv pip install --system transformers peft hf_transfer 'datasets==4.3.0' 'trl==0.26.1'
uv pip install --system --no-deps unsloth unsloth_zoo bitsandbytes
```

**Prerequisites** (GPU operators only — not needed for the introspection verbs):

- CUDA 13 drivers
- `nvidia-container-toolkit` installed and configured
- Docker with GPU access: `docker run --gpus all` must succeed

**Two audiences:**

- **GPU operators** running `sloth train` / `sloth eval` / `sloth export`: you need
  the NGC image and the prerequisites above. The verbs pull the image and orchestrate
  the container automatically; the dep layer is installed inside the container on each
  run.
- **Introspection-only users** running `sloth whoami` / `sloth learn` / `sloth explain`
  / `sloth doctor`: no GPU, no Docker, no torch required.
  `uv tool install unsloth-cli` installs only the pure-stdlib introspection CLI, which
  works on every architecture including aarch64 / DGX Spark GB10.

**Why the NGC container?** Earlier versions of unsloth-cli listed torch + unsloth as
base `[project].dependencies`. On aarch64 (DGX Spark GB10, Blackwell), `uv sync`
resolved to `torch==2.10.0+cpu` — the CPU-only wheel — and the real training path
aborted with `"cannot find any torch accelerator"`. Moving the GPU stack into the NGC
container removes the wheel-resolution problem: the container already ships a
Blackwell-native torch, and the introspection CLI installs cleanly everywhere again.

**UMA / out-of-memory note** (DGX Spark unified memory architecture): if a training
run exhausts unified memory, flush the page cache before retrying:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Dataset schemas

Two JSONL schemas are supported. Validation runs before spending any GPU time;
malformed lines are reported with the offending line number and a remediation hint.

**Chat format** — for instruction-following and conversational behavior:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Task format** — for structured input/output tasks:

```json
{"task": "write-issue", "input": "...", "expected_output": "..."}
```

### Run config (TOML) and Spark-friendly defaults

Training runs are driven by a TOML config file. Omitted optional keys fall back
to Spark-friendly defaults tuned for small-GPU (single-card Spark) operation.

```toml
[run]
model   = "unsloth/Qwen3-4B"      # supported: Qwen3 4B / 9B adapter-class targets
method  = "qlora"                 # "lora" or "qlora" — the only supported methods (default: qlora)
dataset = "data/train.jsonl"
output  = "adapters/my-lora"

[hyperparameters]
lora_r        = 16                # LoRA rank          (default: 16)
lora_alpha    = 16                # LoRA alpha scaling (default: 16)
lora_dropout  = 0.0               # default: 0.0
learning_rate = 2e-4              # default: 2e-4
max_seq_len   = 2048              # default: 2048
batch_size    = 2                 # default: 2  (Spark-friendly: keeps VRAM low)
grad_accum    = 4                 # default: 4
max_steps     = 60                # default: 60 (quick smoke-run; raise for production)
seed          = 3407              # default: 3407
load_in_4bit  = true              # default: true (required for qlora)
```

A metadata file is written next to the adapter output recording model, method,
dataset SHA-256 and line count, hyperparameters, and an ISO-8601 timestamp.
Re-running the same config file and dataset reproduces the same training setup.

### What belongs in fine-tuning vs. memory / RAG

This is a design rule, not a footnote. The fine-tune/RAG boundary decides where
a capability lives in the mesh.

**Fine-tune** stores *stable behavior and reflexes* — things that should be
baked into how the model responds, not looked up on every call:

- CLI-contract discipline (error/hint format, exit-code policy, stream split)
- AgentCulture / CULTURE.DEV terminology and patterns
- Agent-first habits (prefer action verbs, emit structured `--json`, route errors correctly)
- Issue-writing format and AgentCulture PR/review norms
- Teacher behavior for `learn` and `explain` responses

**Memory / RAG** stores *changing facts* — things that vary per session, user,
or deployment and would become stale if baked into weights:

- Current project state, open issues, branch status, recent commits
- Secrets, tokens, credentials, or any per-deployment configuration
- User-specific preferences or operator-specific memory
- Facts better served by retrieval (live documentation, changelogs, external APIs)

**Decision rule for contributors:** *"Would this still be correct six months from
now on any deployment of the mesh?"* If yes, consider fine-tuning. If it changes
over time or is per-user, use memory / RAG.

### Role-specific adapters

The design targets small, role-specific adapters rather than one large mixed blob.
Example adapter names that map to discrete behaviors:

- `culture-contract-lora` — CLI-contract discipline and AgentCulture norms
- `agentculture-cli-teacher-lora` — teacher behavior for `learn` / `explain`
- `repo-maintainer-lora` — issue-writing format and PR review norms
- `tool-router-lora` — tool selection and routing decisions
- `agent-first-coach-lora` — agent-first habits and patterns

The resulting adapters are written in standard PEFT / safetensors layout so
[lobes](https://github.com/agentculture/lobes-cli) can serve them and
[colleague](https://github.com/agentculture/colleague) can run them as model backends.

## Make it your own

1. Rename the package `sloth/` and the `unsloth-cli`
   CLI/dist name throughout `pyproject.toml`, the package, `tests/`,
   `sonar-project.properties`, and this `README.md`. The name is hard-coded in
   ~100 places, so list every occurrence first — see the `git grep` discovery
   command in [`CLAUDE.md`](CLAUDE.md), the authoritative rename procedure.
2. Edit `culture.yaml` with your `suffix` and `backend`.
3. Rewrite `CLAUDE.md` for your agent and run `/init`.
4. Re-vendor only the skills you need from guildmaster (see
   [`docs/skill-sources.md`](docs/skill-sources.md)).

See [`CLAUDE.md`](CLAUDE.md) for the full conventions (version-bump-every-PR,
the `cicd` PR lane, deploy setup).

## License

MIT — see [`LICENSE`](LICENSE).
