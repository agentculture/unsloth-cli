# Build Plan — unsloth-cli runs real LoRA/QLoRA fine-tuning on NVIDIA DGX Spark by orchestrating NVIDIA's official NGC PyTorch container, instead of carrying torch/unsloth as pip dependencies

slug: `unsloth-cli-runs-real-lora-qlora-fine-tuning-on-nv` · status: `exported` · from frame: `unsloth-cli-runs-real-lora-qlora-fine-tuning-on-nv`

> unsloth-cli runs real LoRA/QLoRA fine-tuning on NVIDIA DGX Spark by orchestrating NVIDIA's official NGC PyTorch container, instead of carrying torch/unsloth as pip dependencies

## Tasks

### t1 — Packaging: drop torch+unsloth from deps, register gpu pytest marker, version-bump + CHANGELOG

- covers: c6, c9
- acceptance:
  - pyproject [project].dependencies contains neither torch nor unsloth; a fresh 'uv sync' installs neither (pip list shows no torch/unsloth)
  - a 'gpu' pytest marker is registered in [tool.pytest.ini_options] markers (no PytestUnknownMarkWarning when used)
  - version bumped via the version-bump skill and a Keep-a-Changelog entry prepended to CHANGELOG.md

### t2 — Orchestration core: new pure-stdlib sloth/tune/container.py (preflight + docker-run builder + uv dep-layer + bind-mount)

- covers: c15, c16, c22
- acceptance:
  - sloth/tune/container.py imports cleanly with torch NOT installed (stdlib + subprocess only, imports no torch)
  - the docker-command builder yields 'docker run --gpus all' with '--ulimit memlock=-1' '--ulimit stack=67108864', the pinned image nvcr.io/nvidia/pytorch:25.11-py3, a bind-mount of the workdir+checkout, and 'uv pip install --system' (never a bare 'pip install')
  - preflight() raises CliError(code=2) whose remediation names nvcr.io/nvidia/pytorch:25.11-py3 + nvidia-container-toolkit when docker is absent / image unpullable / nvidia runtime missing (stubbed)

### t3 — Trainer real path: _trainer.py wraps records as datasets.Dataset.from_list, maps no-accelerator to code=2, becomes in-container entrypoint

- covers: c1, c7, c8
- acceptance:
  - _run_real wraps train records via datasets.Dataset.from_list(...) before constructing SFTTrainer (asserted with a fake backend capturing the train_dataset type)
  - a NotImplementedError ('cannot find any torch accelerator') from the backend is caught and re-raised as CliError(code=2) with the NGC hint, not surfaced as code=1
  - the real path is invocable as the in-container entrypoint (python -m sloth runs _run_real) and still lazy-imports torch/unsloth inside the function body

### t4 — train handler: host-side GPU-free preflight before container.launch; --dry-run prints the docker command

- depends on: t2, t3
- covers: c14
- acceptance:
  - host-side preflight (dataset schema + TOML config + scope guard) runs BEFORE any container launch; with an invalid dataset or out-of-scope model, train returns the user/scope error and never calls container.launch/docker (asserted by stubbing container)
  - the real (non-dry-run) path calls container.launch with the resolved plan after preflight passes
  - --dry-run prints the resolved docker image + command and launches no container and imports no torch

### t5 — eval handler: load adapter via PeftModel.from_pretrained(base, adapter); route GPU work through the container

- depends on: t2
- covers: c7
- acceptance:
  - eval reads base_model_name_or_path from <adapter>/adapter_config.json and loads via PeftModel.from_pretrained(base, adapter), not AutoModelForCausalLM.from_pretrained(adapter)
  - eval routes its GPU/ML-stack work through container.launch (asserted by stubbing container)

### t6 — export handler: produce standard PEFT/safetensors layout; route ML-stack work through the container if needed

- depends on: t2
- covers: c7
- acceptance:
  - export produces a standard PEFT/safetensors adapter layout for a trained adapter dir
  - if export needs the ML stack it routes through container.launch; if pure-stdlib (file reorg + metadata) it launches no container and records that decision in the task notes

### t7 — GPU smoke test: gpu-marked real-backend tiny train->eval->export, skipped on CPU CI

- depends on: t1, t3, t4, t5, t6
- covers: c17, h4, h5, h11
- acceptance:
  - tests/test_gpu_smoke.py carries @pytest.mark.gpu and skipif(not torch.cuda.is_available()); under CPU-only pytest it is collected-but-skipped and the suite stays green
  - when a CUDA device is present it runs a tiny Qwen LoRA train -> eval -> export and asserts >0 steps, an adapter dir + training metadata, and eval scores

### t8 — Container unit tests: import-light, preflight code=2, deterministic uv-not-pip command

- depends on: t2
- covers: h2, h3, h12, h14
- acceptance:
  - asserts sloth/tune/container.py imports with torch absent (import-light preserved)
  - asserts preflight returns exit code 2 with a hint naming nvcr.io/nvidia/pytorch:25.11-py3 when docker missing / runtime absent (subprocess stubbed)
  - golden-string test asserts the built docker command is deterministic for a given config and uses 'uv pip install --system' (never a bare 'pip install')

### t9 — Train/trainer unit tests: bad input never invokes docker; Dataset wrap + code=2 mapping asserted

- depends on: t3, t4
- covers: h1
- acceptance:
  - asserts a bad dataset / out-of-scope model returns the user/scope error WITHOUT invoking docker (container stubbed and asserted not-called)
  - asserts _run_real passes a datasets.Dataset (not a list[dict]) to SFTTrainer and maps the no-accelerator NotImplementedError to CliError code=2

### t10 — Import-light packaging test: deps free of torch/unsloth; introspection runs without torch

- depends on: t1
- covers: h9, h10, h13
- acceptance:
  - test parses pyproject and asserts neither torch nor unsloth appears in [project].dependencies nor as an optional extra
  - test imports sloth.cli + whoami with torch absent and asserts 'sloth whoami' exits 0; no vendored wheels added; existing scope-guard tests still pass

### t11 — Docs: README DGX Spark/NGC section + both-audiences + before-state; reverse CLAUDE.md base-dep section

- depends on: t1
- covers: c2, c3, c5, h6, h7
- acceptance:
  - README gains a DGX Spark section: the NGC container path (nvcr.io/nvidia/pytorch:25.11-py3), the uv-only dep layer, the UMA drop_caches note, and documents BOTH audiences (GPU operators + import-light introspection-only install)
  - README records the before-state (CPU-only torch 2.10.0+cpu on aarch64 raising 'cannot find any torch accelerator') and why the change was made
  - CLAUDE.md 'Base runtime dependency + lazy imports' section is reversed to reflect the no-torch / manage-docker decision

## Risks

- [follow_up] DGX Spark UMA can OOM even within memory capacity; document NVIDIA's 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"' flush and keep conservative batch_size defaults
- [unknown_nonblocking] uv pip install --system in-container diverges from NVIDIA's pip-tested recipe; the working set (unsloth --no-deps + unsloth_zoo + bitsandbytes + trl==0.26.1 against the container torch) must be validated on GB10 hardware, which CI cannot do (task t2)
- [unknown_nonblocking] the 'demonstrated on hardware' targets (c1/h5/h11) are only verifiable by a manual Spark run; the gpu smoke test is skipped in CI (task t7)
- [unknown_nonblocking] whether 'export' needs the ML stack/container vs being pure-stdlib is unconfirmed; t6 must determine and the plan adapts (task t6)
