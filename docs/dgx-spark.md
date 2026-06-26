# Running unsloth fine-tuning on the DGX Spark (GB10)

This is the operator guide for running `sloth train` / `sloth eval` / `sloth
export` on an NVIDIA **DGX Spark** (GB10, Blackwell, aarch64). It captures the
prerequisites, how the NGC-container orchestration works, the **validated**
dependency set, and the Spark-specific gotchas that this path actually hits —
each one discovered on hardware, not in theory.

For measured numbers see [`benchmarks.md`](benchmarks.md); for the feature/CLI
reference see [`fine-tuning.md`](fine-tuning.md).

## Why a container at all

The GPU stack (torch + unsloth) is **not** a pip dependency of unsloth-cli.
`uv tool install unsloth-cli` installs only the pure-stdlib introspection CLI,
which works on every architecture. The fine-tuning verbs run the GPU work inside
NVIDIA's official **NGC PyTorch container** (`nvcr.io/nvidia/pytorch:25.11-py3`),
which already ships a **Blackwell-native torch 2.10 (CUDA 13)**. This sidesteps
the wheel-resolution trap on aarch64, where a bare `pip/uv install torch` resolves
to the CPU-only wheel (`torch==2.10.0+cpu`) and training aborts with *"cannot find
any torch accelerator."*

`sloth train`/`eval` build a deterministic `docker run`, bind-mount your checkout,
working dir, and Hugging Face cache, install the dependency layer **with uv**
inside the container, and run `python -m sloth … --in-container`. `sloth export`
is pure stdlib and needs no container at all.

## Prerequisites

| Requirement | Check |
|-------------|-------|
| NVIDIA driver + GPU | `nvidia-smi -L` lists the GB10 |
| Docker | `docker --version` |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi -L` succeeds |
| NGC image pulled | `docker pull nvcr.io/nvidia/pytorch:25.11-py3` (~19.5 GB; pull once) |

> On this DGX Spark, `docker --gpus all` works via **CDI** even though Docker's
> default runtime is `runc` (no `nvidia` runtime registered) — Docker 29 + the
> container toolkit handle it. If `--gpus all` fails, install/۰configure
> `nvidia-container-toolkit`.

If any check fails, `sloth train` exits **2** (environment error) with a
remediation naming the NGC image and the toolkit package — it does not start a
doomed run.

## The validated dependency set (and why it's pinned)

Inside the container the dep layer is installed into a **`--system-site-packages`
venv** (see "Gotchas" below). The pins are **load-bearing**, validated against
NGC 25.11's torch 2.10:

```text
transformers==4.57.1   peft==0.18.0   trl==0.24.0   datasets==4.3.0   hf_transfer
unsloth  unsloth_zoo  bitsandbytes        # installed --no-deps
torchao  → left at the container's 0.14.0+git (NOT upgraded)
```

The reason this exact set matters — a real version-matrix deadlock:

- `unsloth 2026.6.9` requires `torch<2.11`, `peft>=0.18.0`, `trl<=0.24.0`
  (so the previous `trl==0.26.1` pin was **out of range**), and a specific
  `transformers` window.
- `peft 0.19+` **hard-requires `torchao>0.16`** at `get_peft_model` time.
- But `torchao>0.16` (0.17+) needs `torch>=2.11` (it imports
  `torch.nn.functional.ScalingType`), which NGC 25.11 does **not** have.
- `unsloth_zoo` only needs `torchao>=0.13`, so the container's 0.14 is fine.

→ Hold **peft at 0.18.x** (≥ unsloth's floor, < the torchao-0.16 demand), pair it
with **transformers 4.57.1** and **trl 0.24.0**, and leave torchao alone. These
are the values in `sloth/tune/container.py::DEP_LAYER_PACKAGES`.

## Gotchas discovered on hardware

These are the things that *will* bite a naive "just pip install unsloth in the
container" attempt. unsloth-cli's orchestration already handles all of them.

### 1. Unified Memory OOM at unsloth import → `PYTORCH_ALLOC_CONF`

On the GB10, GPU memory **is** system memory (Unified Memory Architecture). When
the box is busy, unsloth's GPU probe at import can raise
`AcceleratorError: CUDA error: out of memory` *before any model loads*. Setting
**`PYTORCH_ALLOC_CONF=expandable_segments:True`** (the container sets it
automatically, plus the deprecated `PYTORCH_CUDA_ALLOC_CONF` alias) avoids the
large up-front reservation and lets the run proceed. `torch.cuda.mem_get_info()`
under-reports free memory on UMA (it returned ~3.6 GB while a 6 GB tensor still
allocated), because allocations grow by evicting page cache.

If a run is still killed (exit 137 / SIGKILL), free memory and flush the page
cache, then retry:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

…or drop `batch_size` / `max_seq_len`, or use `method="qlora"` (4-bit).

### 2. `uv pip install --system` fails — use a `--system-site-packages` venv

The NGC image's Python is **PEP-668 externally-managed**, so
`uv pip install --system` fails *as root* (`externally-managed-environment`), and
the system `dist-packages` is **root-owned**, so it also fails under the
host-user (`--user uid:gid`) that the orchestration uses for correct output
ownership. The fix the orchestration uses: create a
`uv venv --system-site-packages` under `$HOME` (writable, inherits the container's
torch/torchao) and install the dep layer into it.

### 3. Mount the Hugging Face cache

The container is `--rm` (ephemeral). Without mounting the host HF cache, every run
re-downloads the base model. The orchestration bind-mounts your
`~/.cache/huggingface` to `/opt/hf-cache` and points `HF_HOME` there, so models
are downloaded once and reused.

### 4. trl/unsloth API specifics (handled in the trainer)

The real trainer (`sloth/tune/_trainer.py`) encodes several API facts that only
surface at run time on this stack:

- **Import unsloth first.** Imported after trl/transformers/peft, unsloth's
  patches don't apply and trl's `SFTConfig` `<EOS_TOKEN>` sentinel is left
  unpatched → `"eos_token '<EOS_TOKEN>' is not found in the vocabulary"`.
- **trl 0.24 renamed `tokenizer=` → `processing_class=`** on `SFTTrainer`.
- **Pre-render a `text` column.** trl/unsloth won't auto-detect the `{"messages":
  …}` conversational format — it errors with *"Unsloth: You must specify a
  formatting_func"*. The trainer renders chat records with the model's chat
  template (and task records with the `Task:/Input:/Output:` shape) before training.
- **`sloth eval` moves inputs to the model's device** before `generate()`, else
  *"Expected all tensors to be on the same device."*

## Quick start

```bash
# from a repo checkout on the Spark
uv run sloth train --config examples/qlora-smoke.toml --dry-run   # GPU-free: prints the plan + the docker command
uv run sloth train --config examples/qlora-smoke.toml             # real QLoRA run in the NGC container
uv run sloth eval  --adapter runs/qlora-smoke --suite examples/eval-suite.jsonl
uv run sloth export --adapter runs/qlora-smoke --output runs/qlora-smoke-export
```

The first real run creates the in-container venv and installs the dep layer
(a few minutes); the HF cache and the venv make subsequent runs faster.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `exit 2` before any container starts | Preflight failed: install Docker + `nvidia-container-toolkit`; `docker pull` the NGC image. |
| `CUDA error: out of memory` at import | UMA pressure. Free memory / `drop_caches`; the run already sets `expandable_segments`. Now mapped to **exit 2** with a memory hint. |
| `exit 137` (SIGKILL) | UMA OOM reclaimer. Flush page cache, reduce batch/seq, or use QLoRA. |
| `Found an incompatible version of torchao` | A drifted dep set. Use the pinned `DEP_LAYER_PACKAGES` (peft 0.18.0). |
| Model re-downloads every run | HF cache not mounted — check `~/.cache/huggingface` exists and is readable. |
