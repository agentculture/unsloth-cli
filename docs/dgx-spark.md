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

| Package | Pin | Why this exact value |
|---------|-----|----------------------|
| `transformers` | `4.57.1` | The window unsloth 2026.6.9 + peft 0.18 agree on. |
| `peft` | `0.18.0` | ≥ unsloth's floor, < the `torchao>0.16` demand of peft 0.19+. |
| `trl` | `0.24.0` | unsloth requires `trl<=0.24.0` (the old `0.26.1` was out of range). |
| `datasets` | `4.3.0` | Validated against transformers 4.57.1. |
| `hf_transfer` | unpinned | Download accelerator only; no API surface. |
| `llmcompressor` | `0.11.0` | Quantized export (NVFP4 / AWQ). See the note below. |
| `compressed-tensors` | `0.16.0` | Must match llmcompressor 0.11.0. Needs a torch-2.11 shim. |
| `unsloth`, `unsloth_zoo`, `bitsandbytes` | unpinned, `--no-deps` | Must not drag their own torch/transformers in. |
| `torchao` | **left** at the container's `0.14.0+git` | Not upgraded — 0.17+ needs torch 2.11. |

The reason this exact set matters — a real version-matrix deadlock:

- `unsloth 2026.6.9` requires `torch<2.11`, `peft>=0.18.0`, `trl<=0.24.0`
  (so the previous `trl==0.26.1` pin was **out of range**), and a specific
  `transformers` window.
- `peft 0.19+` **hard-requires `torchao>0.16`** at `get_peft_model` time.
- But `torchao>0.16` (0.17+) needs `torch>=2.11` (it imports
  `torch.nn.functional.ScalingType`), which NGC 25.11 does **not** have.
- `unsloth_zoo` only needs `torchao>=0.13`, so the container's 0.14 is fine.

**The quantized-export pair (measured live 2026-09-15 on NGC 25.11 / torch 2.10).**
`compressed-tensors 0.16.0` calls `torch.accelerator.get_memory_info`, which only
exists on **torch >= 2.11** — the exporter installs a small API shim for it before
importing the stack. The older pair `llmcompressor 0.10.0.3` + `compressed-tensors
0.14.0.1` needs no shim and runs NVFP4 natively, but its **AWQ** path fails on
**LFM2** (`args[0]` IndexError — the decoder layers are called with kwargs). A shim
is cheaper than a broken AWQ path, so **0.11.0 / 0.16.0** are the pins.

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
`$HOME/.cache/huggingface` to `/opt/hf-cache` and points `HF_HOME` there, so models
are downloaded once and reused.

### 4. Export needs an explicit `HOME` + a mounted llama.cpp cache

Unsloth resolves its llama.cpp checkout from `Path.home()/".unsloth"/"llama.cpp"`
(`unsloth_zoo/llama_cpp.py`). Under `--user uid:gid` the NGC image's `HOME` is not
reliably writable, and on an `--rm` container anything written there is lost — so
every GGUF export would re-install llama.cpp. The orchestration therefore:

- sets **`HOME=/workspace/.home`** explicitly (`container.EXPORT_HOME`);
- bind-mounts a **host-owned** cache dir at exactly `$HOME/.unsloth/llama.cpp`
  (default `~/.cache/unsloth-cli/llama.cpp`, override with
  **`SLOTH_LLAMA_CPP_CACHE`**), created on the host if absent;
- pins the prebuilt release with **`UNSLOTH_LLAMA_TAG=b10909`** (validated live
  2026-09-15).

With the cache populated, unsloth skips the prebuilt install entirely — **measured
48 s on the first run, 28 s cached**. `container.export_launch_kwargs()` returns
exactly this `env` + `extra_mounts` pair.

> **Side effect, and intended:** the in-container venv lives at
> `$HOME/.unsloth-cli-venv`, so an explicit `HOME` under the workdir mount puts the
> venv *under the bind-mounted working directory too*. It then persists across
> `--rm` runs — the dep layer is installed once per working directory instead of
> once per run.

### 5. Low `MemFree` before a run → a non-blocking preflight hint

`preflight()` reads `/proc/meminfo` on Linux and, when **`MemFree` is under 4 GiB**,
writes a `note:` line to **stderr**. It **never blocks** — MemFree is a snapshot and
page cache is reclaimable — but it names the failure it predicts:

> Unsloth import can fail with CUDA out of memory on the Spark's unified memory
> when MemFree is low even with `expandable_segments`; free page cache (e.g.
> touch-and-free a large mmap, or drop caches with root) or stop other GPU
> residents.

A missing or unparsable `/proc/meminfo` (non-Linux hosts) is silent.

### 6. trl/unsloth API specifics (handled in the trainer)

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
| llama.cpp re-installs on every export | The llama.cpp cache is not persisting — check `~/.cache/unsloth-cli/llama.cpp` (or `SLOTH_LLAMA_CPP_CACHE`) is writable. |
| `note: MemFree is …` on stderr | Informational, not a failure: free page cache or stop other GPU residents before a big run. |
