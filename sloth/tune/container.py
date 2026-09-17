"""NGC container orchestration for unsloth-cli fine-tuning (pure stdlib, no torch).

The real LoRA/QLoRA train/eval/export GPU work does **not** run in this process.
Instead the verbs hand off to NVIDIA's official NGC PyTorch container
(:data:`NGC_IMAGE`), where a Blackwell-compatible torch already lives, add the
unsloth dependency layer with **uv** (never ``pip``), bind-mount this checkout +
the working directory, and run ``python -m sloth <args>`` *inside* the container.

This module is the host-side orchestrator. It is **pure stdlib** — it imports
``json``/``os``/``shutil``/``subprocess``/``shlex``/``sys``/``pathlib`` only and
**never** imports torch/unsloth/datasets/trl/peft — so it loads on a machine
with no GPU and no ML stack, keeping the introspection verbs import-light. The
only code that imports the heavy stack is :mod:`sloth.tune._trainer`, which
becomes the in-container entrypoint reached via ``python -m sloth``.

Public API
----------
build_command(sloth_args, *, workdir=None, checkout=None, image=NGC_IMAGE, gpus="all",
              extra_mounts=None, use_host_user=True) -> list[str]
    Build the deterministic ``docker run`` argv that runs *sloth_args* inside the
    NGC container with the uv-installed dependency layer.
preflight(*, image=NGC_IMAGE) -> None
    Validate the host can run the container (docker present, image pullable,
    NVIDIA GPU runtime usable). Raises :class:`CliError` (code 2) on any failure,
    with a remediation naming the NGC image + ``nvidia-container-toolkit``.
export_launch_kwargs() -> dict
    The ``env`` + ``extra_mounts`` an export run needs (explicit ``HOME``, pinned
    ``UNSLOTH_LLAMA_TAG``, persistent llama.cpp cache mount).
launch(sloth_args, *, workdir=None, checkout=None, image=NGC_IMAGE, gpus="all",
       skip_preflight=False, extra_mounts=None, use_host_user=True, env=None) -> dict
    Run :func:`preflight` (unless skipped), build the command, run it while teeing
    the container's stdout to the host's stderr line by line. Returns the parsed
    JSON result object from the container's last result line; raises
    :class:`CliError` (code 1 or 2) on any container or docker-infrastructure
    failure, including an exit-0 run that produced no parseable result line.

Design notes
------------
* The dep layer is pinned to NVIDIA's Spark recipe but installed with **uv**:
  uv is bootstrapped inside the container with the pinned astral standalone
  installer (:data:`UV_INSTALL_URL`, version :data:`UV_INSTALLER_VERSION`) when
  absent, then ``uv pip install`` installs the layer into a
  ``--system-site-packages`` venv. No ``pip install`` runs anywhere.
* Extra bind-mounts: callers may pass ``extra_mounts=[(host, container), ...]``.
  The convention is identity-mounts (``host_path == container_path``) so that
  host-absolute paths in *sloth_args* (dataset, output, adapter, suite dirs)
  resolve unchanged inside the container without any path rewriting.
* Export runs need an explicit ``HOME`` (:data:`EXPORT_HOME`) that is its own
  host-owned bind mount (outside the workdir), because unsloth resolves llama.cpp
  from ``Path.home()``. :func:`export_launch_kwargs` returns exactly that
  ``env``/``extra_mounts`` pair.
* Subprocess calls are isolated in tiny helpers (:func:`_docker_available`,
  :func:`_image_available`, :func:`_gpu_runtime_ok`, :func:`_stream`,
  :func:`_run_quiet`) so tests can monkeypatch them without invoking docker.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess  # nosec B404 - orchestrating docker is this module's whole job
import sys
from pathlib import Path

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic

# ---------------------------------------------------------------------------
# Pinned NVIDIA recipe — image, dependency layer, uv bootstrap (all importable)
# ---------------------------------------------------------------------------

#: NVIDIA's official NGC PyTorch container with a Blackwell-compatible torch.
#: Pinned and deterministic; bumping it is a documented, deliberate change.
NGC_IMAGE: str = "nvcr.io/nvidia/pytorch:25.11-py3"

#: Dependency layer installed with ``uv pip install`` into the in-container venv.
#: Pinned to a set validated against NGC 25.11's torch 2.10 + the container's
#: torchao 0.14 (left untouched). The pins are load-bearing, not cosmetic:
#: unsloth 2026.6.9 requires ``peft>=0.18.0`` and ``trl<=0.24.0`` (so the prior
#: unpinned ``trl==0.26.1`` was out of range), and peft>=0.19 hard-requires
#: torchao>0.16 — which itself needs torch>=2.11 that the container lacks — so peft
#: is held at 0.18.x and transformers at the matching 4.57.1.
#:
#: Quantized-export pins (``llmcompressor`` / ``compressed-tensors``), measured live
#: 2026-09-15 on NGC 25.11 (torch 2.10):
#:
#: * ``llmcompressor==0.11.0`` + ``compressed-tensors==0.16.0`` is the pair that
#:   handles LFM2 correctly. compressed-tensors 0.16.0 calls
#:   ``torch.accelerator.get_memory_info``, which only exists on **torch>=2.11**, so
#:   the exporter installs a small API shim for it before importing the stack.
#: * The older ``llmcompressor==0.10.0.3`` + ``compressed-tensors==0.14.0.1`` pair
#:   runs NVFP4 natively (no shim needed) but its **AWQ** path fails on **LFM2**
#:   (``args[0]`` IndexError, because the decoder layers are called with kwargs).
#:   The shim is the cheaper of the two defects, so 0.11.0/0.16.0 win.
DEP_LAYER_PACKAGES: tuple[str, ...] = (
    "transformers==4.57.1",
    "peft==0.18.0",
    "hf_transfer",
    # datasets 4.8.5 (was 4.3.0): llmcompressor==0.11.0 requires datasets>=4.8.4,<=4.8.5;
    # trl 0.24 only needs >=3.0 (deviation d3, live-verified 2026-09-15).
    "datasets==4.8.5",
    "trl==0.24.0",
    "llmcompressor==0.11.0",
    "compressed-tensors==0.16.0",
)

#: Dependency layer installed with ``uv pip install --no-deps`` — these must NOT
#: drag their own torch/transformers in; the container's torch is used. Pinned
#: (not left floating) to the versions measured live 2026-09-15 on NGC 25.11
#: (torch 2.10) by running the exact install line this module composes
#: (DEP_LAYER_PACKAGES then these, into a ``uv venv --system-site-packages``
#: venv) via ``docker run --rm nvcr.io/nvidia/pytorch:25.11-py3 ...`` followed
#: by ``uv pip list``: unsloth 2026.9.4, unsloth_zoo 2026.9.3 (matching
#: docs/tested.md's 2026-09-15 row), bitsandbytes 0.50.2. See the "Bumping the
#: unsloth / unsloth_zoo / bitsandbytes pins" section in docs/dgx-spark.md for
#: the re-validation procedure before changing these.
DEP_LAYER_NODEPS_PACKAGES: tuple[str, ...] = (
    "unsloth==2026.9.4",
    "unsloth_zoo==2026.9.3",
    "bitsandbytes==0.50.2",
)

#: Benchmark layer installed with a plain ``uv pip install`` *after* the
#: ``--no-deps`` layer: the lm-evaluation-harness (``sloth bench`` / MMLU) and
#: sacrebleu (the lazy GLEU/BLEU scorer in :mod:`sloth.tune.scorers`). Pinned to
#: the versions measured live 2026-09-17 on NGC 25.11 (torch
#: ``2.10.0a0+b558c986e8.nv25.11``, CUDA 13.0) by installing them into the
#: already-built dep-layer venv and diffing ``uv pip list``: the install is
#: purely additive (45 new packages — evaluate 0.4.6, scikit-learn 1.9.1,
#: sqlitedict 2.1.0, rouge-score, ...) and leaves transformers 4.57.1, peft
#: 0.18.0, trl 0.24.0, datasets 4.8.5, accelerate 1.13.0, numpy 2.3.5 and the
#: nv torch untouched; ``import torch`` still reports CUDA available and
#: ``import unsloth`` still patches. Row recorded in docs/tested.md; bump
#: procedure in docs/dgx-spark.md ("Bumping the benchmark-layer pins").
DEP_LAYER_BENCH_PACKAGES: tuple[str, ...] = (
    "lm_eval==0.4.13",
    "sacrebleu==2.6.0",
)

#: Pinned version of the astral uv standalone installer (supply-chain safety).
UV_INSTALLER_VERSION: str = "0.9.2"

#: The pinned astral standalone uv installer URL (versioned, not floating latest).
UV_INSTALL_URL: str = f"https://astral.sh/uv/{UV_INSTALLER_VERSION}/install.sh"

#: NVIDIA-recommended ulimits for PyTorch training containers.
DOCKER_ULIMITS: tuple[str, ...] = ("memlock=-1", "stack=67108864")

#: Container-side mount points for the working dir and this checkout.
WORKDIR_MOUNT: str = "/workspace"
CHECKOUT_MOUNT: str = "/opt/unsloth-cli"

#: In-container venv path (under ``$HOME``, writable by the ``--user`` uid). Created
#: with ``--system-site-packages`` so it inherits the NGC container's torch +
#: torchao while the dep layer installs into a writable location. ``uv pip install
#: --system`` fails both ways: PEP-668 "externally-managed-environment" as root, and
#: permission-denied on the root-owned system site-packages under ``--user``.
VENV_DIR: str = "$HOME/.unsloth-cli-venv"

#: Container path where the host Hugging Face cache is bind-mounted, so models are
#: reused across ephemeral ``--rm`` runs instead of re-downloaded. ``HF_HOME`` is
#: pointed here (see :func:`build_command`).
HF_CACHE_MOUNT: str = "/opt/hf-cache"

#: Default host Hugging Face cache mounted into the container (overridable per call).
DEFAULT_HF_CACHE: Path = Path.home() / ".cache" / "huggingface"

#: Environment always set on the container: expandable CUDA segments to avoid the
#: large up-front reservations that OOM on the DGX Spark's Unified Memory
#: Architecture. Both the current (``PYTORCH_ALLOC_CONF``) and the deprecated alias
#: (``PYTORCH_CUDA_ALLOC_CONF``) are set so it works across torch versions.
DOCKER_ENV: tuple[tuple[str, str], ...] = (
    ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
)

#: Explicit ``HOME`` for container runs that need a stable, writable home — notably
#: GGUF export, because unsloth resolves its llama.cpp checkout from
#: ``Path.home()/".unsloth"/"llama.cpp"`` (``unsloth_zoo/llama_cpp.py``). It is its
#: **own bind mount** (see :func:`export_launch_kwargs`), deliberately *outside* the
#: ``/workspace`` workdir mount: docker creates missing mount-point parents as root,
#: so a HOME nested under the bind-mounted workdir ended up root-owned and the
#: container user could not create ``$HOME/.cache/uv`` (measured 2026-09-15). A
#: dedicated host-owned home also means :data:`VENV_DIR` (``$HOME/.unsloth-cli-venv``)
#: and the llama.cpp cache **persist across ``--rm`` runs** — the dep layer is
#: installed once per host, not once per run.
EXPORT_HOME: str = "/opt/sloth-home"

#: Environment variable overriding the host-side directory mounted at
#: :data:`EXPORT_HOME`.
EXPORT_HOME_ENV: str = "SLOTH_EXPORT_HOME"

#: Default host-owned directory mounted at :data:`EXPORT_HOME`.
DEFAULT_EXPORT_HOME: Path = Path.home() / ".cache" / "unsloth-cli" / "home"

#: Pinned llama.cpp release tag used for unsloth's *prebuilt* llama.cpp install
#: (``UNSLOTH_LLAMA_TAG``). Validated live 2026-09-15. Unsloth skips the prebuilt
#: install entirely when the cache dir is already populated (measured: 48 s first
#: run, 28 s cached). Bumping this tag is a deliberate, re-validated change.
UNSLOTH_LLAMA_TAG: str = "b10909"

#: Environment variable overriding the host-side llama.cpp cache directory. When set,
#: that directory is bind-mounted at ``<EXPORT_HOME>/.unsloth/llama.cpp`` on top of the
#: home mount; when unset the cache simply lives inside the home dir at
#: :data:`DEFAULT_LLAMA_CPP_CACHE`.
LLAMA_CPP_CACHE_ENV: str = "SLOTH_LLAMA_CPP_CACHE"

#: Default llama.cpp cache location (inside the default export home), so the prebuilt
#: install survives ``--rm``.
DEFAULT_LLAMA_CPP_CACHE: Path = DEFAULT_EXPORT_HOME / ".unsloth" / "llama.cpp"

#: Where the Linux kernel reports memory statistics (patched in tests).
MEMINFO_PATH: Path = Path("/proc/meminfo")

#: Below this much ``MemFree`` (in KiB — 4 GiB), :func:`preflight` emits a
#: non-blocking stderr hint about the Unsloth-import OOM on the Spark's UMA.
LOW_MEMFREE_KIB: int = 4 * 1024 * 1024

#: The non-blocking low-memory hint text emitted by :func:`preflight`. It is a
#: *diagnostic* (stderr, ``note:``-prefixed at emission), not a ``CliError`` — the
#: check never blocks a run, because MemFree is a snapshot and page cache is
#: reclaimable.
LOW_MEMORY_HINT: str = (
    "Unsloth import can fail with CUDA out of memory on the Spark's unified "
    "memory when MemFree is low even with expandable_segments; free page cache "
    "(e.g. touch-and-free a large mmap, or drop caches with root) or stop other "
    "GPU residents"
)

#: Single remediation string reused by every :func:`preflight` failure. It names
#: both the pinned image and the ``nvidia-container-toolkit`` package so an agent
#: reading ``hint:`` knows exactly what to install.
NGC_REMEDIATION: str = (
    "Real fine-tuning runs the GPU work inside NVIDIA's official NGC PyTorch "
    f"container '{NGC_IMAGE}'. Install Docker, install the NVIDIA Container "
    "Toolkit (package 'nvidia-container-toolkit') so `docker run --gpus all` can "
    f"reach the GPU, then confirm the image pulls with `docker pull {NGC_IMAGE}`."
)


# ---------------------------------------------------------------------------
# In-container shell script (uv dep layer + python -m sloth entrypoint)
# ---------------------------------------------------------------------------


def _inner_script(sloth_args: list[str]) -> str:
    """Return the ``bash -lc`` script run inside the NGC container.

    Bootstraps uv with the pinned astral installer (:data:`UV_INSTALL_URL`) when
    absent — guarded by a ``curl`` availability check that exits 2 with a clear
    message if curl is missing — installs the pinned dependency layer with
    ``uv pip install`` into a ``--system-site-packages`` venv (never ``pip``), then
    runs ``python -m sloth <args>`` against the bind-mounted checkout. *sloth_args*
    is shell-quoted with :func:`shlex.join`.
    """
    install_deps = "uv pip install " + shlex.join(DEP_LAYER_PACKAGES)
    install_nodeps = "uv pip install --no-deps " + shlex.join(DEP_LAYER_NODEPS_PACKAGES)
    install_bench = "uv pip install " + shlex.join(DEP_LAYER_BENCH_PACKAGES)
    entrypoint = f"PYTHONPATH={CHECKOUT_MOUNT} python -m sloth " + shlex.join(sloth_args)
    curl_guard = (
        "  command -v curl >/dev/null 2>&1"
        ' || { echo "curl is required to bootstrap uv" >&2; exit 2; }'
    )
    return "\n".join(
        [
            "set -euo pipefail",
            # Make a freshly-installed uv (astral default install dir) discoverable.
            'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
            # Bootstrap uv via the pinned astral standalone installer if it is absent.
            "if ! command -v uv >/dev/null 2>&1; then",
            curl_guard,
            f"  curl -LsSf {UV_INSTALL_URL} | sh",
            "fi",
            # Create a writable venv that inherits the container's torch + torchao via
            # --system-site-packages, then install the dep layer into it and activate
            # it. This is required because ``uv pip install --system`` fails on the NGC
            # image both as root (PEP-668 externally-managed) and under --user
            # (root-owned site-packages is not writable).
            f'uv venv --system-site-packages "{VENV_DIR}"',
            f'. "{VENV_DIR}/bin/activate"',
            install_deps,
            # Drop the venv-local torch/torchvision that the dep layer drags in (a
            # PyPI torch==2.x+cu13 wheel) so the container's Blackwell-native nv torch
            # shows through via --system-site-packages. Otherwise the venv torch
            # shadows it and mismatches the system nv torchvision
            # ("RuntimeError: operator torchvision::nms does not exist").
            "uv pip uninstall torch 2>/dev/null || true",
            "uv pip uninstall torchvision 2>/dev/null || true",
            install_nodeps,
            # Benchmark layer (lm-evaluation-harness + sacrebleu) last: a full
            # resolution, measured additive against the layers above (see
            # DEP_LAYER_BENCH_PACKAGES).
            install_bench,
            entrypoint,
        ]
    )


# ---------------------------------------------------------------------------
# Docker command builder (pure — no subprocess, deterministic)
# ---------------------------------------------------------------------------


def _default_checkout() -> Path:
    """Return this unsloth-cli checkout root (the dir holding the ``sloth`` package).

    ``sloth/tune/container.py`` → ``parents[2]`` is the checkout root, which is
    bind-mounted into the container and put on ``PYTHONPATH`` so ``python -m
    sloth`` runs *this* source without any install step.
    """
    return Path(__file__).resolve().parents[2]


def _reject_root_mount(host_path: str, container_path: str) -> None:
    """Refuse a bind-mount touching the filesystem root (host or container side).

    A ``/`` on either side would overlay the whole host filesystem into the run
    (``-v /:/`` or ``-v /:/workspace``), exposing/altering arbitrary host files.
    """
    if host_path == "/" or container_path == "/":
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                "refusing to bind-mount the filesystem root " f"({host_path}:{container_path})"
            ),
            remediation=(
                "Run sloth from a project directory, not '/', and keep the config, "
                "dataset, and output paths out of the filesystem root."
            ),
        )


def _extra_mount_args(
    extra_mounts: list[tuple[str, str]] | None, used_targets: set[str]
) -> list[str]:
    """Return the ``-v host:container`` args for *extra_mounts*.

    Deduped by container target against (and recording into) *used_targets*: a
    tuple whose ``container_path`` is already a mount target is skipped. Any mount
    of the filesystem root is refused via :func:`_reject_root_mount`.
    """
    args: list[str] = []
    for host_path, container_path in extra_mounts or ():
        _reject_root_mount(host_path, container_path)
        if container_path not in used_targets:
            used_targets.add(container_path)
            args += ["-v", f"{host_path}:{container_path}"]
    return args


def build_command(
    sloth_args: list[str],
    *,
    workdir: str | Path | None = None,
    checkout: str | Path | None = None,
    image: str = NGC_IMAGE,
    gpus: str = "all",
    extra_mounts: list[tuple[str, str]] | None = None,
    use_host_user: bool = True,
    hf_cache: str | Path | None = None,
    env: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Build the deterministic ``docker run`` argv that runs *sloth_args* in NGC.

    Parameters
    ----------
    sloth_args:
        Arguments forwarded to ``python -m sloth`` inside the container
        (e.g. ``["train", "--config", "run.toml"]``).
    workdir:
        Host directory bind-mounted at :data:`WORKDIR_MOUNT` and used as the
        container working dir, so relative dataset/output paths resolve. Defaults
        to the current working directory.
    checkout:
        Host path to this unsloth-cli checkout, bind-mounted at
        :data:`CHECKOUT_MOUNT` and put on ``PYTHONPATH``. Defaults to the
        checkout containing this module.
    image:
        Container image. Defaults to the pinned :data:`NGC_IMAGE`.
    gpus:
        Value for ``--gpus``. Defaults to ``"all"``.
    extra_mounts:
        Optional list of ``(host_path, container_path)`` tuples to bind-mount in
        addition to the standard workdir and checkout mounts. Deduplication is
        applied by container-target: any tuple whose ``container_path`` already
        exists as a mount target (including :data:`WORKDIR_MOUNT` and
        :data:`CHECKOUT_MOUNT`) is silently skipped.

        **Convention — identity mounts:** pass ``host_path == container_path``
        (absolute host path identical to the container path) so that
        host-absolute paths forwarded in *sloth_args* (dataset files, output
        dirs, adapter dirs, eval suites) resolve unchanged inside the container
        without any path rewriting.
    use_host_user:
        When ``True`` (default) **and** running on a POSIX system, adds
        ``--user <uid>:<gid>`` to the ``docker run`` argv so bind-mounted outputs
        are owned by the calling user instead of root. Set to ``False`` when the
        container image requires root or when running on non-POSIX hosts.
    hf_cache:
        Host Hugging Face cache directory to bind-mount at :data:`HF_CACHE_MOUNT`
        (with ``HF_HOME`` pointed at it) so models/datasets are reused across
        ephemeral ``--rm`` runs instead of re-downloaded. Defaults to
        :data:`DEFAULT_HF_CACHE`; the mount is added only when the directory
        exists. Pass a non-existent path to skip the mount.
    env:
        Extra ``(key, value)`` environment pairs to set with ``-e`` in addition to
        the always-on :data:`DOCKER_ENV` (Spark UMA allocator tuning).

    Returns
    -------
    list[str]
        A ready-to-run ``docker run`` argv (list form — no shell on the host).
        Calling this twice with the same inputs on the same host yields an
        identical list (the HF-cache mount depends on whether the dir exists).
    """
    workdir_path = Path(workdir) if workdir is not None else Path.cwd()
    checkout_path = Path(checkout) if checkout is not None else _default_checkout()
    hf_cache_path = Path(hf_cache) if hf_cache is not None else DEFAULT_HF_CACHE
    mount_hf = hf_cache_path.is_dir()

    # The workdir is bind-mounted as the container workspace; the filesystem root
    # would mount the entire host fs (``-v /:/workspace``) into the run. Refuse it.
    if str(workdir_path) == "/":
        raise CliError(
            code=EXIT_USER_ERROR,
            message="refusing to run with the filesystem root as the working directory",
            remediation="Run sloth from a project directory, not '/'.",
        )

    # Track used container mount targets for deduplication (extra_mounts dedup).
    used_container_targets: set[str] = {WORKDIR_MOUNT, CHECKOUT_MOUNT}

    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpus,
        # PyTorch DataLoaders need a large /dev/shm; host IPC avoids shm OOMs.
        "--ipc=host",
    ]
    for limit in DOCKER_ULIMITS:
        cmd += ["--ulimit", limit]

    # Always-on environment (Spark UMA allocator tuning), then HF_HOME (only when the
    # cache is mounted), then any caller-supplied env.
    for key, value in DOCKER_ENV:
        cmd += ["-e", f"{key}={value}"]
    if mount_hf:
        cmd += ["-e", f"HF_HOME={HF_CACHE_MOUNT}"]
    for key, value in env or ():
        cmd += ["-e", f"{key}={value}"]

    # Host-user ownership: bind-mounted outputs are written as the calling user.
    if use_host_user and os.name == "posix":
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

    cmd += [
        "-v",
        f"{workdir_path}:{WORKDIR_MOUNT}",
        "-v",
        f"{checkout_path}:{CHECKOUT_MOUNT}",
    ]

    # Hugging Face cache mount (model/dataset reuse across runs).
    if mount_hf:
        used_container_targets.add(HF_CACHE_MOUNT)
        cmd += ["-v", f"{hf_cache_path}:{HF_CACHE_MOUNT}"]

    # Extra bind-mounts, deduped by container target (filesystem-root mounts refused).
    cmd += _extra_mount_args(extra_mounts, used_container_targets)

    cmd += [
        "-w",
        WORKDIR_MOUNT,
        image,
        "bash",
        "-lc",
        _inner_script(list(sloth_args)),
    ]
    return cmd


# ---------------------------------------------------------------------------
# Subprocess seams (isolated so tests can monkeypatch them)
# ---------------------------------------------------------------------------


def _run_quiet(cmd: list[str]) -> int:
    """Run *cmd* with output suppressed; return its exit code (127 if not found).

    Used for the cheap preflight probes (``docker image inspect`` / ``pull`` /
    ``run --gpus all``). Output is discarded — preflight only needs the verdict.
    """
    try:
        proc = subprocess.run(  # nosec B607 - docker is resolved from PATH on purpose
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return 127
    return proc.returncode


class StreamResult(int):
    """A container exit code that also carries the container's captured stdout.

    Subclasses :class:`int` so every existing exit-code comparison (and every
    test that stubs :func:`_stream` with a bare ``int``) keeps working; the
    captured lines ride along in :attr:`lines`.
    """

    lines: tuple[str, ...]

    def __new__(cls, code: int, lines: "tuple[str, ...] | list[str] | None" = None):
        obj = super().__new__(cls, code)
        obj.lines = tuple(lines or ())
        return obj


def _stream(cmd: list[str]) -> StreamResult:
    """Run *cmd*, teeing its stdout to host **stderr** line by line; capture it too.

    The container's stdout is piped (so the host can read the in-container
    ``sloth --json`` result off the last line) while its stderr is inherited
    untouched. Every line read — JSON or banner noise — is written to
    ``sys.stderr`` as it arrives and flushed, so a long training run still
    streams live for the human watching. Nothing is written to the host's
    stdout here: the caller decides what the *result* is.

    Returns a :class:`StreamResult` (the exit code, ``.lines`` the captured
    stdout lines without their trailing newline); ``127`` with no lines when
    ``docker`` cannot be executed at all (``OSError``).
    """
    lines: list[str] = []
    try:
        proc = subprocess.Popen(  # nosec B607 - docker resolved from PATH on purpose
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
    except OSError:
        return StreamResult(127)
    with proc.stdout as out:
        for raw in out:
            line = raw.rstrip("\n")
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            lines.append(line)
    return StreamResult(proc.wait(), lines)


def _last_json_object(lines: "tuple[str, ...] | list[str]") -> dict | None:
    """Return the last line that parses as a JSON **object**, or ``None``."""
    for line in reversed(list(lines)):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _docker_available() -> bool:
    """Return ``True`` when a ``docker`` executable is on PATH."""
    return shutil.which("docker") is not None


def _image_available(image: str = NGC_IMAGE) -> bool:
    """Return ``True`` when *image* is present locally or can be pulled."""
    if _run_quiet(["docker", "image", "inspect", image]) == 0:
        return True
    return _run_quiet(["docker", "pull", image]) == 0


def _gpu_runtime_ok(image: str = NGC_IMAGE) -> bool:
    """Return ``True`` when Docker can attach the GPU via the NVIDIA runtime.

    Probes by attaching all GPUs to a throwaway container and listing them with
    ``nvidia-smi -L``. Reuses the pinned NGC image — no second image is pulled.
    """
    cmd = ["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi", "-L"]
    return _run_quiet(cmd) == 0


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _export_home_dir() -> Path:
    """Return the host dir mounted at :data:`EXPORT_HOME` (:data:`EXPORT_HOME_ENV` overrides)."""
    override = os.environ.get(EXPORT_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_EXPORT_HOME


def _llama_cpp_cache_override() -> Path | None:
    """Return the :data:`LLAMA_CPP_CACHE_ENV` override as a path, or ``None``."""
    override = os.environ.get(LLAMA_CPP_CACHE_ENV)
    return Path(override).expanduser() if override else None


def export_launch_kwargs() -> dict:
    """Return the :func:`launch` kwargs an export run needs (env + home mount).

    Export (GGUF) asks unsloth to fetch/build llama.cpp, and unsloth resolves that
    checkout from ``Path.home()/".unsloth"/"llama.cpp"``. So ``HOME`` is set explicitly
    to :data:`EXPORT_HOME` and a **host-owned** directory is bind-mounted there. The
    host side pre-creates ``<home>/.unsloth/llama.cpp`` so docker never has to create
    a mount-point parent as root (the failure mode that made ``$HOME`` unwritable
    when HOME lived under the workdir mount). With the cache populated, unsloth skips
    the prebuilt install on subsequent runs (measured 48 s first, 28 s cached).
    :data:`UNSLOTH_LLAMA_TAG` pins which prebuilt release is fetched.

    When :data:`LLAMA_CPP_CACHE_ENV` is set, that directory is additionally mounted
    at ``<EXPORT_HOME>/.unsloth/llama.cpp`` (nested inside the home mount).

    Returns
    -------
    dict
        ``{"env": [("HOME", EXPORT_HOME), ("UNSLOTH_LLAMA_TAG", UNSLOTH_LLAMA_TAG)],
        "extra_mounts": [(host_home, EXPORT_HOME), ...]}`` — splat straight into
        :func:`launch` or :func:`build_command`.
    """
    home = _export_home_dir()
    os.makedirs(home / ".unsloth" / "llama.cpp", exist_ok=True)
    mounts: list[tuple[str, str]] = [(str(home), EXPORT_HOME)]
    cache = _llama_cpp_cache_override()
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        mounts.append((str(cache), EXPORT_HOME + "/.unsloth/llama.cpp"))
    return {
        "env": [("HOME", EXPORT_HOME), ("UNSLOTH_LLAMA_TAG", UNSLOTH_LLAMA_TAG)],
        "extra_mounts": mounts,
    }


def _read_memfree_kib() -> int | None:
    """Return ``MemFree`` in KiB from :data:`MEMINFO_PATH`, or ``None`` if unreadable.

    Never raises: a missing file (non-Linux hosts), an unreadable one, or a line that
    does not parse all yield ``None`` so the caller stays non-blocking.
    """
    try:
        text = MEMINFO_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("MemFree:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _memory_hint() -> None:
    """Emit the non-blocking low-memory diagnostic when ``MemFree`` is under 4 GiB."""
    mem_free = _read_memfree_kib()
    if mem_free is None or mem_free >= LOW_MEMFREE_KIB:
        return
    emit_diagnostic(f"note: MemFree is {mem_free / 1024 / 1024:.1f} GiB — {LOW_MEMORY_HINT}")


def preflight(*, image: str = NGC_IMAGE) -> None:
    """Validate the host can run the NGC container; raise ``CliError`` otherwise.

    Checks, in order: a **non-blocking** low-memory diagnostic (:func:`_memory_hint`,
    which only ever writes a ``note:`` line to stderr — emitted first so it survives
    a later hard failure), then docker present → image present/pullable → NVIDIA GPU
    runtime usable. Every *failure* raises :class:`CliError` with
    ``code=EXIT_ENV_ERROR`` (2) and :data:`NGC_REMEDIATION` — never a code-1
    "file a bug" — so an agent knows to install docker + nvidia-container-toolkit.

    Raises
    ------
    CliError(code=2)
        When docker is absent, the image cannot be pulled, or the GPU runtime is
        unavailable.
    """
    _memory_hint()
    if not _docker_available():
        raise CliError(
            code=EXIT_ENV_ERROR,
            message="Docker was not found on PATH; the fine-tuning GPU stack runs in a container.",
            remediation=NGC_REMEDIATION,
        )
    if not _image_available(image):
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"The NGC container image could not be pulled: {image}",
            remediation=NGC_REMEDIATION,
        )
    if not _gpu_runtime_ok(image):
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                "Docker cannot access the GPU; `docker run --gpus all` failed "
                "(the NVIDIA container runtime is unavailable)."
            ),
            remediation=NGC_REMEDIATION,
        )


def launch(
    sloth_args: list[str],
    *,
    workdir: str | Path | None = None,
    checkout: str | Path | None = None,
    image: str = NGC_IMAGE,
    gpus: str = "all",
    skip_preflight: bool = False,
    extra_mounts: list[tuple[str, str]] | None = None,
    use_host_user: bool = True,
    env: list[tuple[str, str]] | None = None,
) -> dict:
    """Preflight, build the docker command, run it, and return the captured result.

    Calls :func:`preflight` first (unless *skip_preflight*), so a host that
    cannot run the container fails fast with ``CliError(code=2)`` before any
    container starts. Then builds the command with :func:`build_command` and runs
    it via :func:`_stream`, which tees every stdout line to the host's **stderr**
    as it arrives (logs still stream live) while capturing it.

    The in-container ``sloth`` run always ends with a single-line JSON result on
    stdout, so the **last captured line that parses as a JSON object is the
    result** and is returned to the caller — which renders it (text or JSON) on
    the host's stdout. This is fail-closed: a container that exits 0 without such
    a line raises ``CliError(code=2)`` naming how many lines were captured,
    rather than reporting a success it cannot substantiate. The error is a
    single-line summary, not a quote of those lines — :func:`_stream` already
    teed every one of them to the host's stderr as the container ran, so the
    human watching already saw them there.

    Parameters mirror :func:`build_command` (including *env* and *extra_mounts*,
    forwarded verbatim — see :func:`export_launch_kwargs` for the export-run pair);
    *skip_preflight* lets a caller that already validated the environment (or a
    test) bypass the docker probes.

    Returns
    -------
    dict
        The parsed JSON result object from the container's last result line.

    Raises
    ------
    CliError(code=2)
        From :func:`preflight` when the host cannot run the container, when the
        container exits 0 without a parseable JSON result line, or when the
        container exits due to an environment/infrastructure error (exit
        codes: 2, 137 OOM/SIGKILL, or any other non-{0,1,2} docker infra code).
    CliError(code=1)
        When the container exits with code 1 (in-container user-input error;
        the container's own ``error:``/``hint:`` output was already streamed).
    """
    if not skip_preflight:
        preflight(image=image)
    cmd = build_command(
        sloth_args,
        workdir=workdir,
        checkout=checkout,
        image=image,
        gpus=gpus,
        extra_mounts=extra_mounts,
        use_host_user=use_host_user,
        env=env,
    )
    outcome = _stream(cmd)
    code = int(outcome)
    lines: tuple[str, ...] = tuple(getattr(outcome, "lines", ()))
    if code == 0:
        result = _last_json_object(lines)
        if result is not None:
            return result
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"container exited 0 without printing a JSON result ({len(lines)} lines "
                "captured; see the container output above on stderr)"
            ),
            remediation=(
                "Re-run and check the container output above on stderr for why the "
                "in-container sloth verb did not emit its result."
            ),
        )
    if code == 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"Container exited with status {code}; its own output was shown above.",
            remediation="Review the error:/hint: output above from the in-container sloth run.",
        )
    if code == 2:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Container exited with status {code}; its own output was shown above.",
            remediation="Review the environment error output above from the in-container run.",
        )
    if code == 137:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"Container was killed (exit {code}): likely OOM / SIGKILL from the "
                "DGX Spark Unified Memory Architecture (UMA) reclaimer."
            ),
            remediation=(
                "Free host memory and flush the page cache before retrying: "
                'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches". '
                "Then reduce batch size or model size."
            ),
        )
    raise CliError(
        code=EXIT_ENV_ERROR,
        message=(
            f"Container exited with status {code} (docker infrastructure error); "
            "its output was shown above."
        ),
        remediation=(
            "Check that Docker is running and the NVIDIA Container Toolkit is installed "
            "(nvidia-container-toolkit); see `docker run` exit codes 125/126/127."
        ),
    )
