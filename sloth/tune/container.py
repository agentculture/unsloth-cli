"""NGC container orchestration for unsloth-cli fine-tuning (pure stdlib, no torch).

The real LoRA/QLoRA train/eval/export GPU work does **not** run in this process.
Instead the verbs hand off to NVIDIA's official NGC PyTorch container
(:data:`NGC_IMAGE`), where a Blackwell-compatible torch already lives, add the
unsloth dependency layer with **uv** (never ``pip``), bind-mount this checkout +
the working directory, and run ``python -m sloth <args>`` *inside* the container.

This module is the host-side orchestrator. It is **pure stdlib** — it imports
``os``/``shutil``/``subprocess``/``shlex``/``pathlib`` only and **never** imports
torch/unsloth/datasets/trl/peft — so it loads on a machine with no GPU and no ML
stack, keeping the introspection verbs import-light. The only code that imports
the heavy stack is :mod:`sloth.tune._trainer`, which becomes the in-container
entrypoint reached via ``python -m sloth``.

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
launch(sloth_args, *, workdir=None, checkout=None, image=NGC_IMAGE, gpus="all",
       skip_preflight=False, extra_mounts=None, use_host_user=True) -> int
    Run :func:`preflight` (unless skipped), build the command, run it streaming
    its output to the parent stdio. Returns 0 on success; raises :class:`CliError`
    (code 1 or 2) on any container or docker-infrastructure failure.

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
* Subprocess calls are isolated in tiny helpers (:func:`_docker_available`,
  :func:`_image_available`, :func:`_gpu_runtime_ok`, :func:`_stream`,
  :func:`_run_quiet`) so tests can monkeypatch them without invoking docker.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess  # nosec B404 - orchestrating docker is this module's whole job
from pathlib import Path

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

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
DEP_LAYER_PACKAGES: tuple[str, ...] = (
    "transformers==4.57.1",
    "peft==0.18.0",
    "hf_transfer",
    "datasets==4.3.0",
    "trl==0.24.0",
)

#: Dependency layer installed with ``uv pip install --no-deps`` — these must NOT
#: drag their own torch/transformers in; the container's torch is used.
DEP_LAYER_NODEPS_PACKAGES: tuple[str, ...] = (
    "unsloth",
    "unsloth_zoo",
    "bitsandbytes",
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

    # Extra bind-mounts, deduped by container target. A ``container_path`` of "/" is
    # refused outright: it would overlay the container root with a host directory
    # (``-v /:/`` if a caller resolved a path against the filesystem root), exposing the
    # whole host fs inside the run.
    if extra_mounts:
        for host_path, container_path in extra_mounts:
            if container_path == "/" or host_path == "/":
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=(
                        "refusing to bind-mount the filesystem root "
                        f"({host_path}:{container_path})"
                    ),
                    remediation=(
                        "Run sloth from a project directory, not '/', and keep the config, "
                        "dataset, and output paths out of the filesystem root."
                    ),
                )
            if container_path not in used_container_targets:
                used_container_targets.add(container_path)
                cmd += ["-v", f"{host_path}:{container_path}"]

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


def _stream(cmd: list[str]) -> int:
    """Run *cmd* inheriting the parent's stdio (live streaming); return exit code."""
    try:
        proc = subprocess.run(cmd, check=False)  # nosec B607 - docker resolved from PATH
    except OSError:
        return 127
    return proc.returncode


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


def preflight(*, image: str = NGC_IMAGE) -> None:
    """Validate the host can run the NGC container; raise ``CliError`` otherwise.

    Checks, in order: docker present → image present/pullable → NVIDIA GPU
    runtime usable. Every failure raises :class:`CliError` with
    ``code=EXIT_ENV_ERROR`` (2) and :data:`NGC_REMEDIATION` — never a code-1
    "file a bug" — so an agent knows to install docker + nvidia-container-toolkit.

    Raises
    ------
    CliError(code=2)
        When docker is absent, the image cannot be pulled, or the GPU runtime is
        unavailable.
    """
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
) -> int:
    """Preflight, build the docker command, run it streaming output, return 0 or raise.

    Calls :func:`preflight` first (unless *skip_preflight*), so a host that
    cannot run the container fails fast with ``CliError(code=2)`` before any
    container starts. Then builds the command with :func:`build_command` and runs
    it via :func:`_stream`, inheriting the parent's stdio so logs stream live.

    Parameters mirror :func:`build_command`; *skip_preflight* lets a caller that
    already validated the environment (or a test) bypass the docker probes.

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    CliError(code=2)
        From :func:`preflight` when the host cannot run the container, or when
        the container exits due to an environment/infrastructure error (exit
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
    )
    code = _stream(cmd)
    if code == 0:
        return 0
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
