"""Tests for sloth.tune.container — host-side NGC container orchestration.

This module is the pure-stdlib orchestrator: it builds the ``docker run`` argv,
validates the host (preflight), and launches the container — and it must do all
of that **without** importing torch/unsloth (import-light preserved). The tests
cover, without docker or a GPU:

  1. ``import sloth.tune.container`` succeeds and is torch-free (top-level import
     here + an AST guard + a subprocess import guard).
  2. ``build_command`` is deterministic and contains every required token:
     ``docker run``, ``--gpus all``, the two ulimits, the pinned NGC image, the
     two bind-mounts, ``uv pip install --system``, the uv bootstrap, and the
     ``python -m sloth`` entrypoint — and NEVER a bare ``pip install``.
  3. ``preflight`` raises ``CliError(code=2)`` with the NGC remediation when
     docker is absent / the image is unpullable / the GPU runtime is missing
     (the subprocess seams are monkeypatched).
  4. ``launch`` preflights, builds, and streams — all via stubbable seams.
"""

from __future__ import annotations

import ast
import inspect
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Acceptance criterion 1a: the bare import must succeed (import-light).
import sloth.tune.container as container  # noqa: E402
from sloth.cli._errors import CliError
from sloth.tune.container import (
    CHECKOUT_MOUNT,
    DEP_LAYER_NODEPS_PACKAGES,
    DEP_LAYER_PACKAGES,
    NGC_IMAGE,
    NGC_REMEDIATION,
    UV_INSTALL_URL,
    UV_INSTALLER_VERSION,
    WORKDIR_MOUNT,
    build_command,
    launch,
    preflight,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _joined(cmd: list[str]) -> str:
    """Join an argv into one inspectable string (mirrors how docker reads it)."""
    return " ".join(cmd)


# ---------------------------------------------------------------------------
# 1. Import-light: no torch, ever
# ---------------------------------------------------------------------------


class TestImportLight:
    def test_import_succeeds(self) -> None:
        # The top-level `import sloth.tune.container` already ran at collection
        # time; this asserts the module object is usable.
        assert container.NGC_IMAGE == "nvcr.io/nvidia/pytorch:25.11-py3"

    def test_no_module_level_heavy_imports(self) -> None:
        source = inspect.getsource(container)
        tree = ast.parse(source)
        heavy = {"torch", "unsloth", "unsloth_zoo", "datasets", "trl", "peft", "transformers"}
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (roots & heavy), f"heavy import at module level: {roots & heavy}"

    def test_importing_container_does_not_load_torch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        code = (
            "import sloth.tune.container; import sys; "
            "assert 'torch' not in sys.modules, 'torch imported at module top'; "
            "assert 'unsloth' not in sys.modules, 'unsloth imported at module top'; "
            "print('PASS')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        assert result.returncode == 0, (
            f"Expected returncode 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# 2. build_command golden-string contract
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def _cmd(self, tmp_path: Path) -> list[str]:
        return build_command(
            ["train", "--config", "run.toml"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
        )

    def test_returns_list_of_str(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path)
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)

    def test_starts_with_docker_run(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path)
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "docker run" in _joined(cmd)

    def test_requests_all_gpus(self, tmp_path: Path) -> None:
        assert "--gpus all" in _joined(self._cmd(tmp_path))

    def test_carries_nvidia_ulimits(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        assert "--ulimit memlock=-1" in joined
        assert "--ulimit stack=67108864" in joined

    def test_uses_pinned_ngc_image(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path)
        assert NGC_IMAGE == "nvcr.io/nvidia/pytorch:25.11-py3"
        assert NGC_IMAGE in cmd  # present as its own argv token

    def test_bind_mounts_workdir_and_checkout(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path)
        joined = _joined(cmd)
        assert "-v" in cmd
        # workdir mount
        assert f"{tmp_path}:{WORKDIR_MOUNT}" in joined
        # checkout mount
        assert f"{tmp_path / 'checkout'}:{CHECKOUT_MOUNT}" in joined

    def test_installs_dep_layer_into_a_system_site_packages_venv(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        # Deps install into a --system-site-packages venv (inherits the container's
        # torch/torchao) — NOT `uv pip install --system`, which fails on the NGC
        # image both as root (PEP-668) and under --user (root-owned site-packages).
        assert "uv venv --system-site-packages" in joined
        assert "uv pip install" in joined
        assert "uv pip install --system" not in joined
        # full-resolution layer (pinned, validated set)
        for pkg in DEP_LAYER_PACKAGES:
            assert pkg in joined, f"missing dep-layer package: {pkg}"
        assert "transformers==4.57.1" in joined
        assert "peft==0.18.0" in joined
        assert "datasets==4.8.5" in joined
        assert "trl==0.24.0" in joined
        # --no-deps layer (pinned — see TestNodepsLayerPins below)
        assert "uv pip install --no-deps " + " ".join(DEP_LAYER_NODEPS_PACKAGES) in joined
        for pkg in DEP_LAYER_NODEPS_PACKAGES:
            assert pkg in joined, f"missing --no-deps package: {pkg}"

    def test_sets_expandable_segments_env(self, tmp_path: Path) -> None:
        # Spark UMA allocator tuning is always set (current name + deprecated alias).
        joined = _joined(self._cmd(tmp_path))
        assert "-e PYTORCH_ALLOC_CONF=expandable_segments:True" in joined
        assert "-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in joined

    def test_mounts_hf_cache_and_sets_hf_home_when_present(self, tmp_path: Path) -> None:
        hf = tmp_path / "hfcache"
        hf.mkdir()
        joined = _joined(
            build_command(
                ["train", "--config", "run.toml"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=hf,
            )
        )
        assert f"-v {hf}:{container.HF_CACHE_MOUNT}" in joined
        assert f"-e HF_HOME={container.HF_CACHE_MOUNT}" in joined

    def test_skips_hf_cache_when_absent(self, tmp_path: Path) -> None:
        joined = _joined(
            build_command(
                ["train", "--config", "run.toml"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=tmp_path / "does-not-exist",
            )
        )
        assert container.HF_CACHE_MOUNT not in joined
        assert "HF_HOME" not in joined

    def test_extra_env_pairs_rendered(self, tmp_path: Path) -> None:
        joined = _joined(
            build_command(
                ["train", "--config", "run.toml"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=tmp_path / "none",
                env=[("FOO", "bar")],
            )
        )
        assert "-e FOO=bar" in joined

    def test_never_a_bare_pip_install(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        # Every 'pip install' must be immediately preceded by 'uv '.
        idx = joined.find("pip install")
        while idx != -1:
            assert joined[idx - 3 : idx] == "uv ", f"bare 'pip install' at offset {idx}"
            idx = joined.find("pip install", idx + 1)
        # And, with every 'uv pip install' removed, no 'pip install' remains.
        assert "pip install" not in joined.replace("uv pip install", "")

    def test_bootstraps_uv_via_astral_installer(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        assert UV_INSTALLER_VERSION == "0.9.2"
        assert UV_INSTALL_URL == f"https://astral.sh/uv/{UV_INSTALLER_VERSION}/install.sh"
        assert f"curl -LsSf {UV_INSTALL_URL} | sh" in joined
        # bootstrap is guarded on uv being absent
        assert "command -v uv" in joined

    def test_curl_guard_precedes_curl_call(self, tmp_path: Path) -> None:
        """curl availability is checked before the curl | sh bootstrap call."""
        joined = _joined(self._cmd(tmp_path))
        assert "command -v curl" in joined
        curl_guard_idx = joined.index("command -v curl")
        curl_sh_idx = joined.index("curl -LsSf")
        assert curl_guard_idx < curl_sh_idx

    def test_runs_python_m_sloth_with_forwarded_args(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        assert "python -m sloth" in joined
        assert "train --config run.toml" in joined
        # the checkout is on PYTHONPATH so the bind-mounted source is used
        assert f"PYTHONPATH={CHECKOUT_MOUNT}" in joined

    def test_is_deterministic(self, tmp_path: Path) -> None:
        first = self._cmd(tmp_path)
        second = self._cmd(tmp_path)
        assert first == second

    def test_image_and_gpus_overridable(self, tmp_path: Path) -> None:
        cmd = build_command(
            ["eval"],
            workdir=tmp_path,
            checkout=tmp_path,
            image="example/img:tag",
            gpus="0",
        )
        joined = _joined(cmd)
        assert "example/img:tag" in cmd
        assert "--gpus 0" in joined

    def test_checkout_defaults_to_this_checkout(self, tmp_path: Path) -> None:
        # When checkout is omitted it resolves to the repo holding the package.
        cmd = build_command(["train"], workdir=tmp_path)
        joined = _joined(cmd)
        repo_root = Path(container.__file__).resolve().parents[2]
        assert f"{repo_root}:{CHECKOUT_MOUNT}" in joined


# ---------------------------------------------------------------------------
# 2b. extra_mounts contract
# ---------------------------------------------------------------------------


class TestExtraMounts:
    def test_extra_mounts_appear_in_command(self, tmp_path: Path) -> None:
        cmd = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
            extra_mounts=[("/data/dataset", "/data/dataset"), ("/output", "/output")],
        )
        joined = _joined(cmd)
        assert "-v /data/dataset:/data/dataset" in joined
        assert "-v /output:/output" in joined

    def test_extra_mounts_deduped_against_workdir(self, tmp_path: Path) -> None:
        # WORKDIR_MOUNT (/workspace) target — an extra_mount pointing there is skipped.
        cmd = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
            extra_mounts=[("/other", WORKDIR_MOUNT)],
        )
        joined = _joined(cmd)
        # Standard workdir mount present exactly once; extra duplicate discarded.
        assert joined.count(f":{WORKDIR_MOUNT}") == 1

    def test_extra_mounts_deduped_against_checkout(self, tmp_path: Path) -> None:
        # CHECKOUT_MOUNT (/opt/unsloth-cli) target — an extra_mount is skipped.
        cmd = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
            extra_mounts=[("/other", CHECKOUT_MOUNT)],
        )
        joined = _joined(cmd)
        assert joined.count(f":{CHECKOUT_MOUNT}") == 1

    def test_extra_mounts_deduped_among_themselves(self, tmp_path: Path) -> None:
        # When two extra_mounts share the same container target, only the first wins.
        cmd = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
            extra_mounts=[("/a", "/shared"), ("/b", "/shared")],
        )
        joined = _joined(cmd)
        assert joined.count(":/shared") == 1
        assert "-v /a:/shared" in joined  # first one wins

    def test_no_extra_mounts_same_as_none(self, tmp_path: Path) -> None:
        cmd_default = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
        )
        cmd_none = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path / "checkout",
            extra_mounts=None,
        )
        assert cmd_default == cmd_none

    def test_root_container_mount_refused(self, tmp_path: Path) -> None:
        # A "/" container target would overlay the container root with a host dir
        # (-v /:/ exposes the whole host fs) — build_command must refuse it.
        with pytest.raises(CliError) as exc:
            build_command(
                ["train"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                extra_mounts=[("/", "/")],
            )
        assert exc.value.code == 1
        assert "root" in str(exc.value).lower()

    def test_root_host_mount_refused(self, tmp_path: Path) -> None:
        # Even with a non-root container target, a "/" host path is refused.
        with pytest.raises(CliError):
            build_command(
                ["train"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                extra_mounts=[("/", "/host-root")],
            )

    def test_root_workdir_refused(self, tmp_path: Path) -> None:
        # The workdir is mounted as /workspace; "/" would mount the whole host fs.
        with pytest.raises(CliError) as exc:
            build_command(
                ["train"],
                workdir="/",
                checkout=tmp_path / "checkout",
            )
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# 2c. use_host_user contract
# ---------------------------------------------------------------------------


class TestUseHostUser:
    def test_user_flag_present_on_posix_by_default(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 1234)
        monkeypatch.setattr(os, "getgid", lambda: 5678)
        cmd = build_command(["train"], workdir=tmp_path, checkout=tmp_path)
        # On POSIX (Linux CI), --user uid:gid must appear.
        assert "--user" in cmd
        assert "1234:5678" in cmd

    def test_user_flag_absent_when_disabled(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(os, "getuid", lambda: 1234)
        monkeypatch.setattr(os, "getgid", lambda: 5678)
        cmd = build_command(
            ["train"],
            workdir=tmp_path,
            checkout=tmp_path,
            use_host_user=False,
        )
        assert "--user" not in cmd


# ---------------------------------------------------------------------------
# 3. preflight: every failure → CliError(code=2) + NGC remediation
# ---------------------------------------------------------------------------


def _assert_ngc_env_error(exc: CliError) -> None:
    assert exc.code == 2
    assert "nvcr.io/nvidia/pytorch:25.11-py3" in exc.remediation
    assert "nvidia-container-toolkit" in exc.remediation


class TestPreflight:
    def test_remediation_constant_names_image_and_toolkit(self) -> None:
        assert "nvcr.io/nvidia/pytorch:25.11-py3" in NGC_REMEDIATION
        assert "nvidia-container-toolkit" in NGC_REMEDIATION

    def test_docker_absent_raises_env_error(self, monkeypatch) -> None:
        monkeypatch.setattr(container, "_docker_available", lambda: False)
        # image/runtime probes should not even be reached, but stub them safe.
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: True)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: True)
        with pytest.raises(CliError) as exc_info:
            preflight()
        _assert_ngc_env_error(exc_info.value)

    def test_image_unpullable_raises_env_error(self, monkeypatch) -> None:
        monkeypatch.setattr(container, "_docker_available", lambda: True)
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: False)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: True)
        with pytest.raises(CliError) as exc_info:
            preflight()
        _assert_ngc_env_error(exc_info.value)

    def test_gpu_runtime_missing_raises_env_error(self, monkeypatch) -> None:
        monkeypatch.setattr(container, "_docker_available", lambda: True)
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: True)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: False)
        with pytest.raises(CliError) as exc_info:
            preflight()
        _assert_ngc_env_error(exc_info.value)

    def test_all_ok_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(container, "_docker_available", lambda: True)
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: True)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: True)
        assert preflight() is None


# ---------------------------------------------------------------------------
# Subprocess seams are stubbable (no real docker invoked)
# ---------------------------------------------------------------------------


class TestSubprocessSeams:
    def test_image_available_inspects_then_pulls(self, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_run_quiet(cmd: list[str]) -> int:
            calls.append(cmd)
            # inspect fails (image absent), pull succeeds
            return 1 if cmd[:3] == ["docker", "image", "inspect"] else 0

        monkeypatch.setattr(container, "_run_quiet", fake_run_quiet)
        assert container._image_available("img:x") is True
        assert calls[0][:3] == ["docker", "image", "inspect"]
        assert calls[1][:2] == ["docker", "pull"]

    def test_image_available_inspect_hit_skips_pull(self, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_run_quiet(cmd: list[str]) -> int:
            calls.append(cmd)
            return 0  # inspect succeeds

        monkeypatch.setattr(container, "_run_quiet", fake_run_quiet)
        assert container._image_available("img:x") is True
        assert len(calls) == 1  # no pull attempted

    def test_gpu_runtime_ok_uses_gpus_all_probe(self, monkeypatch) -> None:
        seen: list[list[str]] = []

        def fake_run_quiet(cmd: list[str]) -> int:
            seen.append(cmd)
            return 0

        monkeypatch.setattr(container, "_run_quiet", fake_run_quiet)
        assert container._gpu_runtime_ok("img:x") is True
        assert "--gpus" in seen[0]
        assert "all" in seen[0]


# ---------------------------------------------------------------------------
# h12 cross-check: no-docker path → preflight code=2
# ---------------------------------------------------------------------------


class TestH12CrossCheck:
    """h12 honesty-condition cross-check — the two halves of 'no GPU path → code=2'.

    Host side (no docker): ``preflight()`` raises ``CliError(code=2)`` — asserted
    in ``test_h12_no_docker_yields_preflight_code2`` below.

    In-container side (docker present but no GPU accelerator available inside):
    that half lives in ``tests/test_tune_trainer.py``, which verifies the
    in-container training path exits with code=2 when no accelerator is found.
    The two halves share the same exit-code contract so an agent reading a ``hint:``
    line always knows to look at the environment, not the code.
    """

    def test_h12_no_docker_yields_preflight_code2(self, monkeypatch) -> None:
        """h12 (host side): docker absent → preflight raises CliError(code=2).

        This is the explicit h12 named anchor. The same path is also exercised
        in ``TestPreflight.test_docker_absent_raises_env_error`` with full
        remediation-content assertions; this test documents the *honesty condition*
        that code=2 is the required exit for the no-docker environment failure.

        In-container no-accelerator coverage: see tests/test_tune_trainer.py.
        """
        monkeypatch.setattr(container, "_docker_available", lambda: False)
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: True)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: True)
        with pytest.raises(CliError) as exc_info:
            preflight()
        assert exc_info.value.code == 2, (
            "h12 requires the no-docker path to exit code=2 (env-setup error, "
            f"EXIT_ENV_ERROR); got code={exc_info.value.code}"
        )


# ---------------------------------------------------------------------------
# 4. launch: preflight + build + stream, all via stubbable seams
# ---------------------------------------------------------------------------


def _ok_stream(record: dict | None = None):
    """Return a ``_stream`` stub: exit 0 with a single parseable JSON result line."""

    def stub(cmd: list[str]) -> container.StreamResult:
        if record is not None:
            record.setdefault("cmd", cmd)
        return container.StreamResult(0, ['{"ok": true}'])

    return stub


class TestLaunch:
    def test_runs_preflight_then_streams_exit_code(self, tmp_path: Path, monkeypatch) -> None:
        events: dict[str, object] = {}

        def fake_preflight(*, image: str = NGC_IMAGE) -> None:
            events["preflight"] = image

        def fake_stream(cmd: list[str]) -> container.StreamResult:
            events["streamed"] = cmd
            return container.StreamResult(0, ['{"ok": true}'])  # success path

        monkeypatch.setattr(container, "preflight", fake_preflight)
        monkeypatch.setattr(container, "_stream", fake_stream)

        result = launch(["train", "--config", "run.toml"], workdir=tmp_path, checkout=tmp_path)

        assert result == {"ok": True}
        assert events["preflight"] == NGC_IMAGE
        streamed = events["streamed"]
        assert isinstance(streamed, list)
        assert "docker" in streamed
        assert "python -m sloth" in _joined(streamed)

    def test_skip_preflight_does_not_call_preflight(self, tmp_path: Path, monkeypatch) -> None:
        def boom(*, image: str = NGC_IMAGE) -> None:
            raise AssertionError("preflight must not run when skip_preflight=True")

        monkeypatch.setattr(container, "preflight", boom)
        monkeypatch.setattr(container, "_stream", _ok_stream())
        assert launch(["eval"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True) == {
            "ok": True
        }

    def test_preflight_failure_propagates_before_stream(self, tmp_path: Path, monkeypatch) -> None:
        def fail_preflight(*, image: str = NGC_IMAGE) -> None:
            raise CliError(code=2, message="no docker", remediation=NGC_REMEDIATION)

        def must_not_stream(cmd: list[str]) -> int:
            raise AssertionError("_stream must not run when preflight fails")

        monkeypatch.setattr(container, "preflight", fail_preflight)
        monkeypatch.setattr(container, "_stream", must_not_stream)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path)
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 4b. launch exit-code → CliError mapping contract
# ---------------------------------------------------------------------------


class TestLaunchExitCodeMapping:
    """Exit-code contract: launch() maps container exit codes to CliError.

    _stream/_run_quiet return raw ints internally; the mapping lives only in
    launch(). preflight is stubbed out for all tests in this class.
    """

    def test_exit_0_returns_the_parsed_result(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(container, "_stream", _ok_stream())
        result = launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert result == {"ok": True}

    def test_exit_1_raises_cli_error_code_1(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(container, "_stream", lambda cmd: 1)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 1

    def test_exit_2_raises_cli_error_code_2(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(container, "_stream", lambda cmd: 2)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 2

    def test_exit_137_raises_code_2_with_oom_hint(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(container, "_stream", lambda cmd: 137)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        err = exc_info.value
        assert err.code == 2
        assert "137" in err.message
        assert "drop_caches" in err.remediation

    def test_exit_125_raises_cli_error_code_2(self, tmp_path: Path, monkeypatch) -> None:
        """exit 125 = docker-level error (image not found, bad flags)."""
        monkeypatch.setattr(container, "_stream", lambda cmd: 125)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 2

    def test_exit_127_raises_cli_error_code_2(self, tmp_path: Path, monkeypatch) -> None:
        """exit 127 = command not found inside or outside the container."""
        monkeypatch.setattr(container, "_stream", lambda cmd: 127)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 2

    def test_arbitrary_nonzero_raises_cli_error_code_2(self, tmp_path: Path, monkeypatch) -> None:
        """Any exit code not in {0, 1, 2, 137} maps to CliError(code=2)."""
        monkeypatch.setattr(container, "_stream", lambda cmd: 42)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# 5. t8 — llama.cpp cache mount, explicit HOME, env passthrough, memory preflight
# ---------------------------------------------------------------------------


class TestExportLaunchKwargs:
    """The export-run contract: explicit HOME + a host-owned llama.cpp cache mount.

    Unsloth resolves its llama.cpp checkout from ``Path.home()/.unsloth/llama.cpp``
    (unsloth_zoo/llama_cpp.py), so ``HOME`` must be explicit inside the container and
    the bind-mount target must match it exactly — otherwise the prebuilt llama.cpp
    install is redone on every ``--rm`` run.
    """

    def test_constants(self) -> None:
        # HOME is its own mount, outside /workspace: docker creates missing mount-point
        # parents as root, which made a HOME nested under the workdir unwritable.
        assert container.EXPORT_HOME == "/opt/sloth-home"
        assert not container.EXPORT_HOME.startswith(container.WORKDIR_MOUNT)
        assert container.EXPORT_HOME_ENV == "SLOTH_EXPORT_HOME"
        assert container.DEFAULT_EXPORT_HOME == Path.home() / ".cache" / "unsloth-cli" / "home"
        assert container.UNSLOTH_LLAMA_TAG == "b10909"
        assert container.LLAMA_CPP_CACHE_ENV == "SLOTH_LLAMA_CPP_CACHE"
        assert container.DEFAULT_LLAMA_CPP_CACHE == (
            container.DEFAULT_EXPORT_HOME / ".unsloth" / "llama.cpp"
        )

    def test_kwargs_shape_and_mount_target(self, tmp_path: Path, monkeypatch) -> None:
        home = tmp_path / "home"
        cache = tmp_path / "llama-cache"
        monkeypatch.setenv(container.EXPORT_HOME_ENV, str(home))
        monkeypatch.setenv(container.LLAMA_CPP_CACHE_ENV, str(cache))
        kwargs = container.export_launch_kwargs()
        assert kwargs["env"] == [
            ("HOME", container.EXPORT_HOME),
            ("UNSLOTH_LLAMA_TAG", container.UNSLOTH_LLAMA_TAG),
        ]
        assert kwargs["extra_mounts"] == [
            (str(home), container.EXPORT_HOME),
            (str(cache), container.EXPORT_HOME + "/.unsloth/llama.cpp"),
        ]
        # The nested mount-point is pre-created host-side so docker never makes it as root.
        assert (home / ".unsloth" / "llama.cpp").is_dir()

    def test_home_only_mount_without_cache_override(self, tmp_path: Path, monkeypatch) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv(container.EXPORT_HOME_ENV, str(home))
        monkeypatch.delenv(container.LLAMA_CPP_CACHE_ENV, raising=False)
        kwargs = container.export_launch_kwargs()
        assert kwargs["extra_mounts"] == [(str(home), container.EXPORT_HOME)]
        assert (home / ".unsloth" / "llama.cpp").is_dir()

    def test_creates_host_cache_dir(self, tmp_path: Path, monkeypatch) -> None:
        cache = tmp_path / "nested" / "llama.cpp"
        monkeypatch.setenv(container.EXPORT_HOME_ENV, str(tmp_path / "home"))
        monkeypatch.setenv(container.LLAMA_CPP_CACHE_ENV, str(cache))
        assert not cache.exists()
        container.export_launch_kwargs()
        assert cache.is_dir()
        # Idempotent: a second call on an existing dir must not raise.
        container.export_launch_kwargs()

    def test_default_home_used_without_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv(container.LLAMA_CPP_CACHE_ENV, raising=False)
        monkeypatch.delenv(container.EXPORT_HOME_ENV, raising=False)
        default = tmp_path / "default-home"
        monkeypatch.setattr(container, "DEFAULT_EXPORT_HOME", default)
        kwargs = container.export_launch_kwargs()
        assert kwargs["extra_mounts"] == [(str(default), container.EXPORT_HOME)]
        assert (default / ".unsloth" / "llama.cpp").is_dir()

    def test_kwargs_feed_build_command(self, tmp_path: Path, monkeypatch) -> None:
        cache = tmp_path / "llama-cache"
        monkeypatch.setenv(container.EXPORT_HOME_ENV, str(tmp_path / "home"))
        monkeypatch.setenv(container.LLAMA_CPP_CACHE_ENV, str(cache))
        kwargs = container.export_launch_kwargs()
        joined = _joined(
            build_command(
                ["export", "--adapter", "a"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=tmp_path / "none",
                **kwargs,
            )
        )
        assert f"-e HOME={container.EXPORT_HOME}" in joined
        assert f"-e UNSLOTH_LLAMA_TAG={container.UNSLOTH_LLAMA_TAG}" in joined
        assert f"-v {cache}:{container.EXPORT_HOME}/.unsloth/llama.cpp" in joined


class TestLaunchEnvPassthrough:
    def test_launch_forwards_env_and_extra_mounts(self, tmp_path: Path, monkeypatch) -> None:
        seen: dict[str, list[str]] = {}

        monkeypatch.setattr(container, "_stream", _ok_stream(seen))
        result = launch(
            ["export"],
            workdir=tmp_path,
            checkout=tmp_path,
            skip_preflight=True,
            env=[("HOME", container.EXPORT_HOME)],
            extra_mounts=[("/host/cache", "/workspace/.home/.unsloth/llama.cpp")],
        )
        assert result == {"ok": True}
        joined = _joined(seen["cmd"])
        assert f"-e HOME={container.EXPORT_HOME}" in joined
        assert "-v /host/cache:/workspace/.home/.unsloth/llama.cpp" in joined

    def test_launch_accepts_export_launch_kwargs(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(container.LLAMA_CPP_CACHE_ENV, str(tmp_path / "c"))
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(container, "_stream", _ok_stream(captured))
        assert launch(
            ["export"],
            workdir=tmp_path,
            checkout=tmp_path,
            skip_preflight=True,
            **container.export_launch_kwargs(),
        ) == {"ok": True}
        assert f"-e UNSLOTH_LLAMA_TAG={container.UNSLOTH_LLAMA_TAG}" in _joined(captured["cmd"])


class TestQuantizationPins:
    def test_llmcompressor_and_compressed_tensors_pinned(self) -> None:
        assert "llmcompressor==0.11.0" in DEP_LAYER_PACKAGES
        assert "compressed-tensors==0.16.0" in DEP_LAYER_PACKAGES

    def test_pins_reach_the_install_line(self, tmp_path: Path) -> None:
        joined = _joined(
            build_command(
                ["export"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=tmp_path / "none",
            )
        )
        assert "llmcompressor==0.11.0" in joined
        assert "compressed-tensors==0.16.0" in joined

    def test_pin_matrix_comment_documents_the_choice(self) -> None:
        source = inspect.getsource(container)
        assert "0.10.0.3" in source, "the rejected llmcompressor version must be documented"
        assert "AWQ" in source
        assert "LFM2" in source
        assert "2.11" in source


class TestMemoryPreflight:
    """preflight() emits a low-memory hint from a faked /proc/meminfo; never blocks."""

    @staticmethod
    def _ok_probes(monkeypatch) -> None:
        monkeypatch.setattr(container, "_docker_available", lambda: True)
        monkeypatch.setattr(container, "_image_available", lambda image=NGC_IMAGE: True)
        monkeypatch.setattr(container, "_gpu_runtime_ok", lambda image=NGC_IMAGE: True)

    @staticmethod
    def _meminfo(tmp_path: Path, mem_free_kib: int) -> Path:
        path = tmp_path / "meminfo"
        path.write_text(
            f"MemTotal:      119537664 kB\nMemFree:       {mem_free_kib} kB\n"
            "MemAvailable:   50000000 kB\n",
            encoding="utf-8",
        )
        return path

    def test_low_memfree_emits_hint_and_does_not_block(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        self._ok_probes(monkeypatch)
        monkeypatch.setattr(container, "MEMINFO_PATH", self._meminfo(tmp_path, 1024 * 1024))
        assert preflight() is None
        err = capsys.readouterr().err
        assert "out of memory" in err
        assert "expandable_segments" in err
        assert "page cache" in err
        assert capsys.readouterr().out == ""

    def test_ample_memfree_emits_nothing(self, tmp_path: Path, monkeypatch, capsys) -> None:
        self._ok_probes(monkeypatch)
        monkeypatch.setattr(container, "MEMINFO_PATH", self._meminfo(tmp_path, 64 * 1024 * 1024))
        assert preflight() is None
        assert capsys.readouterr().err == ""

    def test_missing_meminfo_is_silent(self, tmp_path: Path, monkeypatch, capsys) -> None:
        self._ok_probes(monkeypatch)
        monkeypatch.setattr(container, "MEMINFO_PATH", tmp_path / "absent")
        assert preflight() is None
        assert capsys.readouterr().err == ""

    def test_unparsable_meminfo_is_silent(self, tmp_path: Path, monkeypatch, capsys) -> None:
        self._ok_probes(monkeypatch)
        bad = tmp_path / "meminfo"
        bad.write_text("garbage\nMemFree: not-a-number kB\n", encoding="utf-8")
        monkeypatch.setattr(container, "MEMINFO_PATH", bad)
        assert preflight() is None
        assert capsys.readouterr().err == ""

    def test_hint_emitted_before_a_failing_docker_probe(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """The hint is informational: it must survive a preflight that then raises."""
        monkeypatch.setattr(container, "_docker_available", lambda: False)
        monkeypatch.setattr(container, "MEMINFO_PATH", self._meminfo(tmp_path, 1024))
        with pytest.raises(CliError):
            preflight()
        assert "out of memory" in capsys.readouterr().err

    def test_threshold_is_four_gib(self, tmp_path: Path, monkeypatch, capsys) -> None:
        self._ok_probes(monkeypatch)
        assert container.LOW_MEMFREE_KIB == 4 * 1024 * 1024
        # Exactly at the threshold: not low.
        monkeypatch.setattr(container, "MEMINFO_PATH", self._meminfo(tmp_path, 4 * 1024 * 1024))
        preflight()
        assert capsys.readouterr().err == ""
        # One KiB under: low.
        monkeypatch.setattr(container, "MEMINFO_PATH", self._meminfo(tmp_path, 4 * 1024 * 1024 - 1))
        preflight()
        assert "out of memory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. t1 — launch captures the container's stdout (tee to stderr, last JSON wins)
# ---------------------------------------------------------------------------


class _FakeStdout:
    """Lazy line iterator that snapshots host stderr before each yield."""

    def __init__(self, lines: list[str], proc: "_FakeProc") -> None:
        self._lines = lines
        self._proc = proc
        self.closed = False

    def __enter__(self) -> "_FakeStdout":
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def __iter__(self):
        for line in self._lines:
            # Snapshot what the host has already teed, and whether the process
            # has been reaped, *before* handing over the next line.
            self._proc.observations.append((self._proc.stderr_value(), self._proc.waited))
            yield line


class _FakeProc:
    """Minimal stand-in for ``subprocess.Popen`` with a line-yielding stdout."""

    def __init__(self, lines: list[str], code: int, stderr_buf: io.StringIO) -> None:
        self._code = code
        self._stderr = stderr_buf
        self.waited = False
        self.observations: list[tuple[str, bool]] = []
        self.stdout = _FakeStdout(lines, self)

    def stderr_value(self) -> str:
        return self._stderr.getvalue()

    def wait(self) -> int:
        self.waited = True
        return self._code


def _fake_popen(
    monkeypatch, lines: list[str], code: int = 0
) -> tuple[_FakeProc, io.StringIO, dict[str, object]]:
    """Install a fake Popen yielding *lines*; return (proc, stderr buffer, call record)."""
    buf = io.StringIO()
    monkeypatch.setattr(container.sys, "stderr", buf)
    proc = _FakeProc([line + "\n" for line in lines], code, buf)
    record: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        record["cmd"] = cmd
        record["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(container.subprocess, "Popen", fake_popen)
    return proc, buf, record


class TestStreamCapture:
    """``_stream`` tees every line to stderr live and returns the captured lines."""

    def test_popen_pipes_stdout_and_inherits_stderr(self, monkeypatch) -> None:
        _proc, _buf, record = _fake_popen(monkeypatch, ["hello"])
        container._stream(["docker", "run"])
        kwargs = record["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is None, "the container's own stderr must pass straight through"

    def test_lines_reach_stderr_before_the_process_exits(self, monkeypatch) -> None:
        proc, buf, _record = _fake_popen(monkeypatch, ["line-1", "line-2", "line-3"])
        result = container._stream(["docker", "run"])

        # The snapshot taken just before line-2 was handed over already shows line-1
        # on stderr and an unreaped process: streaming is live, not buffered to the end.
        before_second = proc.observations[1]
        assert "line-1" in before_second[0]
        assert before_second[1] is False
        assert "line-2" not in before_second[0]

        assert buf.getvalue() == "line-1\nline-2\nline-3\n"
        assert int(result) == 0
        assert list(result.lines) == ["line-1", "line-2", "line-3"]
        assert proc.waited is True
        assert proc.stdout.closed is True

    def test_oserror_maps_to_127(self, monkeypatch) -> None:
        def boom(cmd, **kwargs):
            raise OSError("docker not found")

        monkeypatch.setattr(container.subprocess, "Popen", boom)
        result = container._stream(["docker", "run"])
        assert int(result) == 127
        assert list(result.lines) == []

    def test_nonzero_exit_code_is_returned_with_lines(self, monkeypatch) -> None:
        _proc, _buf, _record = _fake_popen(monkeypatch, ["boom"], code=1)
        result = container._stream(["docker", "run"])
        assert int(result) == 1
        assert list(result.lines) == ["boom"]


class TestLaunchResultCapture:
    """launch() returns the LAST parseable JSON line, and fails closed without one."""

    def _stub(self, monkeypatch, lines: list[str], code: int = 0) -> None:
        monkeypatch.setattr(
            container,
            "_stream",
            lambda cmd: container.StreamResult(code, lines),
        )

    def test_returns_last_json_line(self, tmp_path: Path, monkeypatch) -> None:
        self._stub(
            monkeypatch,
            [
                "== NGC banner ==",
                '{"ok": false, "stage": "early"}',
                "training: 10/10",
                '{"ok": true, "adapter": "/out/adapter", "steps": 10}',
            ],
        )
        result = launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert result == {"ok": True, "adapter": "/out/adapter", "steps": 10}

    def test_non_json_trailing_lines_do_not_hide_the_result(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._stub(monkeypatch, ['{"ok": true}', "Segmentation-free shutdown", "bye"])
        result = launch(["eval"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert result == {"ok": True}

    def test_non_object_json_is_not_a_result(self, tmp_path: Path, monkeypatch) -> None:
        # A bare scalar (a step counter, say) is not a result payload.
        self._stub(monkeypatch, ['{"ok": true}', "137", "[1, 2]"])
        result = launch(["eval"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert result == {"ok": True}

    def test_exit_0_without_json_fails_closed_with_single_line_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        lines = [f"banner-{i:02d}" for i in range(40)]
        # Go through the real `_stream` (via `_fake_popen`) rather than the
        # `_stub` shortcut, so the captured lines are actually teed to
        # `sys.stderr` as `launch()` documents — the error message must stay a
        # single physical line and rely on that already-streamed output
        # instead of re-quoting it.
        _proc, buf, _record = _fake_popen(monkeypatch, lines)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        err = exc_info.value
        assert err.code == 2
        assert err.message.count("\n") == 0
        assert "40" in err.message
        assert err.remediation.count("\n") == 0
        # The banner lines were already streamed to stderr as they arrived.
        assert "banner-39" in buf.getvalue()

    def test_exit_0_with_no_output_at_all_fails_closed(self, tmp_path: Path, monkeypatch) -> None:
        self._stub(monkeypatch, [])
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 2

    def test_nonzero_exit_still_maps_before_parsing(self, tmp_path: Path, monkeypatch) -> None:
        # A parseable JSON line does NOT rescue a non-zero exit.
        self._stub(monkeypatch, ['{"ok": true}'], code=1)
        with pytest.raises(CliError) as exc_info:
            launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert exc_info.value.code == 1


class TestNodepsLayerPins:
    """DEP_LAYER_NODEPS_PACKAGES pins (t8) — unsloth / unsloth_zoo / bitsandbytes.

    Versions measured 2026-09-15 on NGC 25.11 (torch 2.10) via a plain
    ``docker run --rm`` of the exact install line ``container.py`` composes
    (DEP_LAYER_PACKAGES then DEP_LAYER_NODEPS_PACKAGES into a
    ``uv venv --system-site-packages`` venv), followed by ``uv pip list``.
    Recorded in docs/tested.md (2026-09-15 row) and docs/dgx-spark.md.
    """

    def test_exactly_three_pinned_entries(self) -> None:
        assert len(DEP_LAYER_NODEPS_PACKAGES) == 3
        for pkg in DEP_LAYER_NODEPS_PACKAGES:
            assert "==" in pkg, f"expected a pin, got unpinned entry: {pkg}"

    def test_pins_match_the_live_validated_versions(self) -> None:
        assert "unsloth==2026.9.4" in DEP_LAYER_NODEPS_PACKAGES
        assert "unsloth_zoo==2026.9.3" in DEP_LAYER_NODEPS_PACKAGES
        assert "bitsandbytes==0.50.2" in DEP_LAYER_NODEPS_PACKAGES

    def test_pins_reach_the_built_docker_command(self, tmp_path: Path) -> None:
        joined = _joined(
            build_command(
                ["train"],
                workdir=tmp_path,
                checkout=tmp_path / "checkout",
                hf_cache=tmp_path / "none",
            )
        )
        for pkg in DEP_LAYER_NODEPS_PACKAGES:
            assert pkg in joined, f"pin did not reach the docker command: {pkg}"

    def test_dep_layer_packages_untouched(self) -> None:
        # t1 owns DEP_LAYER_PACKAGES in this task; it must stay byte-identical to
        # main's current pinned set (this task only touches DEP_LAYER_NODEPS_PACKAGES).
        assert DEP_LAYER_PACKAGES == (
            "transformers==4.57.1",
            "peft==0.18.0",
            "hf_transfer",
            "datasets==4.8.5",
            "trl==0.24.0",
            "llmcompressor==0.11.0",
            "compressed-tensors==0.16.0",
        )
