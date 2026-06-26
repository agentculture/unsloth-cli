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

    def test_installs_dep_layer_with_uv_pip_system(self, tmp_path: Path) -> None:
        joined = _joined(self._cmd(tmp_path))
        assert "uv pip install --system" in joined
        # full-resolution layer
        for pkg in DEP_LAYER_PACKAGES:
            assert pkg in joined, f"missing dep-layer package: {pkg}"
        assert "uv pip install --system transformers peft hf_transfer" in joined
        assert "datasets==4.3.0" in joined
        assert "trl==0.26.1" in joined
        # --no-deps layer
        assert "uv pip install --system --no-deps unsloth unsloth_zoo bitsandbytes" in joined
        for pkg in DEP_LAYER_NODEPS_PACKAGES:
            assert pkg in joined, f"missing --no-deps package: {pkg}"

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


class TestLaunch:
    def test_runs_preflight_then_streams_exit_code(self, tmp_path: Path, monkeypatch) -> None:
        events: dict[str, object] = {}

        def fake_preflight(*, image: str = NGC_IMAGE) -> None:
            events["preflight"] = image

        def fake_stream(cmd: list[str]) -> int:
            events["streamed"] = cmd
            return 0  # success path

        monkeypatch.setattr(container, "preflight", fake_preflight)
        monkeypatch.setattr(container, "_stream", fake_stream)

        code = launch(["train", "--config", "run.toml"], workdir=tmp_path, checkout=tmp_path)

        assert code == 0
        assert events["preflight"] == NGC_IMAGE
        streamed = events["streamed"]
        assert isinstance(streamed, list)
        assert "docker" in streamed
        assert "python -m sloth" in _joined(streamed)

    def test_skip_preflight_does_not_call_preflight(self, tmp_path: Path, monkeypatch) -> None:
        def boom(*, image: str = NGC_IMAGE) -> None:
            raise AssertionError("preflight must not run when skip_preflight=True")

        monkeypatch.setattr(container, "preflight", boom)
        monkeypatch.setattr(container, "_stream", lambda cmd: 0)
        assert launch(["eval"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True) == 0

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

    def test_exit_0_returns_0(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(container, "_stream", lambda cmd: 0)
        result = launch(["train"], workdir=tmp_path, checkout=tmp_path, skip_preflight=True)
        assert result == 0

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
