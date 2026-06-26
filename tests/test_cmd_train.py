"""Tests for ``sloth train`` — the integrator verb.

Flow under test: load config -> validate dataset -> scope-guard (emit warning)
-> dry-run OR real train (delegated to ``run_training`` / ``container.launch``).

Covers acceptance criteria for t4 (container orchestration) and the original t10
criteria (host-side GPU-free preflight).

Branch behaviour tested:
1. ``--dry-run`` (host, no GPU, no docker) — validates dataset/config/scope,
   prints the resolved plan + the docker image + command, exits 0.  Calls neither
   ``container.launch`` nor ``container.preflight``.
2. ``--in-container`` (recursion guard) — delegates directly to ``run_training``,
   never calls ``container.launch`` again.
3. host real run (default) — calls ``container.launch`` with ``--in-container``
   forwarded; the container exit code is returned.
4. Bad dataset / out-of-scope model — refuses with ``CliError`` BEFORE calling
   ``container.launch`` or ``run_training``.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import sloth.cli._commands.train as train_mod
import sloth.tune.container as container_mod
from sloth.cli._commands.train import cmd_train, register
from sloth.cli._errors import CliError
from sloth.cli._output import emit_error
from sloth.tune.config import RunConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _make_args(
    config: Path,
    *,
    dry_run: bool = False,
    json_mode: bool = False,
    in_container: bool = False,
) -> argparse.Namespace:
    """Build the Namespace argparse would produce for ``sloth train``."""
    return argparse.Namespace(
        config=str(config),
        dry_run=dry_run,
        json=json_mode,
        in_container=in_container,
    )


def _write_dataset(tmp_path: Path, body: str = _VALID_CHAT, name: str = "train.jsonl") -> Path:
    f = tmp_path / name
    f.write_text(body, encoding="utf-8")
    return f


def _write_toml(
    tmp_path: Path,
    *,
    dataset: Path,
    model: str = "unsloth/Qwen3-4B",
    method: str = "qlora",
    output: str | None = None,
    name: str = "run.toml",
) -> Path:
    out = output if output is not None else str(tmp_path / "adapters" / "out")
    toml = (
        "[run]\n"
        f'model   = "{model}"\n'
        f'method  = "{method}"\n'
        f'dataset = "{dataset}"\n'
        f'output  = "{out}"\n'
    )
    f = tmp_path / name
    f.write_text(toml, encoding="utf-8")
    return f


@pytest.fixture()
def good_config(tmp_path: Path) -> Path:
    """A valid run.toml + valid chat dataset; returns the toml path."""
    dataset = _write_dataset(tmp_path)
    return _write_toml(tmp_path, dataset=dataset)


# ---------------------------------------------------------------------------
# Acceptance 1 — dry-run plan: text + json, exit 0, no docker calls
# ---------------------------------------------------------------------------


def test_dry_run_text_plan(good_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run prints a human-readable resolved plan to stdout and exits 0."""
    rc = cmd_train(_make_args(good_config, dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "unsloth/Qwen3-4B" in out
    assert "qlora" in out
    # Hyperparameters are resolved into the plan.
    assert "lora_r" in out
    assert "dry-run" in out.lower()


def test_dry_run_json_plan(good_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run with --json emits the same plan as a structured JSON object."""
    rc = cmd_train(_make_args(good_config, dry_run=True, json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "unsloth/Qwen3-4B"
    assert payload["method"] == "qlora"
    assert payload["dry_run"] is True
    assert "hyperparameters" in payload
    assert payload["hyperparameters"]["lora_r"] == 16
    assert "scope" in payload


def test_dry_run_does_not_load_backend(good_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must never reach the heavy backend loader.

    Sabotage ``_load_backend`` so any attempt to enter the real training path
    explodes; the dry-run must still return 0, proving it short-circuits before
    any GPU/torch work.
    """
    import sloth.tune._trainer as trainer_mod

    def _boom() -> Any:
        raise AssertionError("dry-run must not load the ML backend")

    monkeypatch.setattr(trainer_mod, "_load_backend", _boom)
    rc = cmd_train(_make_args(good_config, dry_run=True))
    assert rc == 0


def test_dry_run_imports_no_torch_in_fresh_process(good_config: Path) -> None:
    """In a fresh interpreter, a dry-run leaves torch/unsloth out of sys.modules.

    torch *is* installed in this environment, so the only rigorous proof that the
    dry-run path stays torch-free is to run it in a subprocess and inspect that
    process's ``sys.modules`` (mirrors tests/test_lazy_import.py).
    """
    code = (
        "import sys, argparse;"
        "from sloth.cli._commands.train import cmd_train;"
        f"rc = cmd_train(argparse.Namespace(config=r'{good_config}', dry_run=True, json=True));"
        "assert rc == 0, rc;"
        "assert 'torch' not in sys.modules, 'torch imported during dry-run';"
        "assert 'unsloth' not in sys.modules, 'unsloth imported during dry-run';"
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# t4 acceptance — dry-run prints docker image + command; no container calls
# ---------------------------------------------------------------------------


def test_dry_run_prints_docker_image_no_container_calls(
    good_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints the NGC image string; neither launch nor preflight is called."""
    mock_launch = Mock()
    mock_preflight = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)
    monkeypatch.setattr(container_mod, "preflight", mock_preflight)

    rc = cmd_train(_make_args(good_config, dry_run=True))
    assert rc == 0
    mock_launch.assert_not_called()
    mock_preflight.assert_not_called()
    out = capsys.readouterr().out
    assert container_mod.NGC_IMAGE in out


def test_dry_run_json_includes_docker_image_and_command(
    good_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run --json includes docker_image and docker_command in the JSON payload."""
    monkeypatch.setattr(container_mod, "launch", Mock())
    monkeypatch.setattr(container_mod, "preflight", Mock())

    rc = cmd_train(_make_args(good_config, dry_run=True, json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["docker_image"] == container_mod.NGC_IMAGE
    assert isinstance(payload["docker_command"], list)
    assert payload["docker_command"][0] == "docker"
    # --in-container is embedded inside the inner bash -lc script (last argv element).
    inner_script = payload["docker_command"][-1]
    assert "--in-container" in inner_script


# ---------------------------------------------------------------------------
# Acceptance 1 — invalid dataset -> CliError(1), error:/hint: contract
# ---------------------------------------------------------------------------


def test_invalid_dataset_raises_cli_error_1(tmp_path: Path) -> None:
    """A dataset that fails schema validation exits 1 BEFORE any GPU work."""
    bad = _write_dataset(
        tmp_path,
        body='{"messages": [{"role": "wizard", "content": "x"}]}\n',
    )
    toml = _write_toml(tmp_path, dataset=bad)
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(toml, dry_run=True))
    assert exc_info.value.code == 1
    assert exc_info.value.remediation


def test_invalid_dataset_error_hint_contract(tmp_path: Path) -> None:
    """The invalid-dataset CliError renders as ``error:`` / ``hint:`` lines."""
    bad = _write_dataset(tmp_path, body="not valid json\n")
    toml = _write_toml(tmp_path, dataset=bad)
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(toml, dry_run=True))
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_invalid_dataset_validates_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation happens before delegation: container.launch is never called."""
    bad = _write_dataset(tmp_path, body='{"messages": []}\n')  # empty messages list
    toml = _write_toml(tmp_path, dataset=bad)

    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    def _must_not_run(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("run_training called despite invalid dataset")

    monkeypatch.setattr(train_mod, "run_training", _must_not_run)
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(toml, dry_run=False))
    assert exc_info.value.code == 1
    mock_launch.assert_not_called()


# ---------------------------------------------------------------------------
# t4 acceptance — bad dataset / out-of-scope: container.launch NOT called
# ---------------------------------------------------------------------------


def test_bad_dataset_does_not_call_container_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid dataset: host-side validation refuses before calling container.launch."""
    bad = _write_dataset(tmp_path, body='{"messages": []}\n')  # empty messages
    toml = _write_toml(tmp_path, dataset=bad)

    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(toml, dry_run=False))
    assert exc_info.value.code == 1
    mock_launch.assert_not_called()


def test_out_of_scope_does_not_call_container_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out-of-scope model: scope guard refuses before calling container.launch."""
    dataset = _write_dataset(tmp_path)
    cfg = RunConfig(
        model="unsloth/Qwen3-72B",
        dataset=str(dataset),
        output=str(tmp_path / "out"),
        method="full",
    )
    monkeypatch.setattr(train_mod, "load_config", lambda _path: cfg)

    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    args = argparse.Namespace(config="ignored.toml", dry_run=False, json=False, in_container=False)
    with pytest.raises(CliError) as exc_info:
        cmd_train(args)
    assert exc_info.value.code == 1
    mock_launch.assert_not_called()


# ---------------------------------------------------------------------------
# H1 anchor — consolidated named proof for honesty condition h1 (issue #9)
# ---------------------------------------------------------------------------


def test_h1_anchor_bad_dataset_and_out_of_scope_never_invoke_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1 anchor: both bad-dataset (h1a) and out-of-scope (h1b) raise CliError(code=1)
    WITHOUT ever calling sloth.tune.container.launch.

    Honesty condition h1 (issue #9): the host-side ``sloth train`` path must complete
    its dataset validation and scope check *before* delegating to the container layer.
    A misconfigured run must raise ``CliError(code=1)`` and leave
    ``container.launch`` un-called in both scenarios.

    Note: individual scenario tests already exist as
    ``test_bad_dataset_does_not_call_container_launch`` (h1a) and
    ``test_out_of_scope_does_not_call_container_launch`` (h1b) — this test is the
    single explicitly h1-named anchor that ties both conditions to the honesty
    condition label for traceability.
    """
    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    # --- H1a: invalid dataset (empty messages list) ---
    bad = _write_dataset(tmp_path, body='{"messages": []}\n', name="bad_ds.jsonl")
    toml_bad = _write_toml(tmp_path, dataset=bad, name="bad_run.toml")
    with pytest.raises(CliError) as exc_info_a:
        cmd_train(_make_args(toml_bad, dry_run=False))
    assert exc_info_a.value.code == 1, "h1a: expected CliError code=1 for invalid dataset"
    mock_launch.assert_not_called()

    # --- H1b: out-of-scope model + method (full fine-tune of large model) ---
    dataset_ok = _write_dataset(tmp_path, name="ok.jsonl")
    cfg_oos = RunConfig(
        model="unsloth/Qwen3-72B",
        dataset=str(dataset_ok),
        output=str(tmp_path / "out"),
        method="full",
    )
    monkeypatch.setattr(train_mod, "load_config", lambda _path: cfg_oos)
    args_oos = argparse.Namespace(
        config="ignored.toml", dry_run=False, json=False, in_container=False
    )
    with pytest.raises(CliError) as exc_info_b:
        cmd_train(args_oos)
    assert exc_info_b.value.code == 1, "h1b: expected CliError code=1 for out-of-scope model"
    mock_launch.assert_not_called()


# ---------------------------------------------------------------------------
# Bad config propagation
# ---------------------------------------------------------------------------


def test_missing_config_propagates_cli_error(tmp_path: Path) -> None:
    """A non-existent config file surfaces load_config's CliError (code 2)."""
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(tmp_path / "nope.toml", dry_run=True))
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Acceptance 2 — out-of-scope target: explicit warning + refusal
# ---------------------------------------------------------------------------


def test_out_of_scope_emits_warning_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A large-dense full-fine-tune target warns on stderr then refuses (code 1).

    ``load_config`` rejects non-LoRA methods, so the only way the verb sees an
    out-of-scope (model, method) is via a config carrying ``method='full'``;
    feed one through a patched ``load_config`` and assert the verb emits the
    ``check_scope`` warning to stderr AND hard-refuses.
    """
    dataset = _write_dataset(tmp_path)
    cfg = RunConfig(
        model="unsloth/Qwen3-72B",
        dataset=str(dataset),
        output=str(tmp_path / "out"),
        method="full",
    )
    monkeypatch.setattr(train_mod, "load_config", lambda _path: cfg)
    # Guard: neither run_training nor container.launch should be reached.
    monkeypatch.setattr(
        train_mod,
        "run_training",
        lambda *_a, **_k: pytest.fail("run_training reached for out-of-scope request"),
    )
    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    args = argparse.Namespace(config="ignored.toml", dry_run=False, json=False, in_container=False)
    with pytest.raises(CliError) as exc_info:
        cmd_train(args)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    # The warning must be EXPLICIT on stderr, not silent.
    assert "out of scope" in err.lower() or "full fine-tuning" in err.lower()
    mock_launch.assert_not_called()


# ---------------------------------------------------------------------------
# t4 acceptance — host real run calls container.launch with --in-container
# ---------------------------------------------------------------------------


def test_host_real_run_calls_container_launch(
    good_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the host (not in-container, not dry-run), cmd_train calls container.launch."""
    mock_launch = Mock(return_value=0)
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    rc = cmd_train(_make_args(good_config, dry_run=False))
    assert rc == 0
    mock_launch.assert_called_once()
    # The forwarded args must include --in-container to prevent docker recursion.
    call_args = mock_launch.call_args
    sloth_args = call_args[0][0]  # first positional arg
    assert "train" in sloth_args
    assert "--in-container" in sloth_args
    assert "--config" in sloth_args


def test_host_real_run_returns_0_on_launch_success(
    good_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cmd_train returns 0 when container.launch succeeds (returns 0)."""
    monkeypatch.setattr(container_mod, "launch", Mock(return_value=0))
    rc = cmd_train(_make_args(good_config, dry_run=False))
    assert rc == 0


def test_host_real_run_propagates_launch_cli_error(
    good_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CliError raised by container.launch propagates out of cmd_train.

    launch() now raises CliError (code 1 or 2) on container failure instead of
    returning a non-zero int; the handler must not swallow it.
    """
    monkeypatch.setattr(
        container_mod,
        "launch",
        Mock(
            side_effect=CliError(
                code=2,
                message="Container exited with status 2",
                remediation="Check the in-container error output.",
            )
        ),
    )
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(good_config, dry_run=False))
    assert exc_info.value.code == 2


def test_host_real_run_json_forwarded_to_container(
    good_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When --json is set, --json is included in the forwarded container args."""
    mock_launch = Mock(return_value=0)
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    cmd_train(_make_args(good_config, dry_run=False, json_mode=True))
    sloth_args = mock_launch.call_args[0][0]
    assert "--json" in sloth_args


def test_host_real_run_extra_mounts_cover_config_dataset_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity mounts cover the parent dirs of config, dataset, and output.

    FIX 4 (path visibility): host-absolute paths forwarded in sloth_args must
    resolve unchanged inside the container; extra_mounts achieves this by
    mounting each parent dir at the same path (identity mount).
    """
    dataset = tmp_path / "data" / "train.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "hi"},'
        '{"role": "assistant", "content": "hello"}]}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "adapters" / "out"
    toml_path = tmp_path / "run.toml"
    toml_path.write_text(
        "[run]\n"
        'model   = "unsloth/Qwen3-4B"\n'
        'method  = "qlora"\n'
        f'dataset = "{dataset}"\n'
        f'output  = "{out_dir}"\n',
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured.update(kwargs)
        captured["sloth_args"] = list(sloth_args)
        return 0

    monkeypatch.setattr(container_mod, "launch", _capture_launch)

    rc = cmd_train(_make_args(toml_path, dry_run=False))
    assert rc == 0

    extra_mounts = captured.get("extra_mounts") or []
    mounted_container_paths = {ct for _, ct in extra_mounts}

    # The parent dirs of config, dataset, and output must all be identity-mounted.
    assert (
        str(toml_path.parent) in mounted_container_paths
    ), f"config parent {toml_path.parent} not in extra_mounts: {extra_mounts}"
    assert (
        str(dataset.parent) in mounted_container_paths
    ), f"dataset parent {dataset.parent} not in extra_mounts: {extra_mounts}"
    # Forward the ABSOLUTE config path — not rewritten to /workspace.
    assert "--config" in captured["sloth_args"]
    idx = captured["sloth_args"].index("--config")
    forwarded_config = captured["sloth_args"][idx + 1]
    assert forwarded_config == str(
        toml_path.resolve()
    ), f"expected absolute config path, got: {forwarded_config}"


# ---------------------------------------------------------------------------
# t4 acceptance — in-container mode: run_training called, no container.launch
# ---------------------------------------------------------------------------


def test_in_container_calls_run_training_not_launch(
    good_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --in-container, cmd_train delegates to run_training without launching docker."""
    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    fake_result: dict[str, Any] = {
        "model": "unsloth/Qwen3-4B",
        "method": "qlora",
        "dataset": "d",
        "output": "o",
        "hyperparameters": {"lora_r": 16},
        "scope": {"ok": True, "out_of_scope": False, "warning": None},
        "dry_run": False,
        "status": "trained",
        "adapter_dir": "o",
        "metadata_path": "o/training_metadata.json",
    }
    monkeypatch.setattr(train_mod, "run_training", lambda *_a, **_k: fake_result)

    rc = cmd_train(_make_args(good_config, dry_run=False, in_container=True))
    assert rc == 0
    mock_launch.assert_not_called()
    out = capsys.readouterr().out
    assert "trained" in out or "training_metadata.json" in out


def test_in_container_json_delegates_to_run_training(
    good_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--in-container --json emits the run_training result as JSON (no docker)."""
    mock_launch = Mock()
    monkeypatch.setattr(container_mod, "launch", mock_launch)

    fake_result: dict[str, Any] = {
        "model": "unsloth/Qwen3-4B",
        "method": "qlora",
        "dataset": "d",
        "output": "o",
        "hyperparameters": {"lora_r": 16},
        "scope": {"ok": True, "out_of_scope": False, "warning": None},
        "dry_run": False,
        "status": "trained",
        "adapter_dir": "o",
        "metadata_path": "o/training_metadata.json",
    }
    monkeypatch.setattr(train_mod, "run_training", lambda *_a, **_k: fake_result)

    rc = cmd_train(_make_args(good_config, dry_run=False, json_mode=True, in_container=True))
    assert rc == 0
    mock_launch.assert_not_called()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "trained"
    assert payload["metadata_path"].endswith("training_metadata.json")


# ---------------------------------------------------------------------------
# Acceptance 2 — real run delegates to run_training (GPU-free via fake)
# (These tests exercise the in-container code path with in_container=True)
# ---------------------------------------------------------------------------


def test_real_run_delegates_to_run_training(
    good_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In-container mode: cmd_train calls run_training and surfaces its result."""
    calls: list[dict[str, Any]] = []

    def _fake_run_training(config: RunConfig, *, dry_run: bool = False) -> dict[str, Any]:
        calls.append({"config": config, "dry_run": dry_run})
        return {
            "model": config.model,
            "method": config.method,
            "dataset": config.dataset,
            "output": config.output,
            "hyperparameters": {"lora_r": config.lora_r},
            "scope": {"ok": True, "out_of_scope": False, "warning": None},
            "dry_run": False,
            "status": "trained",
            "adapter_dir": config.output,
            "metadata_path": f"{config.output}/training_metadata.json",
        }

    monkeypatch.setattr(train_mod, "run_training", _fake_run_training)
    rc = cmd_train(_make_args(good_config, dry_run=False, in_container=True))
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert calls[0]["config"].model == "unsloth/Qwen3-4B"
    out = capsys.readouterr().out
    # The metadata location (written by the trainer) is surfaced to the user.
    assert "training_metadata.json" in out


def test_real_run_json_surfaces_metadata_path(
    good_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json on in-container real run emits the trainer's result dict, incl. metadata_path."""

    def _fake_run_training(config: RunConfig, *, dry_run: bool = False) -> dict[str, Any]:
        return {
            "model": config.model,
            "method": config.method,
            "dataset": config.dataset,
            "output": config.output,
            "hyperparameters": {"lora_r": config.lora_r},
            "scope": {"ok": True, "out_of_scope": False, "warning": None},
            "dry_run": False,
            "status": "trained",
            "adapter_dir": config.output,
            "metadata_path": f"{config.output}/training_metadata.json",
        }

    monkeypatch.setattr(train_mod, "run_training", _fake_run_training)
    rc = cmd_train(_make_args(good_config, dry_run=False, json_mode=True, in_container=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "trained"
    assert payload["metadata_path"].endswith("training_metadata.json")


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_train_subparser(tmp_path: Path) -> None:
    """register() adds a ``train`` subparser with --config / --dry-run / --json."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    cfg = str(tmp_path / "run.toml")
    args = parser.parse_args(["train", "--config", cfg])
    assert args.command == "train"
    assert args.config == cfg
    assert args.dry_run is False
    assert args.json is False
    assert args.in_container is False  # hidden recursion-guard flag
    assert callable(args.func)


def test_register_dry_run_and_json_flags(tmp_path: Path) -> None:
    """--dry-run and --json flip the parsed defaults."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["train", "--config", str(tmp_path / "r.toml"), "--dry-run", "--json"])
    assert args.dry_run is True
    assert args.json is True


def test_train_help_states_scope_warning() -> None:
    """`sloth train --help` must state the LoRA/QLoRA-only scope up front."""
    parser = argparse.ArgumentParser(prog="sloth")
    sub = parser.add_subparsers(dest="command")
    register(sub)
    train_parser = sub.choices["train"]
    help_text = train_parser.format_help().lower()
    assert "full fine-tuning" in help_text
    assert "out of scope" in help_text
    assert "lora" in help_text and "qlora" in help_text


def test_in_container_flag_suppressed_from_help() -> None:
    """--in-container must not appear in the public help text (it is hidden)."""
    parser = argparse.ArgumentParser(prog="sloth")
    sub = parser.add_subparsers(dest="command")
    register(sub)
    train_parser = sub.choices["train"]
    help_text = train_parser.format_help()
    assert "--in-container" not in help_text
