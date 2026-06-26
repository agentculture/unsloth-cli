"""Tests for ``sloth train`` — the integrator verb.

Flow under test: load config -> validate dataset -> scope-guard (emit warning)
-> dry-run OR real train (delegated to ``run_training``).

Covers the two acceptance criteria for t10:

1. ``sloth train --config run.toml --dry-run`` validates the dataset, prints the
   resolved plan in BOTH text and ``--json`` modes, and exits ``0`` WITHOUT
   importing torch. An invalid dataset exits ``1`` with the ``error:`` / ``hint:``
   contract.
2. ``sloth train`` pointed at a large-dense full-fine-tune target emits the
   explicit out-of-scope warning (from ``check_scope``) on stderr and refuses
   with ``CliError(code=1)``; a real run delegates to ``run_training`` (which is
   the module that writes metadata next to the adapter).

The real-run path is exercised GPU-free by monkeypatching ``train``'s reference
to ``run_training`` with a fake, so delegation is asserted without a backend.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import sloth.cli._commands.train as train_mod
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
    config: Path, *, dry_run: bool = False, json_mode: bool = False
) -> argparse.Namespace:
    """Build the Namespace argparse would produce for ``sloth train``."""
    return argparse.Namespace(config=str(config), dry_run=dry_run, json=json_mode)


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
# Acceptance 1 — dry-run plan: text + json, exit 0
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
    """Validation happens before delegation: run_training is never called."""
    bad = _write_dataset(tmp_path, body='{"messages": []}\n')  # empty messages list
    toml = _write_toml(tmp_path, dataset=bad)

    def _must_not_run(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("run_training called despite invalid dataset")

    monkeypatch.setattr(train_mod, "run_training", _must_not_run)
    with pytest.raises(CliError) as exc_info:
        cmd_train(_make_args(toml, dry_run=False))
    assert exc_info.value.code == 1


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
    # If we reach delegation, the test is wrong — guard it.
    monkeypatch.setattr(
        train_mod,
        "run_training",
        lambda *_a, **_k: pytest.fail("run_training reached for out-of-scope request"),
    )

    args = argparse.Namespace(config="ignored.toml", dry_run=False, json=False)
    with pytest.raises(CliError) as exc_info:
        cmd_train(args)

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    # The warning must be EXPLICIT on stderr, not silent.
    assert "out of scope" in err.lower() or "full fine-tuning" in err.lower()


# ---------------------------------------------------------------------------
# Acceptance 2 — real run delegates to run_training (GPU-free via fake)
# ---------------------------------------------------------------------------


def test_real_run_delegates_to_run_training(
    good_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-dry-run delegates to run_training and surfaces its result."""
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
    rc = cmd_train(_make_args(good_config, dry_run=False))
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
    """--json on a real run emits the trainer's result dict, incl. metadata_path."""

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
    rc = cmd_train(_make_args(good_config, dry_run=False, json_mode=True))
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
