"""Tests for ``sloth eval`` command.

Covers:
* missing --adapter dir  → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* missing --suite file   → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* valid adapter + suite  → results emitted in text mode and --json mode (mocked backend)
* no-network assertion   → monkeypatching ``socket.socket`` to raise proves no
  network call escapes the mocked code path
* PeftModel load sequence → _default_backend reads adapter_config.json and
  calls PeftModel.from_pretrained(base_model, adapter), NOT
  AutoModelForCausalLM.from_pretrained(adapter)
* container routing      → host path calls container.launch with forwarded args +
  ``--in-container``; in-container path does NOT call launch
* recursion guard        → ``--in-container`` flag prevents docker recursion
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import sloth.cli._commands.eval as eval_mod
from sloth.cli._commands.eval import cmd_eval, register
from sloth.cli._errors import CliError
from sloth.cli._output import emit_error

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs: Any) -> argparse.Namespace:
    """Build a minimal Namespace, defaulting ``json`` and ``in_container`` to False."""
    defaults: dict[str, Any] = {"json": False, "in_container": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _mock_backend(adapter_path: str, record: dict[str, str]) -> str:  # noqa: ARG001
    """Always returns the expected output — perfect score for testing."""
    return record["expected_output"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_adapter(tmp_path: Path) -> Path:
    """A minimal adapter directory (just needs to exist as a directory)."""
    d = tmp_path / "adapter"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_adapter_with_config(tmp_path: Path) -> tuple[Path, str]:
    """An adapter directory with a valid adapter_config.json.

    Returns (adapter_dir, base_model_name).
    """
    d = tmp_path / "adapter"
    d.mkdir()
    base_model_name = "unsloth/Qwen3-4B"
    config = {"base_model_name_or_path": base_model_name, "peft_type": "LORA"}
    (d / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return d, base_model_name


@pytest.fixture()
def tmp_suite(tmp_path: Path) -> Path:
    """A two-record task-schema JSONL eval suite."""
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n'
        '{"task": "upper", "input": "hello", "expected_output": "HELLO"}\n',
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# Error path: missing adapter directory
# ---------------------------------------------------------------------------


def test_missing_adapter_raises_cli_error(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir raises CliError with code=1."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert "adapter" in err.message.lower()
    assert err.remediation


def test_missing_adapter_emits_error_and_hint(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir renders as ``error: …`` / ``hint: …`` lines."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_missing_adapter_json_error(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir renders as structured JSON when json_mode=True."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=True, stream=buf)
    payload = json.loads(buf.getvalue())
    assert payload["code"] == 1
    assert "message" in payload
    assert "remediation" in payload


# ---------------------------------------------------------------------------
# Error path: missing suite file
# ---------------------------------------------------------------------------


def test_missing_suite_raises_cli_error(tmp_adapter: Path, tmp_path: Path) -> None:
    """Missing --suite file raises CliError with code=1."""
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_path / "no_such.jsonl"))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert "suite" in err.message.lower()
    assert err.remediation


def test_missing_suite_emits_error_and_hint(tmp_adapter: Path, tmp_path: Path) -> None:
    """Missing --suite file renders as ``error: …`` / ``hint: …`` lines."""
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_path / "no_such.jsonl"))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


# ---------------------------------------------------------------------------
# Happy path: text output (mocked backend, in-container)
# ---------------------------------------------------------------------------


def test_eval_text_output_contains_summary(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked perfect-score backend → text output includes total / exact / score."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "total" in out
    assert "exact" in out
    assert "score" in out


def test_eval_text_shows_per_item_results(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text mode shows one line per eval record."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    cmd_eval(args)
    out = capsys.readouterr().out
    # Two records → two per-item lines (each starts with "  [ok]" or "  [fail]")
    item_lines = [ln for ln in out.splitlines() if "[ok]" in ln or "[fail]" in ln]
    assert len(item_lines) == 2


# ---------------------------------------------------------------------------
# Happy path: JSON output (mocked backend, in-container)
# ---------------------------------------------------------------------------


def test_eval_json_output_structure(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked backend with --json → well-formed JSON with summary fields."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=True)
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total"] == 2
    assert data["exact_match"] == 2
    assert data["exact_match_pct"] == 100.0
    assert len(data["results"]) == 2


def test_eval_json_result_items(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each result item has the expected fields and correct exact_match flag."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=True)
    cmd_eval(args)
    data = json.loads(capsys.readouterr().out)
    for item in data["results"]:
        assert "index" in item
        assert "task" in item
        assert "input" in item
        assert "expected_output" in item
        assert "prediction" in item
        assert item["exact_match"] is True


def test_eval_json_partial_score(
    tmp_adapter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A backend that always returns 'WRONG' yields exact_match=0, score=0.0."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"task": "t", "input": "x", "expected_output": "y"}\n',
        encoding="utf-8",
    )

    def _always_wrong(adapter_path: str, record: dict[str, str]) -> str:  # noqa: ARG001
        return "WRONG"

    monkeypatch.setattr(eval_mod, "_inference_backend", _always_wrong)
    args = _make_args(adapter=str(tmp_adapter), suite=str(suite), json=True, in_container=True)
    cmd_eval(args)
    data = json.loads(capsys.readouterr().out)
    assert data["exact_match"] == 0
    assert data["exact_match_pct"] == 0.0
    assert data["results"][0]["exact_match"] is False


# ---------------------------------------------------------------------------
# No-network assertion
# ---------------------------------------------------------------------------


def test_no_network_access_with_mocked_backend(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the inference backend is mocked, no socket is opened.

    Replaces ``socket.socket`` with a callable that raises AssertionError, then
    runs the full eval code path and asserts it completes without triggering the
    replacement — proving the code path never touches the network.
    """
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)

    class _NoSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("network access attempted during eval — local_files_only violated")

    monkeypatch.setattr(socket, "socket", _NoSocket)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc == 0  # completed without touching the network


# ---------------------------------------------------------------------------
# Acceptance 1 — PeftModel load sequence (correct adapter loading)
# ---------------------------------------------------------------------------


def test_default_backend_reads_adapter_config(
    tmp_adapter_with_config: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_default_backend reads base_model_name_or_path from adapter_config.json.

    Injects fake torch/transformers/peft into sys.modules and asserts that:
    - AutoModelForCausalLM.from_pretrained is called with the BASE model name
      (not the adapter dir path)
    - PeftModel.from_pretrained is called with (base_model_obj, adapter_path)
    """
    adapter_dir, base_model_name = tmp_adapter_with_config

    calls: dict[str, Any] = {}

    # --- fake base model object (sentinel) ---
    fake_base_model = object()

    # --- fake peft model (supports .eval() and .generate()) ---
    fake_peft_model = MagicMock()
    fake_peft_model.generate.return_value = [[1, 2, 3]]

    # --- fake AutoModelForCausalLM ---
    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> object:
            calls["causal_lm_name"] = name
            return fake_base_model

    # --- fake PeftModel ---
    class FakePeftModel:
        @staticmethod
        def from_pretrained(base: object, adapter_path: str, **kwargs: Any) -> Any:
            calls["peft_base"] = base
            calls["peft_adapter"] = adapter_path
            return fake_peft_model

    # --- fake AutoTokenizer ---
    class _FakeTokenizer:
        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens: Any, **kwargs: Any) -> str:
            return "decoded_output"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            return _FakeTokenizer()

    # --- fake torch (context manager support via MagicMock) ---
    fake_torch = MagicMock()

    # Inject fakes into sys.modules so lazy `import` inside _default_backend picks them up.
    fake_transformers = MagicMock()
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    fake_peft_module = MagicMock()
    fake_peft_module.PeftModel = FakePeftModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft_module)

    record = {"task": "reverse", "input": "abc", "expected_output": "cba"}
    result = eval_mod._default_backend(str(adapter_dir), record)

    # AutoModelForCausalLM must be called with the BASE name, not the adapter dir.
    assert calls["causal_lm_name"] == base_model_name
    assert calls["causal_lm_name"] != str(
        adapter_dir
    ), "AutoModelForCausalLM.from_pretrained must NOT be called with the adapter dir"

    # PeftModel must be called with (base_model_obj, adapter_path).
    assert (
        calls["peft_base"] is fake_base_model
    ), "PeftModel.from_pretrained must receive the base model object as its first arg"
    assert calls["peft_adapter"] == str(
        adapter_dir
    ), "PeftModel.from_pretrained must receive the adapter dir path as its second arg"

    # Result is the tokenizer's decoded output.
    assert isinstance(result, str)


def test_default_backend_missing_adapter_config(
    tmp_adapter: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_default_backend raises CliError(code=1) when adapter_config.json is absent."""
    fake_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", MagicMock())
    monkeypatch.setitem(sys.modules, "peft", MagicMock())

    record = {"task": "t", "input": "x", "expected_output": "y"}
    with pytest.raises(CliError) as exc_info:
        eval_mod._default_backend(str(tmp_adapter), record)
    assert exc_info.value.code == 1
    assert "adapter_config.json" in exc_info.value.message


# ---------------------------------------------------------------------------
# Acceptance 2 — container routing: host path calls container.launch
# ---------------------------------------------------------------------------


def test_host_routes_to_container_launch(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_eval on the host (no --in-container) calls container.launch.

    Asserts that:
    - container.launch IS called
    - the sloth_args list starts with 'eval' and ends with '--in-container'
    - '--adapter' and '--suite' are forwarded
    - the function returns container.launch's exit code, not the eval summary
    """
    launch_calls: list[dict[str, Any]] = []

    def _fake_launch(
        sloth_args: list[str],
        *,
        workdir: str | None = None,
        checkout: str | None = None,
        **kwargs: Any,
    ) -> int:
        launch_calls.append(
            {"sloth_args": list(sloth_args), "workdir": workdir, "checkout": checkout}
        )
        return 0

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    rc = cmd_eval(args)

    assert rc == 0
    assert len(launch_calls) == 1, "container.launch must be called exactly once"

    forwarded = launch_calls[0]["sloth_args"]
    assert forwarded[0] == "eval", "first sloth_arg must be 'eval'"
    assert "--in-container" in forwarded, "recursion guard must be forwarded"
    assert "--adapter" in forwarded
    assert "--suite" in forwarded
    # workdir must be set (points to the adapter's parent)
    assert launch_calls[0]["workdir"] is not None
    # checkout must be set (the repo root)
    assert launch_calls[0]["checkout"] is not None


def test_host_routes_json_flag_when_set(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json is forwarded to the container when set on the host."""
    forwarded_args: list[list[str]] = []

    def _fake_launch(sloth_args: list[str], **kwargs: Any) -> int:
        forwarded_args.append(list(sloth_args))
        return 0

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=False)
    cmd_eval(args)

    assert forwarded_args, "launch must be called"
    assert "--json" in forwarded_args[0], "--json must be forwarded to the container"
    assert "--in-container" in forwarded_args[0]


def test_host_returns_container_exit_code(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_eval returns the exit code from container.launch, not a hardcoded 0."""
    monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: 42)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    rc = cmd_eval(args)
    assert rc == 42


# ---------------------------------------------------------------------------
# Acceptance 2 — recursion guard: --in-container path does NOT call launch
# ---------------------------------------------------------------------------


def test_in_container_does_not_call_launch(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With --in-container set, cmd_eval runs the real eval, never calls launch.

    Stubs container.launch to raise AssertionError; the in-container path must
    complete without triggering it.
    """

    def _must_not_launch(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("container.launch called inside container — recursion guard broken")

    monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc == 0


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_eval_subparser() -> None:
    """``register`` adds an ``eval`` subparser with --adapter, --suite, --json."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--adapter", "/some/dir", "--suite", "/some/file.jsonl"])
    assert args.adapter == "/some/dir"
    assert args.suite == "/some/file.jsonl"
    assert args.json is False
    assert callable(args.func)


def test_register_json_flag() -> None:
    """--json flag is parsed correctly by the eval subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl", "--json"])
    assert args.json is True


def test_register_in_container_flag() -> None:
    """``--in-container`` is a hidden flag that defaults to False."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    # Default: in_container is False.
    args_default = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl"])
    assert args_default.in_container is False

    # When passed explicitly it is True.
    args_set = parser.parse_args(
        ["eval", "--adapter", "/a", "--suite", "/b.jsonl", "--in-container"]
    )
    assert args_set.in_container is True


def test_register_in_container_not_in_help() -> None:
    """--in-container must not appear in the public help text (it is SUPPRESS'd)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    help_text = sub.choices["eval"].format_help()
    assert "--in-container" not in help_text
