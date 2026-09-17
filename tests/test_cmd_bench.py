"""Tests for ``sloth bench`` — the MMLU/lm_eval benchmark verb.

Covers:

* flag validation on the **host**, before any container launch (both/neither
  target, a missing directory, ``--limit 0``, a negative ``--num-fewshot``) —
  each exit 1 through the ``error:``/``hint:`` contract;
* container routing: a fake ``container.launch`` asserts the forwarded
  in-container argv (``bench … --json --in-container``), the identity mount, and
  the ``HF_HUB_OFFLINE=1`` env that ``--offline`` adds;
* the in-container path: ``run_bench`` is faked, and the run must write
  ``eval/mmlu.json`` in the shared result shape and emit the payload;
* ``bench.py`` imports no ML module (torch / peft / transformers / lm_eval).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import sloth.cli._commands.bench as bench_mod
from sloth.cli._commands.bench import cmd_bench, register
from sloth.cli._errors import CliError

# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _make_args(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace with this verb's defaults, overridden by *kwargs*."""
    defaults: dict[str, Any] = {
        "adapter": None,
        "model": None,
        "benchmark": "mmlu",
        "tasks": None,
        "num_fewshot": 5,
        "limit": None,
        "offline": False,
        "json": False,
        "in_container": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


#: What the faked ``run_bench`` / ``launch`` hand back — the documented shape.
_FAKE_PAYLOAD: dict[str, Any] = {
    "acc": 0.5,
    "acc_norm": None,
    "exact_match_pct": 50.0,
    "total": 40,
    "per_subject": {"astronomy": {"acc": 0.6, "acc_norm": None, "total": 20}},
    "harness": {
        "name": "lm_eval",
        "version": "0.4.13",
        "tasks": "mmlu",
        "limit": 20,
        "num_fewshot": 5,
    },
    "base_load_in_4bit": True,
    "model": "unsloth/Qwen3-4B",
}


@pytest.fixture
def tmp_adapter(tmp_path: Path) -> Path:
    """A minimal adapter directory carrying training metadata."""
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "training_metadata.json").write_text(
        json.dumps({"model": "unsloth/Qwen3-4B", "hyperparameters": {"load_in_4bit": True}}),
        encoding="utf-8",
    )
    return directory


# ---------------------------------------------------------------------------
# Host-side validation (exit 1, before any container launch)
# ---------------------------------------------------------------------------


def test_both_targets_is_a_user_error(tmp_adapter: Path) -> None:
    """--adapter and --model together exit 1 with a remediation."""
    args = _make_args(adapter=str(tmp_adapter), model=str(tmp_adapter))
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 1
    assert excinfo.value.remediation


def test_neither_target_is_a_user_error() -> None:
    """Neither --adapter nor --model exits 1 with a remediation."""
    args = _make_args()
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 1


def test_missing_adapter_directory_is_a_user_error(tmp_path: Path) -> None:
    """A path that is not a directory exits 1 naming it."""
    args = _make_args(adapter=str(tmp_path / "nope"))
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 1
    assert "nope" in excinfo.value.message


def test_limit_below_one_is_rejected_before_any_launch(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--limit 0`` costs no docker cycle."""
    launched: list[Any] = []
    monkeypatch.setattr(
        bench_mod.container, "launch", lambda *a, **kw: launched.append(a) or dict(_FAKE_PAYLOAD)
    )
    args = _make_args(adapter=str(tmp_adapter), limit=0)
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 1
    assert launched == []


def test_negative_num_fewshot_is_rejected(tmp_adapter: Path) -> None:
    """A negative few-shot count is a user error."""
    args = _make_args(adapter=str(tmp_adapter), num_fewshot=-1)
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 1


def test_error_contract_is_two_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """Failures render as ``error:`` then ``hint:`` on stderr."""
    from sloth.cli._output import emit_error

    try:
        cmd_bench(_make_args())
    except CliError as err:
        emit_error(err, json_mode=False)
    captured = capsys.readouterr()
    assert captured.err.splitlines()[0].startswith("error:")
    assert captured.err.splitlines()[1].startswith("hint:")
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Container routing (the fake-launcher assertions on the in-container command)
# ---------------------------------------------------------------------------


def test_host_routes_to_container_launch(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host forwards the full bench argv plus the recursion guard."""
    calls: list[dict[str, Any]] = []

    def _fake_launch(sloth_args: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"sloth_args": list(sloth_args), **kwargs})
        return dict(_FAKE_PAYLOAD)

    monkeypatch.setattr(bench_mod.container, "launch", _fake_launch)

    cmd_bench(_make_args(adapter=str(tmp_adapter), limit=20))

    assert len(calls) == 1
    forwarded = calls[0]["sloth_args"]
    assert forwarded[0] == "bench"
    assert forwarded[1:3] == ["--adapter", str(tmp_adapter.resolve())]
    assert forwarded[forwarded.index("--benchmark") + 1] == "mmlu"
    assert forwarded[forwarded.index("--tasks") + 1] == "mmlu"
    assert forwarded[forwarded.index("--num-fewshot") + 1] == "5"
    assert forwarded[forwarded.index("--limit") + 1] == "20"
    assert forwarded[-2:] == ["--json", "--in-container"]
    # Identity mount so the host-absolute adapter path resolves in-container.
    parent = str(tmp_adapter.resolve().parent)
    assert (parent, parent) in calls[0]["extra_mounts"]
    assert calls[0]["checkout"]


def test_offline_forwards_the_hub_offline_env(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--offline`` sets HF_HUB_OFFLINE=1 in the container and forwards the flag."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bench_mod.container,
        "launch",
        lambda sloth_args, **kw: calls.append({"sloth_args": list(sloth_args), **kw})
        or dict(_FAKE_PAYLOAD),
    )

    cmd_bench(_make_args(adapter=str(tmp_adapter), offline=True))

    assert "--offline" in calls[0]["sloth_args"]
    assert calls[0]["env"] == [("HF_HUB_OFFLINE", "1")]


def test_host_without_offline_sets_no_env(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--offline`` no env override is added (a warm cache is enough)."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bench_mod.container,
        "launch",
        lambda sloth_args, **kw: calls.append(kw) or dict(_FAKE_PAYLOAD),
    )
    cmd_bench(_make_args(adapter=str(tmp_adapter)))
    assert "env" not in calls[0]


def test_host_emits_launch_result_as_json(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host renders the payload the container returned, honouring its own --json."""
    monkeypatch.setattr(bench_mod.container, "launch", lambda *a, **kw: dict(_FAKE_PAYLOAD))
    cmd_bench(_make_args(adapter=str(tmp_adapter), json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["exact_match_pct"] == 50.0
    assert payload["harness"]["name"] == "lm_eval"


def test_host_emits_launch_result_as_text(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Text mode prints the benchmark, the target and the score on stdout."""
    monkeypatch.setattr(bench_mod.container, "launch", lambda *a, **kw: dict(_FAKE_PAYLOAD))
    cmd_bench(_make_args(adapter=str(tmp_adapter)))
    out = capsys.readouterr().out
    assert "benchmark:  mmlu" in out
    assert "50.0%" in out
    assert "astronomy" in out


def test_launch_failure_propagates(tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CliError from the container is not swallowed."""

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise CliError(code=2, message="docker missing", remediation="install docker")

    monkeypatch.setattr(bench_mod.container, "launch", _boom)
    args = _make_args(adapter=str(tmp_adapter))
    with pytest.raises(CliError) as excinfo:
        cmd_bench(args)
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# In-container path: run_bench → eval/<benchmark>.json
# ---------------------------------------------------------------------------


def test_in_container_writes_eval_mmlu_json(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The in-container run writes eval/mmlu.json in the shared result shape."""
    seen: list[dict[str, Any]] = []

    def _fake_run_bench(target: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append({"target": str(target), **kwargs})
        return dict(_FAKE_PAYLOAD)

    monkeypatch.setattr(bench_mod, "run_bench", _fake_run_bench)
    launched: list[Any] = []
    monkeypatch.setattr(bench_mod.container, "launch", lambda *a, **kw: launched.append(a))

    cmd_bench(_make_args(adapter=str(tmp_adapter), in_container=True, limit=20, json=True))

    assert launched == [], "the in-container path must never launch docker"
    assert seen[0]["kind"] == "adapter"
    assert seen[0]["benchmark"] == "mmlu"
    assert seen[0]["tasks"] == "mmlu"
    assert seen[0]["num_fewshot"] == 5
    assert seen[0]["limit"] == 20

    written = json.loads((tmp_adapter / "eval" / "mmlu.json").read_text(encoding="utf-8"))
    assert written["suite"] == "mmlu"
    assert written["schema_version"] == 2
    assert written["target"] == "adapter"
    assert written["base_load_in_4bit"] is True
    assert written["acc"] == 0.5
    assert written["exact_match_pct"] == 50.0
    assert written["per_subject"]["astronomy"]["acc"] == 0.6
    assert written["harness"]["tasks"] == "mmlu"
    assert json.loads(capsys.readouterr().out)["acc"] == 0.5


def test_in_container_uses_the_benchmark_name_for_the_file(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different --benchmark writes a sibling file, leaving mmlu.json alone."""
    monkeypatch.setattr(bench_mod, "run_bench", lambda *a, **kw: dict(_FAKE_PAYLOAD))
    cmd_bench(_make_args(adapter=str(tmp_adapter), in_container=True, benchmark="hellaswag"))
    assert (tmp_adapter / "eval" / "hellaswag.json").is_file()
    assert not (tmp_adapter / "eval" / "mmlu.json").exists()


def test_summarize_picks_up_a_bench_result(
    tmp_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The written file folds into `sloth summarize`'s eval block unchanged."""
    from sloth.tune.summary import read_eval

    monkeypatch.setattr(bench_mod, "run_bench", lambda *a, **kw: dict(_FAKE_PAYLOAD))
    cmd_bench(_make_args(adapter=str(tmp_adapter), in_container=True))

    suites = read_eval(tmp_adapter)
    assert "mmlu" in suites
    assert suites["mmlu"]["acc"] == 0.5


# ---------------------------------------------------------------------------
# Import hygiene + registration
# ---------------------------------------------------------------------------


def test_bench_module_imports_no_ml_stack() -> None:
    """Importing the verb must not pull torch / peft / transformers / lm_eval in."""
    heavy = [m for m in ("torch", "peft", "transformers", "lm_eval") if m in sys.modules]
    assert not heavy, f"bench.py pulled in {heavy}"


def test_register_adds_the_bench_subparser() -> None:
    """``register`` wires the verb with ``--json`` and the hidden recursion guard."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["bench", "--adapter", "a", "--json", "--limit", "3"])
    assert args.adapter == "a"
    assert args.json is True
    assert args.limit == 3
    assert args.in_container is False
    assert args.func is cmd_bench
