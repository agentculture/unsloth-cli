"""Tests for ``sloth compare <a> <b>``.

Covers:
* config/hyperparameter deltas between two runs' training_metadata.json
* identical configs -> empty deltas
* both full summaries are included in the report
* --json shape; unresolvable target -> CliError(code=1)
* register() wiring + main()-level end-to-end (regression pattern from
  test_cmd_validate.py's test_main_validate_end_to_end)

Also covers (t6): an ``eval`` delta block (exact_match_pct + f1 for each
side) shown when BOTH runs have an ``eval.json``, absent otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import sloth.cli._commands.compare as compare_mod
from sloth.cli._commands.compare import cmd_compare, register
from sloth.cli._errors import CliError
from sloth.tune.metadata import write_metadata

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl", body: str = _VALID_CHAT) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _args(a: str, b: str, *, runs_root: str | None = None, json_mode: bool = False):
    return argparse.Namespace(a=a, b=b, runs_root=runs_root, json=json_mode)


# ---------------------------------------------------------------------------
# Config deltas
# ---------------------------------------------------------------------------


def test_compare_reports_hyperparameter_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 16, "learning_rate": 2e-4},
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 32, "learning_rate": 2e-4},
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["lora_r"] == {"a": 16, "b": 32}
    assert "learning_rate" not in payload["deltas"]


def test_compare_reports_model_delta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={},
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b,
        model="unsloth/Qwen3-9B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={},
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["model"] == {"a": "unsloth/Qwen3-4B", "b": "unsloth/Qwen3-9B"}


def test_compare_reports_dataset_delta_on_sha_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_a = _write_dataset(tmp_path, name="a.jsonl", body=_VALID_CHAT)
    dataset_b = _write_dataset(
        tmp_path,
        name="b.jsonl",
        body='{"messages": [{"role": "user", "content": "different"}]}\n',
    )
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset_a, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset_b, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dataset" in payload["deltas"]


def test_compare_identical_configs_empty_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"] == {}


def test_compare_text_mode_no_deltas_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "configs match" in out


# ---------------------------------------------------------------------------
# Both full summaries are included
# ---------------------------------------------------------------------------


def test_compare_includes_both_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m-a", method="lora", dataset_path=dataset, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m-b", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["a"]["metadata"]["model"] == "m-a"
    assert payload["b"]["metadata"]["model"] == "m-b"


# ---------------------------------------------------------------------------
# Export presence/format deltas (t10)
# ---------------------------------------------------------------------------


def _write_export_index(adapter: Path, fmt: str) -> None:
    adapter_dir = adapter
    (adapter_dir / "exports.json").write_text(
        json.dumps(
            [
                {
                    "format": fmt,
                    "quant": [],
                    "base": "m",
                    "adapter": str(adapter_dir),
                    "files": {"f": 1},
                    "calibration": None,
                    "versions": {},
                    "timestamp": "2026-07-06T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_compare_reports_export_delta_when_one_side_has_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_export_index(dir_a, "gguf")

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "exports" in payload["deltas"]
    assert payload["deltas"]["exports"]["a"]["count"] == 1
    assert payload["deltas"]["exports"]["a"]["formats"] == ["gguf"]
    assert payload["deltas"]["exports"]["b"]["count"] == 0


def test_compare_no_export_delta_when_both_sides_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_export_index(dir_a, "gguf")

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_export_index(dir_b, "gguf")

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "exports" not in payload["deltas"]


# ---------------------------------------------------------------------------
# Eval delta block (t6)
# ---------------------------------------------------------------------------


def _write_eval_json(directory: Path, *, exact_match_pct: float, f1: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "total": 4,
        "exact_match": 3,
        "exact_match_pct": exact_match_pct,
        "f1": f1,
        "results": [],
        "files": [
            {
                "path": "a.jsonl",
                "total": 4,
                "exact_match": 3,
                "exact_match_pct": exact_match_pct,
                "f1": f1,
            }
        ],
        "suite_paths": ["a.jsonl"],
        "target": "adapter",
        "written_at": "2026-07-06T02:00:00+00:00",
    }
    (directory / "eval.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compare_shows_eval_delta_when_both_sides_have_eval_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_a, exact_match_pct=75.0, f1=0.8)

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_b, exact_match_pct=90.0, f1=0.95)

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["eval"]["a"] == {"exact_match_pct": 75.0, "f1": 0.8}
    assert payload["deltas"]["eval"]["b"] == {"exact_match_pct": 90.0, "f1": 0.95}


def test_compare_no_eval_delta_when_metrics_are_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both sides have an eval.json with IDENTICAL exact_match_pct/f1: no
    eval delta should be reported (qodo finding: previously any two eval
    blocks were reported as differing without comparing values)."""
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_a, exact_match_pct=75.0, f1=0.8)

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_b, exact_match_pct=75.0, f1=0.8)

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "eval" not in payload["deltas"]


def test_compare_no_eval_delta_when_one_side_missing_eval_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_a, exact_match_pct=75.0, f1=0.8)

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "eval" not in payload["deltas"]


def test_compare_text_mode_shows_eval_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_a, exact_match_pct=75.0, f1=0.8)

    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    _write_eval_json(dir_b, exact_match_pct=90.0, f1=0.95)

    rc = cmd_compare(_args(str(dir_a), str(dir_b)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "eval" in out
    assert "75.0" in out
    assert "90.0" in out


# ---------------------------------------------------------------------------
# Unresolvable target
# ---------------------------------------------------------------------------


def test_compare_unresolvable_a_raises_cli_error_1(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    args = _args("totally-bogus", str(dir_b), runs_root=str(tmp_path))
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_compare_subparser(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["compare", "a-target", "b-target", "--json"])
    assert args.command == "compare"
    assert args.a == "a-target"
    assert args.b == "b-target"
    assert args.json is True
    assert callable(args.func)


def test_register_second_target_is_optional_for_base_mode() -> None:
    """``<b>`` became optional so ``--base`` can take a single positional (t9).

    A bare ``compare <a>`` still parses, and the handler rejects it with
    ``CliError(code=1)`` — the same exit code and ``hint:`` argparse produced
    before (see test_compare_without_base_still_requires_two_targets).
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["compare", "only-one"])
    assert args.a == "only-one"
    assert args.b is None
    with pytest.raises(SystemExit):
        parser.parse_args(["compare"])


# ---------------------------------------------------------------------------
# End-to-end through main()
# ---------------------------------------------------------------------------


def test_main_compare_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth compare`` resolves through sloth.cli.main (regression: the verb
    must be both imported AND registered in _build_parser)."""
    from sloth.cli import main

    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 8}
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )

    rc = main(["compare", str(dir_a), str(dir_b), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["lora_r"] == {"a": 8, "b": 16}


# ---------------------------------------------------------------------------
# --base mode: adapter vs. its base model (t9)
# ---------------------------------------------------------------------------


def _base_args(
    adapter: str,
    base: str,
    *,
    config: str | None = None,
    regression_suite: list[str] | None = None,
    json_mode: bool = False,
    b: str | None = None,
):
    return argparse.Namespace(
        a=adapter,
        b=b,
        base=base,
        config=config,
        regression_suite=regression_suite,
        runs_root=None,
        json=json_mode,
    )


def _suite_payload(name: str, suite_file: Path, **metrics: object) -> dict:
    """One recorded eval result file, shaped like metrics.write_eval_json's
    suite-keyed record (the per-file ``files`` entry carries the provenance
    ``compare --base`` re-runs the base model over)."""
    payload: dict = {
        "schema_version": 2,
        "suite": name,
        "batch_size": 4,
        "target": "adapter",
        "written_at": "2026-09-17T00:00:00+00:00",
        "base_load_in_4bit": True,
        "total": 120,
        "exact_match": 96,
        "exact_match_pct": 80.0,
        "f1": 0.8,
        "results": [],
        "files": [{"path": str(suite_file), "total": 120, "exact_match_pct": 80.0, "f1": 0.8}],
    }
    payload.update(metrics)
    return payload


def _write_suite_result(directory: Path, name: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _flag_value(sloth_args: list[str], flag: str) -> str | None:
    return sloth_args[sloth_args.index(flag) + 1] if flag in sloth_args else None


class _FakeLaunch:
    """Stands in for ``sloth.tune.container.launch``.

    Records every invocation's argv and writes the per-suite result files the
    real in-container ``sloth eval`` would have written: into ``<adapter>/eval``
    for the ``--adapter`` run, into ``--results-dir`` for the base run. No
    docker, no GPU. ``fail_on`` (1-based call index) raises instead — the
    simulated OOM.
    """

    def __init__(
        self,
        adapter_results: dict[str, dict],
        base_results: dict[str, dict],
        *,
        fail_on: int | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.adapter_results = adapter_results
        self.base_results = base_results
        self.fail_on = fail_on

    def __call__(self, sloth_args: list[str], **kwargs: object) -> dict:
        self.calls.append({"sloth_args": list(sloth_args), "kwargs": kwargs})
        if self.fail_on == len(self.calls):
            raise CliError(
                code=2,
                message="CUDA out of memory while evaluating",
                remediation=(
                    "Lower --batch-size, or set "
                    "PYTORCH_ALLOC_CONF=expandable_segments:True and retry."
                ),
            )
        results_dir = _flag_value(sloth_args, "--results-dir")
        if results_dir is None:
            out = Path(_flag_value(sloth_args, "--adapter")) / "eval"
            results = self.adapter_results
        else:
            out = Path(results_dir)
            results = self.base_results
        for name, payload in results.items():
            _write_suite_result(out, name, payload)
        return {"suites": dict(results)}


@pytest.fixture
def base_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """An adapter dir with metadata plus one recorded eval suite, and its suite file."""
    dataset = _write_dataset(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_metadata(
        adapter, model="unsloth/Qwen3-4B", method="qlora", dataset_path=dataset, hyperparameters={}
    )
    suite_file = tmp_path / "regression.jsonl"
    suite_file.write_text('{"task": "t", "input": "i", "expected_output": "o"}\n', encoding="utf-8")
    _write_suite_result(adapter / "eval", "regression", _suite_payload("regression", suite_file))
    return adapter, suite_file


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeLaunch) -> _FakeLaunch:
    monkeypatch.setattr(compare_mod.container, "launch", fake)
    return fake


def test_compare_base_makes_two_sequential_container_invocations(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    adapter_results = {"regression": _suite_payload("regression", suite_file)}
    base_results = {"regression": _suite_payload("regression", suite_file, exact_match_pct=79.0)}
    fake = _install(monkeypatch, _FakeLaunch(adapter_results, base_results))

    rc = cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True))
    assert rc == 0
    assert len(fake.calls) == 2, "one container invocation per model, never both at once"

    first, second = (c["sloth_args"] for c in fake.calls)
    assert first[:2] == ["eval", "--adapter"]
    assert _flag_value(first, "--adapter") == str(adapter.resolve())
    assert "--results-dir" not in first
    assert second[:2] == ["eval", "--model"]
    assert _flag_value(second, "--model") == "unsloth/Qwen3-4B"
    assert _flag_value(second, "--results-dir") == str((adapter / "eval-base").resolve())
    for argv in (first, second):
        assert _flag_value(argv, "--suite") == str(suite_file.resolve())
        assert argv[-1] == "--in-container"
        assert "--json" in argv

    capsys.readouterr()


def test_compare_base_emits_per_suite_per_metric_deltas(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    adapter_results = {"regression": _suite_payload("regression", suite_file, exact_match_pct=81.0)}
    base_results = {
        "regression": _suite_payload("regression", suite_file, exact_match_pct=80.0, f1=0.75)
    }
    _install(monkeypatch, _FakeLaunch(adapter_results, base_results))

    rc = cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    suite = payload["deltas"]["suites"]["regression"]
    assert suite["exact_match_pct"] == {"a": 80.0, "b": 81.0, "delta": pytest.approx(1.0)}
    assert suite["f1"]["a"] == 0.75
    assert suite["f1"]["b"] == 0.8
    assert payload["verdict"] == {"passed": True, "failures": []}
    assert payload["a"]["model"] == "unsloth/Qwen3-4B"
    assert payload["b"]["output_dir"] == str(adapter)


def test_compare_base_writes_base_results_under_eval_base(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file)},
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=79.0)},
        ),
    )
    cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True))
    capsys.readouterr()
    assert (adapter / "eval-base" / "regression.json").is_file()
    assert (adapter / "eval" / "regression.json").is_file()


def test_compare_base_regression_drop_exits_1(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=70.0)},
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=80.0)},
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    assert "regression" in exc_info.value.remediation

    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["passed"] is False
    assert [f["check"] for f in payload["verdict"]["failures"]] == ["regression"]
    assert payload["thresholds"]["regression_drop_pp"] == 2.0
    assert payload["thresholds"]["source"] == "default"


def test_compare_base_regression_flag_tags_another_suite(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A suite not named 'regression' is only gated when --regression-suite names it."""
    adapter, suite_file = base_fixture
    holdout = suite_file.parent / "holdout.jsonl"
    holdout.write_text(suite_file.read_text(encoding="utf-8"), encoding="utf-8")
    (adapter / "eval" / "regression.json").unlink()
    _write_suite_result(adapter / "eval", "holdout", _suite_payload("holdout", holdout))

    fake = _FakeLaunch(
        {"holdout": _suite_payload("holdout", holdout, exact_match_pct=70.0)},
        {"holdout": _suite_payload("holdout", holdout, exact_match_pct=80.0)},
    )
    _install(monkeypatch, fake)
    assert cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)) == 0
    capsys.readouterr()

    fake.calls.clear()
    args = _base_args(
        str(adapter), "unsloth/Qwen3-4B", regression_suite=["holdout"], json_mode=True
    )
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["thresholds"]["regression_suites"] == ["holdout"]


def test_compare_base_compliance_below_minimum_exits_1(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, compliance_pct=90.0)},
            {"regression": _suite_payload("regression", suite_file, compliance_pct=99.0)},
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert [f["check"] for f in payload["verdict"]["failures"]] == ["compliance"]
    assert payload["thresholds"]["compliance_min_pct"] == 95.0


def test_compare_base_latency_ratio_over_max_exits_1(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, median_latency_ms=150.0)},
            {"regression": _suite_payload("regression", suite_file, median_latency_ms=100.0)},
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert [f["check"] for f in payload["verdict"]["failures"]] == ["latency"]
    assert payload["thresholds"]["latency_max_ratio"] == 1.10


def test_compare_base_suite_under_min_rows_exits_1(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, total=12)},
            {"regression": _suite_payload("regression", suite_file, total=12)},
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert {f["check"] for f in payload["verdict"]["failures"]} == {"min_suite_rows"}
    assert payload["thresholds"]["min_suite_rows"] == 100


def test_compare_base_precision_mismatch_exits_1(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, base_load_in_4bit=True)},
            {"regression": _suite_payload("regression", suite_file, base_load_in_4bit=False)},
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    assert "base_load_in_4bit" in exc_info.value.remediation
    payload = json.loads(capsys.readouterr().out)
    assert [f["check"] for f in payload["verdict"]["failures"]] == ["precision"]


def test_compare_base_config_thresholds_override_the_defaults(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A 10 pp drop passes under a config that allows 20 pp — and the applied
    thresholds are echoed in --json."""
    adapter, suite_file = base_fixture
    config = tmp_path / "run.toml"
    config.write_text(
        "[run]\n"
        'model = "unsloth/Qwen3-4B"\n'
        'dataset = "train.jsonl"\n'
        'output = "adapters/x"\n'
        "\n[eval.thresholds]\n"
        "regression_drop_pp = 20.0\n"
        "min_suite_rows = 1\n",
        encoding="utf-8",
    )
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=70.0)},
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=80.0)},
        ),
    )
    rc = cmd_compare(
        _base_args(str(adapter), "unsloth/Qwen3-4B", config=str(config), json_mode=True)
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["thresholds"]["regression_drop_pp"] == 20.0
    assert payload["thresholds"]["min_suite_rows"] == 1
    assert payload["thresholds"]["source"] == str(config)


def test_compare_base_oom_on_second_run_exits_2_and_keeps_first_results(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    fake = _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=81.0)},
            {"regression": _suite_payload("regression", suite_file)},
            fail_on=2,
        ),
    )
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 2
    assert "memory" in exc_info.value.message.lower()
    assert "batch-size" in exc_info.value.remediation

    assert len(fake.calls) == 2
    first = adapter / "eval" / "regression.json"
    assert first.is_file(), "the first run's results must survive the second run's failure"
    assert json.loads(first.read_text(encoding="utf-8"))["exact_match_pct"] == 81.0
    assert not (adapter / "eval-base" / "regression.json").exists()
    capsys.readouterr()


def test_compare_base_text_mode_renders_suites_and_verdict(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=81.0)},
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=80.0)},
        ),
    )
    assert cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B")) == 0
    out = capsys.readouterr().out
    assert "suite regression:" in out
    assert "exact_match_pct" in out
    assert "verdict: passed" in out
    assert "regression_drop_pp=2.0" in out


def test_compare_base_without_adapter_eval_results_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    def _must_not_launch(*args: object, **kwargs: object) -> dict:
        raise AssertionError("no container may be launched without suites to score")

    monkeypatch.setattr(compare_mod.container, "launch", _must_not_launch)
    args = _base_args(str(adapter), "unsloth/Qwen3-4B")
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    assert "sloth eval --adapter" in exc_info.value.remediation


def test_compare_base_rejects_a_second_positional(
    base_fixture: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = base_fixture

    def _must_not_launch(*args: object, **kwargs: object) -> dict:
        raise AssertionError("argument validation must happen before any container launch")

    monkeypatch.setattr(compare_mod.container, "launch", _must_not_launch)
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", b=str(adapter))
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1


def test_compare_without_base_still_requires_two_targets(tmp_path: Path) -> None:
    """``sloth compare <a>`` (no --base) is a user error with a hint — the same
    exit 1 argparse used to produce, now routed through the CliError contract
    so --base can take a single positional."""
    args = _args(str(tmp_path), None)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    assert "compare <a> <b>" in exc_info.value.remediation


def test_register_base_flags() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(
        [
            "compare",
            "adapters/x",
            "--base",
            "unsloth/Qwen3-4B",
            "--config",
            "run.toml",
            "--regression-suite",
            "holdout",
            "--json",
        ]
    )
    assert args.a == "adapters/x"
    assert args.b is None
    assert args.base == "unsloth/Qwen3-4B"
    assert args.config == "run.toml"
    assert args.regression_suite == ["holdout"]


def test_main_compare_base_end_to_end(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sloth.cli import main

    adapter, suite_file = base_fixture
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=60.0)},
            {"regression": _suite_payload("regression", suite_file, exact_match_pct=80.0)},
        ),
    )
    rc = main(["compare", str(adapter), "--base", "unsloth/Qwen3-4B", "--json"])
    assert rc == 1, "a regression beyond the allowance exits 1"
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["verdict"]["passed"] is False
    assert json.loads(captured.err)["code"] == 1


def test_compare_base_does_not_gate_a_suite_only_one_side_has(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A legacy ``eval.json`` (read back as the pseudo-suite ``legacy``) has no
    base-side counterpart: it shows up in the deltas but never fails a gate."""
    adapter, suite_file = base_fixture
    _write_eval_json(adapter, exact_match_pct=80.0, f1=0.8)  # 4 rows, under min_suite_rows
    _install(
        monkeypatch,
        _FakeLaunch(
            {"regression": _suite_payload("regression", suite_file)},
            {"regression": _suite_payload("regression", suite_file)},
        ),
    )
    rc = cmd_compare(_base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["passed"] is True
    legacy = payload["deltas"]["suites"]["legacy"]
    assert legacy["exact_match_pct"]["a"] is None
    assert legacy["exact_match_pct"]["b"] == 80.0


def test_compare_base_precreates_results_dir_and_fails_on_empty_base_side(
    base_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live-measured 2026-09-17 (plan risk r15): docker created the bind-mounted
    eval-base/ as root, the in-container writer got EACCES, and compare passed
    with no base side. The dir must exist before the second launch, and a base
    run that yields no results is exit 1, never a vacuous pass."""
    adapter, suite_file = base_fixture
    adapter_results = {"regression": _suite_payload("regression", suite_file)}
    fake = _install(monkeypatch, _FakeLaunch(adapter_results, {}))  # base writes nothing
    seen_existing: list[bool] = []
    orig_call = fake.__call__

    def _spy(sloth_args: list[str], **kwargs: object) -> dict:
        if sloth_args[:2] == ["eval", "--model"]:
            seen_existing.append((adapter / "eval-base").is_dir())
        return orig_call(sloth_args, **kwargs)

    monkeypatch.setattr(compare_mod.container, "launch", _spy)
    args = _base_args(str(adapter), "unsloth/Qwen3-4B", json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_compare(args)
    assert exc_info.value.code == 1
    assert "produced no results" in exc_info.value.message
    assert exc_info.value.remediation
    assert seen_existing == [True], "eval-base/ must be created by the host user before launch"
