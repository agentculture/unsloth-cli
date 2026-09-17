"""Tests for :mod:`sloth.tune._bench` — the in-container lm_eval runner.

Covers:

* the composed in-container command (the exact ``lm_eval`` argv, including the
  ``--model_args`` string for a plain model, a LoRA adapter and a QLoRA adapter);
* base-model / ``load_in_4bit`` resolution out of ``training_metadata.json``,
  with the ``adapter_config.json`` fallback and the code-1 failure;
* lm_eval results-document parsing into this repo's eval payload shape;
* the park-v4 failure path: a non-zero harness exit is ``CliError(code=2)``
  whose remediation names the merged-16-bit fallback;
* the module is stdlib-only at import (mirrors the metrics/scorers guard in
  ``tests/test_lazy_import.py``).

No docker, no GPU, no lm_eval: the subprocess seam ``_bench._run`` is
monkeypatched throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import sloth.tune._bench as bench
from sloth.cli._errors import CliError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_adapter(
    directory: Path, *, model: str = "unsloth/Qwen3-4B", load_in_4bit: bool | None = True
) -> Path:
    """Create an adapter dir with a training_metadata.json (and adapter_config.json)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metadata.json").write_text(
        json.dumps({"model": model, "hyperparameters": {"load_in_4bit": load_in_4bit}}),
        encoding="utf-8",
    )
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": model, "peft_type": "LORA"}), encoding="utf-8"
    )
    return directory


#: A minimal lm_eval results document with a group row and two subject rows.
_RESULTS_DOC: dict[str, Any] = {
    "results": {
        "mmlu": {"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "mmlu"},
        "mmlu_astronomy": {"acc,none": 0.6, "alias": " - astronomy"},
        "mmlu_world_religions": {"acc,none": 0.4, "acc_norm,none": 0.45},
    },
    "n-samples": {
        "mmlu": {"original": 14042, "effective": 40},
        "mmlu_astronomy": {"original": 152, "effective": 20},
        "mmlu_world_religions": {"original": 171, "effective": 20},
    },
}


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def test_resolve_adapter_target_reads_training_metadata(tmp_path: Path) -> None:
    """The base model id and the QLoRA flag come from training_metadata.json."""
    adapter = _write_adapter(tmp_path / "adapter")
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", True)


def test_resolve_adapter_target_reports_a_lora_run_as_not_4bit(tmp_path: Path) -> None:
    """A plain LoRA run leaves ``load_in_4bit`` off the model args."""
    adapter = _write_adapter(tmp_path / "adapter", load_in_4bit=False)
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", False)


def test_resolve_adapter_target_falls_back_to_adapter_config(tmp_path: Path) -> None:
    """An adapter produced outside `sloth train` still benches, via adapter_config."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-9B"}), encoding="utf-8"
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-9B", False)


def test_resolve_adapter_target_without_a_base_is_a_user_error(tmp_path: Path) -> None:
    """No base model anywhere is exit 1 with a remediation."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(CliError) as excinfo:
        bench.resolve_adapter_target(adapter)
    assert excinfo.value.code == 1
    assert excinfo.value.remediation


# ---------------------------------------------------------------------------
# The in-container command
# ---------------------------------------------------------------------------


def test_model_args_for_a_plain_model_directory() -> None:
    """``--model <dir>`` benches the directory itself; no peft, no 4-bit."""
    assert bench.build_model_args("/exports/merged16") == "pretrained=/exports/merged16"


def test_model_args_for_a_qlora_adapter() -> None:
    """A QLoRA adapter composes pretrained= + peft= + load_in_4bit=True, in order."""
    assert (
        bench.build_model_args("unsloth/Qwen3-4B", peft="/a/lora", load_in_4bit=True)
        == "pretrained=unsloth/Qwen3-4B,peft=/a/lora,load_in_4bit=True"
    )


def test_lm_eval_command_is_the_documented_argv() -> None:
    """The composed command matches docs/dgx-spark.md and `explain bench` verbatim."""
    cmd = bench.lm_eval_command(
        pretrained="unsloth/Qwen3-4B",
        peft="/a/lora",
        load_in_4bit=True,
        output_path="/tmp/out",
        limit=20,
    )
    assert cmd == [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        "pretrained=unsloth/Qwen3-4B,peft=/a/lora,load_in_4bit=True",
        "--tasks",
        "mmlu",
        "--num_fewshot",
        "5",
        "--limit",
        "20",
        "--output_path",
        "/tmp/out",
        "--log_samples",
    ]


def test_lm_eval_command_omits_limit_when_unset() -> None:
    """A full run passes no ``--limit`` at all."""
    cmd = bench.lm_eval_command(pretrained="m", output_path="/tmp/out")
    assert "--limit" not in cmd
    assert cmd[-1] == "--log_samples"


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------


def test_build_payload_has_the_documented_keys() -> None:
    """The payload carries the flat metrics summarize/compare fold, plus provenance."""
    payload = bench.build_payload(_RESULTS_DOC, limit=20, version="0.4.13")
    assert payload["acc"] == 0.5
    assert payload["exact_match_pct"] == 50.0
    assert payload["total"] == 40
    assert payload["acc_norm"] is None  # MMLU reports plain acc only
    assert payload["per_subject"]["astronomy"] == {"acc": 0.6, "acc_norm": None, "total": 20}
    assert payload["per_subject"]["world_religions"]["acc_norm"] == 0.45
    assert payload["harness"] == {
        "name": "lm_eval",
        "version": "0.4.13",
        "tasks": "mmlu",
        "limit": 20,
        "num_fewshot": 5,
    }


def test_build_payload_averages_subjects_when_no_group_row() -> None:
    """Without an overall row the subjects' mean accuracy stands in for it."""
    document = {
        "results": {"mmlu_astronomy": {"acc,none": 0.6}, "mmlu_logic": {"acc,none": 0.4}},
        "n-samples": {"mmlu_astronomy": {"effective": 5}, "mmlu_logic": {"effective": 5}},
    }
    payload = bench.build_payload(document)
    assert payload["acc"] == 0.5
    assert payload["total"] == 10


def test_build_payload_on_an_empty_document_is_well_formed() -> None:
    """A results-less document scores zero rather than raising."""
    payload = bench.build_payload({})
    assert payload["acc"] == 0.0
    assert payload["total"] == 0
    assert payload["per_subject"] == {}


def test_find_results_file_locates_the_nested_output(tmp_path: Path) -> None:
    """lm_eval nests results under a sanitised model dir; the glob finds it."""
    nested = tmp_path / "unsloth__Qwen3-4B"
    nested.mkdir()
    target = nested / "results_2026-09-17T00-00-00.json"
    target.write_text("{}", encoding="utf-8")
    assert bench.find_results_file(tmp_path) == target


def test_find_results_file_returns_none_when_nothing_was_written(tmp_path: Path) -> None:
    """A harness that wrote nothing is reported as nothing, not as an error."""
    assert bench.find_results_file(tmp_path) is None


# ---------------------------------------------------------------------------
# run_bench — the seam, with the subprocess call faked
# ---------------------------------------------------------------------------


def _fake_run_writing(document: dict[str, Any], recorder: list[list[str]]):
    """Return a ``_run`` double that records the argv and writes *document*."""

    def _run(cmd: list[str]) -> int:
        recorder.append(list(cmd))
        output_path = Path(cmd[cmd.index("--output_path") + 1])
        nested = output_path / "model"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "results_run.json").write_text(json.dumps(document), encoding="utf-8")
        return 0

    return _run


def test_run_bench_on_an_adapter_runs_the_expected_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QLoRA adapter is benched as pretrained=<base>,peft=<dir>,load_in_4bit=True."""
    adapter = _write_adapter(tmp_path / "adapter")
    calls: list[list[str]] = []
    monkeypatch.setattr(bench, "_run", _fake_run_writing(_RESULTS_DOC, calls))

    payload = bench.run_bench(adapter, kind="adapter", limit=20)

    assert len(calls) == 1
    argv = calls[0]
    assert argv[:4] == ["lm_eval", "--model", "hf", "--model_args"]
    assert argv[4] == f"pretrained=unsloth/Qwen3-4B,peft={adapter},load_in_4bit=True"
    assert "--log_samples" in argv
    assert payload["exact_match_pct"] == 50.0
    assert payload["base_load_in_4bit"] is True
    assert payload["model"] == "unsloth/Qwen3-4B"


def test_run_bench_on_a_model_directory_passes_it_as_pretrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--model <dir>`` needs no metadata: the directory itself is the model."""
    merged = tmp_path / "merged16"
    merged.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(bench, "_run", _fake_run_writing(_RESULTS_DOC, calls))

    payload = bench.run_bench(merged, kind="model")

    assert calls[0][4] == f"pretrained={merged}"
    assert payload["base_load_in_4bit"] is None


def test_run_bench_failure_names_the_merged16_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Park v4: a harness that cannot load 4-bit + peft exits 2 with the fallback hint."""
    adapter = _write_adapter(tmp_path / "adapter")
    monkeypatch.setattr(bench, "_run", lambda cmd: 1)

    with pytest.raises(CliError) as excinfo:
        bench.run_bench(adapter, kind="adapter")

    assert excinfo.value.code == 2
    assert "merged16" in excinfo.value.remediation
    assert "--model" in excinfo.value.remediation


def test_run_bench_without_a_results_file_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness that exits 0 but writes nothing is still an environment error."""
    adapter = _write_adapter(tmp_path / "adapter")
    monkeypatch.setattr(bench, "_run", lambda cmd: 0)

    with pytest.raises(CliError) as excinfo:
        bench.run_bench(adapter, kind="adapter")
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Argv validation (the injection guard in front of the lm_eval argv)
# ---------------------------------------------------------------------------


def test_run_bench_rejects_a_task_string_with_shell_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --tasks spec carrying anything but task names never reaches the harness."""
    merged = tmp_path / "merged16"
    merged.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(bench, "_run", _fake_run_writing(_RESULTS_DOC, calls))

    with pytest.raises(CliError) as excinfo:
        bench.run_bench(merged, kind="model", tasks="mmlu; rm -rf /")

    assert excinfo.value.code == 1
    assert excinfo.value.remediation
    assert calls == []  # the subprocess seam was never reached


def test_run_bench_rejects_a_model_reference_that_is_neither_path_nor_repo_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model reference forging a second --model_args fragment is a user error."""
    calls: list[list[str]] = []
    monkeypatch.setattr(bench, "_run", _fake_run_writing(_RESULTS_DOC, calls))

    with pytest.raises(CliError) as excinfo:
        bench.run_bench(tmp_path / "nope,peft=/etc", kind="model")

    assert excinfo.value.code == 1
    assert "," in excinfo.value.remediation
    assert calls == []


def test_run_bench_rejects_a_negative_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--limit`` must be a non-negative whole number."""
    merged = tmp_path / "merged16"
    merged.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(bench, "_run", _fake_run_writing(_RESULTS_DOC, calls))

    with pytest.raises(CliError) as excinfo:
        bench.run_bench(merged, kind="model", limit=-1)

    assert excinfo.value.code == 1
    assert calls == []


# ---------------------------------------------------------------------------
# Letter-choice scoring (sloth.tune._trainer scoring helpers)
#
# These live beside the benchmark tests rather than in test_tune_trainer.py
# because they are the *other half* of t13: `sloth bench` runs MMLU through
# lm_eval, and this path scores the committed MMLU-style subset through the
# ordinary `sloth eval --suite` seam.
# ---------------------------------------------------------------------------


def _choice_records() -> list[dict[str, str]]:
    return [
        {"task": "mcq", "input": "q1\nA. x\nB. y", "expected_output": "B"},
        {"task": "mcq", "input": "q2\nA. x\nB. y", "expected_output": "A"},
    ]


def test_is_choice_suite_detects_single_letter_answers() -> None:
    """Every row answering with one A-D letter makes it a choice suite."""
    from sloth.tune._trainer import is_choice_suite

    assert is_choice_suite(_choice_records()) is True


def test_is_choice_suite_is_all_or_nothing() -> None:
    """One prose row means an ordinary task suite, not a choice suite."""
    from sloth.tune._trainer import is_choice_suite

    records = _choice_records() + [{"task": "t", "input": "i", "expected_output": "a sentence"}]
    assert is_choice_suite(records) is False
    assert is_choice_suite([]) is False


def test_choice_scoring_adds_choice_match_without_touching_exact_match() -> None:
    """``Answer: B`` is a choice hit but not an exact match — both are reported."""
    from sloth.tune import metrics
    from sloth.tune._trainer import _extra_metrics_for, choice_acc_pct

    records = _choice_records()
    extra = _extra_metrics_for("task", records=records)
    scored = metrics.score_records(records, ["Answer: B", "C"], extra_metrics=extra)

    assert [row["choice_match"] for row in scored] == [True, False]
    assert [row["exact_match"] for row in scored] == [False, False]
    assert choice_acc_pct(scored) == 50.0


def test_choice_scoring_is_off_for_an_ordinary_task_suite() -> None:
    """A prose suite gets no choice key at all, so no misleading 0%."""
    from sloth.tune import metrics
    from sloth.tune._trainer import _extra_metrics_for, choice_acc_pct

    records = [{"task": "t", "input": "i", "expected_output": "a full answer"}]
    assert _extra_metrics_for("task", records=records) is None
    scored = metrics.score_records(records, ["a full answer"])
    assert choice_acc_pct(scored) is None


def test_choice_scoring_rolls_up_into_the_file_entry() -> None:
    """``choice_acc_pct`` reaches the suite payload alongside ``exact_match_pct``."""
    from sloth.tune import metrics
    from sloth.tune._trainer import _extra_metrics_for, build_file_entry, build_summary

    records = _choice_records()
    extra = _extra_metrics_for("task", records=records)
    scored = metrics.score_records(records, ["B", "(A)"], extra_metrics=extra)
    entry = build_file_entry("mmlu-subset.jsonl", scored)
    assert entry["choice_acc_pct"] == 100.0

    summary = build_summary([entry], batch_size=8, base_load_in_4bit=None)
    assert summary["choice_acc_pct"] == 100.0


def test_choice_scoring_composes_with_a_schema_scorer() -> None:
    """A richer schema keeps its compliance key and gains the choice key."""
    from sloth.tune import metrics
    from sloth.tune._trainer import _extra_metrics_for

    records = [
        {
            "task": "mcq",
            "input": "q",
            "expected_output": "B",
            "constraints": [{"max_words": 3}],
        }
    ]
    extra = _extra_metrics_for("instruction", records=records)
    scored = metrics.score_records(records, ["B"], extra_metrics=extra)
    assert scored[0]["choice_match"] is True
    assert scored[0]["constraints_passed"] is True


# ---------------------------------------------------------------------------
# QLoRA precision resolution (qodo PR #29 finding 4)
# ---------------------------------------------------------------------------


def _write_metadata(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_qlora_method_benches_in_4bit_even_without_the_hyperparameter(tmp_path: Path) -> None:
    """Training forces 4-bit for `method = "qlora"`; the bench must match it."""
    adapter = _write_metadata(
        tmp_path / "adapter",
        {
            "model": "unsloth/Qwen3-4B",
            "method": "qlora",
            "hyperparameters": {"load_in_4bit": False},
        },
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", True)


def test_lora_method_with_the_flag_on_still_benches_in_4bit(tmp_path: Path) -> None:
    adapter = _write_metadata(
        tmp_path / "adapter",
        {"model": "unsloth/Qwen3-4B", "method": "lora", "hyperparameters": {"load_in_4bit": True}},
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", True)


def test_lora_method_without_the_flag_benches_in_full_precision(tmp_path: Path) -> None:
    adapter = _write_metadata(
        tmp_path / "adapter",
        {"model": "unsloth/Qwen3-4B", "method": "lora", "hyperparameters": {"load_in_4bit": False}},
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", False)


def test_resolved_load_in_4bit_wins_over_the_raw_hyperparameter(tmp_path: Path) -> None:
    """``resolved.load_in_4bit`` is what the run actually trained at — prefer it."""
    adapter = _write_metadata(
        tmp_path / "adapter",
        {
            "model": "unsloth/Qwen3-4B",
            "method": "lora",
            "hyperparameters": {"load_in_4bit": False},
            "resolved": {"load_in_4bit": True},
        },
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", True)


def test_resolved_load_in_4bit_false_wins_over_a_qlora_method(tmp_path: Path) -> None:
    adapter = _write_metadata(
        tmp_path / "adapter",
        {
            "model": "unsloth/Qwen3-4B",
            "method": "qlora",
            "hyperparameters": {"load_in_4bit": True},
            "resolved": {"load_in_4bit": False},
        },
    )
    assert bench.resolve_adapter_target(adapter) == ("unsloth/Qwen3-4B", False)
