"""Tests for sloth.tune.registry — the run registry (pure stdlib, no torch).

Covers:
  * runs-root rule: registry_path_for(output) == Path(output).parent / "runs.jsonl"
  * compute_config_hash / make_run_id: deterministic, and vary with the config
  * start_run: appends ONE "running" line; creates runs-root if missing
  * finish_run: atomically rewrites ONLY the matching line to ok/failed,
    leaving every other line (including a corrupt one) untouched
  * a killed train (start_run with no matching finish_run) leaves status
    "running" forever — reported honestly, never reclassified
  * read_registry: missing file -> ([], []); corrupt lines skipped with a
    diagnostic; valid lines parsed
  * find_run / resolve_target: run_id lookup and directory-vs-run_id resolution
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from sloth.cli._errors import EXIT_ENV_ERROR, CliError
from sloth.tune.config import RunConfig
from sloth.tune.registry import (
    RUNS_FILENAME,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_RUNNING,
    compute_config_hash,
    find_run,
    finish_run,
    make_run_id,
    read_registry,
    registry_path,
    registry_path_for,
    resolve_target,
    runs_root_for,
    start_run,
)

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl", body: str = _VALID_CHAT) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _make_config(tmp_path: Path, *, output: str | None = None, **overrides: object) -> RunConfig:
    dataset = overrides.pop("dataset", None) or _write_dataset(tmp_path)
    fields: dict[str, object] = {
        "model": "unsloth/Qwen3-4B",
        "dataset": str(dataset),
        "output": output or str(tmp_path / "adapters" / "out"),
        "method": "qlora",
    }
    fields.update(overrides)
    return RunConfig(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runs-root rule
# ---------------------------------------------------------------------------


class TestRunsRootRule:
    def test_runs_root_is_parent_of_output(self, tmp_path: Path) -> None:
        output = tmp_path / "adapters" / "qwen3-4b-qlora"
        assert runs_root_for(output) == output.parent

    def test_registry_path_for_is_parent_slash_runs_jsonl(self, tmp_path: Path) -> None:
        output = tmp_path / "adapters" / "qwen3-4b-qlora"
        assert registry_path_for(output) == output.parent / RUNS_FILENAME

    def test_registry_path_joins_explicit_runs_root(self, tmp_path: Path) -> None:
        assert registry_path(tmp_path) == tmp_path / RUNS_FILENAME


# ---------------------------------------------------------------------------
# config_hash / run_id
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_deterministic_for_identical_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        assert compute_config_hash(cfg) == compute_config_hash(cfg)

    def test_differs_when_a_hyperparameter_differs(self, tmp_path: Path) -> None:
        cfg_a = _make_config(tmp_path, lora_r=16)
        cfg_b = _make_config(tmp_path, lora_r=32)
        assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)

    def test_differs_when_model_differs(self, tmp_path: Path) -> None:
        cfg_a = _make_config(tmp_path, model="unsloth/Qwen3-4B")
        cfg_b = _make_config(tmp_path, model="unsloth/Qwen3-9B")
        assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)

    def test_is_a_hex_digest(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        digest = compute_config_hash(cfg)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


class TestRunId:
    def test_format_is_hash_prefix_dash_compact_ts(self) -> None:
        run_id = make_run_id("abcdef0123456789", "2026-07-06T12:34:56+00:00")
        assert run_id == "abcdef012345-20260706T123456Z"

    def test_deterministic_given_same_inputs(self) -> None:
        a = make_run_id("abcdef0123456789", "2026-07-06T12:34:56+00:00")
        b = make_run_id("abcdef0123456789", "2026-07-06T12:34:56+00:00")
        assert a == b

    def test_timezone_normalised_to_utc(self) -> None:
        # +02:00 local -> UTC compact timestamp shifts back 2 hours.
        run_id = make_run_id("abcdef0123456789", "2026-07-06T14:34:56+02:00")
        assert run_id == "abcdef012345-20260706T123456Z"


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


class TestStartRun:
    def test_appends_one_running_line(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)

        assert record.status == STATUS_RUNNING
        assert record.finished is None
        assert record.model == cfg.model
        assert record.method == cfg.method
        # output_dir is stored ABSOLUTE (resolved), even though cfg.output
        # here already happens to be absolute (pytest's tmp_path) — the
        # resolve() call must not alter an already-absolute, symlink-free path.
        assert record.output_dir == str(Path(cfg.output).resolve(strict=False))
        assert Path(record.output_dir).is_absolute()
        assert record.dataset["line_count"] == 1

        path = registry_path_for(cfg.output)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        on_disk = json.loads(lines[0])
        assert on_disk["run_id"] == record.run_id
        assert on_disk["status"] == "running"

    def test_relative_output_is_stored_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative ``config.output`` must be resolved to an absolute path
        before being recorded, so a later lookup from a different CWD works."""
        monkeypatch.chdir(tmp_path)
        cfg = _make_config(tmp_path, output=str(Path("adapters") / "out"))
        record = start_run(cfg)

        assert record.output_dir == str((tmp_path / "adapters" / "out").resolve(strict=False))
        assert Path(record.output_dir).is_absolute()

    def test_creates_runs_root_if_missing(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, output=str(tmp_path / "brand-new" / "out"))
        assert not (tmp_path / "brand-new").exists()
        start_run(cfg)
        assert registry_path_for(cfg.output).is_file()

    def test_two_starts_append_two_distinct_lines(self, tmp_path: Path) -> None:
        # run_id is only second-precision (see module docstring: "Run-id
        # rule") — two starts within the same wall-clock second for an
        # IDENTICAL config would collide, so pass explicit distinct
        # timestamps to prove the append behavior deterministically rather
        # than depending on real-clock timing.
        cfg = _make_config(tmp_path)
        r1 = start_run(cfg, timestamp="2026-07-06T00:00:00+00:00")
        r2 = start_run(cfg, timestamp="2026-07-06T00:00:01+00:00")
        assert r1.run_id != r2.run_id

        path = registry_path_for(cfg.output)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_missing_dataset_raises_cli_error_2(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, dataset=str(tmp_path / "does_not_exist.jsonl"))
        with pytest.raises(CliError) as exc_info:
            start_run(cfg)
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# finish_run — atomic rewrite, other lines untouched
# ---------------------------------------------------------------------------


class TestFinishRun:
    def test_rewrites_status_and_sets_finished(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)
        updated = finish_run(record, STATUS_OK, finished="2026-07-06T13:00:00+00:00")

        assert updated.status == STATUS_OK
        assert updated.finished == "2026-07-06T13:00:00+00:00"

        path = registry_path_for(cfg.output)
        on_disk = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert on_disk["status"] == "ok"
        assert on_disk["finished"] == "2026-07-06T13:00:00+00:00"

    def test_failed_status_recorded(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)
        finish_run(record, STATUS_FAILED)
        records, _ = read_registry(runs_root_for(cfg.output))
        assert records[0]["status"] == "failed"

    def test_only_the_matching_line_is_rewritten(self, tmp_path: Path) -> None:
        """Two runs share one registry file; finishing one must not touch the other."""
        cfg_a = _make_config(tmp_path, output=str(tmp_path / "adapters" / "a"))
        cfg_b = _make_config(tmp_path, output=str(tmp_path / "adapters" / "b"))
        record_a = start_run(cfg_a)
        record_b = start_run(cfg_b)

        finish_run(record_a, STATUS_OK)

        records, _ = read_registry(tmp_path / "adapters")
        by_id = {r["run_id"]: r for r in records}
        assert by_id[record_a.run_id]["status"] == "ok"
        assert by_id[record_b.run_id]["status"] == "running"  # untouched

    def test_corrupt_line_preserved_verbatim(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)

        path = registry_path_for(cfg.output)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("not valid json at all\n")

        finish_run(record, STATUS_OK)

        raw_lines = path.read_text(encoding="utf-8").splitlines()
        assert raw_lines[-1] == "not valid json at all"

    def test_missing_registry_file_returns_updated_record_without_crash(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)
        registry_path_for(cfg.output).unlink()  # simulate the file vanishing

        updated = finish_run(record, STATUS_OK)
        assert updated.status == STATUS_OK


# ---------------------------------------------------------------------------
# Crash honesty — a killed train leaves status "running" (no pid tracking, v1)
# ---------------------------------------------------------------------------


class TestKilledTrainLeavesRunningStatus:
    def test_never_finished_stays_running_and_is_reported_as_is(self, tmp_path: Path) -> None:
        """Simulates a killed `train`: start_run runs, finish_run never does."""
        cfg = _make_config(tmp_path)
        record = start_run(cfg)  # process "dies" here — finish_run never called

        records, diagnostics = read_registry(runs_root_for(cfg.output))
        assert diagnostics == []
        assert len(records) == 1
        assert records[0]["run_id"] == record.run_id
        assert records[0]["status"] == "running"
        assert records[0]["finished"] is None

        found = find_run(runs_root_for(cfg.output), record.run_id)
        assert found is not None
        assert found["status"] == "running"


# ---------------------------------------------------------------------------
# read_registry
# ---------------------------------------------------------------------------


class TestReadRegistry:
    def test_missing_file_is_honest_empty(self, tmp_path: Path) -> None:
        records, diagnostics = read_registry(tmp_path / "nope")
        assert records == []
        assert diagnostics == []

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        start_run(cfg)
        path = registry_path_for(cfg.output)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n   \n")
        records, diagnostics = read_registry(runs_root_for(cfg.output))
        assert len(records) == 1
        assert diagnostics == []

    def test_corrupt_json_line_skipped_with_diagnostic(self, tmp_path: Path) -> None:
        root = tmp_path / "adapters"
        root.mkdir()
        path = registry_path(root)
        path.write_text("not json\n", encoding="utf-8")

        records, diagnostics = read_registry(root)
        assert records == []
        assert len(diagnostics) == 1
        assert "line 1" in diagnostics[0]
        assert "invalid JSON" in diagnostics[0]

    def test_line_missing_run_id_skipped_with_diagnostic(self, tmp_path: Path) -> None:
        root = tmp_path / "adapters"
        root.mkdir()
        path = registry_path(root)
        path.write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

        records, diagnostics = read_registry(root)
        assert records == []
        assert len(diagnostics) == 1
        assert "run_id" in diagnostics[0]

    def test_valid_and_corrupt_lines_mixed(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        start_run(cfg)
        path = registry_path_for(cfg.output)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{broken\n")
            fh.write(json.dumps({"status": "ok"}) + "\n")  # missing run_id

        records, diagnostics = read_registry(runs_root_for(cfg.output))
        assert len(records) == 1
        assert len(diagnostics) == 2

    def test_unreadable_file_raises_cli_error_env(self, tmp_path: Path) -> None:
        if os.geteuid() == 0:
            pytest.skip("running as root bypasses file permissions")
        root = tmp_path / "adapters"
        root.mkdir()
        path = registry_path(root)
        path.write_text(json.dumps({"run_id": "x"}) + "\n", encoding="utf-8")
        path.chmod(0o000)
        try:
            with pytest.raises(CliError) as exc_info:
                read_registry(root)
            assert exc_info.value.code == EXIT_ENV_ERROR
            assert exc_info.value.remediation
        finally:
            path.chmod(0o644)


# ---------------------------------------------------------------------------
# find_run
# ---------------------------------------------------------------------------


class TestFindRun:
    def test_finds_by_run_id(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)
        found = find_run(runs_root_for(cfg.output), record.run_id)
        assert found is not None
        assert found["run_id"] == record.run_id

    def test_unknown_run_id_returns_none(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        start_run(cfg)
        assert find_run(runs_root_for(cfg.output), "nonexistent-run-id") is None

    def test_missing_registry_returns_none(self, tmp_path: Path) -> None:
        assert find_run(tmp_path / "nowhere", "any-id") is None


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_existing_directory_resolves_directly(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "adapters" / "out"
        out_dir.mkdir(parents=True)
        resolved = resolve_target(str(out_dir))
        assert resolved == out_dir

    def test_run_id_resolves_via_registry(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        record = start_run(cfg)
        resolved = resolve_target(record.run_id, runs_root_for(cfg.output))
        assert resolved == Path(cfg.output)

    def test_unresolvable_target_raises_cli_error_1(self, tmp_path: Path) -> None:
        with pytest.raises(CliError) as exc_info:
            resolve_target("totally-bogus-target", tmp_path)
        assert exc_info.value.code == 1
        assert exc_info.value.remediation

    def test_default_runs_root_is_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _make_config(tmp_path, output=str(tmp_path / "out"))
        record = start_run(cfg)
        resolved = resolve_target(record.run_id)  # no runs_root passed -> cwd
        assert resolved == Path(cfg.output)

    def test_legacy_relative_output_dir_resolves_under_runs_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward-compat: a pre-existing registry may carry a RELATIVE
        ``output_dir`` (written before this fix). Querying it with an
        absolute ``--runs-root`` from an unrelated CWD must still resolve to
        ``runs_root/<basename>`` (the runs-root invariant: runs_root is the
        parent of output)."""
        runs_root = tmp_path / "adapters"
        runs_root.mkdir()
        legacy_record = {
            "run_id": "legacy0000000-20260706T000000Z",
            "config_hash": "legacy0000000",
            "output_dir": "out",  # relative — pre-fix shape
            "model": "unsloth/Qwen3-4B",
            "method": "qlora",
            "dataset": {"sha256": "deadbeef", "line_count": 1},
            "started": "2026-07-06T00:00:00+00:00",
            "finished": None,
            "status": "running",
        }
        registry_path(runs_root).write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        resolved = resolve_target(legacy_record["run_id"], runs_root.resolve())
        assert resolved == runs_root.resolve() / "out"
