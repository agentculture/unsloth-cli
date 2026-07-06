"""Run registry for unsloth-cli fine-tuning jobs (pure stdlib, no torch).

Today a completed ``sloth train`` run lands as a bare directory holding
``training_metadata.json`` (see :mod:`sloth.tune.metadata`) — there is no
registry, so enumerating past runs means walking directories by hand. This
module gives every run a **registry line** an agent can list/show/resolve
without touching the filesystem tree directly.

Runs-root rule
--------------
The registry file lives at ``<runs-root>/runs.jsonl`` where **runs-root is the
PARENT directory of the run's ``output`` dir** — i.e. ``runs_root_for(output)
== Path(output).parent``. For example ``output = "adapters/qwen3-4b-qlora"``
puts the registry at ``adapters/runs.jsonl``. Every run whose ``output`` lives
under the same parent directory shares one registry file; moving/renaming that
parent directory moves its registry with it. There is no global/home-dir
registry — it is always resolved relative to a specific run's output location
(or an explicit ``--runs-root`` the CLI verbs accept).

Run-id rule
-----------
``run_id = "<config_hash[:12]>-<started-compact-utc>"`` — the first 12 hex
characters of :func:`compute_config_hash` joined to the ``started`` timestamp
formatted as ``%Y%m%dT%H%M%SZ``. This keeps ``run_id`` traceable back to the
config that produced it while staying unique even when the SAME output
directory is retrained repeatedly (a plain output-dir basename would collide
on a rerun, and matching :func:`finish_run` to the wrong line would then
corrupt a different run's status). **Honest limit:** the compact timestamp is
second-precision (``%Y%m%dT%H%M%SZ``), so two runs of the exact SAME config
started within the same wall-clock second would produce the same ``run_id`` —
an accepted, documented v1 edge case (sub-second concurrent identical retrains
are not a realistic operator workflow), not one this module works around.

Status lifecycle (v1 — no pid tracking, by design)
---------------------------------------------------
A registry line's ``status`` field only ever transitions
``"running" -> "ok" | "failed"``. :func:`start_run` appends the ``"running"``
line atomically at train start; :func:`finish_run` atomically rewrites that
SAME line at completion. There is **deliberately no pid tracking** in v1: if
the training process is killed (SIGKILL, OOM, power loss) between those two
calls, the line is left at ``status: "running"`` forever, and ``runs
list``/``runs show`` report it exactly as recorded — honest and simple, not a
guessed "stale"/"incomplete" classification (which would require knowing
whether the recorded pid is still alive — out of scope for v1; a documented
follow-up).

Atomicity
---------
:func:`start_run` appends via a *single* ``os.write()`` call on a file opened
with ``O_APPEND | O_CREAT``. POSIX guarantees a single ``write()`` under the
platform's atomic-write limit (``PIPE_BUF`` — 4096 bytes on Linux) is never
interleaved with a concurrent writer's own ``O_APPEND`` write, so two
concurrent ``sloth train`` processes appending to the same registry cannot
corrupt each other's line — a registry line staying well under that limit is
a documented v1 assumption, not independently enforced.
:func:`finish_run` rewrites the WHOLE file (every other line preserved
byte-for-byte, only the target line replaced) into a temp file in the same
directory, then ``os.replace()``s it over the original — ``os.replace`` is
atomic on POSIX and Windows, so a concurrent reader never observes a
half-written file.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.tune.config import RunConfig
from sloth.tune.metadata import dataset_digest

RUNS_FILENAME = "runs.jsonl"

STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_FAILED = "failed"

_RUN_ID_TS_FORMAT = "%Y%m%dT%H%M%SZ"


# ---------------------------------------------------------------------------
# Path helpers (the runs-root rule)
# ---------------------------------------------------------------------------


def runs_root_for(output: str | Path) -> Path:
    """Return the runs-root for a run's ``output`` dir: its PARENT directory."""
    return Path(output).parent


def registry_path(runs_root: str | Path) -> Path:
    """Return the ``runs.jsonl`` path inside an explicit *runs_root*."""
    return Path(runs_root) / RUNS_FILENAME


def registry_path_for(output: str | Path) -> Path:
    """Return the ``runs.jsonl`` path for a run's ``output`` dir (its parent)."""
    return registry_path(runs_root_for(output))


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One ``runs.jsonl`` line — the registry's stable, documented record shape."""

    run_id: str
    config_hash: str
    output_dir: str
    model: str
    method: str
    dataset: dict[str, Any]
    started: str
    finished: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "output_dir": self.output_dir,
            "model": self.model,
            "method": self.method,
            "dataset": dict(self.dataset),
            "started": self.started,
            "finished": self.finished,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# config_hash / run_id
# ---------------------------------------------------------------------------


def compute_config_hash(config: RunConfig) -> str:
    """Return a stable sha256 hex digest fingerprinting *config*'s resolved fields.

    Built from ``dataclasses.asdict(config)`` serialized with sorted keys, so
    the hash is deterministic regardless of field-declaration order and stays
    in sync automatically if :class:`RunConfig` grows new fields. This is an
    opaque fingerprint (not decoded back into hyperparameters) — ``sloth
    compare`` reads the actual hyperparameters from each run's
    ``training_metadata.json`` instead.
    """
    canonical = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_run_id(config_hash: str, started: str) -> str:
    """Build ``run_id`` from *config_hash*'s first 12 hex chars + a compact UTC
    timestamp derived from the ISO-8601 *started* string (see module docstring:
    "Run-id rule")."""
    dt = datetime.fromisoformat(started)
    compact = dt.astimezone(timezone.utc).strftime(_RUN_ID_TS_FORMAT)
    return f"{config_hash[:12]}-{compact}"


# ---------------------------------------------------------------------------
# start_run / finish_run — the atomic append + atomic rewrite
# ---------------------------------------------------------------------------


def start_run(config: RunConfig, *, timestamp: str | None = None) -> RunRecord:
    """Append a new ``status: "running"`` line to the run's registry.

    Returns the :class:`RunRecord` — the caller MUST hold onto it and pass it
    to :func:`finish_run` at completion, since that is how the same line is
    located again. Runs-root (the parent dir of ``config.output``) is created
    if missing.

    Raises
    ------
    CliError
        code=2 when the dataset cannot be digested (propagated from
        :func:`~sloth.tune.metadata.dataset_digest`) or the registry file
        cannot be written (permission/environment error).
    """
    started = timestamp or datetime.now(timezone.utc).isoformat()
    config_hash = compute_config_hash(config)
    run_id = make_run_id(config_hash, started)
    sha256, line_count = dataset_digest(Path(config.dataset))

    # Store the ABSOLUTE output path so a later `resolve_target()`/`runs show`
    # invoked from a different CWD (even with an explicit `--runs-root`) still
    # resolves correctly — a verbatim, often-relative `config.output` breaks
    # that. `config.output` itself and the config_hash input stay untouched.
    output_abs = Path(config.output).resolve(strict=False)

    record = RunRecord(
        run_id=run_id,
        config_hash=config_hash,
        output_dir=str(output_abs),
        model=config.model,
        method=config.method,
        dataset={"sha256": sha256, "line_count": line_count},
        started=started,
        finished=None,
        status=STATUS_RUNNING,
    )

    path = registry_path_for(output_abs)
    line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not append to run registry at {path}: {exc}",
            remediation="Check that the runs-root directory is writable.",
        ) from exc

    return record


def finish_run(
    record: RunRecord,
    status: str,
    *,
    finished: str | None = None,
) -> RunRecord:
    """Atomically rewrite *record*'s line to *status* (+ *finished*, default now UTC).

    Every other line in the registry — including one that fails to parse as
    JSON — is carried through UNCHANGED, byte-for-byte, so this call can never
    corrupt a different run's line. If the started line cannot be found (e.g.
    the registry file was rotated/truncated externally), the completion record
    is appended instead of silently dropped.

    Raises
    ------
    CliError
        code=2 on any I/O failure while rewriting the registry.
    """
    finished = finished or datetime.now(timezone.utc).isoformat()
    # cast(): dataclasses.replace() is typed as returning a bare DataclassInstance,
    # which trips SonarPython's return-type checks (S5886/S5890); the cast asserts
    # the concrete RunRecord type without reconstructing every field by hand.
    updated = cast(RunRecord, dataclasses.replace(record, status=status, finished=finished))

    path = registry_path_for(record.output_dir)
    if not path.is_file():
        # Nothing on disk to rewrite (e.g. the registry vanished underneath
        # us) — still return the updated in-memory record for the caller.
        return updated

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not read run registry at {path}: {exc}",
            remediation="Check that the registry file is readable.",
        ) from exc

    new_lines: list[str] = []
    matched = False
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            new_lines.append(raw)
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            new_lines.append(raw)
            continue
        if not matched and isinstance(parsed, dict) and parsed.get("run_id") == record.run_id:
            new_lines.append(json.dumps(updated.to_dict(), sort_keys=True) + "\n")
            matched = True
        else:
            new_lines.append(raw)

    if not matched:
        new_lines.append(json.dumps(updated.to_dict(), sort_keys=True) + "\n")

    try:
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".runs.", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.writelines(new_lines)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not rewrite run registry at {path}: {exc}",
            remediation="Check that the runs-root directory is writable.",
        ) from exc

    return updated


# ---------------------------------------------------------------------------
# Reading the registry
# ---------------------------------------------------------------------------


def read_registry(runs_root: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every line of ``<runs_root>/runs.jsonl``.

    Returns ``(records, diagnostics)``:

    * ``records`` — parsed dicts, in file order (callers sort as needed).
    * ``diagnostics`` — one human-readable string per skipped/corrupt line
      (1-indexed line number + reason), meant to be echoed via
      ``emit_diagnostic``. Corrupt input is SKIPPED, never a crash.

    A missing registry file returns ``([], [])`` — an honest empty list, not
    an error (a repo that has never trained anything has no ``runs.jsonl``
    yet).
    """
    path = registry_path(runs_root)
    if not path.is_file():
        return [], []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"could not read run registry at {path}: {exc}",
            remediation="Check that the registry file is readable.",
        ) from exc

    records: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"{path}:{line_no}: skipped (invalid JSON: {exc})")
            continue
        if not isinstance(parsed, dict) or "run_id" not in parsed:
            diagnostics.append(f"{path}:{line_no}: skipped (not a run record — missing 'run_id')")
            continue
        records.append(parsed)
    return records, diagnostics


def find_run(runs_root: str | Path, run_id: str) -> dict[str, Any] | None:
    """Return the LATEST registry record matching *run_id*, or ``None``.

    Normal operation (:func:`start_run` + :func:`finish_run`) keeps exactly
    one line per ``run_id``; "latest" only matters for a manually-edited or
    externally-appended registry that carries duplicates.
    """
    records, _ = read_registry(runs_root)
    matches = [r for r in records if r.get("run_id") == run_id]
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Shared target resolution — run_id OR a literal directory path
# ---------------------------------------------------------------------------


def resolve_target(target: str, runs_root: str | Path | None = None) -> Path:
    """Resolve *target* (a ``run_id`` or a filesystem path) to an output directory.

    Resolution order:

    1. *target* names an EXISTING directory on disk -> use it directly (the
       ``<output_dir>`` form). Directories take precedence, so a run_id that
       happens to collide with a real directory name resolves to the
       directory.
    2. Otherwise, look *target* up as a ``run_id`` in the registry at
       *runs_root* (default: the current working directory) -> use its
       recorded ``output_dir``.

    Raises
    ------
    CliError
        code=1 when neither resolves.
    """
    as_path = Path(target)
    if as_path.is_dir():
        return as_path

    root = Path(runs_root) if runs_root is not None else Path.cwd()
    record = find_run(root, target)
    if record is not None:
        output_dir = record["output_dir"]
        if Path(output_dir).is_absolute():
            return Path(output_dir)
        # Backward-compat: a pre-existing registry line recorded a RELATIVE
        # output_dir. The runs-root invariant (runs_root == parent of output)
        # means the output dir's basename lives directly under runs_root.
        return Path(root) / Path(output_dir).name

    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"could not resolve '{target}' to a run_id or an existing output directory",
        remediation=(
            f"Pass an existing output directory, or a run_id from "
            f"'sloth runs list --runs-root {root}'."
        ),
    )
