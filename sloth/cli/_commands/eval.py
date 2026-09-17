"""``sloth eval`` — score an adapter or a merged/quantized model on an eval suite.

Evaluates against one or more **named** eval suites — each ``--suite`` flag names
one suite, whose name is derived from the given path's stem (sanitised via
:func:`~sloth.tune.metrics.sanitize_suite_name`). ``--suite`` accepts a single
file or a directory — a directory is expanded to its sorted ``*.jsonl``
children — and is repeatable (``--suite a.jsonl --suite dir/``). Two ``--suite``
entries whose stems sanitise to the same name is a collision and exits ``1``
with a ``hint:`` before anything else runs.

Every resolved file is schema-detected (chat / task / instruction / structured /
toolcall — see :func:`~sloth.tune.datasets.detect_schema`) and validated
*before any container is launched* (see :func:`_resolve_named_suites`), so a
malformed suite costs no docker/GPU spend. When a training dataset can be
found (``--train-dataset``, or ``dataset.path`` from the adapter/model's
``training_metadata.json``), a train/eval overlap check
(:func:`~sloth.tune.datasets.overlap_check`) also runs on the host, before any
container launch — an overlap exits ``1`` naming every duplicated
``file:line``. All inference is local and offline.

**Two mutually-exclusive targets.** ``--adapter DIR`` scores a LoRA/QLoRA adapter
against its base model (via :func:`~sloth.tune._trainer.run_eval`, which wraps the
base with ``PeftModel.from_pretrained(base, adapter)``). ``--model DIR`` scores a
merged / quantized (awq, nvfp4) or GGUF directory produced by ``sloth export``
(via :func:`~sloth.tune._exporter.run_eval_model`). Passing both — or neither —
is a user error (exit 1 with a ``hint:``). Both report the same score fields;
``--model`` adds ``model_dir``, ``quant_method`` and ``quant_format``.

This module is **ML-free** — it imports no torch, peft, or transformers at module
level or anywhere.  The heavy ML work lives entirely behind those two seams,
which lazy-import the ML stack inside their bodies.  This mirrors how ``train.py``
delegates to :func:`~sloth.tune._trainer.run_training`.

**Host vs in-container routing**

On the host (no ``--in-container`` flag) :func:`cmd_eval` resolves the target and
every named suite, runs the overlap check, then hands off GPU/ML work to the NGC
container via :func:`sloth.tune.container.launch`, forwarding all original args
(including ``--perplexity`` / ``--tool-call-family`` when given) plus
``--in-container`` to prevent docker recursion.  Identity bind-mounts are added
for the parent directories of the target and suite so their host-absolute paths
resolve unchanged inside the container; a ``--model`` run additionally takes
:func:`sloth.tune.container.export_launch_kwargs` (the llama.cpp cache mount plus
``HOME``/``UNSLOTH_LLAMA_TAG``), because a GGUF directory is scored with
``llama-completion`` out of that cache.

**Output shape.** Results are always reported keyed by suite name:
``{"suites": {<name>: <payload>, ...}}`` in ``--json`` mode, and one text block
per suite otherwise. A single ``sloth eval --adapter X --suite A --suite B``
invocation writes ``eval/A.json`` and ``eval/B.json`` under the target
directory (see :func:`_write_named_eval_json`).

Usage::

    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    sloth eval --model exports/qwen3-4b-awq --suite data/eval.jsonl --json
    sloth eval --adapter adapters/qwen3-4b-qlora --suite a.jsonl --suite b.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune import container, datasets, metadata, metrics
from sloth.tune._exporter import run_eval_model
from sloth.tune._trainer import run_eval

#: Default generation batch size for the eval loop (forwarded to the in-container seam).
DEFAULT_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Checkout locator (repo root for container bind-mount)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the unsloth-cli checkout root by walking up from this module.

    ``sloth/cli/_commands/eval.py`` → ``parents[3]`` is the checkout root (the
    dir containing the ``sloth/`` package), which is bind-mounted inside the NGC
    container so ``python -m sloth`` resolves without an install step.
    """
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Target resolution (--adapter XOR --model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EvalTarget:
    """What is being evaluated: an adapter directory or a whole model directory.

    ``flag`` is the CLI flag it came from (``--adapter`` / ``--model``) and is what
    gets forwarded to the in-container run, so the container evaluates the same
    kind of thing the host was asked about.
    """

    flag: str
    path: Path

    @property
    def kind(self) -> str:
        """``"adapter"`` or ``"model"`` — the flag without its dashes."""
        return self.flag.lstrip("-")

    @property
    def directory(self) -> Path:
        """The directory results are written under (the parent, for a bare .gguf file)."""
        return self.path if self.path.is_dir() else self.path.parent


def _resolve_target(args: argparse.Namespace) -> _EvalTarget:
    """Return the single evaluation target, or raise ``CliError(code=1)``.

    ``--adapter`` and ``--model`` are mutually exclusive **and** one of them is
    required: passing both or neither is a user error with a ``hint:``. The check
    lives here rather than in an argparse mutually-exclusive group so that calling
    :func:`cmd_eval` directly (as an agent or a test does) gets the same contract.
    """
    adapter = getattr(args, "adapter", None)
    model = getattr(args, "model", None)
    if adapter and model:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--adapter and --model are mutually exclusive; pass exactly one",
            remediation=(
                "Use --adapter <dir> to score a LoRA/QLoRA adapter against its base "
                "model, or --model <dir> to score a merged / quantized (awq, nvfp4) "
                "or GGUF directory produced by `sloth export`."
            ),
        )
    if not adapter and not model:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="one of --adapter or --model is required",
            remediation=(
                "Pass --adapter <dir> to score a LoRA/QLoRA adapter, or --model <dir> "
                "to score a merged / quantized (awq, nvfp4) or GGUF directory produced "
                "by `sloth export`."
            ),
        )

    flag, value = ("--adapter", adapter) if adapter else ("--model", model)
    path = Path(value)
    # --model also accepts a single .gguf file (disambiguates multi-quant export dirs).
    gguf_file = flag == "--model" and path.is_file() and path.suffix == ".gguf"
    if not path.is_dir() and not gguf_file:
        remediation = (
            "Pass an existing adapter directory with --adapter <path>. "
            "Run `sloth train` to produce an adapter."
            if adapter
            else (
                "Pass an existing model directory with --model <path>. "
                "Run `sloth export` to produce one."
            )
        )
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{'adapter' if adapter else 'model'} directory not found: {path}",
            remediation=remediation,
        )
    return _EvalTarget(flag=flag, path=path)


# ---------------------------------------------------------------------------
# Suite resolution + validation (host AND in-container: both call this before
# doing anything expensive)
# ---------------------------------------------------------------------------


def _peek_first_record(path: Path) -> Any:
    """Return the first non-blank line of *path*, parsed as JSON, or ``None``.

    ``None`` covers an empty/blank-only file, an unparsable first line, or a
    first record that isn't a JSON object — all of which make the schema
    undetectable from this file alone.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = _json.loads(line)
                except _json.JSONDecodeError:
                    return None
                return record if isinstance(record, dict) else None
    except OSError:
        return None
    return None


def _detect_file_schema(path: Path) -> str | None:
    """Guess *path*'s schema from its first record.

    Reuses :func:`sloth.tune.datasets.detect_schema` for the chat case, and
    extends it — from the same module's exported key-set constants — to the
    other four schemas (task / instruction / structured / toolcall), without
    needing any change to ``sloth/tune/datasets.py`` itself. An instruction
    record without a ``"constraints"`` key is indistinguishable from a plain
    task record by key set alone, and is reported as ``"task"`` (both
    validators accept it identically in that case). Returns ``None`` when no
    known schema matches.
    """
    record = _peek_first_record(path)
    if record is None:
        return None
    if datasets.detect_schema(record) == "chat":
        return "chat"
    if not isinstance(record, dict):
        return None
    keys = set(record.keys())
    if keys == datasets.TASK_KEYS:
        return "task"
    if keys == datasets.INSTRUCTION_KEYS:
        return "instruction"
    if keys == datasets.STRUCTURED_KEYS:
        return "structured"
    if keys == datasets.TOOLCALL_KEYS:
        return "toolcall"
    return None


def _validate_suite_file(path: Path) -> None:
    """Detect *path*'s schema and validate every record against it.

    Raises
    ------
    CliError(code=1)
        When the schema cannot be detected, or a record fails validation
        (the message is prefixed with ``<path>:`` so a multi-file suite names
        both the offending file and the line within it).
    """
    schema = _detect_file_schema(path)
    if schema is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path}: could not detect a known eval schema from the first record",
            remediation=(
                "Each suite file must be chat, task, instruction, structured, or "
                "toolcall schema — see `sloth explain eval`."
            ),
        )
    try:
        datasets.validate_dataset(path, schema)
    except CliError as exc:
        raise CliError(
            code=exc.code,
            message=f"{path}: {exc.message}",
            remediation=exc.remediation,
        ) from exc


def _resolve_named_suites(raw_entries: list[str]) -> dict[str, list[Path]]:
    """Resolve, name, and validate every ``--suite`` entry, in order given.

    Each entry may be a single ``.jsonl`` file or a directory (expanded to its
    sorted ``*.jsonl`` children by :func:`~sloth.tune.datasets.resolve_suite_paths`).
    The suite's name is its own path stem, sanitised via
    :func:`~sloth.tune.metrics.sanitize_suite_name` — two entries that sanitise
    to the same name is a collision and raises before any file is even resolved.
    Every resolved file is schema-detected and validated (see
    :func:`_validate_suite_file`) before any container launch or ML seam call.

    Returns
    -------
    dict[str, list[Path]]
        ``{suite_name: [resolved_file, ...]}``, insertion-ordered to match
        *raw_entries*.

    Raises
    ------
    CliError(code=1)
        On a name collision, a missing/empty suite path, or the first
        invalid/undetectable file.
    """
    named: dict[str, list[Path]] = {}
    sources: dict[str, str] = {}
    for raw in raw_entries:
        name = metrics.sanitize_suite_name(raw)
        if name in named:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f"--suite name collision: {sources[name]!r} and {raw!r} "
                    f"both resolve to suite name {name!r}"
                ),
                remediation=(
                    "Rename one of the suite files/directories so their stems "
                    "differ — the suite name is derived from the path stem."
                ),
            )
        files = datasets.resolve_suite_paths(raw)
        for file_path in files:
            _validate_suite_file(file_path)
        named[name] = files
        sources[name] = raw
    return named


def _validate_batch_size(batch_size: int) -> None:
    """Raise ``CliError(code=1)`` unless *batch_size* is a positive integer.

    Called before suite validation or any container launch so a bogus
    ``--batch-size`` (0 or negative) never reaches the ML seam or spends a
    docker/GPU cycle. ``1`` is a legitimate value — it is the explicit
    unbatched (one prompt per ``generate()`` call) mode.
    """
    if batch_size < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--batch-size must be >= 1, got {batch_size}",
            remediation=(
                "Pass a positive integer with --batch-size (1 for the explicit "
                "unbatched mode, the default is "
                f"{DEFAULT_BATCH_SIZE})."
            ),
        )


def _render_suite_label(suite_paths: list[Path]) -> str:
    """A short display label for the resolved suite: the single path, or a count."""
    if len(suite_paths) == 1:
        return str(suite_paths[0])
    return f"{len(suite_paths)} files"


# ---------------------------------------------------------------------------
# Train/eval overlap check (host, before any container launch)
# ---------------------------------------------------------------------------


def _resolve_train_dataset(target_dir: Path, train_dataset_arg: str | None) -> Path | None:
    """Return the training dataset path to overlap-check against, or ``None``.

    ``--train-dataset`` (when given) always wins. Otherwise, when
    ``<target_dir>/training_metadata.json`` exists, its ``dataset.path`` field
    (written by :func:`sloth.tune.metadata.write_metadata`) is used. Returns
    ``None`` when neither source is available — the caller skips the check and
    emits a diagnostic rather than failing.
    """
    if train_dataset_arg:
        return Path(train_dataset_arg)
    try:
        record = metadata.read_metadata(target_dir)
    except CliError:
        return None
    dataset_path = record.get("dataset", {}).get("path")
    return Path(dataset_path) if dataset_path else None


def _check_train_eval_overlap(
    target_dir: Path,
    train_dataset_arg: str | None,
    suite_paths: list[Path],
) -> None:
    """Run the cross-schema train/eval overlap check, before any container launch.

    Raises ``CliError(code=1)`` with every duplicated ``file:line`` location
    named in the hint when an overlap is found. When no training dataset can be
    resolved (neither ``--train-dataset`` nor a readable
    ``training_metadata.json``), the check is skipped and a diagnostic is
    emitted on stderr instead of failing.
    """
    train_dataset = _resolve_train_dataset(target_dir, train_dataset_arg)
    if train_dataset is None:
        emit_diagnostic(
            "note: no --train-dataset given and no training_metadata.json found; "
            "skipping train/eval overlap check"
        )
        return
    overlaps = datasets.overlap_check(train_dataset, suite_paths)
    if overlaps:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"train/eval overlap detected between {train_dataset} and the eval "
                f"suite(s) at {len(overlaps)} location(s)"
            ),
            remediation=(
                "Remove or replace the overlapping row(s), or use a different eval "
                "suite. Overlapping locations: " + "; ".join(overlaps)
            ),
        )


# ---------------------------------------------------------------------------
# Container routing
# ---------------------------------------------------------------------------


def _merge_mounts(
    own: list[tuple[str, str]], extra: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Merge two mount lists, deduplicating by container target (first wins)."""
    merged: list[tuple[str, str]] = []
    seen: set[str] = set()
    for host, target in list(own) + list(extra):
        if target in seen:
            continue
        seen.add(target)
        merged.append((host, target))
    return merged


def _needs_llama_cpp(path: Path) -> bool:
    """True when the --model target is a GGUF file or a directory holding one."""
    if path.is_file():
        return path.suffix == ".gguf"
    return any(path.glob("*.gguf"))


def _launch_container(
    target: _EvalTarget,
    suite_paths: list[Path],
    *,
    quant: str | None,
    batch_size: int,
    perplexity: bool = False,
    tool_call_family: str | None = None,
) -> dict[str, Any]:
    """Re-run this eval inside the NGC container with ``--in-container``.

    Every resolved suite file is forwarded as its own ``--suite <abs path>`` flag
    (a directory is already expanded to individual files by
    :func:`_resolve_named_suites` before this is called), matching the
    in-container argv contract: ``eval --in-container --json --suite <p> [--suite
    <p> ...] [--quant q] [--batch-size n] [--perplexity] [--tool-call-family f]``
    plus ``--adapter``/``--model``. ``--json`` is always forwarded to the
    container (unconditionally, regardless of the host's own ``--json`` flag) so
    the container always prints a structured result line for
    :func:`~sloth.tune.container.launch` to parse and return; the caller
    re-renders that dict for the host's own ``--json`` flag.

    Identity mounts (``host == container``) for the target's and every suite file's
    parent dirs make the host-absolute paths in *sloth_args* resolve unchanged
    inside the container; ``sorted`` keeps the docker argv deterministic. A
    ``--model`` run also takes everything :func:`container.export_launch_kwargs`
    contributes (the llama.cpp cache mount plus the ``HOME`` / ``UNSLOTH_LLAMA_TAG``
    env), because a GGUF directory is scored with ``llama-completion`` from that
    cache — exactly how ``export.py`` sets up its own run.

    Returns the dict :func:`~sloth.tune.container.launch` returns (the parsed
    JSON result line the container printed); raises :class:`CliError` on any
    container failure.
    """
    target_abs = target.path.resolve()
    suite_abs = [p.resolve() for p in suite_paths]
    sloth_args = ["eval", target.flag, str(target_abs)]
    for suite_file in suite_abs:
        sloth_args += ["--suite", str(suite_file)]
    if quant:
        sloth_args += ["--quant", str(quant)]
    sloth_args += ["--batch-size", str(batch_size)]
    if perplexity:
        sloth_args.append("--perplexity")
    if tool_call_family:
        sloth_args += ["--tool-call-family", str(tool_call_family)]
    sloth_args.append("--json")
    sloth_args.append("--in-container")

    own_mounts = [
        (str(p), str(p)) for p in sorted({target_abs.parent, *(s.parent for s in suite_abs)})
    ]
    kwargs: dict[str, Any] = {
        "workdir": str(target_abs.parent),
        "checkout": str(_repo_root()),
    }
    supplied: dict[str, Any] = {}
    if target.flag == "--model" and _needs_llama_cpp(target.path):
        # Only GGUF scoring needs the llama.cpp cache (and its writable export home);
        # transformers-loadable dirs (bf16 / awq / nvfp4) run without it.
        supplied = getattr(container, "export_launch_kwargs", lambda: {})() or {}
    kwargs["extra_mounts"] = _merge_mounts(own_mounts, list(supplied.get("extra_mounts") or []))
    for key, value in supplied.items():
        if key != "extra_mounts":
            kwargs[key] = value
    return container.launch(sloth_args, **kwargs)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_text(target: _EvalTarget, suite_label: str, summary: dict[str, Any]) -> str:
    """Render one suite's eval summary for stdout in text mode."""
    lines = [
        f"eval suite: {suite_label}",
        f"{target.kind + ':':<11} {target.path}",
    ]
    if target.flag == "--model":
        lines.append(f"quant:      {summary.get('quant_method') or 'none (bf16/gguf)'}")
        lines.append(f"format:     {summary.get('quant_format') or 'none (bf16/gguf)'}")
    lines += [
        f"total:      {summary['total']}",
        f"exact:      {summary['exact_match']}",
        f"score:      {summary['exact_match_pct']}%",
    ]
    for r in summary.get("results", []):
        mark = "[ok]" if r["exact_match"] else "[fail]"
        lines.append(f"  {mark} #{r['index']} {r['task']!r}")
    return "\n".join(lines)


def _render_named_report_text(
    target: _EvalTarget,
    named_suites: dict[str, list[Path]],
    named_results: dict[str, dict[str, Any]],
) -> str:
    """Render one text block per suite, in the order *named_results* provides."""
    blocks = []
    for name, payload in named_results.items():
        suite_label = _render_suite_label(named_suites.get(name, []))
        blocks.append(f"suite: {name}\n" + _render_text(target, suite_label, payload))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# ML-seam call — bridges the CURRENT run_eval/run_eval_model signature and the
# richer one t6 (a sibling task) is adding
# ---------------------------------------------------------------------------


def _call_eval_seam(
    func: Any,
    target_path: Path,
    suite_paths: list[Path],
    *,
    quant: str | None,
    batch_size: int,
    perplexity: bool = False,
    tool_call_family: str | None = None,
) -> dict[str, Any]:
    """Call *func* (``run_eval`` or ``run_eval_model``) with the full suite contract.

    Both seams accept ``suite_paths`` (every resolved file, scored per file and
    in aggregate), ``quant`` (GGUF selector; ignored by the adapter path), and
    ``batch_size`` as keyword arguments. ``perplexity`` and ``tool_call_family``
    are forwarded the same way, but only when *func*'s signature actually
    accepts them (via :func:`inspect.signature`) — this shim tolerates the
    older seam signature (without those two keywords) that ships until a
    sibling task adds them, and any signature carrying ``**kwargs``.
    """
    candidate_kwargs: dict[str, Any] = {
        "suite_paths": suite_paths,
        "quant": quant,
        "batch_size": batch_size,
        "perplexity": perplexity,
        "tool_call_family": tool_call_family,
    }
    try:
        parameters = inspect.signature(func).parameters
        accepts_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
    except (TypeError, ValueError):
        parameters = {}
        accepts_var_keyword = True

    if accepts_var_keyword:
        kwargs = candidate_kwargs
    else:
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in parameters}

    return func(str(target_path), **kwargs)


def _normalize_named_results(
    raw_result: dict[str, Any],
    suite_names: list[str],
    named_suites: dict[str, list[Path]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize a seam/launch return value into ``{suite_name: payload}``.

    Three input shapes are accepted, in order of preference:

    * already suite-keyed (``{"suites": {...}}`` — this module's own in-container
      JSON emission): the inner mapping is returned as-is;
    * the seams' aggregate shape carrying per-file entries (``{"files": [{"path",
      ...}, ...], ...}`` — what :func:`sloth.tune._trainer.run_eval` and
      :func:`sloth.tune._exporter.run_eval_model` return): each named suite gets
      **its own** files' scores — the single file entry when the suite named one
      file, or :func:`metrics.aggregate` over its files when it named a
      directory — so a multi-suite run never reports one suite's numbers under
      another suite's name. Run-level fields the file entries lack
      (``batch_size``, ``base_load_in_4bit``, ``model_dir``, ``quant_*``) are
      copied onto every suite payload;
    * a flat payload with no ``files`` (a legacy/test double): duplicated under
      every name in *suite_names* — a faithful, if not per-suite-precise, fallback.
    """
    if isinstance(raw_result, dict) and "suites" in raw_result:
        return dict(raw_result["suites"])
    files = raw_result.get("files") if isinstance(raw_result, dict) else None
    if not (isinstance(files, list) and files and named_suites):
        return {name: raw_result for name in suite_names}

    by_path: dict[str, dict[str, Any]] = {}
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if isinstance(path, str):
            by_path[str(Path(path).resolve())] = entry
    run_level = {
        key: value
        for key, value in raw_result.items()
        if key in _RUN_LEVEL_KEYS and key not in ("results", "files")
    }
    named: dict[str, dict[str, Any]] = {}
    for name in suite_names:
        entries = [
            by_path[str(p.resolve())]
            for p in named_suites.get(name, [])
            if str(p.resolve()) in by_path
        ]
        if not entries:
            named[name] = raw_result
            continue
        payload = dict(entries[0]) if len(entries) == 1 else metrics.aggregate(entries)
        for key, value in run_level.items():
            payload.setdefault(key, value)
        named[name] = payload
    return named


#: Run-level keys the seams set once per invocation (not per file) that every
#: per-suite payload should still carry.
_RUN_LEVEL_KEYS = frozenset(
    {"batch_size", "base_load_in_4bit", "model_dir", "quant_method", "quant_format", "target"}
)


def _write_named_eval_json(
    target_dir: Path,
    named_results: dict[str, dict[str, Any]],
    *,
    batch_size: int,
    base_load_in_4bit: bool | None = None,
) -> None:
    """Write ``<target_dir>/eval/<name>.json`` for every suite in *named_results*.

    Mirrors :func:`sloth.tune._trainer.write_eval_json`'s failure handling: an
    :class:`OSError` (e.g. a read-only target directory) becomes a stderr
    diagnostic rather than losing an otherwise-complete eval run.
    """
    for name, payload in named_results.items():
        try:
            metrics.write_eval_json(
                target_dir,
                name,
                payload,
                batch_size=batch_size,
                base_load_in_4bit=base_load_in_4bit,
            )
        except OSError as exc:
            emit_diagnostic(f"note: could not write eval/{name}.json: {exc}")


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth eval``.

    On the **host** (``--in-container`` not set): resolves the single target
    (``--adapter`` XOR ``--model``) and every named suite, runs the train/eval
    overlap check, then delegates GPU/ML work to the NGC container via
    :func:`sloth.tune.container.launch` (forwarding all args plus
    ``--in-container`` to prevent recursion).  Identity bind-mounts are added
    for the parent directories of the target and suite paths so the
    host-absolute paths forwarded in sloth_args resolve unchanged inside the
    container (see :func:`_launch_container`).
    Returns ``None`` on success (implicit fall-through); :class:`CliError` is
    raised (and propagated) on any container failure — ``launch()`` raises rather
    than returning a non-zero int.

    **Inside the container** (``--in-container`` is set): validates inputs, calls
    the ML seam for the resolved target — :func:`~sloth.tune._trainer.run_eval`
    for ``--adapter``, :func:`~sloth.tune._exporter.run_eval_model` for
    ``--model`` — writes ``eval/<suite>.json`` for every named suite, and emits
    results (keyed by suite name) via the output contract.

    Parameters
    ----------
    args:
        Parsed namespace with ``adapter``, ``model``, ``suite`` (a list — one
        entry per ``--suite`` flag; a bare string is also accepted for direct
        callers), ``quant``, ``batch_size``, ``perplexity``, ``tool_call_family``,
        ``train_dataset``, ``json``, and ``in_container`` attributes.

    Returns
    -------
    int | None
        ``None`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))
    quant = getattr(args, "quant", None)
    perplexity = bool(getattr(args, "perplexity", False))
    tool_call_family = getattr(args, "tool_call_family", None) or None
    train_dataset_arg = getattr(args, "train_dataset", None)
    _raw_batch_size = getattr(args, "batch_size", None)
    batch_size = DEFAULT_BATCH_SIZE if _raw_batch_size is None else int(_raw_batch_size)

    # --- validate --batch-size BEFORE suite validation or container launch ---
    _validate_batch_size(batch_size)

    # --- resolve the target: exactly one of --adapter / --model --------------
    target = _resolve_target(args)

    # --- resolve, name, and validate every suite ------------------------------
    # A directory is expanded to its sorted *.jsonl children and every file is
    # schema-detected and validated BEFORE any container launch — a malformed
    # suite anywhere in a directory costs no docker/GPU spend (fails fast,
    # names the file and line). ``args.suite`` is normally a list (argparse
    # ``action="append"``); a bare string is accepted too so direct callers
    # (and existing single-suite tests) keep working unchanged.
    raw_suite = args.suite
    if isinstance(raw_suite, str):
        raw_suite = [raw_suite]
    named_suites = _resolve_named_suites(list(raw_suite))
    suite_names = list(named_suites.keys())
    suite_paths = [p for files in named_suites.values() for p in files]

    # --- HOST PATH: overlap check, then route GPU/ML work through the NGC container
    if not in_container:
        # Runs before any docker invocation — an overlap is a user error that
        # should never cost a container launch.
        _check_train_eval_overlap(target.directory, train_dataset_arg, suite_paths)

        # launch() raises CliError on any container/docker failure and otherwise
        # returns the parsed JSON result the container printed — already
        # suite-keyed, since the container's own cmd_eval always runs --json.
        # normalize() tolerates a flat/legacy dict too (e.g. a test double).
        raw_result = _launch_container(
            target,
            [p.resolve() for p in suite_paths],
            quant=quant,
            batch_size=batch_size,
            perplexity=perplexity,
            tool_call_family=tool_call_family,
        )
        named_results = _normalize_named_results(raw_result, suite_names, named_suites)
        if json_mode:
            emit_result({"suites": named_results}, json_mode=True)
        else:
            emit_result(
                _render_named_report_text(target, named_suites, named_results), json_mode=False
            )
        return

    # --- IN-CONTAINER PATH: delegate to the ML seam -------------------------
    # Both seams lazy-import torch/transformers/peft inside their bodies; this
    # module stays ML-free. Every suite file was already validated above.
    if target.flag == "--model":
        raw_result = _call_eval_seam(
            run_eval_model,
            target.path,
            suite_paths,
            quant=quant,
            batch_size=batch_size,
            perplexity=perplexity,
            tool_call_family=tool_call_family,
        )
    else:
        raw_result = _call_eval_seam(
            run_eval,
            target.path,
            suite_paths,
            quant=quant,
            batch_size=batch_size,
            perplexity=perplexity,
            tool_call_family=tool_call_family,
        )

    named_results = _normalize_named_results(raw_result, suite_names, named_suites)
    _write_named_eval_json(
        target.directory,
        named_results,
        batch_size=batch_size,
        base_load_in_4bit=(
            raw_result.get("base_load_in_4bit") if isinstance(raw_result, dict) else None
        ),
    )

    # --- emit results (always suite-keyed) ------------------------------------
    if json_mode:
        emit_result({"suites": named_results}, json_mode=True)
    else:
        emit_result(_render_named_report_text(target, named_suites, named_results), json_mode=False)


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``eval`` subparser on *sub*."""
    p = sub.add_parser(
        "eval",
        help=("Run a LoRA/QLoRA adapter against a local task-schema eval suite (offline)."),
    )
    p.add_argument(
        "--adapter",
        help="Path to the adapter directory produced by ``sloth train``.",
    )
    p.add_argument(
        "--model",
        help=(
            "Path to a merged / quantized (awq, nvfp4) or GGUF model directory "
            "produced by ``sloth export``. Mutually exclusive with --adapter."
        ),
    )
    p.add_argument(
        "--suite",
        required=True,
        action="append",
        metavar="PATH",
        help=(
            "Path to a JSONL eval suite (chat/task/instruction/structured/toolcall "
            "schema, auto-detected), or a directory of them (every *.jsonl child is "
            "scored). Repeatable: pass --suite more than once to score several named "
            "suites in one invocation (name = the path's stem; a stem collision "
            "between two --suite entries exits 1)."
        ),
    )
    p.add_argument(
        "--quant",
        default=None,
        metavar="NAME",
        help=(
            "When --model holds several GGUF files, the quant tag to score "
            "(case-insensitive, e.g. q4_k_m); ignored for a single-GGUF or "
            "non-GGUF --model directory, and for --adapter."
        ),
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Generation batch size for the eval loop (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--perplexity",
        action="store_true",
        help="Also compute held-out perplexity/loss during eval (forwarded in-container).",
    )
    p.add_argument(
        "--tool-call-family",
        dest="tool_call_family",
        default=None,
        metavar="FAMILY",
        help=(
            "Tool-call parsing family to score toolcall-schema suites against "
            "(forwarded in-container); ignored when no toolcall suite is present."
        ),
    )
    p.add_argument(
        "--train-dataset",
        dest="train_dataset",
        default=None,
        metavar="PATH",
        help=(
            "Training dataset to check every --suite against for train/eval overlap "
            "before any container launch. Defaults to the dataset recorded in the "
            "target's training_metadata.json when present; skipped (with a stderr "
            "diagnostic) when neither is available."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_eval)
