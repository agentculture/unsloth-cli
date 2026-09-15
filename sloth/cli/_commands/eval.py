"""``sloth eval`` — score an adapter or a merged/quantized model on an eval suite.

Evaluates against one or more JSONL files whose records conform to the **task**
schema (``{"task": …, "input": …, "expected_output": …}``).  ``--suite`` accepts a
single file or a directory — a directory is expanded to its sorted ``*.jsonl``
children — and is repeatable (``--suite a.jsonl --suite dir/``).  Every resolved
file is validated against the task schema *before any container is launched*
(see :func:`_resolve_and_validate_suite`), so a malformed suite costs no
docker/GPU spend.  All inference is local and offline.

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
validates the suite file, then hands off GPU/ML work to the NGC container via
:func:`sloth.tune.container.launch`, forwarding all original args plus
``--in-container`` to prevent docker recursion.  Identity bind-mounts are added
for the parent directories of the target and suite so their host-absolute paths
resolve unchanged inside the container; a ``--model`` run additionally takes
:func:`sloth.tune.container.export_launch_kwargs` (the llama.cpp cache mount plus
``HOME``/``UNSLOTH_LLAMA_TAG``), because a GGUF directory is scored with
``llama-completion`` out of that cache.

Usage::

    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    sloth eval --model exports/qwen3-4b-awq --suite data/eval.jsonl --json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune import container
from sloth.tune._exporter import run_eval_model
from sloth.tune._trainer import run_eval
from sloth.tune.datasets import validate_suite

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


def _resolve_and_validate_suite(raw_entries: list[str]) -> list[Path]:
    """Expand and task-schema-validate every ``--suite`` entry, in order given.

    Each entry may be a single ``.jsonl`` file or a directory (expanded to its
    sorted ``*.jsonl`` children by :func:`~sloth.tune.datasets.validate_suite`).
    Validation happens here — before any container launch or ML seam call — so
    a malformed suite (anywhere in a directory) costs no docker/GPU spend and
    names the offending file *and* line.

    Raises
    ------
    CliError(code=1)
        On the first invalid file/line, or when an entry resolves to no files.
    """
    resolved: list[Path] = []
    for raw in raw_entries:
        report = validate_suite(raw, schema="task")
        resolved.extend(Path(str(f["path"])) for f in report["files"])
    return resolved


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
) -> dict[str, Any]:
    """Re-run this eval inside the NGC container with ``--in-container``.

    Every resolved suite file is forwarded as its own ``--suite <abs path>`` flag
    (a directory is already expanded to individual files by
    :func:`_resolve_and_validate_suite` before this is called), matching the
    in-container argv contract: ``eval --in-container --json --suite <p> [--suite
    <p> ...] [--quant q] [--batch-size n]`` plus ``--adapter``/``--model``.
    ``--json`` is always forwarded to the container (unconditionally, regardless
    of the host's own ``--json`` flag) so the container always prints a
    structured result line for :func:`~sloth.tune.container.launch` to parse and
    return; the caller re-renders that dict for the host's own ``--json`` flag.

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
    """Render the eval summary for stdout in text mode."""
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


# ---------------------------------------------------------------------------
# ML-seam call — bridges the CURRENT run_eval/run_eval_model signature and the
# richer one t5 (a sibling task) is adding
# ---------------------------------------------------------------------------


def _call_eval_seam(
    func: Any,
    target_path: Path,
    suite_paths: list[Path],
    *,
    quant: str | None,
    batch_size: int,
) -> dict[str, Any]:
    """Call *func* (``run_eval`` or ``run_eval_model``) with the full suite contract.

    Both seams accept ``suite_paths`` (every resolved file, scored per file and in
    aggregate), ``quant`` (GGUF selector; ignored by the adapter path) and
    ``batch_size`` as keyword arguments.
    """
    return func(str(target_path), suite_paths=suite_paths, quant=quant, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth eval``.

    On the **host** (``--in-container`` not set): resolves the single target
    (``--adapter`` XOR ``--model``) and the suite file, then delegates GPU/ML work
    to the NGC container via :func:`sloth.tune.container.launch` (forwarding all
    args plus ``--in-container`` to prevent recursion).  Identity bind-mounts are
    added for the parent directories of the target and suite paths so the
    host-absolute paths forwarded in sloth_args resolve unchanged inside the
    container (see :func:`_launch_container`).
    Returns ``None`` on success (implicit fall-through); :class:`CliError` is
    raised (and propagated) on any container failure — ``launch()`` raises rather
    than returning a non-zero int.

    **Inside the container** (``--in-container`` is set): validates inputs, calls
    the ML seam for the resolved target — :func:`~sloth.tune._trainer.run_eval`
    for ``--adapter``, :func:`~sloth.tune._exporter.run_eval_model` for
    ``--model`` — and emits results via the output contract.

    Parameters
    ----------
    args:
        Parsed namespace with ``adapter``, ``model``, ``suite`` (a list — one
        entry per ``--suite`` flag; a bare string is also accepted for direct
        callers), ``quant``, ``batch_size``, ``json``, and ``in_container``
        attributes.

    Returns
    -------
    int | None
        ``None`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))
    quant = getattr(args, "quant", None)
    _raw_batch_size = getattr(args, "batch_size", None)
    batch_size = DEFAULT_BATCH_SIZE if _raw_batch_size is None else int(_raw_batch_size)

    # --- validate --batch-size BEFORE suite validation or container launch ---
    _validate_batch_size(batch_size)

    # --- resolve the target: exactly one of --adapter / --model --------------
    target = _resolve_target(args)

    # --- resolve + validate the suite (file(s) or directory(ies)) -------------
    # A directory is expanded to its sorted *.jsonl children and every file is
    # task-schema-validated BEFORE any container launch — a malformed suite
    # anywhere in a directory costs no docker/GPU spend (fails fast, names the
    # file and line). ``args.suite`` is normally a list (argparse
    # ``action="append"``); a bare string is accepted too so direct callers
    # (and existing single-suite tests) keep working unchanged.
    raw_suite = args.suite
    if isinstance(raw_suite, str):
        raw_suite = [raw_suite]
    suite_paths = _resolve_and_validate_suite(list(raw_suite))

    # --- HOST PATH: route GPU/ML work through the NGC container --------------
    if not in_container:
        # launch() raises CliError on any container/docker failure and otherwise
        # returns the parsed JSON result the container printed (the same summary
        # dict the in-container branch below builds). Emit it through the HOST's
        # own output contract so a host caller sees the same shape the
        # in-container path would have printed.
        summary = _launch_container(
            target,
            [p.resolve() for p in suite_paths],
            quant=quant,
            batch_size=batch_size,
        )
        if json_mode:
            emit_result(summary, json_mode=True)
        else:
            emit_result(
                _render_text(target, _render_suite_label(suite_paths), summary), json_mode=False
            )
        return

    # --- IN-CONTAINER PATH: delegate to the ML seam -------------------------
    # Both seams lazy-import torch/transformers/peft inside their bodies; this
    # module stays ML-free. The suite was already task-schema-validated above.
    if target.flag == "--model":
        summary: dict[str, Any] = _call_eval_seam(
            run_eval_model, target.path, suite_paths, quant=quant, batch_size=batch_size
        )
    else:
        summary = _call_eval_seam(
            run_eval, target.path, suite_paths, quant=quant, batch_size=batch_size
        )

    # --- emit results --------------------------------------------------------
    if json_mode:
        emit_result(summary, json_mode=True)
    else:
        emit_result(
            _render_text(target, _render_suite_label(suite_paths), summary), json_mode=False
        )


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
            "Path to a task-schema JSONL eval suite, or a directory of them "
            "(every *.jsonl child is scored). Repeatable: pass --suite more than "
            "once to combine several files/directories."
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
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_eval)
