"""``sloth eval`` — score an adapter or a merged/quantized model on an eval suite.

Evaluates against a JSONL file whose records conform to the **task** schema
(``{"task": …, "input": …, "expected_output": …}``).  All inference is local and
offline.

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
from sloth.tune.datasets import validate_dataset

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
    if not path.is_dir():
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


def _launch_container(target: _EvalTarget, suite_abs: Path, *, json_mode: bool) -> None:
    """Re-run this eval inside the NGC container with ``--in-container``.

    Identity mounts (``host == container``) for the target's and the suite's parent
    dirs make the host-absolute paths in *sloth_args* resolve unchanged inside the
    container; ``sorted`` keeps the docker argv deterministic. A ``--model`` run also
    takes everything :func:`container.export_launch_kwargs` contributes (the
    llama.cpp cache mount plus the ``HOME`` / ``UNSLOTH_LLAMA_TAG`` env), because a
    GGUF directory is scored with ``llama-completion`` from that cache — exactly how
    ``export.py`` sets up its own run.
    """
    target_abs = target.path.resolve()
    sloth_args = ["eval", target.flag, str(target_abs), "--suite", str(suite_abs)]
    if json_mode:
        sloth_args.append("--json")
    sloth_args.append("--in-container")

    own_mounts = [(str(p), str(p)) for p in sorted({target_abs.parent, suite_abs.parent})]
    kwargs: dict[str, Any] = {
        "workdir": str(target_abs.parent),
        "checkout": str(_repo_root()),
    }
    supplied: dict[str, Any] = {}
    if target.flag == "--model":
        supplied = getattr(container, "export_launch_kwargs", lambda: {})() or {}
    kwargs["extra_mounts"] = _merge_mounts(own_mounts, list(supplied.get("extra_mounts") or []))
    for key, value in supplied.items():
        if key != "extra_mounts":
            kwargs[key] = value
    container.launch(sloth_args, **kwargs)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_text(target: _EvalTarget, suite: Path, summary: dict[str, Any]) -> str:
    """Render the eval summary for stdout in text mode."""
    lines = [
        f"eval suite: {suite}",
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
        Parsed namespace with ``adapter``, ``model``, ``suite``, ``json``, and
        ``in_container`` attributes.

    Returns
    -------
    int | None
        ``None`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))

    # --- resolve the target: exactly one of --adapter / --model --------------
    target = _resolve_target(args)

    # --- validate suite file -------------------------------------------------
    suite = Path(args.suite)
    if not suite.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"suite file not found: {suite}",
            remediation=(
                "Pass an existing JSONL file with --suite <path>. "
                "Each line must be a task-schema record: "
                '{"task": "…", "input": "…", "expected_output": "…"}.'
            ),
        )

    # --- HOST PATH: route GPU/ML work through the NGC container --------------
    if not in_container:
        _launch_container(target, suite.resolve(), json_mode=json_mode)
        return

    # --- IN-CONTAINER PATH: delegate to the ML seam -------------------------
    # Both seams lazy-import torch/transformers/peft inside their bodies; this
    # module stays ML-free.
    validate_dataset(suite, schema="task")  # fast pre-check before heavy ML load
    if target.flag == "--model":
        summary: dict[str, Any] = run_eval_model(str(target.path), str(suite))
    else:
        summary = run_eval(str(target.path), str(suite))

    # --- emit results --------------------------------------------------------
    if json_mode:
        emit_result(summary, json_mode=True)
    else:
        emit_result(_render_text(target, suite, summary), json_mode=False)


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
        help="Path to a task-schema JSONL eval suite.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_eval)
