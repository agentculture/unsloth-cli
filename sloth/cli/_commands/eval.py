"""``sloth eval`` — run a LoRA/QLoRA adapter against a local task-schema eval suite.

Evaluates an adapter directory against a JSONL file whose records conform to the
**task** schema (``{"task": …, "input": …, "expected_output": …}``).  All
inference is local and offline.

This module is **ML-free** — it imports no torch, peft, or transformers at module
level or anywhere.  The heavy ML work (model loading, PeftModel wrapping, eval
loop) lives entirely in :func:`sloth.tune._trainer.run_eval`, which lazy-imports
the ML stack inside its body.  This mirrors how ``train.py`` delegates to
:func:`~sloth.tune._trainer.run_training`.

**Host vs in-container routing**

On the host (no ``--in-container`` flag) :func:`cmd_eval` validates the adapter
directory and suite file, then hands off GPU/ML work to the NGC container via
:func:`sloth.tune.container.launch`, forwarding all original args plus
``--in-container`` to prevent docker recursion.  Identity bind-mounts are added
for the parent directories of the adapter and suite so their host-absolute paths
resolve unchanged inside the container.

Inside the container (``--in-container`` is set) it runs the real eval by calling
:func:`~sloth.tune._trainer.run_eval`, which loads the base model, wraps it with
the LoRA adapter via ``PeftModel.from_pretrained(base, adapter)``, and scores
the suite.

Usage::

    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl --json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune import container
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
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Handler for ``sloth eval``.

    On the **host** (``--in-container`` not set): validates the adapter directory
    and suite file, then delegates GPU/ML work to the NGC container via
    :func:`sloth.tune.container.launch` (forwarding all args plus
    ``--in-container`` to prevent recursion).  Identity bind-mounts are added for
    the parent directories of the adapter and suite paths so the host-absolute
    paths forwarded in sloth_args resolve unchanged inside the container.
    Returns ``0`` on success; :class:`CliError` is raised (and propagated) on any
    container failure — ``launch()`` raises rather than returning a non-zero int.

    **Inside the container** (``--in-container`` is set): validates inputs, calls
    :func:`~sloth.tune._trainer.run_eval` (the ML seam), and emits results via the
    output contract.

    Parameters
    ----------
    args:
        Parsed namespace with ``adapter``, ``suite``, ``json``, and
        ``in_container`` attributes.

    Returns
    -------
    int
        ``0`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))

    # --- validate adapter dir ------------------------------------------------
    adapter = Path(args.adapter)
    if not adapter.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"adapter directory not found: {adapter}",
            remediation=(
                "Pass an existing adapter directory with --adapter <path>. "
                "Run `sloth train` to produce an adapter."
            ),
        )

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
        adapter_abs = adapter.resolve()
        suite_abs = suite.resolve()
        sloth_args = [
            "eval",
            "--adapter",
            str(adapter_abs),
            "--suite",
            str(suite_abs),
        ]
        if json_mode:
            sloth_args.append("--json")
        sloth_args.append("--in-container")
        # Identity mounts so the host-absolute paths in sloth_args resolve
        # unchanged inside the container without any path rewriting.
        mount_parents = {adapter_abs.parent, suite_abs.parent}
        extra_mounts = [(str(p), str(p)) for p in mount_parents]
        container.launch(
            sloth_args,
            workdir=str(adapter_abs.parent),
            checkout=str(_repo_root()),
            extra_mounts=extra_mounts,
        )
        return 0

    # --- IN-CONTAINER PATH: delegate to the ML seam -------------------------
    # run_eval lazy-imports torch/transformers/peft inside its body; this
    # module stays ML-free.
    validate_dataset(suite, schema="task")  # fast pre-check before heavy ML load
    summary: dict[str, Any] = run_eval(str(adapter), str(suite))

    # --- emit results --------------------------------------------------------
    if json_mode:
        emit_result(summary, json_mode=True)
    else:
        results = summary.get("results", [])
        lines = [
            f"eval suite: {suite}",
            f"adapter:    {adapter}",
            f"total:      {summary['total']}",
            f"exact:      {summary['exact_match']}",
            f"score:      {summary['exact_match_pct']}%",
        ]
        for r in results:
            mark = "[ok]" if r["exact_match"] else "[fail]"
            lines.append(f"  {mark} #{r['index']} {r['task']!r}")
        emit_result("\n".join(lines), json_mode=False)

    return 0


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
        required=True,
        help="Path to the adapter directory produced by ``sloth train``.",
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
