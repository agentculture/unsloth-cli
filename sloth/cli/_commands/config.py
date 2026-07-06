"""``sloth config init`` — write a starting run.toml.

Generates a minimal ``run.toml`` with the ``[run]`` section (model, dataset,
output, method) and a ``[hyperparameters]`` section populated with the
documented defaults from :mod:`sloth.tune.config`.  The generated file always
passes :func:`~sloth.tune.config.load_config` validation.

Usage::

    sloth config init --model unsloth/Qwen3-4B --dataset data/train.jsonl --output adapters/out
    sloth config init --model unsloth/Qwen3-4B --dataset data/train.jsonl --output adapters/out --method lora
    sloth config init --model unsloth/Qwen3-4B --dataset data/train.jsonl --output adapters/out --json

Exit codes:
    0 — config written (report emitted to stdout)
    1 — user-input error (bad args, file exists without --force)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sloth.cli._errors import CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_STEPS,
    DEFAULT_METHOD,
    DEFAULT_SEED,
)

# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_config_init(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth config init``.

    Writes a starting ``run.toml`` to ``<output>/run.toml`` (or the path given
    by ``--path``) with validated defaults.  Refuses to overwrite an existing
    file unless ``--force`` is set.

    Returns ``None`` (exit 0) on success; raises :class:`CliError` on failure.
    """
    json_mode = bool(getattr(args, "json", False))
    model = args.model
    dataset = args.dataset
    output = args.output
    method = args.method or DEFAULT_METHOD
    force = args.force
    path = Path(args.path) if args.path else Path(output) / "run.toml"

    # --- refuse to overwrite without --force --------------------------------
    if path.is_file() and not force:
        raise CliError(
            code=1,
            message=f"config file already exists: {path}",
            remediation="Use --force to overwrite, or pass a different --output / --path.",
        )

    # --- build TOML ----------------------------------------------------------
    toml_lines = [
        "[run]",
        f'model   = "{model}"',
        f'method  = "{method}"',
        f'dataset = "{dataset}"',
        f'output  = "{output}"',
        "",
        "[hyperparameters]",
        f"lora_r         = {DEFAULT_LORA_R}",
        f"lora_alpha     = {DEFAULT_LORA_ALPHA}",
        f"lora_dropout   = {DEFAULT_LORA_DROPOUT}",
        f"learning_rate  = {DEFAULT_LEARNING_RATE}",
        f"max_seq_len    = {DEFAULT_MAX_SEQ_LEN}",
        f"batch_size     = {DEFAULT_BATCH_SIZE}",
        f"grad_accum     = {DEFAULT_GRAD_ACCUM}",
        f"max_steps      = {DEFAULT_MAX_STEPS}",
        f"seed           = {DEFAULT_SEED}",
        f"load_in_4bit   = {str(DEFAULT_LOAD_IN_4BIT).lower()}",
    ]
    toml_text = "\n".join(toml_lines) + "\n"

    # --- write file ----------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml_text, encoding="utf-8")
    emit_diagnostic(f"wrote {path}")

    # --- emit report --------------------------------------------------------
    report: dict[str, Any] = {
        "config": str(path),
        "model": model,
        "method": method,
        "dataset": dataset,
        "output": output,
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [
            f"config:  {path}",
            f"model:   {model}",
            f"method:  {method}",
            f"dataset: {dataset}",
            f"output:  {output}",
            "status:  written",
        ]
        emit_result("\n".join(lines), json_mode=False)

    return None


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``config`` subparser on *sub*."""
    p = sub.add_parser(
        "config",
        help="Configuration management subcommands.",
        description="Configuration management subcommands (e.g. init).",
    )
    sub_config = p.add_subparsers(dest="config_command")

    # --- config init --------------------------------------------------------
    p_init = sub_config.add_parser(
        "init",
        help="Write a starting run.toml with validated defaults.",
        description=(
            "Write a starting run.toml with validated defaults. "
            "The generated file always passes sloth.tune.config.load_config validation."
        ),
    )
    p_init.add_argument(
        "--model",
        required=True,
        metavar="ID",
        help="Model identifier (e.g. unsloth/Qwen3-4B).",
    )
    p_init.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to the JSONL dataset file.",
    )
    p_init.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for the adapter.",
    )
    p_init.add_argument(
        "--method",
        choices=["lora", "qlora"],
        default=None,
        help=f"Adapter method (default: {DEFAULT_METHOD!r}).",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing run.toml.",
    )
    p_init.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help=("Override the output path for run.toml " "(default: <output>/run.toml)."),
    )
    p_init.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p_init.set_defaults(func=cmd_config_init)
