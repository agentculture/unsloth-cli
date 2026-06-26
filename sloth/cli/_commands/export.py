"""``sloth export`` — export a trained adapter to a standard PEFT/safetensors layout.

Copies or normalises a LoRA/QLoRA adapter directory into the canonical PEFT
layout that the ``lobes`` server can serve and ``colleague`` can run:

    <output>/
      adapter_config.json
      adapter_model.safetensors

No torch or ML runtime is required at import time; this is a pure
stdlib file-system operation. Torch / Unsloth are never imported here.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result

# The only format supported today. Extend this set as new formats land.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"safetensors"})

# Canonical PEFT file names that must be present in the output directory.
PEFT_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]


def _export_safetensors(adapter: Path, output: Path) -> list[str]:
    """Copy standard PEFT files from *adapter* into *output*.

    If *adapter* and *output* resolve to the same directory, the copy is skipped
    and the files are reported as-is (normalise-in-place semantics).

    Returns a list of absolute string paths for the files present in *output*.
    """
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for fname in PEFT_FILES:
        src = adapter / fname
        if not src.exists():
            continue
        dst = output / fname
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        written.append(str(dst.resolve()))
    return written


def cmd_export(args: argparse.Namespace) -> int:
    """Handler for ``sloth export``."""
    json_mode = bool(getattr(args, "json", False))

    # --- Validate adapter directory ---
    adapter = Path(args.adapter)
    if not adapter.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"adapter directory not found: {adapter}",
            remediation="pass an existing adapter directory via --adapter <dir>",
        )

    # --- Require the canonical PEFT files BEFORE exporting ---
    # Without this guard a directory missing adapter_config.json /
    # adapter_model.safetensors would "export" to an empty file list and report
    # success, silently producing an unusable (un-servable, un-runnable) export.
    missing = [fname for fname in PEFT_FILES if not (adapter / fname).is_file()]
    if missing:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"adapter directory {adapter} is missing required PEFT files: {missing}",
            remediation=(
                "Point --adapter at a trained adapter directory containing "
                f"{PEFT_FILES} (e.g. the output of `sloth train`)."
            ),
        )

    # --- Validate format ---
    fmt = args.format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unsupported format: {fmt!r}",
            remediation=(
                f"supported formats: {', '.join(sorted(SUPPORTED_FORMATS))} — "
                "pass one with --format safetensors"
            ),
        )

    # --- Determine output directory (default: normalise in place) ---
    output = Path(args.output) if args.output else adapter

    # --- Perform the export ---
    if fmt == "safetensors":
        files = _export_safetensors(adapter, output)

    result = {
        "output": str(output.resolve()),
        "format": fmt,
        "files": files,
    }

    if json_mode:
        emit_result(result, json_mode=True)
    else:
        files_display = ", ".join(files) if files else "(none)"
        emit_result(
            f"exported adapter to {output}\nformat: {fmt}\nfiles: {files_display}",
            json_mode=False,
        )

    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``export`` subparser."""
    p = sub.add_parser(
        "export",
        help="Export a trained adapter to a standard PEFT/safetensors layout.",
    )
    p.add_argument(
        "--adapter",
        required=True,
        metavar="DIR",
        help="Path to the adapter directory to export.",
    )
    p.add_argument(
        "--format",
        default="safetensors",
        metavar="FMT",
        help="Output format (default: safetensors).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory (default: normalise in place inside --adapter).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON result to stdout.",
    )
    p.set_defaults(func=cmd_export)
