"""``sloth export`` — export a trained adapter to a standard PEFT/safetensors layout.

Copies or normalises a LoRA/QLoRA adapter directory into the canonical PEFT
layout that the ``lobes`` server can serve and ``colleague`` can run:

    <output>/
      adapter_config.json          # required — PEFT adapter config
      adapter_model.safetensors    # required — LoRA/QLoRA weight deltas
      tokenizer.json               # optional — copied when present
      tokenizer_config.json        # optional — copied when present
      special_tokens_map.json      # optional — copied when present
      vocab.json                   # optional — copied when present
      merges.txt                   # optional — copied when present
      tokenizer.model              # optional — copied when present (SentencePiece)

Container / ML-stack decision (resolves risk r4 from the build plan)
----------------------------------------------------------------------
``export`` is **pure stdlib — no container, no torch, no peft**.

The rationale: unsloth/PEFT write the adapter weights in safetensors format
*during training*, so by the time the trainer exits the adapter directory
already contains the canonical PEFT files.  Nothing in the export step requires
loading or converting weights — the verb reorganises and validates file-system
artefacts.  ``sloth.tune.container`` is **not imported here**; no docker process
is launched.

Contrast with ``train`` (GPU compute → container) and ``eval`` (PeftModel load
→ container): those verbs genuinely need the ML stack.  ``export`` does not.
Torch / Unsloth are never imported anywhere in this module.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result

# The only format supported today. Extend this set as new formats land.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"safetensors"})

# Canonical PEFT file names that MUST be present in the adapter directory.
# Their absence is a hard error — an export without these files is unusable.
PEFT_FILES: list[str] = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

# Tokenizer files that are copied when present but are NOT required.
# A trained adapter may bundle the tokenizer alongside the weights; if it does,
# lobes / colleague benefit from having those files in the export directory.
OPTIONAL_TOKENIZER_FILES: list[str] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
]


def _export_safetensors(adapter: Path, output: Path) -> list[str]:
    """Copy required PEFT files and any optional tokenizer files from *adapter* to *output*.

    If *adapter* and *output* resolve to the same directory, copies are skipped
    and the files are reported as-is (normalise-in-place semantics).

    Required files (``PEFT_FILES``) are assumed already validated present by the
    caller; optional files (``OPTIONAL_TOKENIZER_FILES``) are silently skipped
    when absent — they are not mandatory for a valid PEFT layout.

    Returns a list of absolute string paths for all files written/present in *output*.
    """
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Copy all files that should appear in the output: required first, then optional.
    for fname in PEFT_FILES + OPTIONAL_TOKENIZER_FILES:
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

    # --- Perform the export (pure filesystem — no container, no torch) ---
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
