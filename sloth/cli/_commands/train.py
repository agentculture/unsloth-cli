"""``sloth train`` — the integrator verb that ties the ``sloth.tune`` core together.

Flow::

    load_config  ->  validate dataset  ->  scope-guard  ->  dry-run | train

1. **load_config** — parse the ``run.toml`` into a :class:`~sloth.tune.config.RunConfig`
   (``CliError`` propagates on a missing/invalid config).
2. **validate dataset** — sniff the schema and run
   :func:`~sloth.tune.datasets.validate_dataset` *before any GPU work* so a
   malformed dataset fails fast with ``CliError(code=1)`` ("validate before
   spending GPU").
3. **scope-guard** — :func:`~sloth.tune.scope.check_scope` classifies the
   (model, method) request. An out-of-scope request (e.g. full fine-tuning of a
   large dense model) emits its warning EXPLICITLY to stderr via
   :func:`emit_diagnostic` and is then hard-refused with ``CliError(code=1)``.
4. **dry-run | train** — three branches depending on execution context:

   * ``--dry-run``: resolve the plan on the host (no GPU, no docker) and also
     print the docker command that would run the real job.
   * ``--in-container`` (hidden, recursion guard): running *inside* the NGC
     container — delegates directly to :func:`~sloth.tune._trainer.run_training`
     without launching another container.
   * default (host real run): call :func:`~sloth.tune.container.launch` to
     orchestrate the NGC container, forwarding the same args plus
     ``--in-container``.

This module imports no torch/unsloth — the heavy stack lives only inside the
trainer's ``_load_backend`` seam, lazily. Importing ``train`` stays torch-free.

Dataset-schema choice
---------------------
The schema is **inferred from the first non-blank record** via
:func:`~sloth.tune.datasets.detect_schema` (``"chat"`` when a ``messages`` key is
present, ``"task"`` for the ``task``/``input``/``expected_output`` shape).  When
the file is empty, unreadable, or the first record is inconclusive, validation
falls back to the **chat** schema (the documented default) and lets
``validate_dataset`` raise the authoritative ``CliError``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sloth.tune.container as container_mod
from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune._trainer import run_training
from sloth.tune.config import load_config
from sloth.tune.datasets import detect_schema, validate_dataset
from sloth.tune.scope import check_scope

#: Schema assumed when the dataset's first record cannot be classified.
DEFAULT_SCHEMA = "chat"

#: Always-visible scope statement. Shown in ``sloth train --help`` (and echoed by
#: ``explain train``) so the LoRA/QLoRA-only boundary is stated up front, before a
#: run starts — not only when an out-of-scope request is rejected at runtime.
SCOPE_NOTICE = (
    "Scope: LoRA and QLoRA adapter training only. Full fine-tuning of large "
    "dense models is out of scope and will be refused — use method='lora' or "
    "method='qlora'."
)


# ---------------------------------------------------------------------------
# Schema inference (pure — no torch)
# ---------------------------------------------------------------------------


def _resolve_schema(dataset: str) -> str:
    """Infer the dataset schema from its first non-blank record.

    Returns ``"chat"`` or ``"task"`` when the first record is classifiable,
    otherwise the :data:`DEFAULT_SCHEMA`. Never raises: any read/parse failure
    is deferred to :func:`validate_dataset`, which surfaces the authoritative
    ``CliError`` once it re-reads the file.
    """
    path = Path(dataset)
    try:
        with path.open(encoding="utf-8") as fh:
            record: Any = None
            for line in fh:
                stripped = line.strip()
                if stripped:
                    record = json.loads(stripped)
                    break
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SCHEMA
    if record is None:
        return DEFAULT_SCHEMA
    return detect_schema(record) or DEFAULT_SCHEMA


# ---------------------------------------------------------------------------
# Plan rendering (text mode)
# ---------------------------------------------------------------------------


def _render_plan_text(plan: dict[str, Any]) -> str:
    """Render the resolved training plan as human-readable text for stdout."""
    mode = "dry-run" if plan.get("dry_run") else "train"
    lines = [
        f"plan: {mode}",
        f"model:   {plan.get('model')}",
        f"method:  {plan.get('method')}",
        f"dataset: {plan.get('dataset')}",
        f"output:  {plan.get('output')}",
        "hyperparameters:",
    ]
    for key, value in plan.get("hyperparameters", {}).items():
        lines.append(f"  {key}: {value}")

    # Real-run extras (present only after the trainer ran the job).
    if plan.get("status"):
        lines.append(f"status:   {plan['status']}")
    if plan.get("adapter_dir"):
        lines.append(f"adapter:  {plan['adapter_dir']}")
    if plan.get("metadata_path"):
        lines.append(f"metadata: {plan['metadata_path']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Handler for ``sloth train``.

    Branching logic after the host-side GPU-free preflight (steps 1–3):

    * **dry-run** (``--dry-run``): resolves the plan without any GPU work and
      prints the docker command that would launch the real job.  Returns ``0``.
    * **in-container** (``--in-container``, hidden recursion guard): running
      *inside* the NGC container — calls :func:`run_training` directly.  Returns
      ``0`` on success.
    * **host real run** (default): validates on the host, then calls
      :func:`container_mod.launch` to orchestrate the NGC container, forwarding
      the same train args plus ``--in-container``.  Returns the container's exit
      code.

    Every failure raises :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    in_container = bool(getattr(args, "in_container", False))

    # -------------------------------------------------------------------
    # Steps 1–3: host-side GPU-free preflight — always runs, so bad
    # configs / datasets / scope are caught before any docker or GPU work.
    # -------------------------------------------------------------------

    # 1) Load + validate the run config (CliError propagates on bad/missing file).
    config = load_config(args.config)

    # 2) Validate the dataset BEFORE any GPU work ("validate before spending GPU").
    schema = _resolve_schema(config.dataset)
    validate_dataset(config.dataset, schema)

    # 3) Scope-guard: warn explicitly, then hard-refuse an out-of-scope request.
    scope = check_scope(config.model, config.method)
    if scope.warning:
        emit_diagnostic(scope.warning)
    if scope.out_of_scope:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"refusing to train: {scope.message}",
            remediation=(
                scope.warning or "Switch to an adapter method: set method='lora' or method='qlora'."
            ),
        )

    # -------------------------------------------------------------------
    # Step 4 — branch on execution context
    # -------------------------------------------------------------------

    # 4a) Dry-run: resolve the plan on the host; also show the docker command.
    if dry_run:
        plan = run_training(config, dry_run=True)
        config_path = Path(args.config).resolve()
        workdir = config_path.parent
        config_in_container = str(Path(container_mod.WORKDIR_MOUNT) / config_path.name)
        dry_run_args: list[str] = ["train", "--config", config_in_container, "--in-container"]
        if json_mode:
            dry_run_args.append("--json")
        checkout = Path(__file__).resolve().parents[3]
        cmd = container_mod.build_command(dry_run_args, workdir=workdir, checkout=checkout)
        if json_mode:
            result: dict[str, Any] = dict(plan)
            result["docker_image"] = container_mod.NGC_IMAGE
            result["docker_command"] = cmd
            emit_result(result, json_mode=True)
        else:
            text = _render_plan_text(plan)
            text += f"\ndocker-image:   {container_mod.NGC_IMAGE}"
            text += f"\ndocker-command: {' '.join(cmd)}"
            emit_result(text, json_mode=False)
        return 0

    # 4b) In-container: recursion guard — run the real trainer, no docker launch.
    if in_container:
        plan = run_training(config, dry_run=False)
        if json_mode:
            emit_result(plan, json_mode=True)
        else:
            emit_result(_render_plan_text(plan), json_mode=False)
        return 0

    # 4c) Host real run: orchestrate via the NGC container.
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    # Resolve dataset and output to absolute paths (TOML values may be relative;
    # the convention is relative-to-config-dir, not relative-to-CWD).
    dataset_path = Path(config.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (config_dir / dataset_path).resolve()
    output_path = Path(config.output)
    if not output_path.is_absolute():
        output_path = (config_dir / output_path).resolve()

    # Identity mounts so host-absolute paths forwarded in sloth_args resolve
    # unchanged inside the container (host_path == container_path).
    mount_parents = {config_path.parent, dataset_path.parent, output_path.parent}
    extra_mounts = [(str(p), str(p)) for p in mount_parents]

    # Forward the ABSOLUTE config path; identity mounts make it resolve inside
    # the container without rewriting to /workspace.
    sloth_args: list[str] = ["train", "--config", str(config_path), "--in-container"]
    if json_mode:
        sloth_args.append("--json")
    checkout = Path(__file__).resolve().parents[3]
    # launch() raises CliError on any non-zero container exit; returns 0 on success.
    container_mod.launch(
        sloth_args, workdir=str(config_dir), checkout=checkout, extra_mounts=extra_mounts
    )
    return 0


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``train`` subparser on *sub*."""
    p = sub.add_parser(
        "train",
        help="Validate a dataset and run (or plan) a LoRA/QLoRA adapter job.",
        description=(
            "Validate a dataset and run (or plan) a LoRA/QLoRA adapter job. " + SCOPE_NOTICE
        ),
        epilog=SCOPE_NOTICE,
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="TOML",
        help="Path to the run.toml describing the model, dataset, output, and method.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and resolve the plan without importing torch or training.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_train)
