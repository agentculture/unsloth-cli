"""``sloth train`` — the integrator verb that ties the ``sloth.tune`` core together.

Flow::

    load_config  ->  validate dataset  ->  model preflight  ->  scope-guard
                 ->  dry-run | train

1. **load_config** — parse the ``run.toml`` into a :class:`~sloth.tune.config.RunConfig`
   (``CliError`` propagates on a missing/invalid config).
2. **validate dataset** — sniff the schema and run
   :func:`~sloth.tune.datasets.validate_dataset` *before any GPU work* so a
   malformed dataset fails fast with ``CliError(code=1)`` ("validate before
   spending GPU").
2b. **model preflight** — :func:`_preflight_model` (host path only, stdlib only):
   recommends ``target_modules = "preset:lfm2"`` for an LFM2 model that leaves it
   unset, notes that lobes' hand lane caps LoRA rank at 32, and — for a ``chat``
   dataset — refuses (``CliError(code=1)``) when the *locally cached* model ships
   no chat template. It never imports transformers/huggingface_hub, and is skipped
   under ``--in-container`` so each diagnostic is emitted once per invocation.
3. **scope-guard** — :func:`~sloth.tune.scope.check_scope` classifies the
   (model, method) request. An out-of-scope request (e.g. full fine-tuning of a
   large dense model) emits its warning EXPLICITLY to stderr via
   :func:`emit_diagnostic` and is then hard-refused with ``CliError(code=1)``.
4. **dry-run | train** — three branches depending on execution context:

   * ``--dry-run``: resolve the plan on the host (no GPU, no docker) and also
     print the docker command that would run the real job.
   * ``--in-container`` (hidden, recursion guard): running *inside* the NGC
     container — delegates directly to :func:`~sloth.tune._trainer.run_training`
     without launching another container. This is the run's **real-run path**,
     so it is the ONE place that writes to the run registry (see "Run
     registry" below); ``--dry-run`` never reaches this branch and therefore
     never appends a registry line.
   * default (host real run): call :func:`~sloth.tune.container.launch` to
     orchestrate the NGC container, forwarding the same args plus
     ``--in-container``.

This module imports no torch/unsloth — the heavy stack lives only inside the
trainer's ``_load_backend`` seam, lazily. Importing ``train`` stays torch-free.

Run registry (issue #12 / colleague#291 S4b)
---------------------------------------------
Every REAL (non-dry-run) training attempt — whether it reaches
:func:`run_training` directly (tests, or the ``--in-container`` recursion
guard) or via the NGC container recursing back into this same branch — gets
one line appended to ``<runs-root>/runs.jsonl`` (:mod:`sloth.tune.registry`;
runs-root = the PARENT dir of ``config.output``): ``status: "running"`` is
appended BEFORE :func:`run_training` runs, then atomically rewritten to
``"ok"``/``"failed"`` after it returns/raises. ``--dry-run`` (branch 4a) never
reaches this code path, so a dry-run plan never touches the registry — the
plan is resolved read-only.

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
import os
from pathlib import Path
from typing import Any

import sloth.tune.container as container_mod
from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune import registry as registry_mod
from sloth.tune._trainer import run_training
from sloth.tune.config import RunConfig, load_config
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


#: Substring (matched case-insensitively) that identifies an LFM2-family model id.
LFM2_MARKER = "lfm2"

#: LoRA rank ceiling accepted by lobes' "hand" serving lane. A higher rank still
#: trains — it just may not be servable there — so exceeding it is a DIAGNOSTIC,
#: never an error.
LOBES_HAND_LANE_RANK_CAP = 32

#: One-line recommendation emitted when an LFM2 model is trained with no explicit
#: ``target_modules``: Unsloth's default adapts only the attention projections and
#: silently misses LFM2's short-conv blocks (see sloth/tune/presets.py).
LFM2_PRESET_HINT = (
    "note: {model} looks like an LFM2 model and target_modules is unset — set "
    'target_modules = "preset:lfm2" in [hyperparameters] to adapt the short-conv '
    "and feed-forward blocks too (the default adapts attention projections only)."
)

#: One-line diagnostic emitted when an LFM2 run asks for a rank above the cap.
LFM2_RANK_HINT = (
    "note: lora_r = {rank} — lobes' hand lane caps LoRA rank at "
    "{cap}; the adapter will still train but may not be servable there."
)

#: One-line diagnostic when the chat-template check cannot run (model not cached).
CHAT_TEMPLATE_UNKNOWN_HINT = (
    "note: {model} is not in the local Hugging Face cache, so its chat template "
    "could not be verified before the run; a base model without a chat template "
    "cannot train on the chat schema."
)


# ---------------------------------------------------------------------------
# Host-side model preflight (pure stdlib — no transformers, no huggingface_hub)
# ---------------------------------------------------------------------------


def _hf_hub_root() -> Path:
    """Return the local Hugging Face hub cache root, honouring ``HF_HOME``."""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _model_snapshot_dir(model: str) -> Path | None:
    """Return the local directory holding *model*'s files, or ``None`` if not cached.

    A *local path* that is a directory counts as cached and is returned as-is.
    Otherwise the HF cache layout is walked:
    ``<hub>/models--<org>--<name>/snapshots/<rev>/`` — the most recently modified
    revision wins when several are present. Purely ``pathlib`` — the heavy HF
    libraries are never imported on the host path.
    """
    local = Path(model).expanduser()
    if local.is_dir():
        return local

    repo_dir = _hf_hub_root() / f"models--{model.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    try:
        revisions = [d for d in snapshots.iterdir() if d.is_dir()]
    except OSError:
        return None
    if not revisions:
        return None
    return max(revisions, key=lambda d: d.stat().st_mtime)


def _has_chat_template(snapshot: Path) -> bool:
    """True when *snapshot* ships a chat template in either supported form.

    Two forms exist in the wild: a ``chat_template`` key inside
    ``tokenizer_config.json``, and a standalone ``chat_template.jinja`` file
    (which is what LiquidAI/LFM2.5-1.2B-Base ships — measured 2026-09-15).
    """
    if (snapshot / "chat_template.jinja").is_file():
        return True
    try:
        with (snapshot / "tokenizer_config.json").open(encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and bool(config.get("chat_template"))


def _preflight_model(config: RunConfig, schema: str) -> None:
    """Emit host-side model diagnostics and refuse a chat run the model cannot do.

    Runs between dataset validation and the scope guard, on the HOST path only
    (the ``--in-container`` recursion skips it so each invocation emits each
    diagnostic exactly once). Three checks:

    1. **LFM2 target_modules hint** — an LFM2 model with no ``target_modules``
       gets one line recommending ``"preset:lfm2"``.
    2. **Hand-lane rank cap** — an LFM2 run with ``lora_r`` above
       :data:`LOBES_HAND_LANE_RANK_CAP` gets one line; never an error.
    3. **Chat-template check** — for a ``chat`` dataset, a *cached* model with no
       chat template raises ``CliError(code=1)`` pointing at the task schema; an
       uncached model only emits a diagnostic and proceeds.
    """
    is_lfm2 = LFM2_MARKER in config.model.lower()

    if is_lfm2 and config.target_modules is None:
        emit_diagnostic(LFM2_PRESET_HINT.format(model=config.model))

    if is_lfm2 and config.lora_r > LOBES_HAND_LANE_RANK_CAP:
        emit_diagnostic(LFM2_RANK_HINT.format(rank=config.lora_r, cap=LOBES_HAND_LANE_RANK_CAP))

    if schema != "chat":
        return

    snapshot = _model_snapshot_dir(config.model)
    if snapshot is None:
        emit_diagnostic(CHAT_TEMPLATE_UNKNOWN_HINT.format(model=config.model))
        return
    if not _has_chat_template(snapshot):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"model {config.model} ships no chat template (neither a "
                f"'chat_template' key in tokenizer_config.json nor a "
                f"chat_template.jinja under {snapshot}), so it cannot train on a "
                f"chat-schema dataset"
            ),
            remediation=(
                "Convert the dataset to the task schema "
                '({"task", "input", "expected_output"} per line) and rerun, or point '
                "the config at an instruct/chat variant of the model that ships a "
                "chat template."
            ),
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
# Container invocation resolver (shared by dry-run and host real-run)
# ---------------------------------------------------------------------------


def _resolve_container_invocation(
    config_path: Path, config: RunConfig
) -> tuple[Path, list[tuple[str, str]], list[str]]:
    """Resolve absolute paths and build the container invocation components.

    Returns *(config_dir, extra_mounts, sloth_args)* for use in both the
    dry-run (passed to :func:`~sloth.tune.container.build_command`) and the
    host real-run (passed to :func:`~sloth.tune.container.launch`) branches,
    so both branches produce an identical docker command.

    ``sloth_args`` is ``["train", "--config", <abs-config-path>, "--json",
    "--in-container"]`` — ``--json`` is always forwarded (regardless of the
    host's own ``--json`` flag) so the container always emits a structured
    result the host can capture from :func:`~sloth.tune.container.launch`'s
    return value and re-render for its own caller.  Passing the absolute
    config path together with identity mounts (``host_path == container_path``)
    means the path resolves unchanged inside the container without any
    ``/workspace/<name>`` rewriting.
    """
    # Relative dataset/output paths resolve against the current working directory
    # (where ``sloth train`` was invoked) — the SAME base the host-side
    # ``validate_dataset`` uses and the dir bind-mounted as the container workdir.
    # Resolving against the config's *parent* instead (the prior behavior) made the
    # host check and the in-container run disagree whenever the config lived in a
    # subdirectory, e.g. ``examples/run.toml`` referencing ``examples/data.jsonl``
    # double-resolved to ``examples/examples/data.jsonl`` inside the container.
    base_dir = Path.cwd()

    dataset_path = Path(config.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (base_dir / dataset_path).resolve()
    output_path = Path(config.output)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    # Identity mounts so host-absolute paths (the forwarded config, the dataset, and
    # the output dir) resolve unchanged inside the container (host_path == container_path).
    # ``base_dir`` is deliberately NOT identity-mounted: it is already bind-mounted as the
    # container workdir (``-v base_dir:/workspace``), so an identity mount would be
    # redundant — and when ``sloth train`` runs from ``/`` it would emit a dangerous
    # ``-v /:/`` overlaying the container root with the host root filesystem. ``sorted``
    # makes the docker argv deterministic (a set has no stable iteration order).
    mount_parents = {config_path.parent, dataset_path.parent, output_path.parent}
    extra_mounts: list[tuple[str, str]] = [(str(p), str(p)) for p in sorted(mount_parents)]

    # Forward the ABSOLUTE config path; identity mounts make it resolve inside
    # the container without rewriting to /workspace. ``--json`` is always
    # forwarded (unconditionally) so the container always prints a structured
    # result line for container.launch() to parse and return — the host then
    # renders it for its own caller according to the HOST's --json flag.
    sloth_args: list[str] = ["train", "--config", str(config_path), "--json", "--in-container"]

    return base_dir, extra_mounts, sloth_args


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth train``.

    Branching logic after the host-side GPU-free preflight (steps 1–3):

    * **dry-run** (``--dry-run``): resolves the plan without any GPU work and
      prints the docker command that would launch the real job.  Returns ``None``.
    * **in-container** (``--in-container``, hidden recursion guard): running
      *inside* the NGC container — calls :func:`run_training` directly.  Returns
      ``None`` on success.
    * **host real run** (default): validates on the host, then calls
      :func:`container_mod.launch` to orchestrate the NGC container, forwarding
      the same train args plus ``--in-container``.  Returns ``None`` on success.

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

    # 2b) Host model preflight (stdlib only): LFM2 target_modules hint, lobes'
    # hand-lane rank cap, and the chat-template check. Skipped under
    # --in-container so the container recursion never repeats a diagnostic the
    # host already emitted (one line per invocation).
    if not in_container:
        _preflight_model(config, schema)

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
        config_dir, extra_mounts, sloth_args = _resolve_container_invocation(config_path, config)
        checkout = Path(__file__).resolve().parents[3]
        cmd = container_mod.build_command(
            sloth_args, workdir=config_dir, checkout=checkout, extra_mounts=extra_mounts
        )
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
        return

    # 4b) In-container: recursion guard — run the real trainer, no docker launch.
    # This is the real-run path (see the "Run registry" module-docstring
    # section): a registry line is appended "running" before run_training and
    # atomically rewritten "ok"/"failed" after — dry-run (4a) never reaches here.
    if in_container:
        run_record = registry_mod.start_run(config)
        try:
            plan = run_training(config, dry_run=False)
        except Exception:
            registry_mod.finish_run(run_record, status=registry_mod.STATUS_FAILED)
            raise
        registry_mod.finish_run(run_record, status=registry_mod.STATUS_OK)
        if json_mode:
            emit_result(plan, json_mode=True)
        else:
            emit_result(_render_plan_text(plan), json_mode=False)
        return

    # 4c) Host real run: orchestrate via the NGC container.
    config_path = Path(args.config).resolve()
    config_dir, extra_mounts, sloth_args = _resolve_container_invocation(config_path, config)
    checkout = Path(__file__).resolve().parents[3]
    # launch() raises CliError on any non-zero container exit, and otherwise returns
    # the parsed JSON result the container printed (its own train result — see the
    # "Run registry" module docstring). Emit it through the host's own output
    # contract so a host caller sees the same shape the in-container path would
    # have printed, honouring the HOST's --json flag (independent of the --json
    # unconditionally forwarded into the container above).
    result = container_mod.launch(
        sloth_args, workdir=str(config_dir), checkout=checkout, extra_mounts=extra_mounts
    )
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(_render_plan_text(result), json_mode=False)


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
