"""``sloth export`` — export a trained adapter to a servable model layout.

Two lanes, one verb
-------------------

**1. ``--format safetensors`` (the default) — pure stdlib, no container.**
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

The rationale (risk r4): unsloth/PEFT write the adapter weights in safetensors
format *during training*, so by the time the trainer exits the adapter directory
already contains the canonical PEFT files.  Nothing in this step loads or
converts weights — the verb reorganises and validates file-system artefacts.
This lane launches **no** container and imports **no** torch.

**2. ``merged-16bit`` / ``merged-4bit`` / ``gguf`` / ``awq`` / ``nvfp4`` —
host→container.**
These genuinely need the ML stack (base-model load, merge, ggml conversion,
llm-compressor one-shot quantization), so the host side of this module does what
``eval.py`` does: validate everything cheaply, then hand off to the NGC container
via :func:`sloth.tune.container.launch`, forwarding the same args plus the hidden
``--in-container`` recursion guard.  Identity bind-mounts (``host == container``)
are added for the parents of the adapter, output and calibration paths so the
host-absolute paths in the forwarded argv resolve unchanged inside the container.

``sloth.tune.container`` and :mod:`sloth.tune._exporter` are **never imported at
module level** — both are imported lazily inside the branch that needs them, so
importing this module stays import-light (and ML-free) for the introspection
verbs and for the ``safetensors`` lane.

Safety rails on the container lane
----------------------------------
* **No-clobber** — a non-empty ``--output`` is refused (exit 1) unless ``--force``.
* **Atomic output** — the container writes into ``<output>.partial`` and the host
  renames it to ``<output>`` only after a clean (exit 0) run, so a killed/OOM run
  never leaves a half-written model directory behind.
* **Fail-closed disk check** — the estimated artifact size is compared against the
  free space under ``--output``; too little space exits 2 with a hint *before* any
  GPU spend.  ``--dry-run`` reports both numbers instead of failing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result

# Note: ``_parse_largest_param_count`` is the hand-rolled (regex-free) scanner the
# scope guard uses to read "4b"/"27b" out of a model id.  Reused here rather than
# duplicated so the disk estimate and the scope guard read model ids identically.
from sloth.tune.scope import _parse_largest_param_count

#: Every format this verb can produce. ``safetensors`` is the pure-stdlib lane;
#: all others run inside the NGC container.
SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"safetensors", "merged-16bit", "merged-4bit", "gguf", "awq", "nvfp4"}
)

#: Formats that route host→container (everything except the stdlib lane).
CONTAINER_FORMATS: frozenset[str] = SUPPORTED_FORMATS - {"safetensors"}

#: Formats that take calibration data (llm-compressor one-shot quantization).
CALIBRATED_FORMATS: frozenset[str] = frozenset({"awq", "nvfp4"})

#: The ggml quantization names accepted by ``--quant`` (Unsloth's documented list).
GGML_QUANTS: frozenset[str] = frozenset(
    {"q4_k_m", "q5_k_m", "q8_0", "f16", "q2_k", "q3_k_m", "q4_0", "q4_1", "q5_0", "q6_k"}
)

#: Bytes per parameter for the disk estimate, per format.  ``merged-16bit`` is
#: bf16 (2 bytes/param); ``gguf`` doubles it because Unsloth writes an F16
#: intermediate alongside the requested quant; the 4-bit / compressed-tensors
#: formats land near 0.6 bytes/param.  These constants are heuristics — t13
#: tightens them against measured artifact sizes.
BYTES_PER_PARAM: dict[str, float] = {
    "merged-16bit": 2.0,
    "merged-4bit": 0.6,
    "gguf": 4.0,
    "awq": 0.6,
    "nvfp4": 0.6,
}

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

#: Suffix of the staging directory the container writes into (atomic output).
PARTIAL_SUFFIX: str = ".partial"


# ---------------------------------------------------------------------------
# Checkout locator (repo root for the container bind-mount) — mirrors eval.py
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the unsloth-cli checkout root by walking up from this module.

    ``sloth/cli/_commands/export.py`` → ``parents[3]`` is the checkout root (the
    dir containing the ``sloth/`` package), bind-mounted inside the NGC container
    so ``python -m sloth`` resolves without an install step.
    """
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# safetensors lane (pure stdlib)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Small JSON readers (never raise — a missing/!bad file just means "unknown")
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return the JSON object at *path*, or ``None`` when absent/unreadable/not a dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _adapter_base_model(adapter: Path) -> str | None:
    """Return ``base_model_name_or_path`` from the adapter's ``adapter_config.json``."""
    config = _read_json(adapter / "adapter_config.json") or {}
    base = config.get("base_model_name_or_path")
    return base if isinstance(base, str) and base.strip() else None


def _final_output(output: Path) -> str:
    """Strip the ``.partial`` staging suffix the host adds around a container run."""
    text = str(output)
    if text.endswith(PARTIAL_SUFFIX):
        return text[: -len(PARTIAL_SUFFIX)]
    return text


def _training_dataset(adapter: Path) -> str | None:
    """Return the training dataset as an **absolute host path**, or ``None``.

    ``training_metadata.json`` records the dataset path exactly as the run config
    gave it (deviation d1) — usually relative to the directory ``sloth train`` ran
    in. The container's workdir is the adapter's parent, so a relative path must be
    resolved here on the host: current directory first, then the adapter's parent,
    then the adapter dir itself. Unresolvable or absent means the dataset is
    unknown and calibration falls back to ``--calib``.
    """
    meta = _read_json(adapter / "training_metadata.json") or {}
    dataset = meta.get("dataset")
    raw: str | None = None
    if isinstance(dataset, str) and dataset.strip():
        raw = dataset
    elif isinstance(dataset, dict):
        path = dataset.get("path")
        if isinstance(path, str) and path.strip():
            raw = path
    if raw is None:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        # Passed through as recorded; the in-container calibration validation
        # reports a missing file with a hint before any model load.
        return str(candidate)
    for root in (Path.cwd(), adapter.parent, adapter):
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return str(resolved)
    return None


# ---------------------------------------------------------------------------
# Disk estimate
# ---------------------------------------------------------------------------


def _hf_hub_root() -> Path:
    """Return the Hugging Face hub cache root (honours ``HF_HOME``)."""
    hf_home = os.environ.get("HF_HOME")
    root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return root / "hub"


def _cached_config(base: str) -> dict[str, Any] | None:
    """Return the base model's ``config.json`` when it is a local dir or HF-cached."""
    local = Path(base)
    if local.is_dir():
        cfg = _read_json(local / "config.json")
        if cfg is not None:
            return cfg

    snapshots = _hf_hub_root() / f"models--{base.replace('/', '--')}" / "snapshots"
    try:
        candidates = sorted(snapshots.iterdir())
    except OSError:
        return None
    for snapshot in candidates:
        cfg = _read_json(snapshot / "config.json")
        if cfg is not None:
            return cfg
    return None


def _params_from_config(config: dict[str, Any]) -> float | None:
    """Estimate the parameter count from a transformers ``config.json``.

    Counts the embedding matrices, per-layer attention projections (honouring
    grouped-query ``num_key_value_heads``) and a 3-matrix MLP.  Returns ``None``
    when the config lacks the shape keys needed for the arithmetic.
    """
    try:
        hidden = int(config["hidden_size"])
        layers = int(config["num_hidden_layers"])
        vocab = int(config["vocab_size"])
        intermediate = int(config["intermediate_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(hidden, layers, vocab, intermediate) <= 0:
        return None

    heads = int(config.get("num_attention_heads") or 0)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    if heads and head_dim:
        attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    else:  # shapes unavailable — the square-projection approximation
        attn = 4 * hidden * hidden

    mlp = 3 * hidden * intermediate
    embeddings = vocab * hidden * (1 if config.get("tie_word_embeddings") else 2)
    return float(layers * (attn + mlp) + embeddings)


def _param_count(base: str) -> float | None:
    """Return the base model's parameter count: cached ``config.json`` first, id second."""
    config = _cached_config(base)
    if config is not None:
        params = _params_from_config(config)
        if params:
            return params
    billions = _parse_largest_param_count(base)
    return billions * 1e9 if billions else None


def _dir_bytes(path: Path) -> int:
    """Return the total size in bytes of the regular files directly under *path*."""
    try:
        return sum(entry.stat().st_size for entry in path.iterdir() if entry.is_file())
    except OSError:
        return 0


def _estimate_bytes(base: str | None, fmt: str, adapter: Path) -> int | None:
    """Estimate the bytes the export will occupy, or ``None`` when unknowable.

    ``safetensors`` copies the adapter, so its estimate is the adapter's own size.
    Every other format is ``parameter_count × BYTES_PER_PARAM[fmt]``, with the
    parameter count taken from the base model's cached ``config.json`` when
    available and otherwise parsed out of the model id (``…-4B`` → 4e9).
    """
    if fmt == "safetensors":
        return _dir_bytes(adapter)
    if not base:
        return None
    params = _param_count(base)
    if not params:
        return None
    return int(params * BYTES_PER_PARAM[fmt])


def _free_bytes(path: Path) -> int | None:
    """Return the free bytes on the filesystem holding *path*'s nearest existing parent."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        stat = os.statvfs(probe)
    except (OSError, AttributeError):  # pragma: no cover - non-POSIX hosts
        return None
    return int(stat.f_bavail) * int(getattr(stat, "f_frsize", 0) or stat.f_bsize)


# ---------------------------------------------------------------------------
# Argument validation (everything below exits 1 before any container launch)
# ---------------------------------------------------------------------------


def _validate_adapter(adapter: Path) -> None:
    """Raise ``CliError(1)`` unless *adapter* is a directory holding the PEFT files."""
    if not adapter.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"adapter directory not found: {adapter}",
            remediation="pass an existing adapter directory via --adapter <dir>",
        )
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


def _validate_format(raw_format: str) -> str:
    """Return the normalised format, raising ``CliError(1)`` for an unknown one."""
    fmt = raw_format.strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unsupported format: {fmt!r}",
            remediation=(
                f"supported formats: {', '.join(sorted(SUPPORTED_FORMATS))} — "
                "pass one with --format <fmt>"
            ),
        )
    return fmt


def _validate_quant(raw_quant: str | None, fmt: str) -> list[str]:
    """Parse/validate the ``--quant`` comma list against :data:`GGML_QUANTS`."""
    if not raw_quant:
        return []
    quants = [item.strip().lower() for item in raw_quant.split(",") if item.strip()]
    unknown = [q for q in quants if q not in GGML_QUANTS]
    if unknown:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unsupported ggml quantization(s): {', '.join(unknown)}",
            remediation=(
                "Pick from the ggml allowlist: "
                f"{', '.join(sorted(GGML_QUANTS))} (--quant q4_k_m)."
            ),
        )
    if quants and fmt != "gguf":
        emit_diagnostic(f"note: --quant applies to --format gguf only; ignored for {fmt}")
    return quants


def _validate_calib(raw_calib: str | None, samples: int | None, fmt: str) -> Path | None:
    """Validate the calibration file and sample count; return the resolved path."""
    if samples is not None and samples <= 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--calib-samples must be a positive integer, got {samples}",
            remediation="Pass a positive sample count, e.g. --calib-samples 128.",
        )
    if raw_calib is None:
        return None
    calib = Path(raw_calib)
    if not calib.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"calibration file not found: {calib}",
            remediation=(
                "Pass an existing JSONL file with --calib <path>, or omit it to "
                "calibrate on the run's training dataset."
            ),
        )
    if fmt not in CALIBRATED_FORMATS:
        emit_diagnostic(
            f"note: --calib applies to {', '.join(sorted(CALIBRATED_FORMATS))} only; "
            f"ignored for {fmt}"
        )
    return calib.resolve()


def _resolve_base(explicit: str | None, adapter: Path, fmt: str) -> str | None:
    """Resolve the base model id: ``--base`` first, then the adapter config."""
    base = explicit.strip() if isinstance(explicit, str) and explicit.strip() else None
    base = base or _adapter_base_model(adapter)
    if base is None and fmt in CONTAINER_FORMATS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"cannot resolve the base model for --format {fmt}: "
                f"{adapter / 'adapter_config.json'} has no base_model_name_or_path"
            ),
            remediation="Name the base model explicitly, e.g. --base unsloth/Qwen3-4B.",
        )
    return base


def _resolve_output(raw_output: str | None, adapter: Path, fmt: str) -> Path:
    """Resolve ``--output``; container formats must name one explicitly."""
    if raw_output:
        return Path(raw_output)
    if fmt in CONTAINER_FORMATS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--format {fmt} writes a new model directory and needs an output path",
            remediation="Name a destination directory with --output <dir>.",
        )
    return adapter  # safetensors: normalise in place (today's behaviour)


def _check_clobber(output: Path, force: bool) -> None:
    """Refuse a non-empty *output* unless *force* (container lane only)."""
    if not output.is_dir():
        return
    try:
        non_empty = any(output.iterdir())
    except OSError:
        non_empty = False
    if non_empty and not force:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"output directory is not empty: {output}",
            remediation="Pick an empty/new --output directory, or pass --force to overwrite it.",
        )


def _check_disk(estimated: int | None, free: int | None, output: Path, fmt: str) -> None:
    """Fail closed (exit 2) when the free space is below the estimate."""
    if estimated is None:
        emit_diagnostic(
            f"note: could not estimate the disk footprint of a {fmt} export "
            "(unknown parameter count); skipping the free-space check"
        )
        return
    if free is None or free >= estimated:
        return
    raise CliError(
        code=EXIT_ENV_ERROR,
        message=(
            f"not enough free space under {output}: a {fmt} export needs about "
            f"{estimated} bytes, {free} bytes are free"
        ),
        remediation=(
            "Free up disk space or pass an --output on a larger filesystem. "
            "Run the same command with --dry-run to see the estimate without exporting."
        ),
    )


# ---------------------------------------------------------------------------
# Container invocation (host lane)
# ---------------------------------------------------------------------------


def _sloth_args(
    *,
    fmt: str,
    adapter: Path,
    output: Path,
    base: str | None,
    quant: list[str],
    calib: Path | None,
    calib_samples: int | None,
    keep_intermediate: bool,
    json_mode: bool,
) -> list[str]:
    """Build the ``python -m sloth export …`` argv forwarded into the container.

    All paths are host-absolute; identity bind-mounts make them resolve unchanged
    inside the container.  ``--in-container`` is the recursion guard.
    """
    args = [
        "export",
        "--adapter",
        str(adapter),
        "--format",
        fmt,
        "--output",
        str(output),
    ]
    if base:
        args += ["--base", base]
    if quant:
        args += ["--quant", ",".join(quant)]
    if calib is not None:
        args += ["--calib", str(calib)]
    if calib_samples is not None:
        args += ["--calib-samples", str(calib_samples)]
    if keep_intermediate:
        args.append("--keep-intermediate")
    if json_mode:
        args.append("--json")
    args.append("--in-container")
    return args


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


def _container_kwargs(
    container: Any,
    adapter: Path,
    output: Path,
    calib: Path | None,
    dataset: Path | None = None,
) -> dict[str, Any]:
    """Return the keyword arguments shared by ``build_command`` and ``launch``.

    Identity mounts (``host == container``) for the adapter, output and calibration
    parents mirror ``eval.py``; ``sorted`` keeps the docker argv deterministic.
    Anything :func:`container.export_launch_kwargs` contributes (the llama.cpp
    cache mount and the ``HOME`` / ``UNSLOTH_LLAMA_TAG`` env, added by t8) is
    merged in — the ``getattr`` fallback keeps this working against a container
    module that does not have it yet.
    """
    parents = {adapter.parent, output.parent}
    if calib is not None:
        parents.add(calib.parent)
    if dataset is not None:
        parents.add(dataset.parent)
    own_mounts = [(str(p), str(p)) for p in sorted(parents)]

    supplied = getattr(container, "export_launch_kwargs", lambda: {})() or {}
    extra_mounts = list(supplied.get("extra_mounts") or [])

    kwargs: dict[str, Any] = {
        "workdir": str(adapter.parent),
        "checkout": str(_repo_root()),
        "extra_mounts": _merge_mounts(own_mounts, extra_mounts),
    }
    for key, value in supplied.items():
        if key != "extra_mounts":
            kwargs[key] = value
    return kwargs


# ---------------------------------------------------------------------------
# Plan rendering
# ---------------------------------------------------------------------------


def _render_plan_text(plan: dict[str, Any]) -> str:
    """Render the dry-run plan as human-readable text for stdout."""
    lines = [
        "plan: dry-run",
        f"format:          {plan['format']}",
        f"base:            {plan['base']}",
        f"adapter:         {plan['adapter']}",
        f"output:          {plan['output']}",
        f"quant:           {', '.join(plan['quant']) if plan['quant'] else '(none)'}",
        f"estimated-bytes: {plan['estimated_bytes']}",
        f"free-bytes:      {plan['free_bytes']}",
    ]
    if plan["docker_command"]:
        lines.append(f"docker-command:  {plan['docker_command']}")
    else:
        lines.append("docker-command:  (none — safetensors runs on the host)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    """Handler for ``sloth export``.

    Validates everything cheaply (adapter, format, quant, calibration, base,
    output) and then takes one of four branches: ``--dry-run`` (plan only),
    ``safetensors`` (pure stdlib copy), ``--in-container`` (delegate to
    :func:`sloth.tune._exporter.run_export`), or the host lane (atomic
    ``<output>.partial`` write inside the NGC container).
    """
    json_mode = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    in_container = bool(getattr(args, "in_container", False))
    force = bool(getattr(args, "force", False))
    keep_intermediate = bool(getattr(args, "keep_intermediate", False))

    # --- validation (exits 1 before anything is launched or written) ---------
    adapter = Path(args.adapter)
    _validate_adapter(adapter)
    fmt = _validate_format(args.format)
    quant = _validate_quant(getattr(args, "quant", None), fmt)
    calib_samples = getattr(args, "calib_samples", None)
    calib = _validate_calib(getattr(args, "calib", None), calib_samples, fmt)
    base = _resolve_base(getattr(args, "base", None), adapter, fmt)
    output = _resolve_output(getattr(args, "output", None), adapter, fmt)

    adapter_abs = adapter.resolve()
    output_abs = output.resolve()
    estimated = _estimate_bytes(base, fmt, adapter_abs)
    free = _free_bytes(output_abs)
    dataset_str = _training_dataset(adapter_abs) if fmt in CALIBRATED_FORMATS else None
    dataset_path = Path(dataset_str) if dataset_str else None

    # --- dry-run: resolve the plan, launch nothing, write nothing -----------
    if dry_run:
        docker_command: str | None = None
        if fmt in CONTAINER_FORMATS:
            import sloth.tune.container as container  # lazy: keeps the stdlib lane clean

            partial = Path(str(output_abs) + PARTIAL_SUFFIX)
            kwargs = _container_kwargs(container, adapter_abs, output_abs, calib, dataset_path)
            sloth_args = _sloth_args(
                fmt=fmt,
                adapter=adapter_abs,
                output=partial,
                base=base,
                quant=quant,
                # The resolved training dataset rides into the container as the
                # calibration source; the container never re-resolves host paths.
                calib=calib if calib is not None else dataset_path,
                calib_samples=calib_samples,
                keep_intermediate=keep_intermediate,
                json_mode=json_mode,
            )
            docker_command = shlex.join(container.build_command(sloth_args, **kwargs))
        plan: dict[str, Any] = {
            "dry_run": True,
            "format": fmt,
            "quant": quant,
            "base": base,
            "adapter": str(adapter_abs),
            "output": str(output_abs),
            "estimated_bytes": estimated,
            "free_bytes": free,
            "docker_command": docker_command,
        }
        emit_result(plan if json_mode else _render_plan_text(plan), json_mode=json_mode)
        return 0

    # --- safetensors: pure stdlib, no container, today's behaviour ----------
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

    # --- in-container: the ML seam (lazy import, never reached on the host) --
    if in_container:
        from sloth.tune._exporter import run_export  # lazy: imports the heavy stack

        summary: dict[str, Any] = run_export(
            {
                "format": fmt,
                "quant": quant,
                "base": base,
                "adapter": str(adapter_abs),
                "output": str(output_abs),
                "calib": str(calib) if calib is not None else None,
                "calib_samples": calib_samples,
                "keep_intermediate": keep_intermediate,
                "dataset": _training_dataset(adapter_abs),
                # The host renames <output>.partial -> <output> on success; record
                # the final path so export.json and the result never name .partial.
                "final_output": _final_output(output_abs),
            }
        )
        result = {"output": _final_output(output_abs), "format": fmt, **summary}
        if json_mode:
            emit_result(result, json_mode=True)
        else:
            files = summary.get("files") or {}
            lines = [f"exported {fmt} model to {output_abs}"]
            lines += [f"  {name}: {size} bytes" for name, size in sorted(files.items())]
            emit_result("\n".join(lines), json_mode=False)
        return 0

    # --- host lane: no-clobber → disk gate → atomic container run -----------
    _check_clobber(output_abs, force)
    # Fail closed on the host: a calibrated format with no resolvable calibration
    # source must not cost a container launch (dry-run is exempt so plans render).
    if fmt in CALIBRATED_FORMATS and calib is None and dataset_path is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"--format {fmt} needs calibration data and the adapter's training dataset "
                "could not be resolved from training_metadata.json"
            ),
            remediation="Pass --calib <jsonl> (chat or task schema), or export from the "
            "directory the adapter was trained in so its relative dataset path resolves.",
        )
    _check_disk(estimated, free, output_abs, fmt)

    import sloth.tune.container as container  # lazy: module level stays container-free

    partial = Path(str(output_abs) + PARTIAL_SUFFIX)
    if partial.exists():
        emit_diagnostic(
            f"note: clearing a leftover staging directory from an earlier run: {partial}"
        )
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    kwargs = _container_kwargs(container, adapter_abs, output_abs, calib, dataset_path)
    sloth_args = _sloth_args(
        fmt=fmt,
        adapter=adapter_abs,
        output=partial,
        base=base,
        quant=quant,
        calib=calib if calib is not None else dataset_path,
        calib_samples=calib_samples,
        keep_intermediate=keep_intermediate,
        json_mode=json_mode,
    )

    # launch() raises CliError on any non-zero container exit; a non-zero return is
    # treated the same way. Either way the staging dir is LEFT IN PLACE and <output>
    # is never created, so a killed/OOM run cannot be mistaken for a finished export.
    code = container.launch(sloth_args, **kwargs)
    if code:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"the {fmt} export container exited with status {code}; "
                f"the partial output was left at {partial}"
            ),
            remediation=(
                "Review the container output above, then re-run the export "
                f"(the staging directory {partial} is cleared automatically)."
            ),
        )

    if output_abs.exists():  # only reachable with --force (no-clobber checked above)
        shutil.rmtree(output_abs)
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(output_abs)
    emit_diagnostic(f"export complete: {output_abs}")
    return 0


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``export`` subparser."""
    p = sub.add_parser(
        "export",
        help=("Export a trained adapter: PEFT/safetensors, merged 16/4-bit, gguf, awq or nvfp4."),
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
        help=(
            "Output format: " + ", ".join(sorted(SUPPORTED_FORMATS)) + " (default: safetensors)."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory (default: normalise in place inside --adapter).",
    )
    p.add_argument(
        "--quant",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated ggml quantizations for --format gguf: "
            + ", ".join(sorted(GGML_QUANTS))
            + "."
        ),
    )
    p.add_argument(
        "--calib",
        default=None,
        metavar="PATH",
        help="JSONL calibration data for awq/nvfp4 (default: the run's training dataset).",
    )
    p.add_argument(
        "--calib-samples",
        dest="calib_samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of calibration samples used for awq/nvfp4.",
    )
    p.add_argument(
        "--base",
        default=None,
        metavar="ID",
        help="Base model id (default: adapter_config.json base_model_name_or_path).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty --output directory.",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Resolve and print the export plan (format, base, disk estimate) without exporting.",
    )
    p.add_argument(
        "--keep-intermediate",
        dest="keep_intermediate",
        action="store_true",
        help="Keep intermediate artifacts (e.g. the F16 GGUF) instead of deleting them.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON result to stdout.",
    )
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_export)
