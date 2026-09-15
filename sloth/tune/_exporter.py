"""Lazy in-container export seam — merged/GGUF via Unsloth, AWQ/NVFP4 via llm-compressor.

Like :mod:`sloth.tune._trainer`, this module is allowed to touch the heavy ML
stack (``unsloth``/``torch``/``transformers``/``peft``/``llmcompressor``) but
**only inside** :func:`_load_backend` — never at module top level. Importing
this module (or the ``sloth`` package) stays torch-free so the introspection
verbs keep working on a machine with no GPU and no ML stack.

Public entry point
------------------
:func:`run_export` takes the export *plan* dict produced by
``sloth/cli/_commands/export.py`` and performs the in-container work:

``plan`` keys
    ``format`` (``merged-16bit`` | ``merged-4bit`` | ``gguf`` | ``awq`` |
    ``nvfp4``), ``quant`` (``list[str]``, may be empty), ``base``, ``adapter``,
    ``output`` (already the ``.partial`` directory — write directly into it),
    ``calib``, ``calib_samples``, ``keep_intermediate``, ``dataset``.

Returns ``{"format", "output", "files", "export_json", "calibration",
"versions"}``.

Live-measured facts encoded here (DGX Spark, NGC 25.11, 2026-09-15 — memory
record ``unsloth-cli-quant-export-livetest-2026-09-15``)
--------------------------------------------------------------------------
* Unsloth writes GGUF into ``f"{dir}_gguf/"`` (a *sibling* suffix directory,
  not the requested one) with names like ``LFM2.5-1.2B-Base.Q4_K_M.gguf`` plus
  an ``F16``/``BF16`` intermediate. :func:`_collect_gguf` moves the requested
  quants into the requested dir and drops the intermediate.
* llm-compressor's **default sequential pipeline dies under ``torch.fx``** for
  ``Lfm2`` (``create_causal_mask`` → ``'NoneType' object has no attribute
  'get_mask_sizes'``). ``pipeline="basic"`` is therefore **mandatory**.
* AWQ on LFM2.5-1.2B (GQA 32/8) with a ``v_proj -> out_proj`` mapping fails with
  ``"size of tensor a (512) must match ... b (2048)"`` — so that pair is
  **deliberately absent** from :func:`_lfm2_awq_mappings`.
* compressed-tensors ``0.16.0`` calls ``torch.accelerator.get_memory_info``,
  which only exists in ``torch >= 2.11``; NGC 25.11 ships torch 2.10. See
  :func:`_apply_memory_info_shim`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - scoring a GGUF means invoking llama-completion
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic
from sloth.tune._trainer import _detect_dataset_schema, _format_records, _is_gpu_oom
from sloth.tune.datasets import validate_dataset

# Formats understood by this seam.
MERGED_FORMATS: dict[str, str] = {
    "merged-16bit": "merged_16bit",
    # Unsloth refuses plain "merged_4bit" with an advisory RuntimeError (accuracy
    # warning for later re-saves); the forced variant is the documented opt-in.
    "merged-4bit": "merged_4bit_forced",
}
COMPRESSED_FORMATS: frozenset[str] = frozenset({"awq", "nvfp4"})
SUPPORTED_FORMATS: tuple[str, ...] = (
    "merged-16bit",
    "merged-4bit",
    "gguf",
    "awq",
    "nvfp4",
)

# Default GGUF quantisation when the plan carries no explicit list.
DEFAULT_GGUF_QUANT: tuple[str, ...] = ("q4_k_m",)

# Quant tags Unsloth emits for the un-quantised intermediate it converts *from*.
_INTERMEDIATE_TAGS: frozenset[str] = frozenset({"f16", "bf16", "f32"})

# Below this many calibration rows the quantisation scales get noisy; warn once.
MIN_CALIBRATION_SAMPLES = 64

_NGC_HINT = (
    "The export backend (unsloth + torch + transformers + llmcompressor) is only "
    "available inside the NVIDIA NGC container: nvcr.io/nvidia/pytorch:25.11-py3. "
    "Run `sloth export` on the host (it orchestrates the container for you) rather "
    "than calling the in-container seam directly."
)

_OOM_HINT = (
    "The GPU ran out of memory during export. On the DGX Spark's Unified Memory "
    "Architecture, free host memory and flush the page cache (sudo sh -c 'sync; echo 3 > "
    "/proc/sys/vm/drop_caches'), stop other GPU processes (a running vLLM server holds "
    "tens of GB), then retry. Exporting a smaller quant or fewer calibration samples "
    "(--calib-samples) also lowers the peak."
)

_PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "peft",
    "llmcompressor",
    "compressed_tensors",
)


# ---------------------------------------------------------------------------
# Heavy backend (the ONLY place torch/unsloth/llmcompressor are imported)
# ---------------------------------------------------------------------------


@dataclass
class _Backend:
    """Bundle of the lazily-imported ML callables used by the real export paths.

    Field names are snake_case (not the PascalCase of the imported classes); each
    holds the corresponding callable/module. ``oneshot``/``awq_modifier``/
    ``quantization_modifier``/``awq_mapping``/``compressed_tensors`` are ``None``
    unless the requested format needs llm-compressor.
    """

    torch: Any
    fast_model: Any  # unsloth.FastModel (or FastLanguageModel on older unsloth)
    auto_model_for_causal_lm: Any  # transformers.AutoModelForCausalLM
    auto_tokenizer: Any  # transformers.AutoTokenizer
    oneshot: Any = None  # llmcompressor.oneshot
    awq_modifier: Any = None  # llmcompressor AWQModifier
    quantization_modifier: Any = None  # llmcompressor QuantizationModifier
    awq_mapping: Any = None  # llmcompressor AWQMapping
    compressed_tensors: Any = None  # the compressed_tensors module (for the shim)


def _load_backend(*, need_compressor: bool = False) -> _Backend:
    """Import the heavy ML stack and return it as a :class:`_Backend`.

    This is the single seam where the heavy stack enters the process — isolated
    exactly like :func:`sloth.tune._trainer._load_backend` so tests can
    monkeypatch *one* loader: raising ``ImportError`` here exercises the
    ``CliError(code=2)`` NGC-hint path, and returning a fake exercises the real
    flow GPU-free.

    Raises:
        ImportError: if any required component of the ML stack is unavailable.
    """
    # Unsloth MUST be imported *before* torch/transformers/peft so its runtime
    # patches apply; imported after them it warns and skips its optimizations.
    # Keep this order — do not let the import sorter reorder the block.
    # isort: off
    import unsloth  # noqa: PLC0415 — intentional lazy import; import FIRST
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # isort: on
    # Newer unsloth exposes the generic ``FastModel`` (the one that loads LFM2);
    # older releases only have ``FastLanguageModel``.
    fast_model = getattr(unsloth, "FastModel", None) or unsloth.FastLanguageModel

    backend = _Backend(
        torch=torch,
        fast_model=fast_model,
        auto_model_for_causal_lm=AutoModelForCausalLM,
        auto_tokenizer=AutoTokenizer,
    )
    if need_compressor:
        _attach_compressor(backend)
    return backend


def _attach_compressor(backend: _Backend) -> None:
    """Import llm-compressor onto *backend* (lazy; only for awq/nvfp4)."""
    import compressed_tensors  # noqa: PLC0415 — intentional lazy import
    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415

    try:
        # llm-compressor 0.11 moved AWQ under modifiers.transform.awq.
        from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415
        from llmcompressor.modifiers.transform.awq.mappings import (  # noqa: PLC0415
            AWQMapping,
        )
    except ImportError:  # pragma: no cover - depends on the installed version
        # 0.10 keeps both in modifiers.awq.
        from llmcompressor.modifiers.awq import AWQMapping, AWQModifier  # noqa: PLC0415

    backend.oneshot = oneshot
    backend.awq_modifier = AWQModifier
    backend.quantization_modifier = QuantizationModifier
    backend.awq_mapping = AWQMapping
    backend.compressed_tensors = compressed_tensors


# ---------------------------------------------------------------------------
# The torch.accelerator.get_memory_info shim
# ---------------------------------------------------------------------------


def _apply_memory_info_shim(torch_mod: Any, compressed_tensors_mod: Any) -> bool:
    """Back-fill ``torch.accelerator.get_memory_info`` for compressed-tensors 0.16.0.

    compressed-tensors 0.16.0 (the version llm-compressor 0.11 requires) calls
    ``torch.accelerator.get_memory_info``, which only landed in torch 2.11. NGC
    25.11 ships torch 2.10, so the call raises ``AttributeError`` mid-run. The
    shim maps it to ``torch.cuda.mem_get_info`` (same ``(free, total)`` shape).

    Deliberately narrow: applied **only** when the attribute is missing **and**
    ``compressed_tensors.__version__ == "0.16.0"``. Any other version — or a torch
    that already has the attribute — is left untouched, so this disappears by
    itself once the image moves to torch >= 2.11.

    Returns:
        True when the shim was installed, False when it was skipped.
    """
    accelerator = getattr(torch_mod, "accelerator", None)
    if accelerator is None or hasattr(accelerator, "get_memory_info"):
        return False
    if getattr(compressed_tensors_mod, "__version__", None) != "0.16.0":
        return False

    def _get_memory_info(device: Any = None) -> Any:
        return torch_mod.cuda.mem_get_info(device)

    accelerator.get_memory_info = _get_memory_info
    return True


# ---------------------------------------------------------------------------
# Version / provenance capture (pure stdlib)
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    """Return the installed version of *name*, or ``None`` when absent."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _llama_cpp_tag() -> str | None:
    """Return the llama.cpp build tag used for GGUF conversion, when known.

    ``UNSLOTH_LLAMA_TAG`` (set when a prebuilt llama.cpp tarball is mounted into
    the container) wins; otherwise the prebuilt id recorded in
    ``$HOME/.unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json`` is used.
    """
    tag = os.environ.get("UNSLOTH_LLAMA_TAG")
    if tag:
        return tag
    info = Path(os.path.expanduser("~")) / ".unsloth" / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json"
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("id", "tag", "build", "version"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _versions() -> dict[str, str | None]:
    """Capture the provenance of every component that shaped the artifacts."""
    captured: dict[str, str | None] = {name: _package_version(name) for name in _PACKAGES}
    captured["llama_cpp_tag"] = _llama_cpp_tag()
    return captured


# ---------------------------------------------------------------------------
# Calibration (pure stdlib up to the tokenizer render)
# ---------------------------------------------------------------------------


@dataclass
class _Calibration:
    """A resolved calibration set: where it came from, and its validated records."""

    source: str
    schema: str
    records: list[dict]

    def as_record(self) -> dict[str, Any]:
        return {"source": self.source, "count": len(self.records)}


def _resolve_calibration(plan: dict[str, Any]) -> _Calibration:
    """Load + cap the calibration records for a quantised export.

    Default source is the run's own dataset (``plan["dataset"]``); ``--calib``
    (``plan["calib"]``) overrides it and ``--calib-samples``
    (``plan["calib_samples"]``) caps the record count. Validation happens here —
    *before* any model load — so a broken calibration file costs no GPU time.
    """
    source = plan.get("calib") or plan.get("dataset")
    if not source:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="a quantised export (awq/nvfp4) needs calibration data, but none was given",
            remediation=(
                "Pass --calib <dataset.jsonl>, or export an adapter whose run metadata "
                "records the training dataset so it can be reused for calibration."
            ),
        )
    path = Path(source)
    schema = _detect_dataset_schema(path)
    records = validate_dataset(path, schema=schema)

    cap = plan.get("calib_samples")
    if cap is not None and cap > 0:
        records = records[:cap]

    if len(records) < MIN_CALIBRATION_SAMPLES:
        # Exactly one diagnostic — results stay on stdout, this belongs on stderr.
        emit_diagnostic(
            f"warning: only {len(records)} calibration samples "
            f"(fewer than {MIN_CALIBRATION_SAMPLES}); quantisation scales may be noisy. "
            "Pass --calib with a larger dataset for a better-calibrated export."
        )
    return _Calibration(source=str(path), schema=schema, records=records)


# ---------------------------------------------------------------------------
# Merged export (Unsloth)
# ---------------------------------------------------------------------------


def _load_adapter(backend: _Backend, plan: dict[str, Any]) -> tuple[Any, Any]:
    """Load the trained adapter (base + LoRA deltas) through Unsloth."""
    return backend.fast_model.from_pretrained(
        model_name=plan["adapter"],
        load_in_4bit=False,
        dtype=None,
    )


def _export_merged(backend: _Backend, plan: dict[str, Any], output: Path) -> None:
    """Merge the adapter into the base weights and save 16-bit or 4-bit."""
    save_method = MERGED_FORMATS[plan["format"]]
    model, tokenizer = _load_adapter(backend, plan)
    model.save_pretrained_merged(str(output), tokenizer, save_method=save_method)


def _requested_gguf_quant(plan: dict[str, Any]) -> list[str]:
    """Return the requested GGUF quantisation methods (documented default applies)."""
    quant = list(plan.get("quant") or [])
    return quant or list(DEFAULT_GGUF_QUANT)


def _collect_gguf(output: Path, quant: list[str], *, keep_intermediate: bool) -> None:
    """Move Unsloth's ``<output>_gguf/*.gguf`` files into *output* and drop the intermediate.

    Unsloth does **not** write into the directory it is handed: it writes into a
    sibling directory with a ``_gguf`` suffix (live-measured; e.g.
    ``LFM2.5-1.2B-Base.Q4_K_M.gguf`` plus the ``.F16.gguf`` intermediate it
    converted from). We normalise that back to the requested directory and delete
    the intermediate unless ``keep_intermediate`` is set.
    """
    staged = Path(str(output) + "_gguf")
    if not staged.is_dir():
        return
    wanted = {q.lower() for q in quant}
    for path in sorted(staged.glob("*.gguf")):
        tag = path.name.rsplit(".", 2)[-2].lower() if path.name.count(".") >= 2 else ""
        is_intermediate = tag in _INTERMEDIATE_TAGS and tag not in wanted
        if is_intermediate and not keep_intermediate:
            path.unlink()
            continue
        shutil.move(str(path), str(output / path.name))
    # Remove the staging directory when nothing is left behind in it.
    if not any(staged.iterdir()):
        staged.rmdir()


def _export_gguf(backend: _Backend, plan: dict[str, Any], output: Path) -> None:
    """Convert the adapter to GGUF, then normalise Unsloth's ``_gguf`` suffix dir."""
    quant = _requested_gguf_quant(plan)
    model, tokenizer = _load_adapter(backend, plan)
    model.save_pretrained_gguf(str(output), tokenizer, quantization_method=quant)
    keep = bool(plan.get("keep_intermediate"))
    _collect_gguf(output, quant, keep_intermediate=keep)
    _drop_merged_weights(output, keep_intermediate=keep)


def _drop_merged_weights(output: Path, *, keep_intermediate: bool) -> None:
    """Delete the merged ``*.safetensors`` Unsloth also writes next to a GGUF export.

    ``save_pretrained_gguf`` first materialises the merged bf16 model **inside** the
    requested directory (live-measured: a 2.34 GB ``model.safetensors`` for
    LFM2.5-1.2B) before converting. For a GGUF export those weights are an
    intermediate, so they are removed unless ``keep_intermediate`` is set; the small
    tokenizer/config files stay.
    """
    if keep_intermediate:
        return
    for path in output.glob("*.safetensors"):
        path.unlink()
    index = output / "model.safetensors.index.json"
    if index.exists():
        index.unlink()


# ---------------------------------------------------------------------------
# Quantised export (llm-compressor)
# ---------------------------------------------------------------------------


def _lfm2_awq_mappings(layer_types: list[str], awq_mapping: Any) -> list[Any]:
    """Build per-layer AWQ smoothing mappings for an LFM2 hybrid model.

    LFM2 alternates ``full_attention`` and short-conv layers (``config.layer_types``
    says which is which), and its norms/projections are named unlike a Llama block,
    so llm-compressor's default mappings match nothing. Per live-tested layout:

    * ``operator_norm`` → ``self_attn.{q,k,v}_proj`` on ``full_attention`` layers
    * ``operator_norm`` → ``conv.in_proj`` on conv layers
    * ``ffn_norm`` → ``feed_forward.{w1,w3}``
    * ``feed_forward.w3`` → ``feed_forward.w2``

    There is deliberately **no ``v_proj`` → ``out_proj`` pair**: LFM2.5-1.2B is GQA
    32/8, so v_proj's output (512) does not match out_proj's input (2048) and AWQ
    aborts with ``"size of tensor a (512) must match the size of tensor b (2048)"``.
    """
    mappings: list[Any] = []
    for index, kind in enumerate(layer_types):
        prefix = rf"re:.*layers\.{index}\."
        if kind == "full_attention":
            balance = [
                prefix + r"self_attn\.q_proj$",
                prefix + r"self_attn\.k_proj$",
                prefix + r"self_attn\.v_proj$",
            ]
        else:
            balance = [prefix + r"conv\.in_proj$"]
        mappings.append(awq_mapping(prefix + r"operator_norm$", balance))
        mappings.append(
            awq_mapping(
                prefix + r"ffn_norm$",
                [prefix + r"feed_forward\.w1$", prefix + r"feed_forward\.w3$"],
            )
        )
        mappings.append(awq_mapping(prefix + r"feed_forward\.w3$", [prefix + r"feed_forward\.w2$"]))
    return mappings


def _build_recipe(backend: _Backend, fmt: str, model: Any) -> list[Any]:
    """Build the llm-compressor recipe for *fmt* against the loaded merged *model*."""
    if fmt == "nvfp4":
        return [
            backend.quantization_modifier(scheme="NVFP4", targets=["Linear"], ignore=["lm_head"])
        ]

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    layer_types = getattr(config, "layer_types", None)
    awq_kwargs: dict[str, Any] = {"duo_scaling": "both"}
    if model_type == "lfm2" and layer_types:
        # Only LFM2 needs hand-built mappings; every other architecture gets
        # llm-compressor's own defaults.
        awq_kwargs["mappings"] = _lfm2_awq_mappings(list(layer_types), backend.awq_mapping)
    return [
        backend.awq_modifier(**awq_kwargs),
        backend.quantization_modifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head"]),
    ]


def _export_compressed(
    backend: _Backend,
    plan: dict[str, Any],
    output: Path,
    calibration: _Calibration,
) -> None:
    """Merge to 16-bit, then one-shot quantise to AWQ W4A16_ASYM or NVFP4."""
    # Step 1 — a 16-bit merged checkpoint is the input llm-compressor quantises.
    merged_dir = output / "_merged-16bit"
    model, tokenizer = _load_adapter(backend, plan)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # Step 2 — reload the merged checkpoint through plain transformers (Unsloth's
    # patched model is not what llm-compressor expects to trace/quantise).
    model = backend.auto_model_for_causal_lm.from_pretrained(  # nosec B615
        str(merged_dir), dtype=backend.torch.bfloat16, local_files_only=True
    )
    tokenizer = backend.auto_tokenizer.from_pretrained(  # nosec B615
        str(merged_dir), local_files_only=True
    )

    _apply_memory_info_shim(backend.torch, backend.compressed_tensors)

    rows = _format_records(calibration.records, calibration.schema, tokenizer)
    backend.oneshot(
        model=model,
        dataset=rows,
        recipe=_build_recipe(backend, plan["format"], model),
        # MANDATORY for LFM2: the default sequential pipeline traces the model with
        # torch.fx and dies in create_causal_mask ("'NoneType' object has no
        # attribute 'get_mask_sizes'") on the hybrid cache.
        pipeline="basic",
        num_calibration_samples=len(rows),
    )
    model.save_pretrained(str(output), save_compressed=True)
    tokenizer.save_pretrained(str(output))

    if not plan.get("keep_intermediate"):
        shutil.rmtree(merged_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Result recording (pure stdlib)
# ---------------------------------------------------------------------------


def _collect_files(output: Path) -> dict[str, int]:
    """Return ``{relative name: size in bytes}`` for every file under *output*."""
    files: dict[str, int] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(output))] = path.stat().st_size
    return files


def _append_index(adapter: Path, record: dict[str, Any]) -> None:
    """Append *record* to ``<adapter>/exports.json`` (a JSON list; created if absent)."""
    index_path = adapter / "exports.json"
    entries: list[Any] = []
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, list):
            entries = loaded
    entries.append(record)
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def _write_export_json(
    plan: dict[str, Any],
    output: Path,
    calibration: _Calibration | None,
    timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``<output>/export.json`` and mirror the record into the adapter's index."""
    final_output = plan.get("final_output") or str(output)
    record = {
        "format": plan["format"],
        "output": final_output,
        "quant": list(plan.get("quant") or []),
        "base": plan.get("base"),
        "adapter": plan.get("adapter"),
        "files": _collect_files(output),
        "calibration": calibration.as_record() if calibration is not None else None,
        "versions": _versions(),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    export_json = output / "export.json"
    export_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _append_index(Path(plan["adapter"]), record)
    return export_json, record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_export(plan: dict[str, Any]) -> dict[str, Any]:
    """Run the in-container export described by *plan* and record what was produced.

    Parameters
    ----------
    plan:
        See the module docstring for the full key list. ``plan["output"]`` is
        already the ``.partial`` directory — artifacts are written directly into
        it; the calling verb performs the atomic rename.

    Returns
    -------
    dict
        ``{"format", "output", "files", "export_json", "calibration", "versions"}``.

    Raises
    ------
    CliError(code=1)
        For an unknown format, or a quantised export with no calibration data.
    CliError(code=2)
        When the ML stack is unavailable (NGC hint) or the GPU runs out of memory
        (memory hint).
    """
    fmt = plan.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unknown export format {fmt!r}",
            remediation=f"Use one of: {', '.join(SUPPORTED_FORMATS)}.",
        )

    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)

    # Calibration is validated BEFORE the backend import — a broken calibration
    # file must not cost a model load.
    calibration = _resolve_calibration(plan) if fmt in COMPRESSED_FORMATS else None

    try:
        backend = _load_backend(need_compressor=fmt in COMPRESSED_FORMATS)
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"The export backend is not installed: {exc}",
            remediation=_NGC_HINT,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Unsloth's GPU probe can raise a CUDA OOM at import time — an environment
        # error (exit 2) with a memory remediation, not a code-1 bug.
        if _is_gpu_oom(exc):
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"GPU out of memory while initializing the export backend: {exc}",
                remediation=_OOM_HINT,
            ) from exc
        raise

    try:
        if fmt in MERGED_FORMATS:
            _export_merged(backend, plan, output)
        elif fmt == "gguf":
            _export_gguf(backend, plan, output)
        else:
            _export_compressed(backend, plan, output, calibration)
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_gpu_oom(exc):
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"GPU out of memory during export: {exc}",
                remediation=_OOM_HINT,
            ) from exc
        raise

    export_json, record = _write_export_json(plan, output, calibration)
    final_output = plan.get("final_output") or str(output)
    return {
        "format": fmt,
        "output": final_output,
        "files": record["files"],
        "export_json": str(Path(final_output) / export_json.name),
        "calibration": record["calibration"],
        "versions": record["versions"],
    }


# ---------------------------------------------------------------------------
# Evaluating a merged / quantized model directory (``sloth eval --model DIR``)
# ---------------------------------------------------------------------------

#: Name of the llama.cpp binary used to score a GGUF. It is ``llama-completion``,
#: **not** ``llama-cli``: ``-no-cnv`` (single-turn, no conversation wrapper) is only
#: valid on ``llama-completion``. It lives in the llama.cpp cache that
#: :func:`sloth.tune.container.export_launch_kwargs` bind-mounts at
#: ``<HOME>/.unsloth/llama.cpp`` — the same directory must also be on
#: ``LD_LIBRARY_PATH`` for the shared libs next to the binary.
LLAMA_COMPLETION_BIN: str = "llama-completion"

#: Default number of tokens generated per eval item (matches ``run_eval``).
EVAL_MAX_NEW_TOKENS: int = 100

_EVAL_NGC_HINT = (
    "The eval backend (torch + transformers, plus compressed-tensors for an "
    "awq/nvfp4 directory) is only available inside the NVIDIA NGC container: "
    "nvcr.io/nvidia/pytorch:25.11-py3. Run `sloth eval --model <dir>` on the host "
    "(it orchestrates the container for you) rather than calling the in-container "
    "seam directly."
)

_LLAMA_HINT = (
    "GGUF scoring runs llama.cpp's `llama-completion` from the cache mounted at "
    "$HOME/.unsloth/llama.cpp. Run `sloth export --format gguf …` once (it populates "
    "that cache), or set SLOTH_LLAMA_CPP_CACHE to a directory holding a llama.cpp "
    "build, then retry."
)


@dataclass
class _EvalBackend:
    """The lazily-imported ML callables used to score a merged/quantized directory."""

    torch: Any
    auto_model_for_causal_lm: Any  # transformers.AutoModelForCausalLM
    auto_tokenizer: Any  # transformers.AutoTokenizer


def _load_eval_backend() -> _EvalBackend:
    """Import torch + transformers and return them as an :class:`_EvalBackend`.

    The single heavy-import seam for ``sloth eval --model`` — mirroring
    :func:`_load_backend`, so a test monkeypatches *one* function: raising
    ``ImportError`` exercises the ``CliError(code=2)`` NGC path and returning a fake
    exercises the real flow GPU-free.

    No unsloth import here: a merged/AWQ/NVFP4 checkpoint is a plain transformers
    model (compressed-tensors quantisation is auto-detected from ``config.json``).

    Raises:
        ImportError: if torch or transformers is unavailable.
    """
    import torch  # noqa: PLC0415 — intentional lazy import
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    return _EvalBackend(
        torch=torch,
        auto_model_for_causal_lm=AutoModelForCausalLM,
        auto_tokenizer=AutoTokenizer,
    )


def _quant_info(model_dir: Path) -> tuple[str | None, str | None]:
    """Return ``(quant_method, quant_format)`` read from ``<model_dir>/config.json``.

    compressed-tensors outputs record their quantisation under
    ``config.json["quantization_config"]`` as ``{"quant_method": "compressed-tensors",
    "format": "pack-quantized"}`` (AWQ W4A16) or ``"nvfp4-pack-quantized"`` (NVFP4).
    A plain bf16 merged checkpoint has no such key, and a GGUF directory has no
    ``config.json`` at all — both yield ``(None, None)``.
    """
    config_file = model_dir / "config.json"
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(config, dict):
        return None, None
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return None, None
    method = quant.get("quant_method")
    fmt = quant.get("format")
    return (method if isinstance(method, str) else None, fmt if isinstance(fmt, str) else None)


def _find_gguf(model_dir: Path) -> Path | None:
    """Return the first ``*.gguf`` file directly inside *model_dir*, or ``None``."""
    candidates = sorted(model_dir.glob("*.gguf"))
    return candidates[0] if candidates else None


def _eval_prompt(record: dict[str, Any]) -> str:
    """Render one task-schema record into the prompt shape used by ``run_eval``."""
    return f"Task: {record['task']}\nInput: {record['input']}\nOutput:"


def _score_predictions(records: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """Score *predictions* against *records* exactly as ``run_eval`` scores an adapter.

    Deliberately a **duplicate** of the comparison run_eval performs inline
    (``prediction.strip() == expected.strip()``, same per-record keys, same
    ``exact_match_pct`` rounding): extracting it from ``_trainer.run_eval`` would mean
    editing that function, and this seam must not change its behaviour.
    """
    results: list[dict[str, Any]] = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        expected = record["expected_output"]
        results.append(
            {
                "index": index,
                "task": record["task"],
                "input": record["input"],
                "expected_output": expected,
                "prediction": prediction,
                "exact_match": prediction.strip() == expected.strip(),
            }
        )
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    return {
        "total": total,
        "exact_match": exact,
        "exact_match_pct": round(exact / total * 100, 2) if total else 0.0,
        "results": results,
    }


def _llama_cpp_dir() -> Path:
    """Return the in-container llama.cpp cache directory (``$HOME/.unsloth/llama.cpp``)."""
    return Path(os.path.expanduser("~")) / ".unsloth" / "llama.cpp"


def _run_llama_completion(gguf: Path, prompt: str, max_tokens: int) -> str:
    """Score one prompt with ``llama-completion`` and return its stdout.

    Invoked once per suite item as
    ``llama-completion -m <gguf> -p <prompt> -n <max_tokens> --temp 0 -no-cnv``
    with ``LD_LIBRARY_PATH`` pointing at the llama.cpp cache (the shared libraries
    sit next to the binary). ``--temp 0`` makes the run deterministic and ``-no-cnv``
    keeps it single-turn.
    """
    cache = _llama_cpp_dir()
    binary = cache / LLAMA_COMPLETION_BIN
    if not binary.is_file():
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"llama.cpp binary not found: {binary}",
            remediation=_LLAMA_HINT,
        )
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = f"{cache}:{existing}" if existing else str(cache)
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        [
            str(binary),
            "-m",
            str(gguf),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            "0",
            "-no-cnv",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"{LLAMA_COMPLETION_BIN} exited with status {proc.returncode} "
                f"while scoring {gguf.name}: {proc.stderr.strip()[:400]}"
            ),
            remediation=_LLAMA_HINT,
        )
    output = proc.stdout
    # Defensive: some llama.cpp builds echo the prompt ahead of the completion.
    if output.startswith(prompt):
        output = output[len(prompt) :]
    return output


def _predict_gguf(gguf: Path, records: list[dict[str, Any]], max_tokens: int) -> list[str]:
    """Return one completion per record, scored through llama.cpp."""
    return [_run_llama_completion(gguf, _eval_prompt(record), max_tokens) for record in records]


def _predict_transformers(
    backend: _EvalBackend,
    model_dir: Path,
    records: list[dict[str, Any]],
    max_tokens: int,
) -> list[str]:
    """Return one completion per record from a transformers-loadable directory.

    ``AutoModelForCausalLM.from_pretrained(dir, dtype=torch.bfloat16)`` auto-detects a
    compressed-tensors checkpoint (AWQ ``pack-quantized`` / NVFP4
    ``nvfp4-pack-quantized``) from ``config.json``; a bf16 merged dir loads plainly.
    ``local_files_only=True`` keeps the run offline.
    """
    tokenizer = backend.auto_tokenizer.from_pretrained(  # nosec B615
        str(model_dir), local_files_only=True
    )
    model = backend.auto_model_for_causal_lm.from_pretrained(  # nosec B615
        str(model_dir), dtype=backend.torch.bfloat16, local_files_only=True
    )
    model.eval()
    # Tokenized inputs must share the model's device, else generate() raises
    # "Expected all tensors to be on the same device" (same constraint as run_eval).
    device = next(model.parameters()).device
    predictions: list[str] = []
    for record in records:
        inputs = tokenizer(_eval_prompt(record), return_tensors="pt").to(device)
        with backend.torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return predictions


def run_eval_model(
    model_dir: str,
    suite_path: str,
    *,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Evaluate a merged / quantized model **directory** against a task-schema suite.

    The ``--model`` counterpart of :func:`sloth.tune._trainer.run_eval` (which scores a
    LoRA *adapter*). Two backends, chosen by what the directory holds:

    * a ``*.gguf`` file → scored with llama.cpp's ``llama-completion`` from the cache
      mounted at ``$HOME/.unsloth/llama.cpp`` (see :func:`_run_llama_completion`);
    * otherwise → loaded with transformers, which auto-detects a compressed-tensors
      AWQ/NVFP4 checkpoint from ``config.json`` (see :func:`_predict_transformers`).

    Parameters
    ----------
    model_dir:
        Directory holding a merged (bf16/4-bit), AWQ, NVFP4 or GGUF export.
    suite_path:
        Path to a task-schema JSONL eval suite.
    max_new_tokens:
        Generation budget per item.

    Returns
    -------
    dict
        The same score fields as adapter eval — ``total``, ``exact_match``,
        ``exact_match_pct``, ``results`` — plus ``model_dir``, ``quant_method`` and
        ``quant_format`` (from ``config.json``'s ``quantization_config``; ``None``
        for a plain bf16 merged dir or a GGUF).

    Raises
    ------
    CliError(code=1)
        When *model_dir* is not a directory (or the suite fails task-schema validation).
    CliError(code=2)
        When the ML stack is unavailable, llama.cpp is missing, or the GPU OOMs.
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"model directory not found: {directory}",
            remediation=(
                "Pass an existing merged/quantized model directory with --model <path>. "
                "Run `sloth export` to produce one."
            ),
        )

    # Validate the suite BEFORE any model load — a broken suite must cost no GPU time.
    records = validate_dataset(Path(suite_path), schema="task")
    quant_method, quant_format = _quant_info(directory)

    gguf = _find_gguf(directory)
    try:
        if gguf is not None:
            predictions = _predict_gguf(gguf, records, max_new_tokens)
        else:
            try:
                backend = _load_eval_backend()
            except ImportError as exc:
                raise CliError(
                    code=EXIT_ENV_ERROR,
                    message=f"The eval backend is not installed: {exc}",
                    remediation=_EVAL_NGC_HINT,
                ) from exc
            predictions = _predict_transformers(backend, directory, records, max_new_tokens)
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_gpu_oom(exc):
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"GPU out of memory during eval: {exc}",
                remediation=_OOM_HINT,
            ) from exc
        raise

    summary = _score_predictions(records, predictions)
    summary["model_dir"] = str(directory)
    summary["quant_method"] = quant_method
    summary["quant_format"] = quant_format
    return summary
