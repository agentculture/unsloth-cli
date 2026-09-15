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
from typing import Any, Sequence

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic
from sloth.tune import metrics
from sloth.tune._trainer import (
    DEFAULT_EVAL_BATCH_SIZE,
    _detect_dataset_schema,
    _format_records,
    _generate_predictions,
    _is_gpu_oom,
    eval_prompt,
    resolve_suite_paths,
    write_eval_json,
)
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

#: Calibration sequence length handed to llm-compressor. Passed explicitly because
#: oneshot otherwise falls back to the tokenizer's ``model_max_length``, and LFM2's
#: is a huge sentinel that overflows ("OverflowError: int too big to convert",
#: live-measured). 512 matches the probe run that validated AWQ/NVFP4 on LFM2.5.
CALIB_MAX_SEQ_LENGTH: int = 512

#: PEFT adapter config filename (read for the base id, rewritten for a --base override).
ADAPTER_CONFIG_NAME: str = "adapter_config.json"

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
    dataset_from_list: Any = None  # datasets.Dataset.from_list (oneshot needs a Dataset)


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
        # llm-compressor 0.11 moved AWQ under modifiers.transform.awq. The old
        # ``modifiers.awq.AWQModifier`` path still exists there but is a shim
        # *function* returning ``[AWQModifier, QuantizationModifier]`` — with only
        # mapping kwargs that second modifier is bare and oneshot fails with
        # "QuantizationModifier requires that quantization fields be specified"
        # (live-measured). Always prefer the class from the new path.
        from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier  # noqa: PLC0415
    except ImportError:  # pragma: no cover - depends on the installed version
        # 0.10 keeps both in modifiers.awq (there AWQModifier is the real class).
        from llmcompressor.modifiers.awq import AWQMapping, AWQModifier  # noqa: PLC0415

    # llmcompressor.oneshot reads ``dataset.column_names``: a plain list of dicts
    # fails with "'list' object has no attribute 'column_names'" (live-measured),
    # so calibration rows are wrapped in a datasets.Dataset (in the dep layer).
    from datasets import Dataset  # noqa: PLC0415

    backend.dataset_from_list = Dataset.from_list
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


def _adapter_for_base(plan: dict[str, Any], output: Path) -> str:
    """Return the adapter dir to load, honouring a ``--base`` override.

    Unsloth resolves the base from ``adapter_config.json``'s
    ``base_model_name_or_path``. When the plan's ``base`` differs, a staged copy of
    the adapter is written under ``<output>/_adapter-override`` with that one key
    rewritten (weights are symlinked), so the proven load path is unchanged and the
    override really is what gets merged — not just what provenance records.
    """
    adapter = Path(plan["adapter"])
    base = plan.get("base")
    config_path = adapter / ADAPTER_CONFIG_NAME
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return str(adapter)
    if not base or config.get("base_model_name_or_path") == base:
        return str(adapter)
    staged = output / "_adapter-override"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for entry in adapter.iterdir():
        if entry.name == ADAPTER_CONFIG_NAME or entry.name.startswith("_"):
            continue
        if entry.is_file():
            os.symlink(entry.resolve(), staged / entry.name)
    config["base_model_name_or_path"] = base
    (staged / ADAPTER_CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(staged)


def _drop_override_staging(output: Path, *, keep_intermediate: bool) -> None:
    """Remove the ``--base`` override staging dir after an export (unless kept)."""
    staged = output / "_adapter-override"
    if staged.exists() and not keep_intermediate:
        shutil.rmtree(staged, ignore_errors=True)


def _load_adapter(
    backend: _Backend, plan: dict[str, Any], output: Path, *, load_in_4bit: bool = False
) -> tuple[Any, Any]:
    """Load the trained adapter (base + LoRA deltas) through Unsloth."""
    return backend.fast_model.from_pretrained(
        model_name=_adapter_for_base(plan, output),
        load_in_4bit=load_in_4bit,
        dtype=None,
    )


def _export_merged(backend: _Backend, plan: dict[str, Any], output: Path) -> None:
    """Merge the adapter into the base weights and save 16-bit or 4-bit.

    ``merged_4bit`` requires the base to be *loaded* quantised — Unsloth raises
    "Model does not appear to be quantized" otherwise (live-measured) — so the
    4-bit format loads the base with ``load_in_4bit=True`` before merging.
    """
    save_method = MERGED_FORMATS[plan["format"]]
    model, tokenizer = _load_adapter(
        backend, plan, output, load_in_4bit=(plan["format"] == "merged-4bit")
    )
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
    model, tokenizer = _load_adapter(backend, plan, output)
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


def _disable_dynamo(torch_mod: Any) -> bool:
    """Run the calibration forward passes eagerly (returns True when applied).

    The export process imports Unsloth first (for the merge), and Unsloth patches
    transformers' forwards with ``torch.compile``. llm-compressor's calibration then
    trips Dynamo on those patched functions ("Unsupported: Missing bytecode handler
    ... MATCH_CLASS", live-measured on NVFP4). Disabling Dynamo makes compiled
    functions fall back to eager execution; quantisation needs no compilation.
    """
    dynamo = getattr(torch_mod, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is None:
        return False
    try:
        config.disable = True
    except Exception:  # noqa: BLE001 - never let a diagnostic knob break an export
        return False
    return True


def _export_compressed(
    backend: _Backend,
    plan: dict[str, Any],
    output: Path,
    calibration: _Calibration,
) -> None:
    """Merge to 16-bit, then one-shot quantise to AWQ W4A16_ASYM or NVFP4."""
    # Step 1 — a 16-bit merged checkpoint is the input llm-compressor quantises.
    merged_dir = output / "_merged-16bit"
    model, tokenizer = _load_adapter(backend, plan, output)
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
    _disable_dynamo(backend.torch)

    rows = _format_records(calibration.records, calibration.schema, tokenizer)
    backend.oneshot(
        model=model,
        dataset=(backend.dataset_from_list(rows) if backend.dataset_from_list else rows),
        recipe=_build_recipe(backend, plan["format"], model),
        # MANDATORY for LFM2: the default sequential pipeline traces the model with
        # torch.fx and dies in create_causal_mask ("'NoneType' object has no
        # attribute 'get_mask_sizes'") on the hybrid cache.
        pipeline="basic",
        max_seq_length=CALIB_MAX_SEQ_LENGTH,
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
    lock_path = adapter / ".exports.json.lock"
    # Serialise concurrent exports of the same adapter (cross-process flock) so a
    # read-modify-write never drops another export's record.
    with open(lock_path, "w", encoding="utf-8") as lock:
        _flock(lock)
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


def _flock(handle: Any) -> None:
    """Take an exclusive advisory lock (POSIX); a no-op where fcntl is unavailable."""
    try:
        import fcntl  # noqa: PLC0415 - POSIX only
    except ImportError:  # pragma: no cover - non-POSIX
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _write_export_json(
    plan: dict[str, Any],
    output: Path,
    calibration: _Calibration | None,
    timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``<output>/export.json`` and mirror the record into the adapter's index."""
    final_output = plan.get("final_output") or str(output)
    quant = list(plan.get("quant") or [])
    if plan["format"] == "gguf" and not quant:
        quant = list(DEFAULT_GGUF_QUANT)  # the default actually used for conversion
    record = {
        "format": plan["format"],
        "output": final_output,
        "quant": quant,
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

    _drop_override_staging(output, keep_intermediate=bool(plan.get("keep_intermediate")))
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


def _quant_matches(name: str, quant: str) -> bool:
    """True when GGUF file *name* carries the quant tag *quant* (case-insensitive)."""
    return quant.strip().lower() in name.lower()


def _find_gguf(model_dir: Path, quant: str | None = None) -> Path | None:
    """Return the ``*.gguf`` inside *model_dir* to score (``None`` if there is none).

    With **one** GGUF present that file is returned and *quant* is ignored — which
    is what ``sloth eval --quant`` documents ("ignored for a single-GGUF or
    non-GGUF --model directory").

    With **several** (e.g. a ``--quant q4_k_m,q8_0`` export, or a kept F16
    intermediate) the directory is ambiguous:

    * *quant* given → the file whose name contains that tag, matched
      case-insensitively (``q4_k_m`` selects ``…-Q4_K_M.gguf``). No match is a
      user error whose hint lists the names actually present; an ambiguous match
      (several files carrying the tag) is a user error too.
    * *quant* omitted → the historical error: exit 1 and ask for the exact file
      via ``--model <file.gguf>``, rather than silently scoring the
      alphabetically first one.
    """
    candidates = sorted(model_dir.glob("*.gguf"))
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    names = ", ".join(c.name for c in candidates)
    if quant:
        matches = [c for c in candidates if _quant_matches(c.name, quant)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"no GGUF file in {model_dir} matches --quant {quant}",
                remediation=(
                    f"The directory holds: {names}. Pass --quant with one of those "
                    "tags, or point --model at the exact file, e.g. "
                    "--model <dir>/<name>.gguf."
                ),
            )
        matched = ", ".join(m.name for m in matches)
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--quant {quant} matches several GGUF files in {model_dir}: {matched}",
            remediation="Point --model at the exact file, e.g. --model <dir>/<name>.gguf.",
        )

    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"{model_dir} holds several GGUF files: {names}",
        remediation=(
            "Select one with --quant <tag> (e.g. --quant q4_k_m), or point --model "
            "at the exact file, e.g. --model <dir>/<name>.gguf."
        ),
    )


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
    """Return one completion per record, scored through llama.cpp.

    llama.cpp's ``llama-completion`` takes exactly one prompt per process, so this
    path is inherently unbatched — ``--batch-size`` applies to the transformers
    path only.
    """
    return [_run_llama_completion(gguf, eval_prompt(record), max_tokens) for record in records]


def _predict_transformers(
    backend: _EvalBackend,
    model_dir: Path,
    records: list[dict[str, Any]],
    max_tokens: int,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> list[str]:
    """Return one completion per record from a transformers-loadable directory.

    ``AutoModelForCausalLM.from_pretrained(dir, dtype=torch.bfloat16)`` auto-detects a
    compressed-tensors checkpoint (AWQ ``pack-quantized`` / NVFP4
    ``nvfp4-pack-quantized``) from ``config.json``; a bf16 merged dir loads plainly.
    ``local_files_only=True`` keeps the run offline.

    Generation itself is delegated to :func:`sloth.tune._trainer._generate_predictions`
    — the same left-padded, batched loop the ``--adapter`` seam uses — so both
    seams slice the prompt off identically and cannot drift apart.
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
    return _generate_predictions(
        backend.torch,
        model,
        tokenizer,
        [eval_prompt(record) for record in records],
        batch_size=batch_size,
        max_new_tokens=max_tokens,
        device=device,
    )


def run_eval_model(
    model_dir: str,
    suite_path: str | None = None,
    *,
    suite_paths: Sequence[str | Path] | None = None,
    quant: str | None = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
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
        Directory holding a merged (bf16/4-bit), AWQ, NVFP4 or GGUF export — or a
        single ``.gguf`` file. ``eval.json`` is written into that directory.
    suite_path:
        A single task-schema JSONL suite — the historical positional argument,
        still honoured for direct/legacy callers.
    suite_paths:
        Every resolved suite file, in order (what ``sloth eval`` passes once it
        detects this signature). Wins over *suite_path* when both are given.
    quant:
        Which GGUF to score when the directory holds several (case-insensitive
        tag, e.g. ``q4_k_m``); ignored for a single-GGUF or transformers-loadable
        directory. See :func:`_find_gguf`.
    batch_size:
        Prompts per ``generate()`` call on the transformers path (left-padded;
        the llama.cpp path is one process per prompt and ignores it).
    max_new_tokens:
        Generation budget per item.

    Returns
    -------
    dict
        The same shape as adapter eval — the aggregate ``total``,
        ``exact_match``, ``exact_match_pct``, ``f1``, ``results`` plus per-file
        ``files`` entries — with ``model_dir``, ``quant_method`` and
        ``quant_format`` added (from ``config.json``'s ``quantization_config``;
        ``None`` for a plain bf16 merged dir or a GGUF).

    Raises
    ------
    CliError(code=1)
        When *model_dir* is not a directory, the suite fails task-schema
        validation, or *quant* matches no GGUF in an ambiguous directory.
    CliError(code=2)
        When the ML stack is unavailable, llama.cpp is missing, or the GPU OOMs.
    """
    resolved_suites = resolve_suite_paths(suite_path, suite_paths)
    directory = Path(model_dir)
    if directory.is_file() and directory.suffix == ".gguf":
        gguf_file = directory
        directory = directory.parent
    else:
        gguf_file = None
    if not directory.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"model directory not found: {directory}",
            remediation=(
                "Pass an existing merged/quantized model directory with --model <path>. "
                "Run `sloth export` to produce one."
            ),
        )

    # Validate every suite file BEFORE any model load — a broken suite must cost no
    # GPU time. Records are then scored as one flat batch so the model is loaded
    # once, and split back per file for the ``files`` entries afterwards.
    records_by_file = [
        (path, validate_dataset(Path(path), schema="task")) for path in resolved_suites
    ]
    records = [record for _, file_records in records_by_file for record in file_records]
    quant_method, quant_format = _quant_info(directory)

    gguf = gguf_file if gguf_file is not None else _find_gguf(directory, quant)
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
            predictions = _predict_transformers(
                backend, directory, records, max_new_tokens, batch_size
            )
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

    files: list[dict[str, Any]] = []
    cursor = 0
    for path, file_records in records_by_file:
        chunk = predictions[cursor : cursor + len(file_records)]
        scored = metrics.score_records(file_records, chunk, start_index=cursor, source=str(path))
        cursor += len(file_records)
        files.append(metrics.file_entry(path, scored))

    summary = metrics.aggregate(files)
    summary["model_dir"] = str(directory)
    summary["quant_method"] = quant_method
    summary["quant_format"] = quant_format
    write_eval_json(directory, summary, resolved_suites, target="model")
    return summary
