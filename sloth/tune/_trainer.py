"""Lazy LoRA/QLoRA trainer adapter — the ONLY module that touches torch/unsloth.

The heavy ML stack (``unsloth``, ``torch``, ``trl``) is imported **only** inside
:func:`_load_backend`, never at module top level. Importing this module — or the
``sloth`` package — stays torch-free, so the introspection verbs keep working on
a machine with no GPU and no ML stack installed (the repo's load-bearing
"zero runtime dependencies" rule).

Public entry point
------------------
:func:`run_training` resolves a :class:`~sloth.tune.config.RunConfig` into a
training *plan* and, unless ``dry_run`` is set, runs the adapter job.

Out-of-scope policy (documented contract)
----------------------------------------
* **Dry-run never raises on scope.** It returns the resolved plan with the
  scope decision embedded under ``plan["scope"]`` so the calling verb can warn,
  downgrade, or decide. A dry-run never imports torch.
* **A real (non-dry-run) run hard-refuses an out-of-scope request** by raising
  ``CliError(code=1)`` *before* importing the heavy backend — so no GPU time is
  spent setting up a job that this tool will not run (full fine-tuning of large
  dense models is explicitly out of scope; use ``lora``/``qlora``).

Missing-backend policy
----------------------
When the ML stack is absent, :func:`_load_backend` raises ``ImportError`` and
:func:`run_training` converts it into ``CliError(code=2)`` carrying the
``uv tool install unsloth-cli`` install hint. Isolating the heavy import in a
tiny helper makes it monkeypatchable: tests inject an ``ImportError`` (to assert
the install-hint path) or a fake backend (to exercise the real flow) without a
GPU.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic
from sloth.tune import metrics, scorers
from sloth.tune.config import RunConfig
from sloth.tune.datasets import detect_schema, render_chat_prompt, validate_dataset
from sloth.tune.metadata import write_metadata
from sloth.tune.presets import resolve_target_modules
from sloth.tune.scope import check_scope

_INSTALL_HINT = (
    "The fine-tuning backend (unsloth + torch + trl) ships with unsloth-cli. "
    "Reinstall it with `uv tool install unsloth-cli` (or `uv sync` in a checkout) "
    "on a CUDA-capable machine, then re-run."
)

_NGC_HINT = (
    "No GPU accelerator was found. Run the trainer inside the NVIDIA NGC container: "
    "nvcr.io/nvidia/pytorch:25.11-py3 "
    "(which includes the required CUDA drivers, unsloth, torch, and trl)."
)

_OOM_HINT = (
    "The GPU ran out of memory. On the DGX Spark's Unified Memory Architecture, free "
    "host memory and flush the page cache (sudo sh -c 'sync; echo 3 > "
    "/proc/sys/vm/drop_caches'), stop other GPU processes, then retry. You can also "
    "reduce batch_size / max_seq_len, or use method='qlora' (4-bit) for a smaller "
    "footprint (the container already sets PYTORCH_ALLOC_CONF=expandable_segments:True)."
)


def _is_gpu_oom(exc: BaseException) -> bool:
    """Return True when *exc* looks like a CUDA/accelerator out-of-memory error.

    Detects by class name and message so we need not import torch to reference its
    ``OutOfMemoryError`` / ``AcceleratorError`` types. Unsloth raises these at
    *import* time (GPU probe) and during training when memory is exhausted — both
    are environment errors (exit 2), not "file a bug" code-1 errors.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "outofmemory" in name
        or "acceleratorerror" in name
        or "out of memory" in msg
        or "cuda error: out of memory" in msg
    )


# ---------------------------------------------------------------------------
# Plan construction (pure — no torch)
# ---------------------------------------------------------------------------


def _resolved_hyperparameters(config: RunConfig) -> dict[str, Any]:
    """Return the fully-resolved hyperparameter mapping for *config*."""
    return {
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.learning_rate,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "load_in_4bit": config.load_in_4bit,
        "target_modules": resolve_target_modules(config.target_modules),
    }


def _build_plan(config: RunConfig, *, dry_run: bool, scope) -> dict[str, Any]:
    """Build the resolved training-plan dict (model, method, hparams, scope)."""
    return {
        "model": config.model,
        "method": config.method,
        "dataset": config.dataset,
        "output": config.output,
        "hyperparameters": _resolved_hyperparameters(config),
        "scope": {
            "ok": scope.ok,
            "out_of_scope": scope.out_of_scope,
            "warning": scope.warning,
            "downgrade_to": scope.downgrade_to,
            "message": scope.message,
        },
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Heavy backend (the ONLY place torch/unsloth/trl are imported)
# ---------------------------------------------------------------------------


@dataclass
class _Backend:
    """Bundle of the lazily-imported ML callables used by the real training path.

    Field names are snake_case (not the PascalCase of the imported classes) to
    satisfy the field-naming convention; each holds the corresponding callable.
    """

    fast_language_model: Any  # unsloth.FastLanguageModel
    sft_trainer: Any  # trl.SFTTrainer
    sft_config: Any  # trl.SFTConfig
    torch: Any


def _load_backend() -> _Backend:
    """Import the heavy ML stack and return it as a :class:`_Backend`.

    This is the single seam where ``unsloth``/``torch``/``trl`` enter the
    process. Isolated so tests can monkeypatch it: raising ``ImportError`` here
    (or from a patched stand-in) is converted by :func:`run_training` into
    ``CliError(code=2)``; returning a fake exercises the real flow GPU-free.

    Raises:
        ImportError: if any component of the ML stack is unavailable.
    """
    # Unsloth MUST be imported *before* trl/transformers/peft so its runtime
    # patches apply. Imported after them, unsloth warns and skips optimizations —
    # and trl's SFTConfig `<EOS_TOKEN>` sentinel is left unpatched, raising
    # "eos_token '<EOS_TOKEN>' is not found in the vocabulary" at train() time.
    # unsloth MUST be imported before torch/trl (see note above); keep this order —
    # do not let the import sorter reorder this block alphabetically.
    # isort: off
    from unsloth import FastLanguageModel  # noqa: PLC0415 — intentional lazy import; import FIRST
    import torch  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    # isort: on
    return _Backend(
        fast_language_model=FastLanguageModel,
        sft_trainer=SFTTrainer,
        sft_config=SFTConfig,
        torch=torch,
    )


# ---------------------------------------------------------------------------
# Dataset loading for the real path (pure — no torch)
# ---------------------------------------------------------------------------


def _first_json_record(path: Path) -> dict | None:
    """Return the first non-blank JSON record of *path* (``None`` for an empty file).

    Shared by the training-side schema sniff (:func:`_detect_dataset_schema`) and
    the eval-side one (:func:`_detect_suite_schema`) so both surface the identical
    ``CliError`` for an unopenable file or a non-JSON first line.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    return json.loads(stripped)
            return None
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"cannot open dataset file {path}",
            remediation="Check that the file exists and is readable.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"dataset {path}: first line is not valid JSON — {exc.msg}",
            remediation="Each line of the dataset must be a JSON object.",
        ) from exc


def _detect_dataset_schema(path: Path) -> str:
    """Sniff the schema (``"chat"``/``"task"``) from the first record of *path*."""
    first_record = _first_json_record(path)
    schema = detect_schema(first_record) if first_record is not None else None
    if schema is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"cannot detect a known schema for dataset {path}",
            remediation=(
                'Use the chat schema ({"messages": [...]}) or the task schema '
                '({"task", "input", "expected_output"}).'
            ),
        )
    return schema


def _format_records(records: list[dict], schema: str, tokenizer: Any) -> list[dict]:
    """Render validated records into ``{"text": ...}`` rows for SFTTrainer.

    Chat records are rendered with the model's chat template; task records use the
    same ``Task:/Input:/Output:`` prompt shape as :func:`run_eval`, terminated with
    the tokenizer's EOS so the model learns to stop. Pre-rendering an explicit
    ``text`` column avoids trl/unsloth conversational auto-detection, which would
    otherwise raise ``"Unsloth: You must specify a `formatting_func`"``.
    """
    if schema == "chat":
        return [
            {"text": tokenizer.apply_chat_template(record["messages"], tokenize=False)}
            for record in records
        ]
    eos = tokenizer.eos_token or ""
    return [
        {
            "text": (
                f"Task: {record['task']}\nInput: {record['input']}\n"
                f"Output: {record['expected_output']}{eos}"
            )
        }
        for record in records
    ]


# ---------------------------------------------------------------------------
# Real training path (uses the lazily-loaded backend; not GPU-tested in CI)
# ---------------------------------------------------------------------------


def _run_real(config: RunConfig, plan: dict[str, Any], backend: _Backend) -> dict[str, Any]:
    """Load the model, apply LoRA/QLoRA, train, save the adapter, write metadata."""
    load_in_4bit = bool(config.load_in_4bit) or config.method == "qlora"

    # Validate + load the dataset BEFORE the expensive model load, so a schema or
    # empty-dataset failure surfaces a CliError without spending any GPU/model-load
    # time ("validate before spending GPU").
    dataset_path = Path(config.dataset)
    schema = _detect_dataset_schema(dataset_path)
    train_records = validate_dataset(dataset_path, schema=schema)

    # Lazy-imported here (not at module top) so the module stays importable without
    # ``datasets`` installed and tests can inject a fake via sys.modules.
    from datasets import Dataset  # noqa: PLC0415 — intentional lazy import

    try:
        model, tokenizer = backend.fast_language_model.from_pretrained(
            model_name=config.model,
            max_seq_length=config.max_seq_len,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        peft_kwargs: dict[str, Any] = {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "random_state": config.seed,
        }
        resolved_target_modules = plan["hyperparameters"]["target_modules"]
        if resolved_target_modules is not None:
            peft_kwargs["target_modules"] = resolved_target_modules
        model = backend.fast_language_model.get_peft_model(model, **peft_kwargs)

        # Render each record into a single ``text`` column so SFTTrainer does not
        # depend on trl/unsloth conversational auto-detection (which otherwise
        # raises "Unsloth: You must specify a `formatting_func`"). Chat records use
        # the model's chat template; task records use the same prompt shape as
        # ``sloth eval``. Done after the tokenizer exists (the chat template lives
        # on it).
        train_dataset = Dataset.from_list(_format_records(train_records, schema, tokenizer))

        sft_config = backend.sft_config(
            output_dir=config.output,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            seed=config.seed,
            dataset_text_field="text",
        )
        # trl >= 0.12 renamed the ``tokenizer`` kwarg to ``processing_class``;
        # trl 0.24 (the pinned in-container version) removed ``tokenizer`` entirely.
        trainer = backend.sft_trainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )
        trainer.train()
    except NotImplementedError as exc:
        # Unsloth raises NotImplementedError (message: "cannot find any torch
        # accelerator") when no GPU is available. Map it to a user-actionable
        # CliError so the CLI can surface a clear remediation instead of a
        # "file a bug" generic error (code=1).
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"No GPU accelerator found — the ML backend raised: {exc}",
            remediation=_NGC_HINT,
        ) from exc

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta_path = write_metadata(
        output_dir,
        model=config.model,
        method=config.method,
        dataset_path=Path(config.dataset),
        hyperparameters=plan["hyperparameters"],
    )

    result = dict(plan)
    result["status"] = "trained"
    result["adapter_dir"] = str(output_dir)
    result["metadata_path"] = str(meta_path)
    return result


# ---------------------------------------------------------------------------
# Shared eval plumbing (suite resolution + batched generation)
#
# Both eval seams use these: ``run_eval`` below (``--adapter``) and
# ``sloth.tune._exporter._predict_transformers`` (``--model``), which imports
# them from here — the dependency runs _exporter → _trainer, never back.
# ---------------------------------------------------------------------------

#: Generation budget per suite item (mirrors ``_exporter.EVAL_MAX_NEW_TOKENS``).
EVAL_MAX_NEW_TOKENS: int = 100

#: Default number of prompts generated per ``generate()`` call.
DEFAULT_EVAL_BATCH_SIZE: int = 8


def resolve_suite_paths(
    suite_path: str | Path | None,
    suite_paths: Sequence[str | Path] | None,
) -> list[Path]:
    """Normalise the two suite arguments into an ordered list of files.

    The eval seams accept **either** the historical single ``suite_path``
    positional (kept so the old two-argument call site and any direct caller
    stay green) **or** the newer ``suite_paths`` keyword the CLI passes once it
    detects the richer signature. When both are given ``suite_paths`` wins; when
    neither is given that is a caller bug, surfaced as ``CliError(code=1)``.
    """
    if suite_paths:
        return [Path(p) for p in suite_paths]
    if suite_path is not None:
        return [Path(suite_path)]
    raise CliError(
        code=EXIT_USER_ERROR,
        message="no eval suite given",
        remediation="Pass --suite <file.jsonl> (or a directory of them) to `sloth eval`.",
    )


def _resolve_eval_batch_size(tokenizer: Any, batch_size: int) -> int:
    """Return the batch size that *tokenizer* can actually pad for.

    Batched generation needs a pad token. Three cases:

    * ``pad_token`` set → use *batch_size* unchanged;
    * ``pad_token is None`` but ``eos_token`` set → adopt EOS as the pad token
      (the standard decoder-only workaround) and use *batch_size*;
    * both ``None`` → padding is impossible, so fall back to **batch size 1**
      after one stderr diagnostic (results stay correct, only slower).
    """
    if batch_size <= 1:
        return 1
    pad_token = getattr(tokenizer, "pad_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if pad_token is not None:
        return batch_size
    if eos_token is not None:
        tokenizer.pad_token = eos_token
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = eos_id
        return batch_size
    emit_diagnostic(
        "note: tokenizer has neither pad_token nor eos_token — batched generation "
        "needs padding, so falling back to batch size 1 (slower, same scores)."
    )
    return 1


def _padded_prompt_width(inputs: Any, row: int) -> int:
    """Return the index where row *row*'s continuation starts in a ``generate()`` output.

    ``generate()`` returns ``[batch, padded_prompt_width + new_tokens]``: every
    row is ``pad * P_i`` + ``prompt * L_i`` + continuation, and under **left**
    padding the pads come first, so ``P_i + L_i`` is the same shared padded width
    for every row. Computing it per row from the attention mask —
    ``L_i = attention_mask[i].sum()`` (the row's real token count) and
    ``P_i = width - L_i`` (its left pad count) — therefore lands on that shared
    width, which is exactly what must be sliced off.

    Slicing at ``L_i`` alone would be **wrong**: for a short row the prompt sits
    at indices ``[P_i, width)``, so ``sequence[L_i:]`` would still contain pad
    tokens and the whole prompt whenever ``L_i < P_i``. The per-row arithmetic is
    spelled out here rather than collapsed to ``width`` so the reasoning is
    visible at the one place it matters.
    """
    ids = inputs["input_ids"] if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    shape = getattr(ids, "shape", None)
    if shape is not None:
        width = int(shape[-1])
    elif ids is not None:
        first = ids[0] if len(ids) and isinstance(ids[0], (list, tuple)) else ids
        width = len(first)
    else:
        return 0
    mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
    if mask is None:
        return width
    try:
        real_len = int(mask[row].sum())
    except (AttributeError, IndexError, TypeError):
        return width
    pad_len = width - real_len  # left padding ⇒ the pads precede the prompt
    return pad_len + real_len


@dataclass
class GenerationRun:
    """One generation pass: the predictions plus what the pass *cost*.

    ``generated_tokens[i]`` is how many new tokens row *i* produced (the decoded
    continuation's token count: the generated row length minus the padded prompt
    width — see :func:`_padded_prompt_width`). ``latency_ms[i]`` is that row's
    share of its batch's wall time — ``time.perf_counter`` measured strictly
    around ``model.generate`` and divided by the number of rows in the batch, so
    a batch of eight rows taking 800 ms attributes 100 ms to each. Summing
    ``latency_ms`` over any subset of rows therefore reconstructs the wall time
    those rows cost, which is what makes the per-file ``tokens_per_s`` rollup in
    :func:`timing_summary` exact even when one flat generation pass is split
    back across several suite files.

    ``generate_seconds`` is the total wall time spent inside ``generate`` across
    every batch of this pass (model load and tokenisation are deliberately
    excluded — this measures generation, not setup).
    """

    predictions: list[str] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    generate_seconds: float = 0.0

    def extend(self, other: "GenerationRun") -> "GenerationRun":
        """Append *other*'s rows onto this run (used to concatenate batches)."""
        self.predictions.extend(other.predictions)
        self.generated_tokens.extend(other.generated_tokens)
        self.latency_ms.extend(other.latency_ms)
        self.generate_seconds += other.generate_seconds
        return self


def _sequence_length(sequence: Any) -> int:
    """Return the token count of a 1-D sequence (tensor row, list, or string)."""
    shape = getattr(sequence, "shape", None)
    if shape is not None:
        try:
            return int(shape[-1])
        except (IndexError, TypeError):
            pass
    try:
        return len(sequence)
    except TypeError:
        return 0


def _generate_predictions(
    torch_mod: Any,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: Any,
) -> GenerationRun:
    """Generate one continuation per prompt, in batches of *batch_size*.

    Batched generation pads with ``padding_side="left"`` so every row's
    continuation begins at the same offset (see :func:`_padded_prompt_width`);
    the prompt is always sliced off before decoding, so a prediction never
    contains its own ``Task:/Input:/Output:`` prefix. When padding is impossible
    (no pad and no EOS token) or the caller asked for ``batch_size <= 1``, this
    degrades to the historical one-prompt-at-a-time path, which needs no padding
    at all.

    Returns a :class:`GenerationRun` — the predictions plus the per-row generated
    token counts and latencies measured with ``time.perf_counter`` around the
    ``generate`` call itself (nothing else is inside the timed region).
    """
    effective = _resolve_eval_batch_size(tokenizer, batch_size)
    run = GenerationRun()

    if effective == 1:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            started = time.perf_counter()
            with torch_mod.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - started
            start = _padded_prompt_width(inputs, 0)
            run.predictions.append(tokenizer.decode(outputs[0][start:], skip_special_tokens=True))
            run.generated_tokens.append(max(_sequence_length(outputs[0]) - start, 0))
            run.latency_ms.append(elapsed * 1000.0)
            run.generate_seconds += elapsed
        return run

    tokenizer.padding_side = "left"
    for offset in range(0, len(prompts), effective):
        chunk = list(prompts[offset : offset + effective])
        inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        started = time.perf_counter()
        with torch_mod.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - started
        # A batch is one wall-clock event; each row is billed its equal share.
        per_row_ms = elapsed * 1000.0 / len(chunk) if chunk else 0.0
        run.generate_seconds += elapsed
        for row in range(len(chunk)):
            start = _padded_prompt_width(inputs, row)
            run.predictions.append(tokenizer.decode(outputs[row][start:], skip_special_tokens=True))
            run.generated_tokens.append(max(_sequence_length(outputs[row]) - start, 0))
            run.latency_ms.append(per_row_ms)
    return run


def timing_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return ``{median_latency_ms, tokens_per_s}`` over scored *rows*.

    ``median_latency_ms`` is the median of the per-row ``latency_ms`` (the
    median, not the mean, because one slow row — a cold first batch — should not
    move the number readers compare across runs). ``tokens_per_s`` is
    ``sum(generated_tokens) / total generate wall time``, where the wall time is
    the sum of the rows' ``latency_ms``: since each row carries its equal share
    of its batch's wall time, that sum is exactly the time those rows cost.
    """
    latencies = [float(r["latency_ms"]) for r in rows if "latency_ms" in r]
    tokens = [int(r["generated_tokens"]) for r in rows if "generated_tokens" in r]
    seconds = sum(latencies) / 1000.0
    return {
        "median_latency_ms": round(statistics.median(latencies), 4) if latencies else 0.0,
        "tokens_per_s": round(sum(tokens) / seconds, 4) if seconds > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Suite schema resolution + per-schema scoring (pure stdlib)
# ---------------------------------------------------------------------------

#: Per-row boolean key each non-``task`` suite schema contributes, and the source
#: of the suite-level ``compliance_pct``.
COMPLIANCE_KEYS: dict[str, str] = {
    "instruction": "constraints_passed",
    "structured": "json_valid",
    "toolcall": "tool_call_matched",
}


def detect_suite_schema(record: dict[str, Any]) -> str:
    """Return the eval-suite schema of a single *record*.

    :func:`sloth.tune.datasets.detect_schema` is the authority for the two
    original schemas (``chat``/``task``); the three newer suite schemas are
    distinguished here by their one discriminating key (``constraints``,
    ``json_schema``, ``expected_tool_call``), because a shared ``detect_schema``
    that knows them is not part of this task's editable surface. A record that
    matches nothing falls back to ``"task"``, so ``validate_dataset`` — not this
    sniff — produces the error message.
    """
    schema = detect_schema(record)
    if schema is not None:
        return schema
    keys = set(record) if isinstance(record, dict) else set()
    if "constraints" in keys:
        return "instruction"
    if "json_schema" in keys:
        return "structured"
    if "expected_tool_call" in keys:
        return "toolcall"
    return "task"


def _detect_suite_schema(path: Path) -> str:
    """Sniff the suite schema of the JSONL file at *path* from its first record."""
    record = _first_json_record(path)
    return detect_suite_schema(record) if record is not None else "task"


def _scorable_row(record: dict[str, Any], schema: str) -> dict[str, Any]:
    """Return *record* with the ``task``/``input``/``expected_output`` trio filled in.

    :func:`sloth.tune.metrics.score_records` scores a task-shaped row, so every
    schema is normalised to one. The original keys are **kept** on the row (the
    scorer's ``extra_metrics`` callback needs ``constraints`` / ``json_schema`` /
    ``expected_tool_call``); ``score_records`` copies only the task trio into the
    result entry, so nothing leaks into the output.

    ``structured`` and ``toolcall`` suites carry no reference *text*, so their
    ``expected_output`` is ``""`` and their ``exact_match``/``f1`` are not
    meaningful — ``compliance_pct`` is the metric for those suites.
    """
    row = dict(record)
    if schema == "chat":
        messages = list(record.get("messages") or [])
        if messages and messages[-1].get("role") == "assistant":
            expected = messages[-1]["content"]
            prompt_messages = messages[:-1]
        else:
            expected = ""
            prompt_messages = messages
        row.update(
            {
                "task": "chat",
                "input": render_chat_prompt(prompt_messages),
                "expected_output": expected,
            }
        )
        return row
    row.setdefault("task", schema)
    row.setdefault("input", "")
    row.setdefault("expected_output", "")
    return row


def _constraint_dicts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a dataset row's ``constraints`` into :mod:`sloth.tune.scorers` shape.

    The dataset schema spells a constraint as a single-key object
    (``{"max_words": 12}``); the scorer takes ``{"kind": ..., "value": ...}``.
    The two flag kinds (``must_refuse``, ``json_only``) carry no value and are
    dropped entirely when set to ``false``, so "not required" never scores as a
    failure.
    """
    converted: list[dict[str, Any]] = []
    for constraint in record.get("constraints") or []:
        for kind, value in constraint.items():
            if kind in ("must_refuse", "json_only"):
                if value:
                    converted.append({"kind": kind})
            else:
                converted.append({"kind": kind, "value": value})
    return converted


def _resolve_tool_call_family(model_id: str | None, override: str | None) -> str:
    """Resolve the tool-call family, turning the scorer's ValueError into a CliError."""
    try:
        return scorers.family_for_model(model_id or "", override)
    except ValueError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=str(exc),
            remediation=(
                "Name the model's tool-call format explicitly — set [eval] "
                "tool_call_family in the run config (supported families: lfm2, qwen3)."
            ),
        ) from exc


def _extra_metrics_for(
    schema: str, *, model_id: str | None = None, tool_call_family: str | None = None
):
    """Return the per-row ``extra_metrics`` callable for *schema* (``None`` if any).

    ``chat``/``task`` suites are scored by exact match and token F1 alone. The
    three richer schemas each add one boolean key — see :data:`COMPLIANCE_KEYS`
    — which :func:`compliance_pct` folds into the suite-level percentage.
    """
    if schema == "instruction":

        def score_instruction(record: dict[str, Any], prediction: str) -> dict[str, Any]:
            outcome = scorers.score_constraints(prediction, _constraint_dicts(record))
            return {"constraints_passed": bool(outcome["passed"])}

        return score_instruction

    if schema == "structured":

        def score_structured(record: dict[str, Any], prediction: str) -> dict[str, Any]:
            outcome = scorers.check_json_subset(prediction, record.get("json_schema") or {})
            return {"json_valid": bool(outcome["valid"])}

        return score_structured

    if schema == "toolcall":
        # Resolved once, before any generation, so an unknown family costs no GPU.
        family = _resolve_tool_call_family(model_id, tool_call_family)

        def score_toolcall(record: dict[str, Any], prediction: str) -> dict[str, Any]:
            outcome = scorers.score_tool_call(
                prediction, record.get("expected_tool_call") or {}, family
            )
            return {"tool_call_matched": bool(outcome["matched"])}

        return score_toolcall

    return None


def compliance_pct(rows: Sequence[dict[str, Any]]) -> float | None:
    """Return the percentage of *rows* whose schema compliance flag is true.

    ``None`` when no row carries one of :data:`COMPLIANCE_KEYS` (a plain
    ``task``/``chat`` suite has nothing to comply with), so the caller can leave
    the key off the payload entirely rather than reporting a misleading ``0``.
    """
    flags = [bool(row[key]) for row in rows for key in COMPLIANCE_KEYS.values() if key in row]
    if not flags:
        return None
    return round(sum(flags) / len(flags) * 100, 2)


def score_suite(
    records: Sequence[dict[str, Any]],
    schema: str,
    generation: GenerationRun,
    *,
    start_index: int = 0,
    source: str | None = None,
    extra_metrics: Any = None,
) -> list[dict[str, Any]]:
    """Score one suite's *generation* and stamp each row with its timing."""
    rows = [_scorable_row(record, schema) for record in records]
    scored = metrics.score_records(
        rows,
        generation.predictions,
        start_index=start_index,
        source=source,
        extra_metrics=extra_metrics,
    )
    for entry, tokens, latency in zip(scored, generation.generated_tokens, generation.latency_ms):
        entry["generated_tokens"] = tokens
        entry["latency_ms"] = round(latency, 4)
    return scored


def build_file_entry(
    suite_file: Any, scored: Sequence[dict[str, Any]], *, perplexity: float | None = None
) -> dict[str, Any]:
    """Build one ``files`` entry: metrics, the timing rollups, compliance, perplexity."""
    entry = metrics.file_entry(suite_file, scored)
    entry.update(timing_summary(scored))
    pct = compliance_pct(scored)
    if pct is not None:
        entry["compliance_pct"] = pct
    if perplexity is not None:
        entry["perplexity"] = perplexity
    return entry


def build_summary(
    files: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    base_load_in_4bit: bool | None,
) -> dict[str, Any]:
    """Aggregate per-file entries and add the suite-level rollups both seams report."""
    summary = metrics.aggregate(files)
    summary.update(timing_summary(summary["results"]))
    pct = compliance_pct(summary["results"])
    if pct is not None:
        summary["compliance_pct"] = pct
    perplexities = [f["perplexity"] for f in files if f.get("perplexity") is not None]
    if perplexities:
        summary["perplexity"] = round(sum(perplexities) / len(perplexities), 4)
    summary["batch_size"] = batch_size
    summary["base_load_in_4bit"] = base_load_in_4bit
    return summary


#: Diagnostic keys already emitted this process (see :func:`_warn_once`).
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit *message* on stderr the first time *key* is seen in this process.

    A per-row fallback notice would otherwise repeat once per suite record and
    bury the rest of the run's diagnostics; the condition is a property of the
    tokenizer, not of the row, so saying it once is saying it fully.
    """
    if key in _WARNED:
        return
    _WARNED.add(key)
    emit_diagnostic(message)


def eval_prompt(record: dict[str, Any], tokenizer: Any = None) -> str:
    """Render one suite record into the prompt the model is asked to continue.

    Task-shaped records (``task``/``instruction``/``structured``/``toolcall``)
    use the shared ``Task:/Input:/Output:`` shape — the same rendering
    ``_format_records`` trains on, so eval matches training.

    **Chat records** are rendered with the *model's own* chat template
    (``tokenizer.apply_chat_template(..., add_generation_prompt=True)``): the
    final assistant turn is the expected output, so only the turns before it are
    rendered, and ``add_generation_prompt`` appends the template's
    assistant-turn header so the model continues rather than re-opening the
    conversation. Without a tokenizer (or with one carrying no chat template)
    this falls back to :func:`sloth.tune.datasets.render_chat_prompt`, the
    stdlib core's documented plain-text rendering.
    """
    if isinstance(record, dict) and "messages" in record:
        messages = list(record.get("messages") or [])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is not None:
            try:
                return apply_template(messages, tokenize=False, add_generation_prompt=True)
            except (AttributeError, KeyError, TypeError, ValueError):
                _warn_once(
                    "chat-template",
                    "note: the tokenizer has no usable chat template — falling back to "
                    "the plain-text chat rendering for this suite.",
                )
        return render_chat_prompt(messages) + "\nassistant:"
    return f"Task: {record['task']}\nInput: {record['input']}\nOutput:"


def _model_device(model: Any) -> Any:
    """Return the model's device, or ``None`` when it does not expose parameters."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _suite_records(suite: Any) -> list[dict[str, Any]]:
    """Normalise *suite* (a JSONL path or an already-parsed record list) to records."""
    if isinstance(suite, (str, Path)):
        path = Path(suite)
        return validate_dataset(path, schema=_detect_suite_schema(path))
    return list(suite)


def run_perplexity(model: Any, tokenizer: Any, suite: Any, *, torch_mod: Any = None) -> float:
    """Return ``exp(mean per-token NLL)`` of *suite* under *model*.

    This is a **labelled forward pass**, never a ``generate`` call: each record is
    rendered to ``prompt + expected_output``, tokenized, and run through
    ``model(**inputs, labels=input_ids)``, whose ``loss`` is the mean
    cross-entropy over that sequence's predicted tokens. The losses are combined
    *token-weighted* (each row's loss times its label count, divided by the total
    label count), so a long row counts for more than a short one — a plain mean
    of per-row losses would silently reweight the suite by row count.

    *suite* is either a JSONL path (validated and parsed here) or an
    already-parsed list of records. *torch_mod* lets a caller that has already
    imported torch (both eval seams have) hand it in; when omitted torch is
    imported lazily inside the function, exactly like this module's other heavy
    seams — so the documented ``run_perplexity(model, tokenizer, suite)`` call
    works on its own.

    An astronomically bad model can overflow ``math.exp``; that is reported as
    ``inf`` rather than raising.
    """
    if torch_mod is None:
        import torch  # noqa: PLC0415 — intentional lazy import

        torch_mod = torch

    records = _suite_records(suite)
    device = _model_device(model)
    total_nll = 0.0
    total_tokens = 0
    for record in records:
        schema = detect_suite_schema(record)
        row = _scorable_row(record, schema)
        prompt = eval_prompt(record, tokenizer)
        expected = row["expected_output"]
        text = f"{prompt} {expected}" if expected else prompt
        inputs = tokenizer(text, return_tensors="pt")
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        with torch_mod.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        # A causal LM predicts token i+1 from token i, so an n-token sequence
        # contributes n-1 scored labels.
        labels = max(_sequence_length(inputs["input_ids"]) - 1, 1)
        total_nll += float(loss) * labels
        total_tokens += labels
    if not total_tokens:
        return 0.0
    try:
        return math.exp(total_nll / total_tokens)
    except OverflowError:
        return math.inf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval(
    adapter_path: str,
    suite_path: str | None = None,
    *,
    suite_paths: Sequence[str | Path] | None = None,
    quant: str | None = None,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    perplexity: bool = False,
    tool_call_family: str | None = None,
    base_load_in_4bit: bool | None = None,
) -> dict[str, Any]:
    """Load a LoRA adapter and evaluate against a task-schema JSONL suite.

    This is the ML-seam entry point for ``sloth eval``.  Heavy imports
    (torch, transformers, peft) are deferred inside this function so
    ``sloth/cli/_commands/eval.py`` stays ML-free at module level.

    Parameters
    ----------
    adapter_path:
        Filesystem path to the adapter directory (must contain
        ``adapter_config.json``). ``eval.json`` is written here.
    suite_path:
        A single task-schema JSONL suite — the historical positional argument,
        still honoured for direct/legacy callers.
    suite_paths:
        Every resolved suite file, in order (what ``sloth eval`` passes once it
        detects this signature). Wins over *suite_path* when both are given.
    quant:
        Accepted and **ignored**: quant selection only means something for a
        ``--model`` directory holding several GGUF files
        (:func:`sloth.tune._exporter.run_eval_model`). The parameter exists so
        the CLI can forward the same keyword set to either seam.
    batch_size:
        Prompts per ``generate()`` call (left-padded). Degrades to 1 when the
        tokenizer cannot pad — see :func:`_resolve_eval_batch_size`.
    max_new_tokens:
        Generation budget per suite item.
    perplexity:
        Also score each suite with :func:`run_perplexity` (a labelled forward
        pass, on top of generation) and report it per file and in the aggregate.
    tool_call_family:
        Overrides tool-call family detection for a ``toolcall`` suite (what a run
        config's ``[eval] tool_call_family`` supplies); otherwise the family is
        detected from the adapter's base model id.
    base_load_in_4bit:
        The base model's load precision, recorded verbatim into the result and
        the written eval files so a score is never read without knowing the
        precision it was measured at. ``None`` means "not stated".

    Returns
    -------
    dict
        The **aggregate** over every suite file — ``total``, ``exact_match``,
        ``exact_match_pct``, ``f1``, ``results`` — plus ``files``, one entry per
        scored file, ``median_latency_ms`` / ``tokens_per_s``, ``batch_size``,
        ``base_load_in_4bit``, and (when the suite schema or *perplexity* asks
        for them) ``compliance_pct`` / ``perplexity``. Each row of ``results``
        additionally carries ``generated_tokens`` and ``latency_ms``. The same
        payload is written to ``<adapter_path>/eval.json`` (the legacy flat file
        ``sloth summarize`` reads) and, per suite, to
        ``<adapter_path>/eval/<suite>.json``.

    Raises
    ------
    CliError(code=1)
        When ``adapter_config.json`` is absent, unreadable, or missing the
        ``base_model_name_or_path`` key.
    CliError(code=2)
        When the ML stack (torch / transformers / peft) is not installed.
    """
    resolved_suites = resolve_suite_paths(suite_path, suite_paths)
    try:
        import torch  # noqa: PLC0415 — intentional lazy import
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"ML stack not installed: {exc}",
            remediation=(
                "Install the tuning stack: uv tool install unsloth-cli "
                "(or run uv sync in a checkout)."
            ),
        ) from exc

    # Read the base model name from the adapter config (pure stdlib).
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"adapter_config.json not found in {adapter_path}",
            remediation=(
                "The adapter directory must contain adapter_config.json "
                "(produced by peft/unsloth during training). "
                "Re-run `sloth train` to produce a valid adapter."
            ),
        )
    try:
        with config_file.open(encoding="utf-8") as fh:
            adapter_cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"could not read adapter_config.json: {exc}",
            remediation="Ensure adapter_config.json is valid JSON.",
        ) from exc
    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="base_model_name_or_path missing in adapter_config.json",
            remediation=(
                "The adapter_config.json must contain 'base_model_name_or_path'. "
                "Re-run `sloth train` to produce a valid adapter."
            ),
        )

    # Correct PEFT load sequence: load BASE model first, then wrap with adapter.
    # local_files_only=True so no Hub access can occur (B615 — unpinned remote
    # revision risk does not apply here since we are loading from local files).
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)  # nosec B615
    base_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model_name, local_files_only=True
    )
    # Wrap the base model with the LoRA adapter weights — calling
    # AutoModelForCausalLM.from_pretrained(adapter_path) directly would fail.
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)  # nosec B615
    model.eval()

    # Eval loop, one suite file at a time so each file gets its own score entry.
    # Tokenized inputs must live on the same device as the (GPU-resident, 4-bit)
    # model, else generate() raises "Expected all tensors to be on the same
    # device" — index_select on a CPU index against CUDA weights.
    device = next(model.parameters()).device
    files: list[dict[str, Any]] = []
    next_index = 0
    for suite_file in resolved_suites:
        # Re-validate inside the container (eval.py already validated on the host).
        schema = _detect_suite_schema(Path(suite_file))
        records = validate_dataset(Path(suite_file), schema=schema)
        # Resolved BEFORE generation so an unknown tool-call family costs no GPU.
        extra_metrics = _extra_metrics_for(
            schema, model_id=base_model_name, tool_call_family=tool_call_family
        )
        generation = _generate_predictions(
            torch,
            model,
            tokenizer,
            [eval_prompt(record, tokenizer) for record in records],
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        scored = score_suite(
            records,
            schema,
            generation,
            start_index=next_index,
            source=str(suite_file),
            extra_metrics=extra_metrics,
        )
        next_index += len(scored)
        files.append(
            build_file_entry(
                suite_file,
                scored,
                perplexity=(
                    run_perplexity(model, tokenizer, records, torch_mod=torch)
                    if perplexity
                    else None
                ),
            )
        )

    summary = build_summary(files, batch_size=batch_size, base_load_in_4bit=base_load_in_4bit)
    write_eval_artifacts(
        Path(adapter_path),
        files,
        resolved_suites,
        summary,
        target="adapter",
        batch_size=batch_size,
        base_load_in_4bit=base_load_in_4bit,
    )
    return summary


def write_eval_json(
    directory: Path,
    summary: dict[str, Any],
    suite_paths: Sequence[str | Path],
    *,
    target: str,
) -> Path | None:
    """Write ``<directory>/eval.json``; warn (never fail) if the directory is read-only.

    Losing an otherwise-complete eval because its artifact directory could not be
    written would be worse than the missing file, so an :class:`OSError` becomes
    a stderr diagnostic and the scores are still returned on stdout.
    """
    try:
        return metrics.write_eval_json(directory, summary, suite_paths=suite_paths, target=target)
    except OSError as exc:
        emit_diagnostic(f"note: could not write {directory / metrics.EVAL_JSON_NAME}: {exc}")
        return None


def write_suite_eval_json(
    directory: Path,
    suite: str | Path,
    payload: dict[str, Any],
    *,
    target: str,
    batch_size: int | None = None,
    base_load_in_4bit: bool | None = None,
) -> Path | None:
    """Write ``<directory>/eval/<suite>.json``; warn (never fail) on a read-only dir."""
    try:
        return metrics.write_eval_json(
            directory,
            suite,
            payload,
            target=target,
            batch_size=batch_size,
            base_load_in_4bit=base_load_in_4bit,
        )
    except OSError as exc:
        emit_diagnostic(f"note: could not write the {metrics.EVAL_JSON_DIR}/ result file: {exc}")
        return None


def write_eval_artifacts(
    directory: Path,
    files: Sequence[dict[str, Any]],
    suite_paths: Sequence[str | Path],
    summary: dict[str, Any],
    *,
    target: str,
    batch_size: int,
    base_load_in_4bit: bool | None,
) -> None:
    """Write both eval layouts: one file per suite, plus the legacy flat ``eval.json``.

    The per-suite files (schema_version 2, carrying ``batch_size`` and
    ``base_load_in_4bit``) are the layout comparisons read; the flat
    ``eval.json`` is kept because :mod:`sloth.tune.summary` — and therefore
    ``sloth summarize``/``sloth compare`` — still reads it.
    """
    for entry, suite_file in zip(files, suite_paths):
        write_suite_eval_json(
            directory,
            suite_file,
            entry,
            target=target,
            batch_size=batch_size,
            base_load_in_4bit=base_load_in_4bit,
        )
    write_eval_json(directory, summary, suite_paths, target=target)


def run_training(config: RunConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Resolve *config* into a training plan and (unless ``dry_run``) run the job.

    Parameters
    ----------
    config:
        The validated run configuration (see :class:`sloth.tune.config.RunConfig`).
    dry_run:
        When ``True``, return the resolved plan immediately without importing the
        heavy backend. The plan carries the scope decision so the caller can warn
        or downgrade.

    Returns
    -------
    dict
        The resolved training plan. For a real run the dict additionally carries
        ``status``, ``adapter_dir``, and ``metadata_path``.

    Raises
    ------
    CliError(code=1)
        For a non-dry-run *out-of-scope* request (hard refusal, before any heavy
        import).
    CliError(code=2)
        When the ML backend is not installed (the ``uv tool install unsloth-cli``
        install hint is attached).
    """
    scope = check_scope(config.model, config.method)
    plan = _build_plan(config, dry_run=dry_run, scope=scope)

    if dry_run:
        return plan

    # Hard refusal for out-of-scope real runs — before importing torch.
    if scope.out_of_scope:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"Refusing to train: {scope.message}",
            remediation=scope.warning or "Switch to method='lora' or method='qlora'.",
        )

    try:
        backend = _load_backend()
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message="The fine-tuning backend (unsloth + torch + trl) is not installed.",
            remediation=_INSTALL_HINT,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Unsloth's GPU probe at import can raise a CUDA/accelerator OOM. That is an
        # environment error (exit 2) with a memory remediation, not a code-1 bug;
        # classify it and re-raise everything else unchanged.
        if _is_gpu_oom(exc):
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"GPU out of memory while initializing the backend: {exc}",
                remediation=_OOM_HINT,
            ) from exc
        raise

    try:
        return _run_real(config, plan, backend)
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Map a CUDA/accelerator OOM during training to exit 2; re-raise the rest.
        if _is_gpu_oom(exc):
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"GPU out of memory during training: {exc}",
                remediation=_OOM_HINT,
            ) from exc
        raise
