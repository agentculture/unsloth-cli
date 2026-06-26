"""``sloth eval`` — run a LoRA/QLoRA adapter against a local task-schema eval suite.

Evaluates an adapter directory against a JSONL file whose records conform to the
**task** schema (``{"task": …, "input": …, "expected_output": …}``).  All
inference is local and offline: every model load uses ``local_files_only=True``
and the heavy ML stack (torch, transformers, unsloth) is imported lazily inside
:func:`_default_backend` so the introspection CLI keeps working on machines with
no GPU or ML stack installed.

The inference backend is exposed at module level as :data:`_inference_backend`
so tests can monkeypatch it without touching the rest of the evaluation logic.

Usage::

    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl --json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune.datasets import validate_dataset

# ---------------------------------------------------------------------------
# Inference backend (mockable)
# ---------------------------------------------------------------------------


def _default_backend(adapter_path: str, record: dict[str, str]) -> str:
    """Load the adapter locally and generate a prediction for *record*.

    Heavy ML imports (torch, transformers) are deferred here so the CLI works
    on machines without the ML stack.  ``local_files_only=True`` is passed to
    every ``from_pretrained`` call so no network access can occur.

    Parameters
    ----------
    adapter_path:
        Filesystem path to the adapter directory (must exist on disk).
    record:
        A validated task-schema record with ``"task"``, ``"input"``, and
        ``"expected_output"`` keys.

    Returns
    -------
    str
        The raw decoded prediction from the model.
    """
    try:
        import torch  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"ML stack not installed: {exc}",
            remediation=(
                "Install the ML extras: pip install 'unsloth-cli[train]' "
                "or install torch and transformers manually."
            ),
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(adapter_path, local_files_only=True)
    model.eval()

    prompt = f"Task: {record['task']}\nInput: {record['input']}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Module-level reference; override in tests via ``monkeypatch.setattr``.
_inference_backend: Callable[[str, dict[str, str]], str] = _default_backend


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate(adapter_path: str, records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run the inference backend on each record and return structured results.

    Parameters
    ----------
    adapter_path:
        Filesystem path to the adapter directory.
    records:
        Validated task-schema records from :func:`sloth.tune.datasets.validate_dataset`.

    Returns
    -------
    list[dict]
        One result dict per record with keys: ``index``, ``task``, ``input``,
        ``expected_output``, ``prediction``, ``exact_match``.
    """
    results: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        prediction = _inference_backend(adapter_path, record)
        expected = record["expected_output"]
        exact_match = prediction.strip() == expected.strip()
        results.append(
            {
                "index": i,
                "task": record["task"],
                "input": record["input"],
                "expected_output": expected,
                "prediction": prediction,
                "exact_match": exact_match,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Handler for ``sloth eval``.

    Validates the adapter directory and suite file, runs the eval suite through
    the (mockable) inference backend, and emits results via the output contract.

    Parameters
    ----------
    args:
        Parsed namespace with ``adapter``, ``suite``, and ``json`` attributes.

    Returns
    -------
    int
        ``0`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))

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

    # --- parse + validate the suite as task-schema JSONL --------------------
    records = validate_dataset(suite, schema="task")

    # --- run inference -------------------------------------------------------
    results = _evaluate(str(adapter), records)

    # --- compute summary -----------------------------------------------------
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    score_pct = round(exact / total * 100, 2) if total else 0.0
    summary: dict[str, Any] = {
        "total": total,
        "exact_match": exact,
        "exact_match_pct": score_pct,
        "results": results,
    }

    # --- emit results --------------------------------------------------------
    if json_mode:
        emit_result(summary, json_mode=True)
    else:
        lines = [
            f"eval suite: {suite}",
            f"adapter:    {adapter}",
            f"total:      {total}",
            f"exact:      {exact}",
            f"score:      {score_pct}%",
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
    p.set_defaults(func=cmd_eval)
