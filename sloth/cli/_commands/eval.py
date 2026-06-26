"""``sloth eval`` — run a LoRA/QLoRA adapter against a local task-schema eval suite.

Evaluates an adapter directory against a JSONL file whose records conform to the
**task** schema (``{"task": …, "input": …, "expected_output": …}``).  All
inference is local and offline: every model load uses ``local_files_only=True``
and the heavy ML stack (torch, transformers, peft) is imported lazily inside
:func:`_default_backend` so the introspection CLI keeps working on machines with
no GPU or ML stack installed.

The inference backend is exposed at module level as :data:`_inference_backend`
so tests can monkeypatch it without touching the rest of the evaluation logic.

**Host vs in-container routing**

On the host (no ``--in-container`` flag) :func:`cmd_eval` validates the adapter
directory and suite file, then hands off GPU/ML work to the NGC container via
:func:`sloth.tune.container.launch`, forwarding all original args plus
``--in-container`` to prevent docker recursion.

Inside the container (``--in-container`` is set) it runs the real eval using
:func:`_default_backend`, which loads the adapter correctly via
``PeftModel.from_pretrained(base_model, adapter)``.

Usage::

    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl
    sloth eval --adapter adapters/qwen3-4b-qlora --suite data/eval.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune import container
from sloth.tune.datasets import validate_dataset

# ---------------------------------------------------------------------------
# Adapter config reader (pure stdlib — no ML imports)
# ---------------------------------------------------------------------------


def _read_base_model_name(adapter_path: str) -> str:
    """Read ``base_model_name_or_path`` from *adapter_path*/adapter_config.json.

    Parameters
    ----------
    adapter_path:
        Filesystem path to the adapter directory.

    Returns
    -------
    str
        The base model name / path recorded in the adapter config.

    Raises
    ------
    CliError(code=1)
        When ``adapter_config.json`` is absent, unreadable, or missing the key.
    """
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
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"could not read adapter_config.json: {exc}",
            remediation="Ensure adapter_config.json is valid JSON.",
        ) from exc
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="base_model_name_or_path missing in adapter_config.json",
            remediation=(
                "The adapter_config.json must contain 'base_model_name_or_path'. "
                "Re-run `sloth train` to produce a valid adapter."
            ),
        )
    return base


# ---------------------------------------------------------------------------
# Inference backend (mockable)
# ---------------------------------------------------------------------------


def _default_backend(adapter_path: str, record: dict[str, str]) -> str:
    """Load the adapter via PeftModel and generate a prediction for *record*.

    Reads ``base_model_name_or_path`` from ``<adapter>/adapter_config.json``,
    loads the BASE model with ``AutoModelForCausalLM.from_pretrained(base_name)``,
    then wraps it with ``PeftModel.from_pretrained(base_model, adapter_path)``.
    This is the correct PEFT load sequence — calling
    ``AutoModelForCausalLM.from_pretrained(adapter_path)`` directly would fail
    (an adapter dir is not a full model checkpoint).

    Heavy ML imports (torch, transformers, peft) are deferred here so the CLI
    works on machines without the ML stack.  ``local_files_only=True`` is passed
    to every ``from_pretrained`` call so no network access can occur.

    Parameters
    ----------
    adapter_path:
        Filesystem path to the adapter directory (must exist on disk and contain
        ``adapter_config.json``).
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
        from peft import PeftModel  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"ML stack not installed: {exc}",
            remediation=(
                "Install the tuning stack: uv tool install unsloth-cli "
                "(or run uv sync in a checkout)."
            ),
        ) from exc

    # Read the base model name from the adapter config before any loading.
    base_model_name = _read_base_model_name(adapter_path)

    # local_files_only=True loads only from on-disk files, never the Hub,
    # so B615's unpinned-remote-revision risk does not apply here.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)  # nosec B615
    base_model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model_name, local_files_only=True
    )
    # Wrap the base model with the LoRA adapter weights — this is the correct
    # PEFT load sequence; passing adapter_path to AutoModelForCausalLM would fail.
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)  # nosec B615
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
# Checkout locator (repo root for container bind-mount)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the unsloth-cli checkout root by walking up from this module.

    ``sloth/cli/_commands/eval.py`` → ``parents[3]`` is the checkout root (the
    dir containing the ``sloth/`` package), which is bind-mounted inside the NGC
    container so ``python -m sloth`` resolves without an install step.
    """
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Handler for ``sloth eval``.

    On the **host** (``--in-container`` not set): validates the adapter directory
    and suite file, then delegates GPU/ML work to the NGC container via
    :func:`sloth.tune.container.launch` (forwarding all args plus
    ``--in-container`` to prevent recursion), and returns the container's exit
    code.

    **Inside the container** (``--in-container`` is set): validates inputs, runs
    the eval suite through the mockable :data:`_inference_backend`, and emits
    results via the output contract.

    Parameters
    ----------
    args:
        Parsed namespace with ``adapter``, ``suite``, ``json``, and
        ``in_container`` attributes.

    Returns
    -------
    int
        ``0`` on success.  Failures raise :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))

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

    # --- HOST PATH: route GPU/ML work through the NGC container --------------
    if not in_container:
        sloth_args = [
            "eval",
            "--adapter",
            str(adapter.resolve()),
            "--suite",
            str(suite.resolve()),
        ]
        if json_mode:
            sloth_args.append("--json")
        sloth_args.append("--in-container")
        return container.launch(
            sloth_args,
            workdir=str(adapter.parent.resolve()),
            checkout=str(_repo_root()),
        )

    # --- IN-CONTAINER PATH: run the real eval --------------------------------
    records = validate_dataset(suite, schema="task")
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
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_eval)
