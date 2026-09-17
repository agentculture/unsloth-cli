"""``sloth bench`` — score an adapter or a model on a standard benchmark (MMLU).

Where ``sloth eval`` scores a *local, hand-written* suite, ``sloth bench`` scores
the public benchmark everyone quotes: it runs the lm-evaluation-harness
(``lm_eval``) **inside the NGC container**, where the benchmark dep layer
(:data:`sloth.tune.container.DEP_LAYER_BENCH_PACKAGES`) lives, and writes the
result as ``eval/<benchmark>.json`` — the *same* result shape
:func:`sloth.tune.metrics.write_eval_json` writes for a suite, so
``sloth summarize`` and ``sloth compare`` pick a benchmark up with no
special-casing.

**Two mutually-exclusive targets**, exactly like ``sloth eval``: ``--adapter DIR``
benches a LoRA/QLoRA adapter against its recorded base model
(``pretrained=<base>,peft=<dir>`` plus ``load_in_4bit=True`` for a QLoRA run),
``--model DIR`` benches a merged / exported directory (``pretrained=<dir>``).
Passing both — or neither — is a user error (exit ``1`` with a ``hint:``).

**Park v4 (the 4-bit base + PEFT adapter question).** Whether ``lm_eval``'s ``hf``
backend can load a 4-bit base wrapped with a PEFT adapter is a live-hardware
question. ``--adapter`` attempts it, because it is the cheap and correct thing
when it works; when the harness cannot load that combination the run exits ``2``
with a ``hint:`` naming the documented fallback — export the adapter merged to
16-bit and bench that directory with ``--model``. Both outcomes are supported by
design; ``docs/dgx-spark.md`` records which one the hardware took.

This module is **ML-free**: it imports no torch/peft/transformers/lm_eval at
module level. The harness work lives behind :mod:`sloth.tune._bench`, reached
only inside the container.

Usage::

    sloth bench --adapter adapters/qwen3-4b-qlora --benchmark mmlu
    sloth bench --model exports/qwen3-4b-merged16 --benchmark mmlu --limit 20 --json
    sloth bench --adapter adapters/qwen3-4b-qlora --offline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune import container, metrics
from sloth.tune._bench import DEFAULT_BENCHMARK, DEFAULT_NUM_FEWSHOT, run_bench

#: Environment variable lm_eval / huggingface_hub honour to forbid any network
#: access, forwarded into the container by ``--offline``.
HF_OFFLINE_ENV = "HF_HUB_OFFLINE"


def _repo_root() -> Path:
    """Return the unsloth-cli checkout root (bind-mounted into the container)."""
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Target resolution (--adapter XOR --model) — same contract as `sloth eval`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BenchTarget:
    """What is being benchmarked: an adapter directory or a model directory."""

    flag: str
    path: Path

    @property
    def kind(self) -> str:
        """``"adapter"`` or ``"model"`` — the flag without its dashes."""
        return self.flag.lstrip("-")

    @property
    def directory(self) -> Path:
        """The directory ``eval/<benchmark>.json`` is written under."""
        return self.path


def _resolve_target(args: argparse.Namespace) -> _BenchTarget:
    """Return the single benchmark target, or raise ``CliError(code=1)``.

    ``--adapter`` and ``--model`` are mutually exclusive **and** one is
    required. The check lives here, not in an argparse mutually-exclusive
    group, so a direct caller (an agent, a test) gets the same contract as the
    command line.
    """
    adapter = getattr(args, "adapter", None)
    model = getattr(args, "model", None)
    if adapter and model:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--adapter and --model are mutually exclusive; pass exactly one",
            remediation=(
                "Use --adapter <dir> to bench a LoRA/QLoRA adapter against its recorded "
                "base model, or --model <dir> to bench a merged / exported directory."
            ),
        )
    if not adapter and not model:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="one of --adapter or --model is required",
            remediation=(
                "Pass --adapter <dir> to bench a LoRA/QLoRA adapter, or --model <dir> "
                "to bench a merged / exported directory produced by `sloth export`."
            ),
        )
    flag, value = ("--adapter", adapter) if adapter else ("--model", model)
    path = Path(value)
    if not path.is_dir():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{'adapter' if adapter else 'model'} directory not found: {path}",
            remediation=(
                "Pass an existing adapter directory with --adapter <path> "
                "(run `sloth train` to produce one)."
                if adapter
                else (
                    "Pass an existing model directory with --model <path> "
                    "(run `sloth export` to produce one)."
                )
            ),
        )
    return _BenchTarget(flag=flag, path=path)


def _validate_limit(limit: int | None) -> None:
    """Raise ``CliError(code=1)`` unless *limit* is unset or a positive integer.

    Checked on the host, before any container launch, so ``--limit 0`` costs no
    docker/GPU cycle.
    """
    if limit is not None and limit < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--limit must be >= 1, got {limit}",
            remediation=(
                "Pass a positive number of documents per task with --limit N, or omit "
                "it to score the whole benchmark."
            ),
        )


def _validate_num_fewshot(num_fewshot: int) -> None:
    """Raise ``CliError(code=1)`` unless *num_fewshot* is zero or positive."""
    if num_fewshot < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--num-fewshot must be >= 0, got {num_fewshot}",
            remediation=(
                "Pass 0 for zero-shot or a positive integer with --num-fewshot "
                f"(MMLU's standard is {DEFAULT_NUM_FEWSHOT})."
            ),
        )


# ---------------------------------------------------------------------------
# Container routing
# ---------------------------------------------------------------------------


def _launch_container(
    target: _BenchTarget,
    *,
    benchmark: str,
    tasks: str,
    num_fewshot: int,
    limit: int | None,
    offline: bool,
) -> dict[str, Any]:
    """Re-run this bench inside the NGC container with ``--in-container``.

    The forwarded argv is ``bench <flag> <abs path> --benchmark <b> --tasks <t>
    --num-fewshot <n> [--limit N] [--offline] --json --in-container``. ``--json``
    is forwarded unconditionally (regardless of the host's own flag) so the
    container always prints a structured result line for
    :func:`sloth.tune.container.launch` to parse and hand back.

    An identity bind-mount (``host == container``) for the target's parent makes
    the host-absolute path resolve unchanged inside the container, and
    ``--offline`` becomes ``HF_HUB_OFFLINE=1`` in the container environment so
    the harness cannot reach the network at all.
    """
    target_abs = target.path.resolve()
    sloth_args = [
        "bench",
        target.flag,
        str(target_abs),
        "--benchmark",
        str(benchmark),
        "--tasks",
        str(tasks),
        "--num-fewshot",
        str(num_fewshot),
    ]
    if limit is not None:
        sloth_args += ["--limit", str(limit)]
    if offline:
        sloth_args.append("--offline")
    sloth_args.append("--json")
    sloth_args.append("--in-container")

    kwargs: dict[str, Any] = {
        "workdir": str(target_abs.parent),
        "checkout": str(_repo_root()),
        "extra_mounts": [(str(target_abs.parent), str(target_abs.parent))],
    }
    if offline:
        kwargs["env"] = [(HF_OFFLINE_ENV, "1")]
    return container.launch(sloth_args, **kwargs)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

#: How many per-subject rows the text report prints before summarising the rest.
_SUBJECT_PREVIEW = 10


def _render_text(target: _BenchTarget, benchmark: str, payload: dict[str, Any]) -> str:
    """Render one benchmark payload for stdout in text mode."""
    harness = payload.get("harness") or {}
    lines = [
        f"benchmark:  {benchmark}",
        f"{target.kind + ':':<11} {target.path}",
        f"harness:    {harness.get('name')} {harness.get('version') or '(unknown version)'}",
        f"tasks:      {harness.get('tasks')} ({harness.get('num_fewshot')}-shot)",
        f"total:      {payload.get('total')}",
        f"acc:        {payload.get('acc')}",
        f"score:      {payload.get('exact_match_pct')}%",
    ]
    per_subject = payload.get("per_subject") or {}
    for name in sorted(per_subject)[:_SUBJECT_PREVIEW]:
        lines.append(f"  {name}: {per_subject[name].get('acc')}")
    if len(per_subject) > _SUBJECT_PREVIEW:
        lines.append(f"  ... {len(per_subject) - _SUBJECT_PREVIEW} more subjects")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_bench(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth bench``.

    On the **host**: validates the flags and the target, then hands the GPU work
    to the NGC container via :func:`sloth.tune.container.launch`, forwarding
    ``--in-container`` to prevent docker recursion.

    **Inside the container**: runs :func:`sloth.tune._bench.run_bench` (the only
    code that touches ``lm_eval``), writes ``<target>/eval/<benchmark>.json`` via
    :func:`sloth.tune.metrics.write_eval_json`, and emits the payload through the
    output contract.

    Returns ``None`` on success; every failure raises :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    in_container = bool(getattr(args, "in_container", False))
    offline = bool(getattr(args, "offline", False))
    benchmark = getattr(args, "benchmark", None) or DEFAULT_BENCHMARK
    tasks = getattr(args, "tasks", None) or benchmark
    raw_fewshot = getattr(args, "num_fewshot", None)
    num_fewshot = DEFAULT_NUM_FEWSHOT if raw_fewshot is None else int(raw_fewshot)
    raw_limit = getattr(args, "limit", None)
    limit = None if raw_limit is None else int(raw_limit)

    _validate_limit(limit)
    _validate_num_fewshot(num_fewshot)
    target = _resolve_target(args)

    if not in_container:
        payload = _launch_container(
            target,
            benchmark=benchmark,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            offline=offline,
        )
        _emit(target, benchmark, payload, json_mode=json_mode)
        return None

    payload = run_bench(
        target.path,
        kind=target.kind,
        benchmark=benchmark,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    _write_eval_json(target, benchmark, payload)
    _emit(target, benchmark, payload, json_mode=json_mode)
    return None


def _write_eval_json(target: _BenchTarget, benchmark: str, payload: dict[str, Any]) -> None:
    """Write ``<target>/eval/<benchmark>.json``, degrading to a diagnostic on OSError.

    A read-only target directory should not lose an otherwise-complete benchmark
    run — the same failure handling ``sloth eval`` uses for its suite files.
    """
    try:
        metrics.write_eval_json(
            target.directory,
            benchmark,
            payload,
            target=target.kind,
            base_load_in_4bit=payload.get("base_load_in_4bit"),
        )
    except OSError as exc:
        emit_diagnostic(f"note: could not write eval/{benchmark}.json: {exc}")


def _emit(
    target: _BenchTarget, benchmark: str, payload: dict[str, Any], *, json_mode: bool
) -> None:
    """Emit *payload* through the output contract (stdout, JSON or text)."""
    if json_mode:
        emit_result(payload, json_mode=True)
    else:
        emit_result(_render_text(target, benchmark, payload), json_mode=False)


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``bench`` subparser on *sub*."""
    p = sub.add_parser(
        "bench",
        help="Score an adapter or model on a standard benchmark (MMLU) via lm_eval.",
    )
    p.add_argument("--adapter", help="Adapter directory produced by ``sloth train``.")
    p.add_argument(
        "--model",
        help=(
            "Merged / exported model directory produced by ``sloth export``. "
            "Mutually exclusive with --adapter."
        ),
    )
    p.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        metavar="NAME",
        help=(
            f"Benchmark to run; names the result file eval/<NAME>.json "
            f"(default: {DEFAULT_BENCHMARK})."
        ),
    )
    p.add_argument(
        "--tasks",
        default=None,
        metavar="SPEC",
        help="lm_eval task spec to run (default: the --benchmark name).",
    )
    p.add_argument(
        "--num-fewshot",
        dest="num_fewshot",
        type=int,
        default=DEFAULT_NUM_FEWSHOT,
        metavar="N",
        help=f"Few-shot examples per task (default: {DEFAULT_NUM_FEWSHOT}, MMLU's standard).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Score only the first N documents per task (smoke runs); must be >= 1.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help=(
            f"Forbid all network access by setting {HF_OFFLINE_ENV}=1 in the container; "
            "requires a warm Hugging Face cache."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--in-container",
        dest="in_container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(func=cmd_bench)
