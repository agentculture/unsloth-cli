"""In-container benchmark runner: MMLU (and friends) via ``lm_eval``.

This is the ``sloth bench`` seam, the benchmark sibling of
:mod:`sloth.tune._trainer`'s ``run_eval``. It runs **inside the NGC container**,
where the benchmark dep layer (:data:`sloth.tune.container.DEP_LAYER_BENCH_PACKAGES`
— ``lm_eval==0.4.13``) is installed; the host-side verb
(:mod:`sloth.cli._commands.bench`) never calls into here directly.

Import discipline
-----------------
Top-level imports are **pure stdlib**. ``lm_eval`` is imported lazily, inside
:func:`run_bench`, and only to read its ``__version__`` for the provenance block
— the evaluation itself is driven through the ``lm_eval`` **console script** as a
subprocess, so a harness crash is an exit code rather than an exception in this
process. ``tests/test_lazy_import.py`` asserts the stdlib-only property, exactly
as it does for :mod:`sloth.tune.metrics` and :mod:`sloth.tune.scorers`.

The in-container command
------------------------
:func:`lm_eval_command` composes, verbatim::

    lm_eval --model hf \\
            --model_args pretrained=<base>[,peft=<adapter>][,load_in_4bit=True] \\
            --tasks mmlu --num_fewshot 5 [--limit N] \\
            --output_path <tmp> --log_samples

For ``--adapter <dir>`` the base model id and the 4-bit flag come from the
adapter's own ``training_metadata.json`` (``model`` and
``hyperparameters.load_in_4bit``, see :func:`resolve_adapter_target`); for
``--model <dir>`` the directory itself is ``pretrained=`` and no ``peft=`` is
passed.

Park v4 — does lm_eval load a 4-bit base + a PEFT adapter?
----------------------------------------------------------
That is a **live-hardware question** this module does not pretend to answer: the
QLoRA path (``peft=<adapter>,load_in_4bit=True``) is attempted first because it
is the cheap, correct thing when it works. When the harness fails to load that
combination, :func:`run_bench` raises :class:`~sloth.cli._errors.CliError` with
``code=2`` whose remediation names the documented fallback — export the adapter
merged to 16-bit (``sloth export --format merged16``) and bench that directory
with ``--model``. Both outcomes are supported by design; ``docs/dgx-spark.md``
records which one the hardware actually took.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess  # nosec B404 - fixed argv, no shell, container-local binary
import tempfile
from pathlib import Path
from typing import Any, Sequence

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic
from sloth.tune.metadata import resolve_load_in_4bit

#: The lm-evaluation-harness console script, as installed by the container's
#: benchmark dep layer. Named (not a ``python -m`` invocation) so the composed
#: command reads exactly like the one in ``docs/dgx-spark.md``.
LM_EVAL_BIN = "lm_eval"

#: The benchmark this verb ships with. ``--tasks`` may name any lm_eval task
#: spec, but ``mmlu`` is the one the acceptance criteria and the docs cover.
DEFAULT_BENCHMARK = "mmlu"

#: MMLU's standard few-shot count (the 5-shot number everyone quotes).
DEFAULT_NUM_FEWSHOT = 5

#: Remediation used whenever the harness could not load the requested model —
#: the documented park-v4 fallback.
MERGE_FALLBACK_HINT = (
    "lm_eval could not load this target. If it is a QLoRA adapter, the "
    "4-bit-base + PEFT-adapter load path may be unsupported by this lm_eval / "
    "peft / bitsandbytes combination: export the adapter merged to 16-bit "
    "(`sloth export --adapter <dir> --format merged16 --output <dir>`) and "
    "bench that directory with `sloth bench --model <dir>`. See the "
    "'Warm-cache check for sloth bench' section of docs/dgx-spark.md."
)


# ---------------------------------------------------------------------------
# Argv validation (every CLI-provided string is checked BEFORE the argv is built)
# ---------------------------------------------------------------------------

#: An lm_eval ``--tasks`` spec: task names, joined by commas, nothing else. No
#: shell metacharacters, no whitespace, no ``=`` (which would let a task string
#: smuggle an extra ``--model_args`` key past the composer).
_TASK_SPEC_RE = re.compile(r"^[A-Za-z0-9_,.-]+$")

#: A Hugging Face repo id, ``org/name``. Deliberately narrower than the hub's own
#: rules: no commas, no ``=``, no shell metacharacters.
_LOCAL_PATH_RE = re.compile(r"^[A-Za-z0-9._~/-]+$")
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Characters that would break out of one ``--model_args`` fragment into another
#: (``key=value`` pairs joined by ``,``) — never allowed inside a model reference.
_MODEL_ARGS_SEPARATORS = (",", "=")


def _validate_tasks(tasks: str) -> str:
    """Return *tasks* when it is a plain lm_eval task spec, else ``CliError(code=1)``."""
    if not _TASK_SPEC_RE.match(tasks):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"invalid --tasks spec: {tasks!r}",
            remediation=(
                "A task spec is one or more lm_eval task names joined by commas, using "
                "only letters, digits, '_', '.', '-' (e.g. `mmlu` or "
                "`mmlu_astronomy,mmlu_logic`). Drop any spaces or shell characters."
            ),
        )
    return tasks


def _validate_count(value: int | None, flag: str) -> int | None:
    """Return *value* when it is a non-negative integer (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"invalid {flag}: {value!r}",
            remediation=f"Pass {flag} as a non-negative whole number, e.g. `{flag} 20`.",
        )
    return value


def _validate_model_reference(value: str, label: str) -> str:
    """Return *value* when it is an existing path or a ``org/name`` repo id.

    Everything that reaches ``--model_args`` as ``pretrained=`` / ``peft=`` goes
    through here: a reference that is neither on disk nor a well-formed hub id —
    or that carries a ``,``/``=`` and could therefore forge a second
    ``--model_args`` fragment — is a user error, not something to hand to the
    harness.
    """
    text = str(value)
    if any(separator in text for separator in _MODEL_ARGS_SEPARATORS):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"invalid {label} reference: {text!r}",
            remediation=(
                "A model reference may not contain ',' or '=' — those separate "
                "lm_eval's --model_args fragments. Rename the directory, or pass a "
                "Hugging Face repo id of the form org/name."
            ),
        )
    # No filesystem probe here (a missing directory surfaces as lm_eval's own
    # load error): a reference is either an HF repo id or a plain path made of
    # safe characters that cannot start a new argv flag.
    if _HF_REPO_ID_RE.match(text) or (_LOCAL_PATH_RE.match(text) and not text.startswith("-")):
        return text
    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"invalid {label} reference: {text!r}",
        remediation=(
            "Pass a local directory path (letters, digits, '/', '.', '_', '~' and '-' "
            "only, not starting with '-'), or a Hugging Face repo id of the form "
            "org/name."
        ),
    )


# ---------------------------------------------------------------------------
# Target resolution (pure stdlib, no model load)
# ---------------------------------------------------------------------------


def resolve_adapter_target(adapter_dir: str | Path) -> tuple[str, bool]:
    """Return ``(base_model_id, load_in_4bit)`` for the adapter at *adapter_dir*.

    Both come from the adapter's ``training_metadata.json`` (written by
    :func:`sloth.tune.metadata.write_metadata`): ``model`` names the base the
    adapter was trained against, and the *effective* 4-bit flag decides whether
    ``load_in_4bit=True`` is added to ``--model_args``. That flag is
    ``resolved.load_in_4bit`` when the record carries one, else
    :func:`sloth.tune.metadata.resolve_load_in_4bit` of the recorded ``method``
    and ``hyperparameters.load_in_4bit`` — a ``qlora`` run trains in 4-bit even
    when the raw hyperparameter is ``false``, and benching it in full precision
    would measure a model that was never trained.

    Falls back to ``adapter_config.json``'s ``base_model_name_or_path`` when the
    metadata file is missing or carries no ``model``, so an adapter produced
    outside ``sloth train`` still benches (with ``load_in_4bit`` defaulting to
    ``False``).

    Raises
    ------
    CliError(code=1)
        When neither file names a base model — there is nothing to benchmark
        the adapter against.
    """
    directory = Path(adapter_dir)
    base: Any = None
    load_in_4bit: Any = None

    metadata = _read_json(directory / "training_metadata.json")
    if metadata is not None:
        base = metadata.get("model")
        hyperparameters = metadata.get("hyperparameters")
        # Training forces 4-bit for a `qlora` method, whatever the configured
        # flag says, so the raw hyperparameter alone would bench a QLoRA adapter
        # at the wrong precision. `resolved.load_in_4bit` (written since the
        # fix) is the effective value and wins whenever it is present.
        load_in_4bit = resolve_load_in_4bit(
            str(metadata.get("method") or ""),
            hyperparameters if isinstance(hyperparameters, dict) else None,
        )
        resolved = metadata.get("resolved")
        if isinstance(resolved, dict) and "load_in_4bit" in resolved:
            load_in_4bit = resolved["load_in_4bit"]

    if not isinstance(base, str) or not base:
        adapter_config = _read_json(directory / "adapter_config.json") or {}
        base = adapter_config.get("base_model_name_or_path")

    if not isinstance(base, str) or not base:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"cannot determine the base model for adapter {directory}",
            remediation=(
                "The adapter directory must carry a training_metadata.json with a "
                "'model' field (written by `sloth train`) or an adapter_config.json "
                "with 'base_model_name_or_path'. Re-run `sloth train`, or bench a "
                "merged export with `sloth bench --model <dir>`."
            ),
        )
    return base, bool(load_in_4bit)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON object, tolerating absence/corruption (never raises)."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Command composition (pure — the fake-launcher tests assert on this)
# ---------------------------------------------------------------------------


def build_model_args(
    pretrained: str, *, peft: str | None = None, load_in_4bit: bool = False
) -> str:
    """Compose lm_eval's ``--model_args`` value.

    ``pretrained=<base>`` always leads; ``peft=<adapter>`` follows when an
    adapter is being benched, and ``load_in_4bit=True`` last when the adapter
    was a QLoRA run. The order is fixed so the composed command is
    byte-deterministic and assertable.
    """
    parts = [f"pretrained={pretrained}"]
    if peft:
        parts.append(f"peft={peft}")
    if load_in_4bit:
        parts.append("load_in_4bit=True")
    return ",".join(parts)


def lm_eval_command(
    *,
    pretrained: str,
    output_path: str | Path,
    peft: str | None = None,
    load_in_4bit: bool = False,
    tasks: str = DEFAULT_BENCHMARK,
    num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    limit: int | None = None,
) -> list[str]:
    """Return the deterministic ``lm_eval`` argv run inside the container.

    See the module docstring for the literal shape. ``--log_samples`` is always
    passed so the per-sample record lands next to the results file and a
    surprising score can be audited after the fact.
    """
    cmd = [
        LM_EVAL_BIN,
        "--model",
        "hf",
        "--model_args",
        build_model_args(pretrained, peft=peft, load_in_4bit=load_in_4bit),
        "--tasks",
        str(tasks),
        "--num_fewshot",
        str(num_fewshot),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += ["--output_path", str(output_path), "--log_samples"]
    return cmd


# ---------------------------------------------------------------------------
# Subprocess seam (isolated so tests can monkeypatch it)
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str]) -> int:
    """Run *cmd* (list form, no shell) and return its exit code, streaming output.

    stdout/stderr are left attached to this process so the harness's own
    progress lines reach the operator's terminal through
    :func:`sloth.tune.container.launch`'s tee.

    Injection surface: *cmd* is always the fixed argv :func:`lm_eval_command`
    composes, run with ``shell=False`` (the list form), and every CLI-provided
    string that reaches it — the task spec, the row/few-shot counts and the
    model/adapter references — is validated by :func:`_validate_tasks`,
    :func:`_validate_count` and :func:`_validate_model_reference` before
    :func:`run_bench` composes the argv. No value can introduce a new flag, a new
    ``--model_args`` fragment or a shell metacharacter.
    """
    emit_diagnostic(f"running: {shlex.join(cmd)}")
    # pythonsecurity:S6350 - shell=False list argv; every user-supplied element is
    # validated in run_bench (see this function's docstring).
    return subprocess.run(list(cmd), check=False).returncode  # nosec B603 # NOSONAR


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def find_results_file(output_path: str | Path) -> Path | None:
    """Return the newest ``results*.json`` lm_eval wrote under *output_path*.

    lm_eval nests its output under a sanitised model-name directory whose exact
    spelling depends on the model id, so the file is found by glob rather than
    by construction. ``None`` when the harness wrote nothing.
    """
    candidates = sorted(Path(output_path).rglob("results*.json"))
    return candidates[-1] if candidates else None


def _metric(entry: Any, name: str) -> float | None:
    """Read one lm_eval metric out of a results entry, tolerating its key suffixes.

    lm_eval 0.4.x keys metrics as ``"<name>,<filter>"`` (e.g. ``"acc,none"``),
    so an exact-key lookup is tried first and a ``"<name>,"``-prefixed key
    second. Returns ``None`` when the metric was not reported — MMLU, for
    instance, reports ``acc`` but no ``acc_norm``.
    """
    if not isinstance(entry, dict):
        return None
    if isinstance(entry.get(name), (int, float)) and not isinstance(entry.get(name), bool):
        return float(entry[name])
    for key, value in entry.items():
        if (
            isinstance(key, str)
            and key.startswith(f"{name},")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return float(value)
    return None


def _sample_count(document: dict[str, Any], task: str) -> int:
    """Return how many samples *task* was actually scored on, or ``0``.

    ``n-samples`` maps each task to ``{"original": N, "effective": M}``; the
    *effective* count is what ``--limit`` shrinks, so it is the honest total.
    """
    samples = document.get("n-samples")
    if not isinstance(samples, dict):
        return 0
    entry = samples.get(task)
    if not isinstance(entry, dict):
        return 0
    value = entry.get("effective", entry.get("original"))
    return int(value) if isinstance(value, (int, float)) else 0


def build_payload(
    document: dict[str, Any],
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    tasks: str = DEFAULT_BENCHMARK,
    num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    limit: int | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Turn lm_eval's results *document* into this repo's eval-result payload.

    The shape deliberately matches what :func:`sloth.tune.metrics.write_eval_json`
    writes for an ordinary suite, so ``sloth summarize`` / ``sloth compare`` fold
    a benchmark run in with no special-casing
    (:func:`sloth.tune.summary._eval_summary` folds every top-level numeric key):

    ``acc``
        The benchmark's overall accuracy as a ``0..1`` fraction.
    ``acc_norm``
        The length-normalised accuracy **when the harness reports one**. MMLU
        does not, so this is ``None`` for an MMLU run — the key is always
        present, never invented (a ``None`` is skipped by the numeric fold
        rather than scored as ``0``).
    ``exact_match_pct``
        ``acc * 100``, so the flat percentage field every other suite carries
        means the same thing here.
    ``total``
        Number of samples actually scored (``--limit`` aware).
    ``per_subject``
        ``{subject: {acc, acc_norm, total}}`` for every sub-task (MMLU's 57
        subjects), with the ``<benchmark>_`` prefix stripped from the name.
    ``harness``
        ``{"name": "lm_eval", "version", "tasks", "limit", "num_fewshot"}`` —
        the provenance of the number.
    """
    results = document.get("results")
    results = results if isinstance(results, dict) else {}

    per_subject: dict[str, dict[str, Any]] = {}
    prefix = f"{benchmark}_"
    for name in sorted(results):
        if name == benchmark or not name.startswith(prefix):
            continue
        entry = results[name]
        acc = _metric(entry, "acc")
        if acc is None:
            # A group row (e.g. ``mmlu_stem``) that aggregates other rows and
            # reports no accuracy of its own: nothing to record.
            continue
        per_subject[name[len(prefix) :]] = {
            "acc": round(acc, 4),
            "acc_norm": _round_or_none(_metric(entry, "acc_norm")),
            "total": _sample_count(document, name),
        }

    overall = results.get(benchmark)
    acc = _metric(overall, "acc")
    if acc is None and per_subject:
        acc = sum(s["acc"] for s in per_subject.values()) / len(per_subject)
    acc = float(acc or 0.0)

    total = _sample_count(document, benchmark)
    if not total:
        total = sum(s["total"] for s in per_subject.values())

    return {
        "acc": round(acc, 4),
        "acc_norm": _round_or_none(_metric(overall, "acc_norm")),
        "exact_match_pct": round(acc * 100, 2),
        "total": total,
        "per_subject": per_subject,
        "harness": {
            "name": "lm_eval",
            "version": version,
            "tasks": tasks,
            "limit": limit,
            "num_fewshot": num_fewshot,
        },
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# ---------------------------------------------------------------------------
# The run function (the only place lm_eval is imported)
# ---------------------------------------------------------------------------


def _lm_eval_version() -> str | None:
    """Return the installed ``lm_eval`` version, or ``None`` when absent.

    This is the module's **only** ``lm_eval`` import, and it lives inside a
    function body — the lazy-import discipline every heavy dependency in
    :mod:`sloth.tune` follows. A missing harness is not fatal here: the
    subprocess call is what needs it, and that failure is reported with its own
    remediation.
    """
    try:
        import lm_eval  # noqa: WPS433  (deliberate lazy import)
    except ImportError:
        return None
    return getattr(lm_eval, "__version__", None)


def run_bench(
    target_path: str | Path,
    *,
    kind: str = "adapter",
    benchmark: str = DEFAULT_BENCHMARK,
    tasks: str | None = None,
    num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    limit: int | None = None,
) -> dict[str, Any]:
    """Benchmark *target_path* with ``lm_eval`` and return the result payload.

    *kind* is ``"adapter"`` (``pretrained=<base>,peft=<dir>``, plus
    ``load_in_4bit=True`` for a QLoRA adapter) or ``"model"``
    (``pretrained=<dir>``). *tasks* defaults to *benchmark*.

    Runs entirely offline when ``HF_HUB_OFFLINE=1`` is set in the environment
    (``sloth bench --offline`` forwards it); otherwise a warm Hugging Face cache
    is what keeps the second run download-free — see the warm-cache check in
    ``docs/dgx-spark.md``.

    Raises
    ------
    CliError(code=2)
        When the harness exits non-zero (including the QLoRA 4-bit + PEFT load
        path failing — see the module docstring's park-v4 note) or writes no
        results file. The remediation names the merged-16-bit fallback.
    """
    target = Path(target_path)
    task_spec = _validate_tasks(tasks or benchmark)
    num_fewshot = _validate_count(num_fewshot, "--num-fewshot") or 0
    limit = _validate_count(limit, "--limit")
    if kind == "adapter":
        base, load_in_4bit = resolve_adapter_target(target)
        peft: str | None = _validate_model_reference(str(target), "adapter")
    else:
        base, load_in_4bit, peft = str(target), False, None
    base = _validate_model_reference(base, "model")

    with tempfile.TemporaryDirectory(prefix="sloth-bench-") as tmp:
        cmd = lm_eval_command(
            pretrained=base,
            output_path=tmp,
            peft=peft,
            load_in_4bit=load_in_4bit,
            tasks=task_spec,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        code = _run(cmd)
        if code != 0:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"lm_eval exited {code} while benchmarking {target}",
                remediation=MERGE_FALLBACK_HINT,
            )
        results_file = find_results_file(tmp)
        if results_file is None:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"lm_eval wrote no results file for {target}",
                remediation=MERGE_FALLBACK_HINT,
            )
        document = _read_json(results_file)

    if document is None:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"lm_eval's results file for {target} was unreadable",
            remediation=MERGE_FALLBACK_HINT,
        )

    payload = build_payload(
        document,
        benchmark=benchmark,
        tasks=task_spec,
        num_fewshot=num_fewshot,
        limit=limit,
        version=_lm_eval_version(),
    )
    payload["base_load_in_4bit"] = load_in_4bit if kind == "adapter" else None
    payload["model"] = base
    return payload
