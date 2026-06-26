"""Model-scope guard for unsloth-cli adapter fine-tuning.

Classifies whether a (model, method) request is an in-scope LoRA/QLoRA adapter
job or an out-of-scope request (e.g. full fine-tuning, unsupported method).

Design principles
-----------------
* Pure stdlib — no torch, no unsloth, no external deps.
* Returns structured data only; never prints. The calling verb emits diagnostics
  via the CLI's output contract.
* Rules are data-driven: thresholds and supported methods live in module
  constants so they can be updated without touching logic.

Thresholds
----------
``LARGE_DENSE_THRESHOLD_B``
    Parameter count (in billions) above which a model is considered "large dense"
    for the purpose of full-FT warnings.  Set to **10** — this keeps 4B and 9B
    Qwen variants firmly in the small/medium bucket while flagging 27B, 32B, 72B.
    (MoE models may report a total parameter count that is large but their
    active-parameter count is much smaller; the guard uses the number it can
    parse from the model name string, so "57b-a14b" reads as 57.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Public constants (exported so tests and CLI verbs can reference them)
# ---------------------------------------------------------------------------

#: Methods that produce LoRA/QLoRA adapters — always in scope.
SUPPORTED_ADAPTER_METHODS: frozenset[str] = frozenset({"lora", "qlora"})

#: Method aliases that mean "full parameter fine-tuning" — out of scope.
FULL_FT_METHODS: frozenset[str] = frozenset({"full", "full_ft", "full-ft", "fullft"})

#: Models with a parsed parameter count *above* this threshold (in billions)
#: are flagged as "large dense" when the method is full fine-tuning.
LARGE_DENSE_THRESHOLD_B: int = 10

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_FULL_FT_WARNING = (
    "Full fine-tuning (method='{method}') is out of scope for unsloth-cli. "
    "This tool supports LoRA and QLoRA adapter training only. "
    "Full fine-tuning of large dense models requires far more GPU memory and "
    "is not supported. "
    "Recommendation: switch to method='lora' or method='qlora'."
)

_UNSUPPORTED_METHOD_WARNING = (
    "Method '{method}' is not supported by unsloth-cli. "
    "Only LoRA and QLoRA adapter methods are in scope. "
    "Recommendation: use method='lora' or method='qlora'."
)

_OK_MESSAGE = "Model '{model}' with method='{method}' is in scope — proceed with adapter training."

_OUT_OF_SCOPE_MESSAGE = (
    "Request is out of scope: model='{model}', method='{method}'. See warning for details."
)


@dataclass
class ScopeResult:
    """Result of a scope check for a (model, method) request.

    Attributes
    ----------
    ok:
        ``True`` when the request is fully in scope and can proceed.
    out_of_scope:
        ``True`` when the request is rejected or must be downgraded.
    warning:
        Human-readable warning string, or ``None`` when ``ok`` is ``True``.
        The CLI verb must emit this via ``emit_diagnostic``; do not print here.
    message:
        Short summary suitable for structured JSON output.
    downgrade_to:
        Suggested replacement method when the requested method is out of scope,
        or ``None`` when the request is already acceptable.
    """

    ok: bool
    out_of_scope: bool
    warning: str | None
    message: str
    downgrade_to: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PARAM_COUNT_RE = re.compile(
    r"(?<![a-z])(\d+(?:\.\d+)?)b(?!\w)",
    re.IGNORECASE,
)


def _parse_largest_param_count(model: str) -> float | None:
    """Return the largest parameter count (in billions) found in *model*, or ``None``."""
    matches = _PARAM_COUNT_RE.findall(model)
    if not matches:
        return None
    return max(float(m) for m in matches)


def _is_large_model(model: str) -> bool:
    """Return ``True`` when *model* appears to exceed the large-dense threshold."""
    count = _parse_largest_param_count(model)
    if count is None:
        return False
    return count > LARGE_DENSE_THRESHOLD_B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_scope(model: str, method: str) -> ScopeResult:
    """Classify a (model, method) fine-tuning request as in-scope or out-of-scope.

    Parameters
    ----------
    model:
        Model identifier or path (e.g. ``"unsloth/Qwen3-4B"``).  A parameter
        count hint such as ``"4b"`` or ``"27b"`` may be embedded in the name
        and is used to assess model size.
    method:
        Training method string, case-insensitive (e.g. ``"lora"``, ``"qlora"``,
        ``"full"``).

    Returns
    -------
    ScopeResult
        Structured verdict.  When ``ok`` is ``True`` the caller may proceed.
        When ``out_of_scope`` is ``True`` the ``warning`` field explains why and
        ``downgrade_to`` (if set) suggests an alternative method.
    """
    method_lower = method.strip().lower()

    # --- Adapter methods: always in scope ------------------------------------
    if method_lower in SUPPORTED_ADAPTER_METHODS:
        return ScopeResult(
            ok=True,
            out_of_scope=False,
            warning=None,
            message=_OK_MESSAGE.format(model=model, method=method),
            downgrade_to=None,
        )

    # --- Full fine-tuning: out of scope -------------------------------------
    if method_lower in FULL_FT_METHODS:
        warning = _OUT_OF_SCOPE_FULL_FT_WARNING.format(method=method)
        return ScopeResult(
            ok=False,
            out_of_scope=True,
            warning=warning,
            message=_OUT_OF_SCOPE_MESSAGE.format(model=model, method=method),
            downgrade_to="lora",
        )

    # --- Anything else: unsupported -----------------------------------------
    warning = _UNSUPPORTED_METHOD_WARNING.format(method=method)
    return ScopeResult(
        ok=False,
        out_of_scope=True,
        warning=warning,
        message=_OUT_OF_SCOPE_MESSAGE.format(model=model, method=method),
        downgrade_to="lora",
    )
