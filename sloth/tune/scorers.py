"""Pure-stdlib-at-import scoring helpers: instruction constraints, a JSON-schema
subset checker, a per-family tool-call parser registry, and lazy GLEU/BLEU.

Like :mod:`sloth.tune.metrics` / :mod:`sloth.tune.datasets` / :mod:`sloth.tune.config`
/ :mod:`sloth.tune.scope`, this module's **top-level imports are pure stdlib**
(``tests/test_lazy_import.py`` asserts this), so it stays importable and testable
without torch, unsloth, or any third-party package installed. The only
third-party import — ``sacrebleu`` — is deferred *inside* :func:`gleu` /
:func:`bleu`, exactly like the rest of the container-only ML stack is
lazy-imported elsewhere in :mod:`sloth.tune`.

Five capabilities
------------------
1. :func:`score_constraints` — evaluate a prediction against a list of
   instruction constraints (``max_words``, ``min_words``, ``must_contain``,
   ``must_not_contain``, ``must_refuse``, ``json_only``).
2. :func:`check_json_subset` — validate a prediction (parsed as JSON) against a
   documented *subset* of JSON Schema keywords (``type``, ``required``,
   ``properties``, ``enum``, ``items``, ``additionalProperties``).
3. :func:`parse_tool_call` / :func:`score_tool_call` — parse a model's raw
   tool-call text via a per-family parser registry (``qwen3``, ``lfm2``) and
   compare it against an expected ``{"name", "arguments"}`` call.
   :func:`family_for_model` maps a model id to a registered family by
   substring, overridable by the caller (e.g. from a run-config's
   ``[eval] tool_call_family``).
4. :func:`gleu` / :func:`bleu` — sentence-level GLEU/BLEU, lazily depending on
   ``sacrebleu`` (installed only in the container dep layer per
   ``docs/dgx-spark.md``); calling either without it installed raises
   :class:`~sloth.cli._errors.CliError` with a remediation hint instead of an
   ``ImportError``.
5. :func:`extract_choice_letter` — pull the answer letter (``A``–``D``) out of a
   multiple-choice prediction, so a letter-choice suite (e.g.
   ``examples/eval/mmlu-subset.jsonl``) is scored on the *choice* the model made
   rather than on a whole-string exact match.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any, Callable

from sloth.cli._errors import CliError

# ---------------------------------------------------------------------------
# 1. Instruction constraints
# ---------------------------------------------------------------------------

#: Constraint kinds recognised by :func:`score_constraints`. Each constraint is
#: a dict ``{"kind": <one of these>, "value": ...}`` — ``must_refuse`` and
#: ``json_only`` take no ``value``.
CONSTRAINT_KINDS = frozenset(
    {
        "max_words",
        "min_words",
        "must_contain",
        "must_not_contain",
        "must_refuse",
        "json_only",
    }
)

#: Regex over common refusal phrasings, used by the ``must_refuse`` constraint.
#: Deliberately conservative (favors precision): it matches first-person
#: "I can't / I'm unable to / I won't / sorry, but I can't"-style refusals, the
#: most common shapes a small instruction-tuned model produces when declining.
REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+can(?:not|'t)\s+(?:help|assist|do\s+that|comply|provide)"
    r"|i'?m\s+(?:not\s+able|unable)\s+to"
    r"|i\s+won'?t\b"
    r"|as\s+an\s+ai\b[^.]*\b(?:cannot|can't|can\s+not)\b"
    r"|sorry,?\s+(?:but\s+)?i\s+(?:can(?:not|'t)|won'?t)"
    r")",
    re.IGNORECASE,
)


def _check_constraint(prediction: str, constraint: dict[str, Any]) -> bool:
    """Return whether *prediction* satisfies a single *constraint*."""
    kind = constraint.get("kind")
    if kind == "max_words":
        return len(prediction.split()) <= constraint["value"]
    if kind == "min_words":
        return len(prediction.split()) >= constraint["value"]
    if kind == "must_contain":
        return constraint["value"] in prediction
    if kind == "must_not_contain":
        return constraint["value"] not in prediction
    if kind == "must_refuse":
        return bool(REFUSAL_RE.search(prediction))
    if kind == "json_only":
        try:
            json.loads(prediction.strip())
        except ValueError:
            return False
        return True
    raise ValueError(
        f"unknown constraint kind {kind!r}; supported kinds: {sorted(CONSTRAINT_KINDS)}"
    )


def score_constraints(prediction: str, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    """Score *prediction* against a list of constraint dicts.

    Each constraint is ``{"kind": <CONSTRAINT_KINDS member>, "value": ...}``
    (``must_refuse`` / ``json_only`` omit ``value``). Returns
    ``{"passed": bool, "failed": [<kind>, ...]}`` — ``passed`` is ``True`` only
    when every constraint is satisfied; ``failed`` lists the ``kind`` of each
    constraint that was not (in order, duplicates preserved if the same kind
    appears more than once).
    """
    failed = [c.get("kind") for c in constraints if not _check_constraint(prediction, c)]
    return {"passed": not failed, "failed": failed}


# ---------------------------------------------------------------------------
# 2. JSON-schema subset checker
# ---------------------------------------------------------------------------

#: The only schema keywords this checker understands. Any other keyword in a
#: schema dict is reported as an error naming it (rather than silently ignored).
SCHEMA_KEYWORDS = frozenset(
    {"type", "required", "properties", "enum", "items", "additionalProperties"}
)

_SCHEMA_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _report_unknown_keywords(schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Name every schema keyword outside :data:`SCHEMA_KEYWORDS` (never silently ignored)."""
    for keyword in sorted(set(schema.keys()) - SCHEMA_KEYWORDS):
        errors.append(f"{path}: unsupported schema keyword {keyword!r}")


def _check_type_keyword(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Apply ``"type"``: unknown type names and type mismatches both report."""
    expected = schema["type"]
    pytype = _SCHEMA_TYPES.get(expected)
    if pytype is None:
        errors.append(f"{path}: unknown type {expected!r}")
    elif isinstance(data, bool) and expected != "boolean":
        # bool is an int subclass in Python; keep "type": "integer"/"number"
        # from accepting True/False.
        errors.append(f"{path}: expected {expected}, got bool")
    elif not isinstance(data, pytype):
        errors.append(f"{path}: expected {expected}, got {type(data).__name__}")


def _check_enum_keyword(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Apply ``"enum"``: *data* must be one of the listed values."""
    if data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in enum {schema['enum']!r}")


def _check_required_keyword(
    data: Any, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    """Apply ``"required"`` to a mapping: every named property must be present."""
    for key in schema["required"]:
        if key not in data:
            errors.append(f"{path}: missing required property {key!r}")


def _check_properties_keyword(
    data: Any, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    """Apply ``"properties"``: recurse into each declared property that is present."""
    for key, subschema in schema["properties"].items():
        if key in data:
            _validate_subset(data[key], subschema, f"{path}.{key}", errors)


def _check_additional_properties_keyword(
    data: Any, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    """Apply ``"additionalProperties"``: ``False`` forbids extras, a dict validates them."""
    additional = schema["additionalProperties"]
    extra = sorted(set(data.keys()) - set(schema.get("properties", {})))
    if additional is False:
        for key in extra:
            errors.append(f"{path}: additional property {key!r} not allowed")
    elif isinstance(additional, dict):
        for key in extra:
            _validate_subset(data[key], additional, f"{path}.{key}", errors)


def _check_items_keyword(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Apply ``"items"`` to a list: recurse into every element."""
    for index, item in enumerate(data):
        _validate_subset(item, schema["items"], f"{path}[{index}]", errors)


_Checker = Callable[[Any, dict[str, Any], str, list[str]], None]

#: Keyword checkers applied to any value, in report order.
_VALUE_CHECKERS: dict[str, _Checker] = {
    "type": _check_type_keyword,
    "enum": _check_enum_keyword,
}

#: Keyword checkers applied only when the value is a mapping, in report order.
_MAPPING_CHECKERS: dict[str, _Checker] = {
    "required": _check_required_keyword,
    "properties": _check_properties_keyword,
    "additionalProperties": _check_additional_properties_keyword,
}


def _apply_checkers(
    checkers: dict[str, _Checker],
    data: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Run each checker in *checkers* whose keyword is present in *schema*, in order."""
    for keyword, check in checkers.items():
        if keyword in schema:
            check(data, schema, path, errors)


def _validate_subset(data: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    _report_unknown_keywords(schema, path, errors)
    _apply_checkers(_VALUE_CHECKERS, data, schema, path, errors)
    if isinstance(data, dict):
        _apply_checkers(_MAPPING_CHECKERS, data, schema, path, errors)
    if "items" in schema and isinstance(data, list):
        _check_items_keyword(data, schema, path, errors)


def check_json_subset(prediction: str | Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate *prediction* against a documented subset of JSON Schema.

    *prediction* is JSON-decoded first when it is a ``str`` (otherwise used
    as-is, so callers may pass an already-parsed value). Supported keywords:
    ``type``, ``required``, ``properties``, ``enum``, ``items``,
    ``additionalProperties`` — applied recursively through ``properties`` /
    ``items`` / ``additionalProperties``. Any other keyword present in a
    schema dict is reported as an error naming it, rather than ignored.

    Returns ``{"valid": bool, "errors": [<str>, ...]}``.
    """
    if isinstance(prediction, str):
        try:
            data = json.loads(prediction)
        except ValueError as exc:
            return {"valid": False, "errors": [f"invalid JSON: {exc}"]}
    else:
        data = prediction
    errors: list[str] = []
    _validate_subset(data, schema, "$", errors)
    return {"valid": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# 3. Per-family tool-call parsers
# ---------------------------------------------------------------------------


def _parse_qwen3_tool_call(prediction: str) -> dict[str, Any]:
    """Parse a qwen3-family tool call.

    Source: ``chat_template.jinja`` shipped with the cached qwen3 snapshots
    under ``~/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B/...`` (and the
    other cached ``Qwen3*`` / ``unsloth--Qwen3*`` snapshots, which share the
    same template). The assistant-turn rendering is::

        <tool_call>
        {"name": "<function-name>", "arguments": <args-json-object>}
        </tool_call>

    i.e. one JSON object per call, wrapped in literal ``<tool_call>`` /
    ``</tool_call>`` tags. ``arguments`` is normally a JSON object but the
    template also accepts (and re-emits verbatim) a JSON-encoded *string*, so
    this parser decodes it a second time when needed.
    """
    # NOSONAR: the reluctant `.*?` is required here -- `arguments` is itself a JSON
    # object, so a negated class like `[^}]*` would stop at the first inner `}` and
    # fail to match any nested tool call. The `\}\s*</tool_call>` tail bounds it.
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction, re.DOTALL)  # NOSONAR
    if not match:
        raise ValueError("no <tool_call>...</tool_call> block found in prediction")
    payload = json.loads(match.group(1))
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    return {"name": name, "arguments": arguments}


def _consume_quoted_char(
    text: str, index: int, current: list[str], quote: str
) -> tuple[int, str | None]:
    """Copy the character at *index* while inside a quoted run.

    Returns the next index and the still-open quote character (``None`` once the
    run closes). A backslash escape copies the following character verbatim, so
    an escaped quote never ends the run.
    """
    char = text[index]
    current.append(char)
    if char == "\\" and index + 1 < len(text):
        current.append(text[index + 1])
        return index + 2, quote
    return index + 1, None if char == quote else quote


def _split_top_level(text: str) -> list[str]:
    """Split *text* on top-level commas, respecting quotes and bracket nesting."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            index, quote = _consume_quoted_char(text, index, current, quote)
            continue
        if char in "'\"":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if current:
        parts.append("".join(current))
    return parts


def _parse_lfm2_value(raw: str) -> Any:
    """Best-effort parse of one LFM2.5-style argument value.

    Values are rendered by ``format_arg_value`` in the tokenizer's chat
    template: a Python-single-quoted string for ``str`` args, ``tojson`` (JSON)
    for mappings/iterables, or Python's ``str()`` for everything else (numbers,
    booleans, ``None``). ``json.loads`` handles the JSON case; Python literal
    syntax (single-quoted strings, ``True``/``False``/``None``, numbers) is
    handled by :func:`ast.literal_eval`. Anything neither parses as is
    returned verbatim, stripped.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_lfm2_tool_call(prediction: str) -> dict[str, Any]:
    """Parse an LFM2.5-family tool call.

    Source: ``chat_template.jinja`` inside the cached
    ``models--LiquidAI--LFM2.5-1.2B-Instruct`` snapshot under
    ``~/.cache/huggingface/hub/`` (``render_tool_calls`` /
    ``format_arg_value`` macros). The assistant-turn rendering wraps a
    Python-call-style expression (**not** JSON) in a pair of dedicated
    special tokens::

        <|tool_call_start|>[func_name(arg1='value', arg2={"k": 1})]<|tool_call_end|>

    i.e. literal ``<|tool_call_start|>`` / ``<|tool_call_end|>`` tokens
    around a bracketed, comma-separated list of ``name(key=value, ...)``
    calls. String arguments are single-quoted (backslash-escaped); mapping/
    iterable arguments are rendered via Jinja's ``tojson`` (so JSON,
    double-quoted); everything else uses Python's ``str()``. This parser
    supports the single-call case (this repo's eval suites issue one tool
    call per turn) and decodes each ``key=value`` pair with
    :func:`_parse_lfm2_value`.
    """
    match = re.search(
        r"<\|tool_call_start\|>\s*\[(.*)\]\s*<\|tool_call_end\|>", prediction, re.DOTALL
    )
    if not match:
        raise ValueError("no <|tool_call_start|>...<|tool_call_end|> block found in prediction")
    body = match.group(1).strip()
    call_match = re.match(r"([A-Za-z_]\w*)\((.*)\)\s*$", body, re.DOTALL)
    if not call_match:
        raise ValueError(f"could not parse LFM2 tool-call syntax: {body!r}")
    name = call_match.group(1)
    args_text = call_match.group(2).strip()
    arguments: dict[str, Any] = {}
    if args_text:
        for part in _split_top_level(args_text):
            key, sep, value = part.partition("=")
            if not sep:
                raise ValueError(f"could not parse LFM2 tool-call argument: {part!r}")
            arguments[key.strip()] = _parse_lfm2_value(value)
    return {"name": name, "arguments": arguments}


#: Registry of per-family tool-call parsers. Each parser takes the raw
#: prediction string and returns ``{"name": str, "arguments": dict}``.
TOOL_CALL_PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "qwen3": _parse_qwen3_tool_call,
    "lfm2": _parse_lfm2_tool_call,
}

#: Substrings (checked against a lowercased model id) that map to a registered
#: tool-call family, in :func:`family_for_model`.
_FAMILY_SUBSTRINGS: dict[str, str] = {
    "qwen3": "qwen3",
    "lfm2": "lfm2",
}


def parse_tool_call(prediction: str, family: str) -> dict[str, Any]:
    """Parse *prediction* as a tool call using the parser registered for *family*.

    Returns ``{"name": str, "arguments": dict}``. Raises ``ValueError`` if
    *family* is not registered (listing the supported families) or if the
    family's parser cannot find/parse a tool-call block.
    """
    try:
        parser = TOOL_CALL_PARSERS[family]
    except KeyError:
        supported = ", ".join(sorted(TOOL_CALL_PARSERS))
        raise ValueError(
            f"unsupported tool-call family {family!r}; supported families: {supported}"
        ) from None
    return parser(prediction)


def family_for_model(model_id: str, override: str | None = None) -> str:
    """Return the registered tool-call family for *model_id*.

    *override* takes precedence when given (truthy) — this is how a run
    config's ``[eval] tool_call_family`` is threaded through without this
    module needing to know about :mod:`sloth.tune.config`. Otherwise the
    family is detected by lowercased substring match against
    :data:`TOOL_CALL_PARSERS`'s registered names (``qwen3``, ``lfm2``).
    Raises ``ValueError`` (listing supported families) if neither an override
    nor a substring match is found.
    """
    if override:
        return override
    lowered = (model_id or "").lower()
    for family, needle in _FAMILY_SUBSTRINGS.items():
        if needle in lowered:
            return family
    supported = ", ".join(sorted(_FAMILY_SUBSTRINGS))
    raise ValueError(
        f"could not determine tool-call family for model {model_id!r}; "
        f"set [eval] tool_call_family explicitly (supported families: {supported})"
    )


def score_tool_call(
    prediction: str, expected_tool_call: dict[str, Any], family: str
) -> dict[str, Any]:
    """Compare a parsed *prediction* tool call against *expected_tool_call*.

    *expected_tool_call* is ``{"name": str, "arguments": dict}``. Returns
    ``{"matched": bool, "name_ok": bool, "arguments_ok": bool}`` — ``name_ok``
    is a plain string comparison, ``arguments_ok`` is a JSON-equal (``==``)
    comparison of the parsed argument dicts, and ``matched`` is both. A
    prediction the family parser cannot parse at all scores all three
    ``False`` rather than raising.
    """
    try:
        parsed = parse_tool_call(prediction, family)
    except ValueError:
        return {"matched": False, "name_ok": False, "arguments_ok": False}
    name_ok = parsed.get("name") == expected_tool_call.get("name")
    arguments_ok = parsed.get("arguments") == expected_tool_call.get("arguments")
    return {"matched": name_ok and arguments_ok, "name_ok": name_ok, "arguments_ok": arguments_ok}


# ---------------------------------------------------------------------------
# 4. Lazy GLEU / BLEU (sacrebleu, container-only dependency)
# ---------------------------------------------------------------------------

_SACREBLEU_HINT = "install sacrebleu in the container dep layer"


def _require_sacrebleu() -> Any:
    try:
        import sacrebleu
    except ImportError as exc:
        raise CliError(code=2, message=str(exc), remediation=_SACREBLEU_HINT) from exc
    return sacrebleu


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def gleu(prediction: str, expected: str, *, min_n: int = 1, max_n: int = 4) -> float:
    """Sentence-level Google-GLEU (Wu et al., 2016) of *prediction* vs *expected*.

    Tokenizes both strings with ``sacrebleu``'s default (13a) tokenizer for
    consistency with :func:`bleu`, then computes GLEU directly: pool all
    n-grams (``min_n..max_n``) from each side into one multiset each, and
    score ``min(precision, recall)`` over the pooled overlap — sacrebleu
    itself has no GLEU metric, so the n-gram counting is done here. The result
    is scaled to ``0..100`` to match sacrebleu's BLEU score convention.

    Raises :class:`~sloth.cli._errors.CliError` (code 2) if ``sacrebleu`` is
    not installed — it ships only in the container dep layer, never as a base
    runtime dependency (see ``docs/dgx-spark.md``).
    """
    sacrebleu = _require_sacrebleu()
    tokenize = sacrebleu.BLEU().tokenizer
    pred_tokens = tokenize(prediction).split()
    ref_tokens = tokenize(expected).split()

    pred_ngrams: Counter[tuple[str, ...]] = Counter()
    ref_ngrams: Counter[tuple[str, ...]] = Counter()
    for n in range(min_n, max_n + 1):
        pred_ngrams.update(_ngrams(pred_tokens, n))
        ref_ngrams.update(_ngrams(ref_tokens, n))

    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((pred_ngrams & ref_ngrams).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    return min(precision, recall) * 100.0


def bleu(prediction: str, expected: str) -> float:
    """Sentence-level BLEU of *prediction* against *expected*, via ``sacrebleu``.

    Thin wrapper over ``sacrebleu.sentence_bleu(prediction, [expected]).score``.
    Raises :class:`~sloth.cli._errors.CliError` (code 2) if ``sacrebleu`` is
    not installed — it ships only in the container dep layer, never as a base
    runtime dependency (see ``docs/dgx-spark.md``).
    """
    sacrebleu = _require_sacrebleu()
    return sacrebleu.sentence_bleu(prediction, [expected]).score


# ---------------------------------------------------------------------------
# 5. Multiple-choice letter extraction (MMLU-style suites)
# ---------------------------------------------------------------------------

#: Matches an ``A``-``D`` (or ``a``-``d``) that stands alone — not glued to
#: another letter or digit. ``"Answer: B"``, ``"(B)"``, ``"B."``, ``"**B**"`` and
#: a bare ``"B"`` all match; the ``A`` inside ``"Answer"`` does not.
_CHOICE_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])")

#: Matches an explicit answer marker followed by the letter, e.g.
#: ``"Answer: B"``, ``"answer is (c)"``, ``"The correct option — D"``. Checked
#: before the bare-letter scan so a prose preamble containing a standalone
#: ``"a"`` (the English article) cannot win over a stated answer.
# The classes below are written upper-case only on purpose: ``re.IGNORECASE``
# already folds case, so spelling ``a-d`` as well would be a duplicate range.
_CHOICE_MARKER_RE = re.compile(
    r"\b(?:answer|option|choice)\b[^A-Z0-9]{0,8}([A-D])(?![A-Z0-9])",
    re.IGNORECASE,
)


def extract_choice_letter(prediction: str) -> str | None:
    """Return the multiple-choice letter (``"A"``-``"D"``) *prediction* picks, or ``None``.

    Letter-choice suites (an ``expected_output`` that is a single ``A``-``D``)
    ask the model to "answer with the letter", but models answer in prose:
    ``"B"``, ``"B."``, ``"(B)"``, ``"Answer: B"``, ``"The answer is b."`` all
    mean the same choice. This is the tolerant reader that turns any of them
    into ``"B"``, so :func:`sloth.tune._trainer._extra_metrics_for` can score a
    ``choice_match`` per row without changing exact-match semantics.

    Resolution order — the first rule that fires wins:

    1. an explicit marker (``answer``/``option``/``choice``) followed within a
       few punctuation/space characters by a standalone ``A``-``D``;
    2. the first standalone **upper-case** ``A``-``D``;
    3. the first standalone **lower-case** ``a``-``d``, upper-cased.

    Upper-case is preferred over lower-case in the unmarked case because a
    lower-case standalone ``"a"`` is far more often the English article than an
    answer. This is a heuristic, not a parser: a prose answer that never states
    a letter returns ``None`` (scored as a miss), and a sentence that opens with
    a standalone capital ``"A"`` before naming its real choice will be read as
    ``"A"``. Terse, letter-first answers — what the suite prompt asks for — are
    read correctly.

    Returns ``None`` for a non-string, an empty string, or text with no
    standalone ``A``-``D`` in it.
    """
    if not isinstance(prediction, str) or not prediction:
        return None

    marked = _CHOICE_MARKER_RE.search(prediction)
    if marked is not None:
        return marked.group(1).upper()

    lower_fallback: str | None = None
    for match in _CHOICE_LETTER_RE.finditer(prediction):
        letter = match.group(1)
        if letter.isupper():
            return letter
        if lower_fallback is None:
            lower_fallback = letter
    return lower_fallback.upper() if lower_fallback else None
