"""Dataset validation for unsloth-cli fine-tuning verbs (pure stdlib, no torch).

Supports five JSONL schemas:

* **chat** — each line is ``{"messages": [{"role": <str>, "content": <str>}, ...]}``.
  Valid roles: ``"system"``, ``"user"``, ``"assistant"``.
* **task** — each line is ``{"task": <str>, "input": <str>, "expected_output": <str>}``.
* **instruction** — the **task** keys, plus an optional ``"constraints"`` list:
  ``{"task": <str>, "input": <str>, "expected_output": <str>, "constraints": [...]}``.
  Each item of ``constraints`` is a single-key object naming one check:
  ``{"max_words": <int>}``, ``{"min_words": <int>}``, ``{"must_contain": <str>}``,
  ``{"must_not_contain": <str>}``, ``{"must_refuse": <bool>}``, or
  ``{"json_only": <bool>}``. Any other key on a constraint object is rejected by name.
* **structured** — ``{"task": <str>, "input": <str>, "json_schema": <object>}`` where
  ``json_schema`` is a JSON-Schema **subset**: only the keywords ``type``,
  ``required``, ``properties``, ``enum``, ``items``, and ``additionalProperties``
  are recognised (nested inside ``properties``/``items`` recursively); any other
  keyword anywhere in the schema is a validation error naming that keyword.
* **toolcall** — ``{"task": <str>, "input": <str>, "expected_tool_call":
  {"name": <str>, "arguments": <object>}}``.

Usage::

    from sloth.tune.datasets import validate_dataset, detect_schema

    records = validate_dataset("train.jsonl", schema="chat")
    # => list[dict] on success, CliError raised on the first invalid line

Two further helpers support cross-dataset hygiene, both **pure stdlib** and
neither one raises ``CliError`` itself — they hand back findings for a CLI
caller (see ``sloth train``/``sloth eval``, wired in a later task) to turn
into a ``CliError`` with a hint naming the offenders:

* :func:`split_holdout` — a seeded, deterministic train/holdout split. Chat
  rows are rendered down into scorable **task** rows: the prompt is every
  message but the final (assistant) turn, rendered as deterministic
  ``"role: content"`` lines joined by newlines (there is no tokenizer in this
  stdlib-only core, so this plain-text rendering is the documented, stable
  contract), and ``expected_output`` is the final message's content.
* :func:`overlap_check` — normalises **chat** and **task** rows (on both
  sides) to a ``(prompt, expected_output)`` pair and reports every ``path:line``
  location that shares a pair with another location in the checked set — this
  is how train/eval leakage across the chat and task schemas is caught.

Public API is intentionally small; validation error paths raise
:class:`CliError` so callers never have to inspect return codes.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = frozenset({"system", "user", "assistant"})
CHAT_KEYS = frozenset({"messages"})
TASK_KEYS = frozenset({"task", "input", "expected_output"})
INSTRUCTION_KEYS = TASK_KEYS | {"constraints"}
STRUCTURED_KEYS = frozenset({"task", "input", "json_schema"})
TOOLCALL_KEYS = frozenset({"task", "input", "expected_tool_call"})
TOOLCALL_INNER_KEYS = frozenset({"name", "arguments"})

CONSTRAINT_VALUE_TYPES: dict[str, type] = {
    "max_words": int,
    "min_words": int,
    "must_contain": str,
    "must_not_contain": str,
    "must_refuse": bool,
    "json_only": bool,
}
ALLOWED_CONSTRAINT_KEYS = frozenset(CONSTRAINT_VALUE_TYPES)

JSON_SCHEMA_SUBSET_KEYWORDS = frozenset(
    {"type", "required", "properties", "enum", "items", "additionalProperties"}
)

KNOWN_SCHEMAS = frozenset({"chat", "task", "instruction", "structured", "toolcall"})


# ---------------------------------------------------------------------------
# Per-schema validators
# ---------------------------------------------------------------------------


def _validate_chat_message(msg: object, idx: int, line_no: int) -> None:
    """Raise CliError if a single chat *msg* (at position *idx*) is malformed.

    Split out of :func:`_validate_chat_record` so the per-message checks live at a
    single nesting level — keeping each function's branching simple.
    """
    if not isinstance(msg, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"line {line_no}: messages[{idx}] must be a JSON object, "
                f"got {type(msg).__name__}"
            ),
            remediation='Each message must be {"role": "user|assistant|system", "content": "..."}',
        )

    if "role" not in msg:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: messages[{idx}] missing required key "role"',
            remediation='Each message must include "role": one of "system", "user", "assistant".',
        )

    if "content" not in msg:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: messages[{idx}] missing required key "content"',
            remediation='Each message must include "content": a string.',
        )

    role = msg["role"]
    if not isinstance(role, str):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f'line {line_no}: messages[{idx}]["role"] must be a string, '
                f"got {type(role).__name__}"
            ),
            remediation=f'"role" must be one of: {sorted(VALID_ROLES)}.',
        )

    if role not in VALID_ROLES:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: messages[{idx}]["role"] {role!r} is not a valid role',
            remediation=f'"role" must be one of: {sorted(VALID_ROLES)}.',
        )

    content = msg["content"]
    if not isinstance(content, str):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f'line {line_no}: messages[{idx}]["content"] must be a string, '
                f"got {type(content).__name__}"
            ),
            remediation='"content" must be a plain string.',
        )


def _validate_chat_record(record: object, line_no: int) -> None:
    """Raise CliError if *record* does not conform to the chat schema."""
    if not isinstance(record, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: expected a JSON object, got {type(record).__name__}",
            remediation='Each line must be a JSON object: {"messages": [...]}',
        )

    extra = set(record.keys()) - CHAT_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: unexpected keys {sorted(extra)!r} in chat record",
            remediation='Chat records must have exactly one key: "messages".',
        )

    if "messages" not in record:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: missing required key "messages"',
            remediation='Add a "messages" list: {"messages": [{"role": "user", "content": "..."}]}',
        )

    messages = record["messages"]
    if not isinstance(messages, list):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: "messages" must be a list, got {type(messages).__name__}',
            remediation='"messages" must be a JSON array of message objects.',
        )

    if len(messages) == 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: "messages" list is empty',
            remediation='Provide at least one message object with "role" and "content".',
        )

    for idx, msg in enumerate(messages):
        _validate_chat_message(msg, idx, line_no)


def _validate_task_record(record: object, line_no: int) -> None:
    """Raise CliError if *record* does not conform to the task schema."""
    if not isinstance(record, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: expected a JSON object, got {type(record).__name__}",
            remediation=(
                "Each line must be a JSON object: "
                '{"task": ..., "input": ..., "expected_output": ...}'
            ),
        )

    extra = set(record.keys()) - TASK_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: unexpected keys {sorted(extra)!r} in task record",
            remediation=(
                'Task records must have exactly these keys: "task", "input", "expected_output".'
            ),
        )

    for key in ("task", "input", "expected_output"):
        if key not in record:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'line {line_no}: missing required key "{key}"',
                remediation=(
                    'Task records must include "task", "input", '
                    'and "expected_output" — all strings.'
                ),
            )
        value = record[key]
        if not isinstance(value, str):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(f'line {line_no}: "{key}" must be a string, got {type(value).__name__}'),
                remediation=f'"{key}" must be a plain string.',
            )


def _validate_constraint(constraint: object, idx: int, line_no: int) -> None:
    """Raise CliError if a single ``constraints[idx]`` object is malformed."""
    if not isinstance(constraint, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"line {line_no}: constraints[{idx}] must be a JSON object, "
                f"got {type(constraint).__name__}"
            ),
            remediation=(
                f"Each constraint must be a single-key object, one of: "
                f"{sorted(ALLOWED_CONSTRAINT_KEYS)}."
            ),
        )

    extra = set(constraint.keys()) - ALLOWED_CONSTRAINT_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: constraints[{idx}] has unknown key(s) {sorted(extra)!r}",
            remediation=f"Constraint keys must be one of: {sorted(ALLOWED_CONSTRAINT_KEYS)}.",
        )

    if len(constraint) != 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"line {line_no}: constraints[{idx}] must have exactly one key, "
                f"got {sorted(constraint.keys())!r}"
            ),
            remediation=f"Constraint keys must be one of: {sorted(ALLOWED_CONSTRAINT_KEYS)}.",
        )

    ((key, value),) = constraint.items()
    expected_type = CONSTRAINT_VALUE_TYPES[key]
    # bool is an int subclass in Python; guard int-typed keys against bool values
    # and vice versa so True/False never silently satisfies an int constraint.
    type_ok = isinstance(value, expected_type) and not (
        expected_type is int and isinstance(value, bool)
    )
    if not type_ok:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f'line {line_no}: constraints[{idx}]["{key}"] must be a '
                f"{expected_type.__name__}, got {type(value).__name__}"
            ),
            remediation=f'"{key}" must be a {expected_type.__name__}.',
        )


def _validate_instruction_record(record: object, line_no: int) -> None:
    """Raise CliError if *record* does not conform to the instruction schema."""
    if not isinstance(record, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: expected a JSON object, got {type(record).__name__}",
            remediation=(
                "Each line must be a JSON object: "
                '{"task": ..., "input": ..., "expected_output": ..., "constraints": [...]}'
            ),
        )

    extra = set(record.keys()) - INSTRUCTION_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: unexpected keys {sorted(extra)!r} in instruction record",
            remediation=(
                'Instruction records may only have keys "task", "input", '
                '"expected_output", and "constraints".'
            ),
        )

    for key in ("task", "input", "expected_output"):
        if key not in record:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'line {line_no}: missing required key "{key}"',
                remediation=(
                    'Instruction records must include "task", "input", '
                    'and "expected_output" — all strings.'
                ),
            )
        value = record[key]
        if not isinstance(value, str):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(f'line {line_no}: "{key}" must be a string, got {type(value).__name__}'),
                remediation=f'"{key}" must be a plain string.',
            )

    if "constraints" in record:
        constraints = record["constraints"]
        if not isinstance(constraints, list):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f'line {line_no}: "constraints" must be a list, '
                    f"got {type(constraints).__name__}"
                ),
                remediation='"constraints" must be a JSON array of constraint objects.',
            )
        for idx, constraint in enumerate(constraints):
            _validate_constraint(constraint, idx, line_no)


def _as_schema_dict(schema: object, line_no: int, path: str) -> dict:
    """Return *schema* as a dict, rejecting a non-object or an unsupported keyword."""
    if not isinstance(schema, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: {path} must be a JSON object, got {type(schema).__name__}",
            remediation=f"{path} must be a JSON-Schema-subset object.",
        )

    extra = set(schema.keys()) - JSON_SCHEMA_SUBSET_KEYWORDS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(f"line {line_no}: {path} uses unsupported keyword(s) {sorted(extra)!r}"),
            remediation=(
                f"Only these JSON-Schema keywords are supported: "
                f"{sorted(JSON_SCHEMA_SUBSET_KEYWORDS)}."
            ),
        )
    return schema


def _check_schema_type(schema: dict, line_no: int, path: str) -> None:
    """``"type"`` must name a JSON-Schema type, i.e. be a string."""
    if "type" in schema and not isinstance(schema["type"], str):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: {path}["type"] must be a string',
            remediation='"type" must be a JSON-Schema type name, e.g. "object" or "string".',
        )


def _check_schema_required(schema: dict, line_no: int, path: str) -> None:
    """``"required"`` must be a list of property-name strings."""
    if "required" not in schema:
        return
    required = schema["required"]
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: {path}["required"] must be a list of strings',
            remediation='"required" must be a JSON array of property-name strings.',
        )


def _check_schema_additional_properties(schema: dict, line_no: int, path: str) -> None:
    """``"additionalProperties"`` must be a boolean in this subset."""
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: {path}["additionalProperties"] must be a boolean',
            remediation='"additionalProperties" must be true or false.',
        )


def _check_schema_enum(schema: dict, line_no: int, path: str) -> None:
    """``"enum"`` must be a list of allowed values."""
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: {path}["enum"] must be a list',
            remediation='"enum" must be a JSON array of allowed values.',
        )


def _check_schema_properties(schema: dict, line_no: int, path: str) -> None:
    """``"properties"`` must map names to schemas; each nested schema recurses."""
    if "properties" not in schema:
        return
    properties = schema["properties"]
    if not isinstance(properties, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: {path}["properties"] must be a JSON object',
            remediation='"properties" must map property names to nested schemas.',
        )
    for prop_name, prop_schema in properties.items():
        _validate_json_schema_subset(prop_schema, line_no, f'{path}["properties"]["{prop_name}"]')


#: Per-keyword checkers run by :func:`_validate_json_schema_subset`, in report order.
_SCHEMA_KEYWORD_CHECKS: tuple[Callable[[dict, int, str], None], ...] = (
    _check_schema_type,
    _check_schema_required,
    _check_schema_additional_properties,
    _check_schema_enum,
    _check_schema_properties,
)


def _validate_json_schema_subset(schema: object, line_no: int, path: str) -> None:
    """Raise CliError if *schema* uses a keyword outside the documented subset.

    Recognised keywords: ``type``, ``required``, ``properties``, ``enum``,
    ``items``, ``additionalProperties``. ``properties`` values and ``items``
    are validated recursively as nested schemas.
    """
    schema_dict = _as_schema_dict(schema, line_no, path)

    for check in _SCHEMA_KEYWORD_CHECKS:
        check(schema_dict, line_no, path)

    if "items" in schema_dict:
        _validate_json_schema_subset(schema_dict["items"], line_no, f'{path}["items"]')


def _validate_structured_record(record: object, line_no: int) -> None:
    """Raise CliError if *record* does not conform to the structured schema."""
    if not isinstance(record, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: expected a JSON object, got {type(record).__name__}",
            remediation=(
                "Each line must be a JSON object: "
                '{"task": ..., "input": ..., "json_schema": {...}}'
            ),
        )

    extra = set(record.keys()) - STRUCTURED_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: unexpected keys {sorted(extra)!r} in structured record",
            remediation='Structured records must have exactly keys "task", "input", "json_schema".',
        )

    for key in ("task", "input"):
        if key not in record:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'line {line_no}: missing required key "{key}"',
                remediation=f'Structured records must include "{key}" (a string).',
            )
        if not isinstance(record[key], str):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f'line {line_no}: "{key}" must be a string, '
                    f"got {type(record[key]).__name__}"
                ),
                remediation=f'"{key}" must be a plain string.',
            )

    if "json_schema" not in record:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: missing required key "json_schema"',
            remediation=(
                'Structured records must include "json_schema": a JSON-Schema-subset object.'
            ),
        )

    _validate_json_schema_subset(record["json_schema"], line_no, "json_schema")


def _validate_toolcall_record(record: object, line_no: int) -> None:
    """Raise CliError if *record* does not conform to the toolcall schema."""
    if not isinstance(record, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: expected a JSON object, got {type(record).__name__}",
            remediation=(
                "Each line must be a JSON object: "
                '{"task": ..., "input": ..., "expected_tool_call": {"name": ..., "arguments": ...}}'
            ),
        )

    extra = set(record.keys()) - TOOLCALL_KEYS
    if extra:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"line {line_no}: unexpected keys {sorted(extra)!r} in toolcall record",
            remediation=(
                'Toolcall records must have exactly keys "task", "input", "expected_tool_call".'
            ),
        )

    for key in ("task", "input"):
        if key not in record:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'line {line_no}: missing required key "{key}"',
                remediation=f'Toolcall records must include "{key}" (a string).',
            )
        if not isinstance(record[key], str):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f'line {line_no}: "{key}" must be a string, '
                    f"got {type(record[key]).__name__}"
                ),
                remediation=f'"{key}" must be a plain string.',
            )

    if "expected_tool_call" not in record:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: missing required key "expected_tool_call"',
            remediation=(
                'Toolcall records must include "expected_tool_call": '
                '{"name": <str>, "arguments": <object>}.'
            ),
        )

    call = record["expected_tool_call"]
    if not isinstance(call, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f'line {line_no}: "expected_tool_call" must be a JSON object, '
                f"got {type(call).__name__}"
            ),
            remediation='"expected_tool_call" must be {"name": <str>, "arguments": <object>}.',
        )

    extra_inner = set(call.keys()) - TOOLCALL_INNER_KEYS
    if extra_inner:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"line {line_no}: unexpected keys {sorted(extra_inner)!r} in expected_tool_call"
            ),
            remediation='"expected_tool_call" must have exactly keys "name", "arguments".',
        )

    if "name" not in call or not isinstance(call.get("name"), str):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: "expected_tool_call.name" must be a string',
            remediation='"expected_tool_call.name" must be the tool name as a string.',
        )

    if "arguments" not in call or not isinstance(call.get("arguments"), dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f'line {line_no}: "expected_tool_call.arguments" must be a JSON object',
            remediation='"expected_tool_call.arguments" must be a JSON object of tool arguments.',
        )


_SCHEMA_VALIDATORS = {
    "chat": _validate_chat_record,
    "task": _validate_task_record,
    "instruction": _validate_instruction_record,
    "structured": _validate_structured_record,
    "toolcall": _validate_toolcall_record,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_schema(record: dict) -> str | None:
    """Guess the schema of a single parsed record.

    Returns one of :data:`KNOWN_SCHEMAS` — ``"chat"``, ``"toolcall"``,
    ``"structured"``, ``"instruction"``, ``"task"`` — or ``None`` when the
    record matches none of them. Detection is by discriminating key, checked
    from the most specific shape to the least: ``messages`` → chat,
    ``expected_tool_call`` → toolcall, ``json_schema`` → structured,
    ``constraints`` → instruction, otherwise the task key set. A task-shaped
    record with no ``constraints`` key is reported as ``"task"`` (the two
    validators accept it identically in that case). This is the single
    detector every host-side and in-container path uses, so a suite file is
    classified the same way by ``sloth validate``, ``sloth eval`` and the
    trainer's scoring.
    """
    if not isinstance(record, dict):
        return None
    keys = set(record.keys())
    if "messages" in keys:
        return "chat"
    if "expected_tool_call" in keys:
        return "toolcall"
    if "json_schema" in keys:
        return "structured"
    if "constraints" in keys:
        return "instruction"
    if keys == TASK_KEYS or (keys <= TASK_KEYS and len(keys) > 0 and "task" in keys):
        return "task"
    return None


def detect_file_schema(path: str | os.PathLike) -> str | None:
    """Detect a JSONL file's schema from its first non-blank, parseable record.

    Returns ``None`` when the file is empty, its first record is not JSON, or
    the record matches no known schema; callers decide whether that is an
    error. Never raises on a missing file — :func:`validate_dataset` reports
    that with the proper ``CliError``.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    return detect_schema(json.loads(line))
                except json.JSONDecodeError:
                    return None
    except OSError:
        return None
    return None


def validate_dataset(
    path: str | os.PathLike,
    schema: str,
) -> list[dict]:
    """Validate a JSONL file against *schema* and return the parsed records.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file (``str`` or ``os.PathLike``).
    schema:
        ``"chat"`` or ``"task"``.

    Returns
    -------
    list[dict]
        The parsed records in order.

    Raises
    ------
    CliError(code=1, ...)
        On the first line that fails schema validation or contains invalid JSON.
    CliError(code=2, ...)
        If the file cannot be opened.
    CliError(code=1, ...)
        If *schema* is not a known schema name.
    CliError(code=1, ...)
        If the file holds no records (empty or blank-only) — caught here so
        ``train`` fails fast instead of after the trainer has loaded the model.
    """
    if schema not in KNOWN_SCHEMAS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unknown schema {schema!r}; must be one of {sorted(KNOWN_SCHEMAS)}",
            remediation='Pass schema="chat" or schema="task".',
        )

    file_path = Path(path)
    try:
        fh = file_path.open(encoding="utf-8")
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"cannot open dataset file {file_path}: {exc.strerror}",
            remediation="Check that the file exists and is readable.",
        ) from exc

    validator = _SCHEMA_VALIDATORS[schema]

    records: list[dict] = []
    with fh:
        for raw_line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines; they don't advance the logical line count

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=f"line {raw_line_no}: invalid JSON — {exc.msg}",
                    remediation=(
                        "Each non-blank line must be valid JSON. " f"Schema expected: {schema!r}."
                    ),
                ) from exc

            validator(record, raw_line_no)
            records.append(record)

    if not records:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"dataset {file_path} contains no records (empty or blank-only)",
            remediation=(
                "Add at least one JSON object record per line, e.g. "
                '{"messages": [{"role": "user", "content": "..."}]} (chat) or '
                '{"task": "...", "input": "...", "expected_output": "..."} (task).'
            ),
        )

    return records


# ---------------------------------------------------------------------------
# Suite validation (file-or-directory) — shared by ``sloth eval`` and
# ``sloth validate --suite``
# ---------------------------------------------------------------------------


def resolve_suite_paths(path: str | os.PathLike) -> list[Path]:
    """Expand *path* into a sorted list of ``.jsonl`` file paths.

    *path* may be a single file (returned as a one-element list) or a
    directory, whose ``*.jsonl`` children are returned sorted (for a
    deterministic, reproducible file order).

    Raises
    ------
    CliError(code=1)
        When *path* does not exist, or is a directory holding no ``.jsonl``
        files.
    """
    suite_path = Path(path)
    if suite_path.is_dir():
        files = sorted(suite_path.glob("*.jsonl"))
        if not files:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"no *.jsonl files found in suite directory: {suite_path}",
                remediation=(
                    "Add at least one .jsonl file to the directory, or pass "
                    "--suite <file.jsonl> directly."
                ),
            )
        return files
    if suite_path.is_file():
        return [suite_path]
    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"suite path not found: {suite_path}",
        remediation=(
            "Pass an existing .jsonl file, or a directory containing .jsonl "
            "files, with --suite <path>."
        ),
    )


#: ``validate_suite(schema=AUTO_SCHEMA)`` detects each file's schema separately.
AUTO_SCHEMA = "auto"


def validate_suite(path: str | os.PathLike, schema: str = "task") -> dict[str, object]:
    """Validate every ``.jsonl`` file under *path* (a single file or a directory).

    Reuses :func:`validate_dataset` per resolved file — the exact same rules
    ``sloth train``/``sloth eval`` apply — so a failure names both the
    offending file and the line within it. Used by both ``sloth eval``
    (pre-launch host validation) and ``sloth validate --suite``.

    Parameters
    ----------
    path:
        A single ``.jsonl`` file, or a directory of them.
    schema:
        One of :data:`KNOWN_SCHEMAS` applied to every file (default ``"task"``,
        the historical eval-suite schema), or :data:`AUTO_SCHEMA` (``"auto"``)
        to detect each file's schema from its first record with
        :func:`detect_file_schema` — the mode ``sloth validate --suite`` and
        ``sloth eval`` use, so a directory can mix task, chat, instruction,
        structured and toolcall suites.

    Returns
    -------
    dict
        ``{"files": [{"path": str, "line_count": int, "schema": str}, ...],
        "total_records": int}``, one entry per resolved file in sorted order.

    Raises
    ------
    CliError(code=1)
        On the first file/line that fails validation, or when *path* resolves
        to no files at all.
    """
    files = resolve_suite_paths(path)
    reported: list[dict[str, object]] = []
    total = 0
    for file_path in files:
        file_schema = schema
        if schema == AUTO_SCHEMA:
            # Undetectable (empty, non-JSON, or unknown keys) falls back to the
            # historical task schema so the error below still names the line.
            file_schema = detect_file_schema(file_path) or "task"
        try:
            records = validate_dataset(file_path, file_schema)
        except CliError as exc:
            # Re-raise with the file name prepended so a directory suite's
            # failure names both the offending file and the line within it.
            raise CliError(
                code=exc.code,
                message=f"{file_path}: {exc.message}",
                remediation=exc.remediation,
            ) from exc
        reported.append({"path": str(file_path), "line_count": len(records), "schema": file_schema})
        total += len(records)
    return {"files": reported, "total_records": total}


# ---------------------------------------------------------------------------
# Seeded holdout split
# ---------------------------------------------------------------------------


def render_chat_prompt(messages: list[dict]) -> str:
    """Render chat *messages* as deterministic ``"role: content"`` lines.

    There is no tokenizer in this stdlib-only core, so this plain-text
    rendering (one line per message, joined by ``"\\n"``, in message order)
    is the documented, stable contract used to turn a chat row into a
    scorable prompt string — used by both :func:`split_holdout` and
    :func:`overlap_check`.
    """
    return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)


def _chat_record_to_task_row(record: dict) -> dict:
    """Convert a valid chat *record* into a scorable task-schema row.

    ``prompt`` (the row's ``"input"``) is every message but the final one,
    rendered via :func:`render_chat_prompt`; ``"expected_output"`` is the
    final message's ``"content"``.
    """
    messages = record["messages"]
    prompt = render_chat_prompt(messages[:-1])
    expected_output = messages[-1]["content"]
    return {"task": "chat-holdout", "input": prompt, "expected_output": expected_output}


def split_holdout(
    path: str | os.PathLike,
    fraction: float,
    seed: int,
) -> dict[str, object]:
    """Split the JSONL dataset at *path* into a seeded train/holdout pair.

    The input file's schema is auto-detected (``"chat"`` or ``"task"``, via
    :func:`detect_schema` on its first record) and every record is validated
    with :func:`validate_dataset` before the split runs. Chat rows are
    rendered into scorable task rows (see :func:`render_chat_prompt`) in
    *both* output files, so a holdout produced from a chat dataset can always
    be scored without a chat-aware harness.

    Parameters
    ----------
    path:
        Path to the source ``.jsonl`` file.
    fraction:
        The holdout's share of records, strictly between 0 and 1.
    seed:
        Seed for the deterministic shuffle. The same *path*, *fraction*, and
        *seed* always produce byte-identical output files — the split order
        comes from ``random.Random(seed).shuffle`` over the record indices,
        not from any process-global RNG state.

    Returns
    -------
    dict
        ``{"train_path": Path, "holdout_path": Path, "train_count": int,
        "holdout_count": int, "schema": str}``.

    Raises
    ------
    CliError(code=1)
        If *fraction* is not strictly between 0 and 1, or the source schema
        cannot be detected.
    CliError(code=1|2)
        Propagated from :func:`validate_dataset` for a malformed or missing
        source file.
    """
    if not 0 < fraction < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"fraction must be strictly between 0 and 1, got {fraction!r}",
            remediation="Pass a fraction like 0.1 for a 10% holdout.",
        )

    file_path = Path(path)

    probe_schema: str | None = None
    try:
        with file_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    probe_schema = detect_schema(json.loads(line))
                except json.JSONDecodeError:
                    probe_schema = None
                break
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"cannot open dataset file {file_path}: {exc.strerror}",
            remediation="Check that the file exists and is readable.",
        ) from exc

    if probe_schema not in ("chat", "task"):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"could not detect a chat or task schema in {file_path} to split",
            remediation=(
                "split_holdout only supports chat and task datasets; "
                "validate the file with `sloth validate` first."
            ),
        )

    records = validate_dataset(file_path, probe_schema)

    indices = list(range(len(records)))
    # Deterministic dataset shuffling, not a security/cryptographic use of
    # randomness — reproducibility (same seed -> same split) is the goal.
    random.Random(seed).shuffle(indices)  # nosec B311  # NOSONAR
    holdout_count = round(len(records) * fraction)
    holdout_count = max(0, min(holdout_count, len(records)))
    holdout_index_set = set(indices[:holdout_count])

    if probe_schema == "chat":
        converted = [_chat_record_to_task_row(r) for r in records]
    else:
        converted = records

    train_records = [converted[i] for i in range(len(records)) if i not in holdout_index_set]
    holdout_records = [converted[i] for i in range(len(records)) if i in holdout_index_set]

    train_path = file_path.with_name(f"{file_path.stem}.train.jsonl")
    holdout_path = file_path.with_name(f"{file_path.stem}.holdout.jsonl")

    for out_path, out_records in ((train_path, train_records), (holdout_path, holdout_records)):
        with out_path.open("w", encoding="utf-8") as fh:
            for record in out_records:
                fh.write(json.dumps(record))
                fh.write("\n")

    return {
        "train_path": train_path,
        "holdout_path": holdout_path,
        "train_count": len(train_records),
        "holdout_count": len(holdout_records),
        "schema": probe_schema,
    }


# ---------------------------------------------------------------------------
# Cross-schema train/eval overlap check
# ---------------------------------------------------------------------------


def _normalize_row_for_overlap(record: object) -> tuple[str, str] | None:
    """Normalise a chat or task *record* to a ``(prompt, expected_output)`` pair.

    Returns ``None`` for anything that isn't a recognisable chat or task-shaped
    row (e.g. instruction/structured/toolcall rows, or malformed JSON) — those
    are simply not compared for overlap.
    """
    if not isinstance(record, dict):
        return None

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if not isinstance(last, dict) or "content" not in last:
            return None
        prompt = render_chat_prompt(messages[:-1])
        return (prompt, last["content"])

    if "input" in record and "expected_output" in record:
        return (record["input"], record["expected_output"])

    return None


def _collect_overlap_locations(
    scan_path: Path, locations: dict[tuple[str, str], list[str]]
) -> None:
    """Record every normalisable row of *scan_path* under its ``(prompt, expected)`` key.

    An unreadable file, a blank line, a line that is not JSON, and a row that
    :func:`_normalize_row_for_overlap` does not recognise are all skipped
    silently — overlap reporting never raises.
    """
    try:
        fh = scan_path.open(encoding="utf-8")
    except OSError:
        return
    with fh:
        for line_no, raw_line in enumerate(fh, start=1):
            record = _decode_overlap_line(raw_line)
            key = None if record is None else _normalize_row_for_overlap(record)
            if key is not None:
                locations[key].append(f"{scan_path}:{line_no}")


def _decode_overlap_line(raw_line: str) -> object | None:
    """Return the JSON value on *raw_line*, or ``None`` when blank or malformed."""
    line = raw_line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def overlap_check(
    train_path: str | os.PathLike,
    suite_paths: list[str | os.PathLike],
) -> list[str]:
    """Find duplicate ``(prompt, expected_output)`` pairs across *train_path* and *suite_paths*.

    Chat and task rows are normalised to a common ``(prompt, expected_output)``
    pair via :func:`_normalize_row_for_overlap`, so a chat training row and a
    task eval row that render to the same prompt/answer are correctly caught
    as the same duplicate. Rows that don't normalise (instruction, structured,
    toolcall rows, or malformed JSON lines) are silently skipped — they are
    not compared.

    This function only reports; it never raises :class:`CliError`. A CLI
    caller (e.g. ``sloth train``) is expected to raise ``CliError(code=1)``
    with a hint naming the offending locations when the returned list is
    non-empty.

    Parameters
    ----------
    train_path:
        Path to the training ``.jsonl`` file.
    suite_paths:
        Paths to one or more eval-suite ``.jsonl`` files to check against
        *train_path* (and against each other).

    Returns
    -------
    list[str]
        A sorted list of ``"path:line"`` strings — every location (in
        *train_path* or any of *suite_paths*) whose normalised pair also
        appears at another location in the checked set. Empty when there is
        no overlap.
    """
    locations: dict[tuple[str, str], list[str]] = defaultdict(list)

    _collect_overlap_locations(Path(train_path), locations)
    for suite_path in suite_paths:
        _collect_overlap_locations(Path(suite_path), locations)

    duplicates: list[str] = []
    for locs in locations.values():
        if len(locs) > 1:
            duplicates.extend(locs)
    return sorted(duplicates)
