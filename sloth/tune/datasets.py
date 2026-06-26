"""Dataset validation for unsloth-cli fine-tuning verbs (pure stdlib, no torch).

Supports two JSONL schemas:

* **chat** — each line is ``{"messages": [{"role": <str>, "content": <str>}, ...]}``.
  Valid roles: ``"system"``, ``"user"``, ``"assistant"``.
* **task** — each line is ``{"task": <str>, "input": <str>, "expected_output": <str>}``.

Usage::

    from sloth.tune.datasets import validate_dataset, detect_schema

    records = validate_dataset("train.jsonl", schema="chat")
    # => list[dict] on success, CliError raised on the first invalid line

Public API is intentionally small; all error paths raise :class:`CliError`
so callers never have to inspect return codes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = frozenset({"system", "user", "assistant"})
CHAT_KEYS = frozenset({"messages"})
TASK_KEYS = frozenset({"task", "input", "expected_output"})
KNOWN_SCHEMAS = frozenset({"chat", "task"})


# ---------------------------------------------------------------------------
# Per-schema validators
# ---------------------------------------------------------------------------


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
        if not isinstance(msg, dict):
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f"line {line_no}: messages[{idx}] must be a JSON object, "
                    f"got {type(msg).__name__}"
                ),
                remediation=(
                    'Each message must be {"role": "user|assistant|system", "content": "..."}'
                ),
            )

        if "role" not in msg:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'line {line_no}: messages[{idx}] missing required key "role"',
                remediation=(
                    'Each message must include "role": one of "system", "user", "assistant".'
                ),
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_schema(record: dict) -> str | None:
    """Guess the schema of a single parsed record; returns ``"chat"``, ``"task"``, or ``None``."""
    if not isinstance(record, dict):
        return None
    keys = set(record.keys())
    if "messages" in keys:
        return "chat"
    if keys == TASK_KEYS or (keys <= TASK_KEYS and len(keys) > 0 and "task" in keys):
        return "task"
    return None


def validate_dataset(
    path: Union[str, os.PathLike],
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

    validator = _validate_chat_record if schema == "chat" else _validate_task_record

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

    return records
