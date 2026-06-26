"""Integration tests: the fine-tuning verbs are wired into the parser + catalog.

These guard task t13 — that ``train``/``eval``/``export`` are registered on the
root parser (so ``--help`` works and dispatch finds them) and that each resolves
through the global ``explain`` catalog.
"""

from __future__ import annotations

import json

import pytest

from sloth.cli import _build_parser, main
from sloth.explain import known_paths

_TUNE_VERBS = ("train", "eval", "export")


@pytest.mark.parametrize("verb", _TUNE_VERBS)
def test_verb_registered_on_parser(verb: str) -> None:
    """Each fine-tuning verb is a registered subcommand of the root parser."""
    parser = _build_parser()
    subactions = [
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    ]
    registered = set()
    for action in subactions:
        registered.update(action.choices)
    assert verb in registered, f"{verb} not registered on the parser"


@pytest.mark.parametrize("verb", _TUNE_VERBS)
def test_verb_help_exits_zero(verb: str, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth <verb> --help`` prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main([verb, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert verb in out


@pytest.mark.parametrize("verb", _TUNE_VERBS)
def test_explain_resolves_for_verb(verb: str, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth explain <verb>`` resolves to a catalog entry."""
    rc = main(["explain", verb])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"# unsloth-cli {verb}" in out


@pytest.mark.parametrize("verb", _TUNE_VERBS)
def test_explain_json_resolves_for_verb(verb: str, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth explain <verb> --json`` returns the path + markdown payload."""
    rc = main(["explain", verb, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == [verb]
    assert f"unsloth-cli {verb}" in payload["markdown"]


def test_tune_verbs_in_known_paths() -> None:
    """The catalog exposes a path tuple for each fine-tuning verb."""
    paths = set(known_paths())
    for verb in _TUNE_VERBS:
        assert (verb,) in paths
