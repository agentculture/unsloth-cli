"""Test that torch and unsloth are not imported at top level.

This guards against heavy ML dependencies leaking into the CLI's import
path. The introspection verbs (whoami, learn, explain, overview, doctor)
must work on machines without torch or unsloth installed.
"""

import subprocess
import sys


def test_sloth_import_does_not_load_torch():
    """Importing sloth must not bring torch into sys.modules."""
    code = (
        "import sloth; import sys; "
        "assert 'torch' not in sys.modules, "
        "'torch was imported at top level'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/home/spark/git/.wt-unsloth/agent-t5",
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_sloth_import_does_not_load_unsloth():
    """Importing sloth must not bring unsloth into sys.modules."""
    code = (
        "import sloth; import sys; "
        "assert 'unsloth' not in sys.modules, "
        "'unsloth was imported at top level'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/home/spark/git/.wt-unsloth/agent-t5",
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_whoami_verb_does_not_load_torch():
    """Running whoami verb must not bring torch into sys.modules."""
    code = (
        "from sloth.cli import main; "
        "import sys; "
        "main(['whoami']); "
        "assert 'torch' not in sys.modules, "
        "'torch was imported during whoami'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/home/spark/git/.wt-unsloth/agent-t5",
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_whoami_verb_does_not_load_unsloth():
    """Running whoami verb must not bring unsloth into sys.modules."""
    code = (
        "from sloth.cli import main; "
        "import sys; "
        "main(['whoami']); "
        "assert 'unsloth' not in sys.modules, "
        "'unsloth was imported during whoami'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/home/spark/git/.wt-unsloth/agent-t5",
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
