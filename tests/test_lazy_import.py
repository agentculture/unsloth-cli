"""Test that torch and unsloth are not imported at top level.

This guards against heavy ML dependencies leaking into the CLI's import
path. The introspection verbs (whoami, learn, explain, overview, doctor)
must work on machines without torch or unsloth installed.
"""

import subprocess
import sys
from pathlib import Path

# Repo root — two levels up from this test file (tests/ -> repo root)
_REPO_ROOT = str(Path(__file__).parent.parent)


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
        cwd=_REPO_ROOT,
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
        cwd=_REPO_ROOT,
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
        cwd=_REPO_ROOT,
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
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_exporter_import_does_not_load_torch():
    """Importing sloth.tune._exporter must not bring torch into sys.modules.

    The export seam (merged/gguf via Unsloth, awq/nvfp4 via llm-compressor) is
    allowed to touch the heavy stack, but only inside ``_load_backend``.
    """
    code = (
        "import sloth.tune._exporter; import sys; "
        "assert 'torch' not in sys.modules, "
        "'torch was imported at top level of _exporter'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_exporter_import_does_not_load_unsloth_or_llmcompressor():
    """Importing sloth.tune._exporter must not bring unsloth/llmcompressor in."""
    code = (
        "import sloth.tune._exporter; import sys; "
        "assert 'unsloth' not in sys.modules, "
        "'unsloth was imported at top level of _exporter'; "
        "assert 'llmcompressor' not in sys.modules, "
        "'llmcompressor was imported at top level of _exporter'; "
        "assert 'compressed_tensors' not in sys.modules, "
        "'compressed_tensors was imported at top level of _exporter'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_metrics_import_loads_no_ml_stack():
    """sloth.tune.metrics is pure stdlib — importing it pulls in no ML module.

    The eval scoring core (exact match, token F1, eval.json) must stay usable on
    a machine with no ML stack at all, exactly like datasets/config/scope: it is
    what lets a suite be scored and written without torch ever being imported.
    """
    code = (
        "import sloth.tune.metrics; import sys; "
        "heavy = [m for m in ('torch', 'unsloth', 'transformers', 'peft') "
        "if m in sys.modules]; "
        "assert not heavy, f'metrics pulled in {heavy}'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_scorers_import_loads_no_ml_stack():
    """sloth.tune.scorers is pure-stdlib-at-import — no ML module leaks in.

    Constraint/JSON-schema/tool-call scoring must stay usable with no ML stack
    installed, exactly like metrics/datasets/config/scope. The only
    third-party import (``sacrebleu``) is deferred inside gleu()/bleu().
    """
    code = (
        "import sloth.tune.scorers; import sys; "
        "heavy = [m for m in ('torch', 'unsloth', 'transformers', 'peft', 'sacrebleu') "
        "if m in sys.modules]; "
        "assert not heavy, f'scorers pulled in {heavy}'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_metrics_module_imports_only_stdlib():
    """Every top-level import in metrics.py resolves to a stdlib module."""
    code = (
        "import ast, sys, pathlib; "
        "import sloth.tune.metrics as m; "
        "tree = ast.parse(pathlib.Path(m.__file__).read_text(encoding='utf-8')); "
        "roots = set(); "
        "[roots.update(a.name.split('.')[0] for a in n.names) "
        "for n in ast.walk(tree) if isinstance(n, ast.Import)]; "
        "[roots.add(n.module.split('.')[0]) for n in ast.walk(tree) "
        "if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module]; "
        "extra = sorted(r for r in roots if r not in sys.stdlib_module_names); "
        "assert not extra, f'non-stdlib imports in metrics.py: {extra}'; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Expected returncode 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
