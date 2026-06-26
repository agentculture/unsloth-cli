"""Test that packaging has no torch/unsloth and introspection is import-light.

This test file verifies:
1. pyproject.toml dependencies do not include torch or unsloth
2. Importing sloth.cli does not import torch into sys.modules
3. The whoami verb runs successfully with exit code 0
"""

import sys
import tomllib
from pathlib import Path


def test_deps_have_no_torch_or_unsloth() -> None:
    """Assert torch/unsloth are not in project.dependencies or
    project.optional-dependencies."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Check project.dependencies
    dependencies = data.get("project", {}).get("dependencies", [])
    for dep in dependencies:
        assert "torch" not in dep.lower(), f"Found 'torch' in dependencies: {dep}"
        assert "unsloth" not in dep.lower(), f"Found 'unsloth' in dependencies: {dep}"

    # Check project.optional-dependencies if it exists
    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    for group_name, group_deps in optional_deps.items():
        for dep in group_deps:
            assert (
                "torch" not in dep.lower()
            ), f"Found 'torch' in optional-dependencies[{group_name}]: {dep}"
            assert (
                "unsloth" not in dep.lower()
            ), f"Found 'unsloth' in optional-dependencies[{group_name}]: {dep}"


def test_introspection_imports_without_torch() -> None:
    """Assert that importing sloth.cli does not import torch into
    sys.modules."""
    # Ensure torch is not already loaded
    assert "torch" not in sys.modules, "torch is already in sys.modules"

    # Import sloth.cli
    import sloth.cli  # noqa: F401

    # Verify torch was not loaded
    assert "torch" not in sys.modules, "torch was imported when importing sloth.cli"


def test_whoami_runs() -> None:
    """Assert that running the whoami verb returns exit code 0."""
    from sloth.cli import main

    rc = main(["whoami"])
    assert rc in (0, None), f"Expected exit code 0 or None, got {rc}"
