"""Test-first check for the ``/finetune`` skill's flag forwarding (plan t12).

``scripts/finetune.sh run`` accepts ``--export-format FMT`` (default
``safetensors``) and ``--quant LIST``, and must forward them to the export
step (``sloth export --format <fmt> [--quant <list>]``). This is a shell
script, not Python, so it is exercised end-to-end via ``subprocess``: a stub
``sloth`` executable is put first on ``PATH`` that fakes ``train --dry-run``
(emits a plan JSON with an ``output`` dir), ``train``/``eval`` (exit 0), and
``export`` (records its argv to a file so the test can assert on it).

No docker, no GPU, no real ``sloth`` package involved — this only checks that
the *wrapper script* builds the right argv.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_SH = REPO_ROOT / ".claude" / "skills" / "finetune" / "scripts" / "finetune.sh"

STUB_SLOTH = """\
#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
verb = args[0] if args else ""
log_path = os.environ["STUB_LOG"]

with open(log_path, "a") as fh:
    fh.write(json.dumps(args) + "\\n")

if verb == "train":
    if "--dry-run" in args:
        adapter_out = os.environ["STUB_ADAPTER_DIR"]
        plan = {"dry_run": True, "output": adapter_out, "format": "n/a"}
        if "--json" in args:
            print(json.dumps(plan))
        else:
            print(f"output: {adapter_out}")
        sys.exit(0)
    sys.exit(0)

if verb == "eval":
    sys.exit(0)

if verb == "export":
    sys.exit(0)

sys.exit(0)
"""


@pytest.fixture()
def stub_env(tmp_path: Path) -> dict[str, str]:
    """Put a stub ``sloth`` first on PATH and point it at a log file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "sloth"
    stub.write_text(STUB_SLOTH)
    stub.chmod(0o755)

    adapter_dir = tmp_path / "adapters" / "run1"
    log_path = tmp_path / "calls.jsonl"

    config = tmp_path / "run.toml"
    config.write_text('[run]\nmodel = "unsloth/Qwen3-4B"\n')
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"task": "t", "input": "i", "expected_output": "o"}\n')

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_LOG"] = str(log_path)
    env["STUB_ADAPTER_DIR"] = str(adapter_dir)
    env["_STUB_CONFIG"] = str(config)
    env["_STUB_SUITE"] = str(suite)
    return env


def _read_calls(log_path: Path) -> list[list[str]]:
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def _run(env: dict[str, str], *extra_args: str) -> subprocess.CompletedProcess:
    cmd = [
        "bash",
        str(FINETUNE_SH),
        "run",
        "--config",
        env["_STUB_CONFIG"],
        "--suite",
        env["_STUB_SUITE"],
        *extra_args,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)


def test_finetune_sh_defaults_export_format_to_safetensors(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """No --export-format given: step 4 forwards --format safetensors, no --quant."""
    result = _run(stub_env)
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    export_calls = [c for c in calls if c and c[0] == "export"]
    assert len(export_calls) == 1, calls
    export_argv = export_calls[0]

    assert "--format" in export_argv
    assert export_argv[export_argv.index("--format") + 1] == "safetensors"
    assert "--quant" not in export_argv


def test_finetune_sh_forwards_export_format_and_quant(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """--export-format gguf --quant q4_k_m,q8_0 is forwarded verbatim to sloth export."""
    result = _run(stub_env, "--export-format", "gguf", "--quant", "q4_k_m,q8_0")
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    export_calls = [c for c in calls if c and c[0] == "export"]
    assert len(export_calls) == 1, calls
    export_argv = export_calls[0]

    assert export_argv[export_argv.index("--format") + 1] == "gguf"
    assert "--quant" in export_argv
    assert export_argv[export_argv.index("--quant") + 1] == "q4_k_m,q8_0"
    # Container formats need --output: the script derives <adapter>-<format> by default.
    assert "--output" in export_argv
    assert export_argv[export_argv.index("--output") + 1].endswith("-gguf")


def test_finetune_sh_forwards_export_format_without_quant(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """--export-format alone (e.g. merged-16bit) needs no --quant to be forwarded."""
    result = _run(stub_env, "--export-format", "merged-16bit")
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    export_calls = [c for c in calls if c and c[0] == "export"]
    assert len(export_calls) == 1, calls
    export_argv = export_calls[0]

    assert export_argv[export_argv.index("--format") + 1] == "merged-16bit"
    assert "--quant" not in export_argv


@pytest.mark.skipif(sys.platform == "win32", reason="bash script, posix only")
def test_finetune_sh_run_help_mentions_export_flags() -> None:
    """--help / help text documents the new flags (keeps docs + script in sync)."""
    result = subprocess.run(
        ["bash", str(FINETUNE_SH), "help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--export-format" in result.stdout
    assert "--quant" in result.stdout
