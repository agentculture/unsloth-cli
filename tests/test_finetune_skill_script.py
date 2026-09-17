"""Test-first check for the ``/finetune`` skill's flag forwarding (plan t12,
t14) and its ``resolve_sloth`` CLI-resolution order (plan t3).

``scripts/finetune.sh run`` accepts ``--export-format FMT`` (default
``safetensors``) and ``--quant LIST``, and must forward them to the export
step (``sloth export --format <fmt> [--quant <list>]``). It also accepts a
repeatable ``--suite`` (one flag per suite, forwarded verbatim to ``sloth
eval``), ``--batch-size``/``--perplexity`` (forwarded to ``sloth eval``), and
``--base REF`` (adds step 5: ``sloth compare --base <ref> <adapter_dir>
--config <c>``). This is a shell script, not Python, so it is exercised
end-to-end via ``subprocess``: a stub ``sloth`` executable is put first on
``PATH`` that fakes ``train --dry-run`` (emits a plan JSON with an ``output``
dir), ``train``/``eval``/``compare`` (exit 0), and ``export`` (records its
argv to a file so the test can assert on it) — every verb call is logged.

No docker, no GPU, no real ``sloth`` package involved — this only checks that
the *wrapper script* builds the right argv.

``resolve_sloth`` resolves the ``sloth`` CLI in order: ``SLOTH_BIN`` (if
non-empty) → a walked-up ``unsloth-cli`` checkout run via
``uv run --project <dir> sloth`` → ``sloth`` on ``PATH``. Each branch gets its
own test below (``test_resolve_sloth_*``), including the empty-``SLOTH_BIN``
fall-through case.
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

if verb == "compare":
    sys.exit(0)

sys.exit(0)
"""

# A fake ``uv`` that unwraps ``run --project <dir> sloth <args...>`` and
# replays the same behaviour as STUB_SLOTH on the unwrapped args — used to
# exercise the checkout branch of resolve_sloth without a real `uv`/`sloth`.
FAKE_UV = """\
#!/usr/bin/env python3
import json
import os
import sys

argv = sys.argv[1:]
assert argv[0] == "run", argv
assert argv[1] == "--project", argv
# argv[2] is the project dir passed to --project; unused by this fake.
assert argv[3] == "sloth", argv
args = argv[4:]

verb = args[0] if args else ""
log_path = os.environ["STUB_LOG"]

with open(log_path, "a") as fh:
    fh.write(json.dumps(args) + "\\n")

if verb == "train" and "--dry-run" in args:
    adapter_out = os.environ.get("STUB_ADAPTER_DIR", "")
    plan = {"dry_run": True, "output": adapter_out, "format": "n/a"}
    if "--json" in args:
        print(json.dumps(plan))
    else:
        print(f"output: {adapter_out}")
    sys.exit(0)

sys.exit(0)
"""


@pytest.fixture
def stub_env(tmp_path: Path) -> dict[str, str]:
    """Point ``SLOTH_BIN`` directly at a stub ``sloth`` script (SLOTH_BIN branch)."""
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
    suite2 = tmp_path / "suite2.jsonl"
    suite2.write_text('{"task": "t2", "input": "i2", "expected_output": "o2"}\n')

    env = dict(os.environ)
    env["SLOTH_BIN"] = str(stub)
    env["STUB_LOG"] = str(log_path)
    env["STUB_ADAPTER_DIR"] = str(adapter_dir)
    env["_STUB_CONFIG"] = str(config)
    env["_STUB_SUITE"] = str(suite)
    env["_STUB_SUITE2"] = str(suite2)
    return env


def _make_checkout(base: Path) -> Path:
    """Build a fake unsloth-cli checkout holding a copy of finetune.sh.

    Returns the path to the copied finetune.sh, whose containing checkout
    (``pyproject.toml`` with ``name = "unsloth-cli"``) sits four directories
    up, matching the real repo layout (``.claude/skills/finetune/scripts/``).
    """
    checkout = base / "unsloth-cli-checkout"
    scripts_dir = checkout / ".claude" / "skills" / "finetune" / "scripts"
    scripts_dir.mkdir(parents=True)
    script_copy = scripts_dir / "finetune.sh"
    script_copy.write_text(FINETUNE_SH.read_text())
    script_copy.chmod(0o755)
    (checkout / "pyproject.toml").write_text('[project]\nname = "unsloth-cli"\n')
    return script_copy


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


def test_finetune_sh_repeated_suite_forwarded_to_eval(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """Repeated --suite flags are each forwarded verbatim to sloth eval."""
    result = _run(
        stub_env,
        "--suite",
        stub_env["_STUB_SUITE2"],
        "--batch-size",
        "16",
        "--perplexity",
    )
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    eval_calls = [c for c in calls if c and c[0] == "eval"]
    assert len(eval_calls) == 1, calls
    eval_argv = eval_calls[0]

    # Two --suite flags: one from _run()'s base invocation, one from extra_args.
    suite_indices = [i for i, a in enumerate(eval_argv) if a == "--suite"]
    assert len(suite_indices) == 2, eval_argv
    suite_values = {eval_argv[i + 1] for i in suite_indices}
    assert suite_values == {stub_env["_STUB_SUITE"], stub_env["_STUB_SUITE2"]}

    assert "--batch-size" in eval_argv
    assert eval_argv[eval_argv.index("--batch-size") + 1] == "16"
    assert "--perplexity" in eval_argv


def test_finetune_sh_no_batch_size_or_perplexity_by_default(
    stub_env: dict[str, str], tmp_path: Path
) -> None:
    """--batch-size / --perplexity are omitted from sloth eval unless given."""
    result = _run(stub_env)
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    eval_calls = [c for c in calls if c and c[0] == "eval"]
    assert len(eval_calls) == 1, calls
    eval_argv = eval_calls[0]

    assert "--batch-size" not in eval_argv
    assert "--perplexity" not in eval_argv


def test_finetune_sh_base_adds_compare_step(stub_env: dict[str, str], tmp_path: Path) -> None:
    """--base REF adds step 5: sloth compare --base <ref> <adapter_dir> --config <c>."""
    result = _run(stub_env, "--base", "unsloth/Qwen3-4B", "--json")
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    compare_calls = [c for c in calls if c and c[0] == "compare"]
    assert len(compare_calls) == 1, calls
    compare_argv = compare_calls[0]

    assert "--base" in compare_argv
    assert compare_argv[compare_argv.index("--base") + 1] == "unsloth/Qwen3-4B"
    assert stub_env["STUB_ADAPTER_DIR"] in compare_argv
    assert "--config" in compare_argv
    assert compare_argv[compare_argv.index("--config") + 1] == stub_env["_STUB_CONFIG"]
    assert "--json" in compare_argv

    # Step banner renumbers to N/5 when --base is given, and step 5 is printed.
    assert "step 1/5" in result.stderr
    assert "step 5/5  compare" in result.stderr


def test_finetune_sh_no_base_skips_compare_step(stub_env: dict[str, str], tmp_path: Path) -> None:
    """Without --base, no compare call is made and banners stay N/4."""
    result = _run(stub_env)
    assert result.returncode == 0, result.stderr

    calls = _read_calls(tmp_path / "calls.jsonl")
    compare_calls = [c for c in calls if c and c[0] == "compare"]
    assert compare_calls == []

    assert "step 1/4" in result.stderr
    assert "step 4/4" in result.stderr
    assert "step 5" not in result.stderr


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
    assert "--batch-size" in result.stdout
    assert "--perplexity" in result.stdout
    assert "--base" in result.stdout


def test_resolve_sloth_uses_sloth_bin_override(stub_env: dict[str, str], tmp_path: Path) -> None:
    """SLOTH_BIN, when non-empty, is used verbatim ahead of checkout/PATH lookup."""
    result = subprocess.run(
        ["bash", str(FINETUNE_SH), "whoami"],
        env=stub_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = _read_calls(tmp_path / "calls.jsonl")
    assert calls == [["whoami"]]


def test_resolve_sloth_falls_back_to_checkout_via_uv(tmp_path: Path) -> None:
    """No SLOTH_BIN: walk up to a checkout's pyproject.toml, run via `uv run --project`."""
    script_copy = _make_checkout(tmp_path)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(FAKE_UV)
    fake_uv.chmod(0o755)

    log_path = tmp_path / "calls.jsonl"
    env = dict(os.environ)
    env.pop("SLOTH_BIN", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(script_copy), "whoami"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = _read_calls(log_path)
    assert calls == [["whoami"]]


def test_resolve_sloth_falls_back_to_path(tmp_path: Path) -> None:
    """No SLOTH_BIN, no enclosing checkout: fall back to `sloth` on PATH."""
    outside = tmp_path / "outside" / "scripts"
    outside.mkdir(parents=True)
    script_copy = outside / "finetune.sh"
    script_copy.write_text(FINETUNE_SH.read_text())
    script_copy.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "sloth"
    stub.write_text(STUB_SLOTH)
    stub.chmod(0o755)

    log_path = tmp_path / "calls.jsonl"
    env = dict(os.environ)
    env.pop("SLOTH_BIN", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(script_copy), "whoami"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = _read_calls(log_path)
    assert calls == [["whoami"]]


def test_resolve_sloth_empty_sloth_bin_falls_through_to_checkout(tmp_path: Path) -> None:
    """SLOTH_BIN='' is treated as unset, falling through to the checkout branch."""
    script_copy = _make_checkout(tmp_path)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(FAKE_UV)
    fake_uv.chmod(0o755)

    log_path = tmp_path / "calls.jsonl"
    env = dict(os.environ)
    env["SLOTH_BIN"] = ""
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["STUB_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(script_copy), "whoami"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = _read_calls(log_path)
    assert calls == [["whoami"]]
