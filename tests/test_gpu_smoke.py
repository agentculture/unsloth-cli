"""GPU smoke test — real train -> eval -> export pipeline.

Collected by pytest everywhere; all tests skip on CPU-only machines.
The skip guard evaluates :func:`_cuda_available` which imports torch lazily
inside the function body, so the module is importable with no torch/unsloth
installed.  On CPU-only CI the test appears as SKIPPED, not ERROR.

When a CUDA device is present the test runs a tiny real loop:
  1. ``sloth train --config <tiny.toml> --in-container``  (QLoRA, max_steps=1)
  2. ``sloth eval  --adapter <adapter_dir> --suite <eval.jsonl> --in-container``
  3. ``sloth export --adapter <adapter_dir> --output <export_dir>``
and asserts:
  * return code 0 at every step
  * adapter directory + training_metadata.json written by train
  * eval summary contains > 0 records and an exact_match_pct score
  * exported directory contains the canonical PEFT/safetensors artefacts

The ``--in-container`` flag is the recursion guard (see
``sloth/cli/_commands/train.py``): it causes the verbs to run the real
ML path directly without launching an NGC container, which is exactly what
we want when the test itself is already running on a GPU host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# CUDA guard — NO torch at module top level
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Return True only when a CUDA GPU is accessible.

    The CUDA check runs in a *subprocess* so that torch is never imported into
    this (the test) interpreter — importing torch at collection time would
    leave it in ``sys.modules`` and break the import-light invariants asserted
    by ``test_packaging_import_light`` and ``test_tune_datasets``. We first use
    ``find_spec`` (which does not import the module) to skip the subprocess
    entirely when torch is not even installed (the CPU-only CI case).
    """
    import importlib.util  # noqa: PLC0415 — local to keep module import-light
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [
                sys.executable,
                "-c",
                "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
            ],
            timeout=120,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — subprocess failure / timeout → treat as no GPU
        return False


_SKIP_NO_CUDA = pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")


# ---------------------------------------------------------------------------
# Tiny in-test dataset strings (written to tmp_path inside the test)
# ---------------------------------------------------------------------------

# Four chat-schema records for the training dataset.
_TINY_TRAIN_JSONL = "\n".join(
    [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ),
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "6"},
                ]
            }
        ),
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 4+4?"},
                    {"role": "assistant", "content": "8"},
                ]
            }
        ),
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 5+5?"},
                    {"role": "assistant", "content": "10"},
                ]
            }
        ),
    ]
)

# Two task-schema records for the eval suite.
_TINY_EVAL_JSONL = "\n".join(
    [
        json.dumps({"task": "arithmetic", "input": "2+2", "expected_output": "4"}),
        json.dumps({"task": "arithmetic", "input": "3+3", "expected_output": "6"}),
    ]
)

# run.toml template — tiny hyperparameters so the smoke run finishes quickly.
_TOML_TEMPLATE = """\
[run]
model   = "unsloth/Qwen3-4B"
method  = "qlora"
dataset = "{dataset}"
output  = "{output}"

[hyperparameters]
max_steps    = 1
batch_size   = 1
grad_accum   = 1
max_seq_len  = 128
lora_r       = 4
lora_alpha   = 4
load_in_4bit = true
"""


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@_SKIP_NO_CUDA
def test_train_eval_export_smoke(tmp_path: Path, capsys) -> None:
    """Real GPU loop: train -> eval -> export; asserts artefacts and scores.

    Drives each verb via ``main()`` with ``--in-container`` to run the real
    ML path directly (bypassing the NGC container orchestration layer), as
    if the test were already executing inside the container on a GPU host.
    """
    from sloth.cli import main  # noqa: PLC0415 — only imported on GPU hosts

    # --- Prepare on-disk artefacts -----------------------------------------
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(_TINY_TRAIN_JSONL, encoding="utf-8")

    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(_TINY_EVAL_JSONL, encoding="utf-8")

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    config_toml = tmp_path / "run.toml"
    config_toml.write_text(
        _TOML_TEMPLATE.format(
            dataset=str(train_jsonl),
            output=str(adapter_dir),
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 1. TRAIN — QLoRA adapter (max_steps=1 keeps the smoke run fast)     #
    # ------------------------------------------------------------------ #
    rc_train = main(["train", "--config", str(config_toml), "--in-container"])
    assert rc_train == 0, "sloth train --in-container exited non-zero"

    # Adapter directory must contain the PEFT required files.
    assert adapter_dir.is_dir(), "adapter directory missing after train"
    assert (adapter_dir / "adapter_config.json").is_file(), "adapter_config.json missing"
    assert (
        adapter_dir / "adapter_model.safetensors"
    ).is_file(), "adapter_model.safetensors missing"

    # Training metadata must be written beside the adapter output.
    metadata_file = adapter_dir / "training_metadata.json"
    assert metadata_file.is_file(), "training_metadata.json not written"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata.get("model"), "training_metadata.json missing 'model'"
    assert metadata.get("method"), "training_metadata.json missing 'method'"
    assert metadata.get("timestamp"), "training_metadata.json missing 'timestamp'"
    hparams = metadata.get("hyperparameters", {})
    assert int(hparams.get("max_steps", 0)) == 1, "hyperparameters not recorded correctly"

    # Clear stdout/stderr written by the train step before the next capture.
    capsys.readouterr()

    # ------------------------------------------------------------------ #
    # 2. EVAL — adapter against the task-schema eval suite (offline)      #
    # ------------------------------------------------------------------ #
    rc_eval = main(
        [
            "eval",
            "--adapter",
            str(adapter_dir),
            "--suite",
            str(eval_jsonl),
            "--in-container",
            "--json",
        ]
    )
    assert rc_eval == 0, "sloth eval --in-container exited non-zero"

    captured = capsys.readouterr()
    eval_result = json.loads(captured.out.strip())
    assert eval_result["total"] > 0, "eval returned zero records"
    assert "exact_match_pct" in eval_result, "eval summary missing exact_match_pct"
    assert isinstance(eval_result["results"], list), "eval results must be a list"
    assert len(eval_result["results"]) == eval_result["total"], "result count mismatch"

    # ------------------------------------------------------------------ #
    # 3. EXPORT — canonical PEFT/safetensors layout                       #
    # ------------------------------------------------------------------ #
    export_dir = tmp_path / "exported"
    rc_export = main(
        [
            "export",
            "--adapter",
            str(adapter_dir),
            "--output",
            str(export_dir),
            "--json",
        ]
    )
    assert rc_export == 0, "sloth export exited non-zero"

    captured = capsys.readouterr()
    export_result = json.loads(captured.out.strip())
    assert export_result["format"] == "safetensors", "wrong export format"
    exported_names = {Path(f).name for f in export_result.get("files", [])}
    assert "adapter_config.json" in exported_names, "adapter_config.json not in export"
    assert "adapter_model.safetensors" in exported_names, "adapter_model.safetensors not in export"
