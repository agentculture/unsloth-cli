"""Tests for sloth.tune._trainer — the lazy LoRA/QLoRA trainer adapter.

This is the ONLY module in the package allowed to touch torch/unsloth, and it
must do so lazily. The tests cover, without a GPU or the ML stack installed:

  1. The dry-run path returns a resolved training plan (model, method, resolved
     hyperparameters, dataset path, output path, scope decision) WITHOUT
     importing torch.
  2. torch/unsloth/trl are never imported at module top level (AST guard +
     subprocess import guard).
  3. A missing backend (``_load_backend`` raising ``ImportError``) surfaces as
     ``CliError(code=2)`` carrying the ``uv tool install unsloth-cli`` hint.
  4. The real path's flow — load model, apply LoRA, train, save adapter, write
     metadata — runs end-to-end against injected fakes (no GPU).
  5. A non-dry-run out-of-scope request is hard-refused with ``CliError(code=1)``
     before any heavy import.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloth.cli._errors import CliError
from sloth.tune import _trainer
from sloth.tune._trainer import run_training
from sloth.tune.config import RunConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, *, model: str = "unsloth/Qwen3-4B", method: str = "qlora") -> RunConfig:
    return RunConfig(
        model=model,
        dataset=str(tmp_path / "train.jsonl"),
        output=str(tmp_path / "adapters" / "out"),
        method=method,
    )


def _write_chat_dataset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _make_fake_backend() -> tuple[SimpleNamespace, dict]:
    """Return a fake backend mimicking _load_backend()'s interface + an event log."""
    events: dict = {
        "from_pretrained": [],
        "get_peft": [],
        "sft_config": [],
        "trainer": [],
        "trained": [],
        "saved": [],
    }

    class FakeModel:
        def save_pretrained(self, path):
            events["saved"].append(("model", str(path)))

    class FakeTokenizer:
        def save_pretrained(self, path):
            events["saved"].append(("tokenizer", str(path)))

    class FakeFLM:
        @staticmethod
        def from_pretrained(**kw):
            events["from_pretrained"].append(kw)
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kw):
            events["get_peft"].append(kw)
            return model

    class FakeTrainer:
        def __init__(self, **kw):
            events["trainer"].append(kw)

        def train(self):
            events["trained"].append(True)

    def fake_sft_config(**kw):
        events["sft_config"].append(kw)
        return kw

    backend = SimpleNamespace(
        FastLanguageModel=FakeFLM,
        SFTTrainer=FakeTrainer,
        SFTConfig=fake_sft_config,
        torch=SimpleNamespace(),
    )
    return backend, events


# ---------------------------------------------------------------------------
# 1. Dry-run plan
# ---------------------------------------------------------------------------


class TestDryRunPlan:
    def test_returns_resolved_plan(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        plan = run_training(config, dry_run=True)

        assert plan["model"] == config.model
        assert plan["method"] == "qlora"
        assert plan["dataset"] == config.dataset
        assert plan["output"] == config.output
        assert plan["dry_run"] is True

    def test_plan_carries_resolved_hyperparameters(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        plan = run_training(config, dry_run=True)
        hp = plan["hyperparameters"]
        assert hp["lora_r"] == config.lora_r
        assert hp["lora_alpha"] == config.lora_alpha
        assert hp["learning_rate"] == config.learning_rate
        assert hp["max_steps"] == config.max_steps
        assert hp["seed"] == config.seed

    def test_plan_carries_scope_decision(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        plan = run_training(config, dry_run=True)
        assert plan["scope"]["ok"] is True
        assert plan["scope"]["out_of_scope"] is False

    def test_dry_run_does_not_import_torch(self, tmp_path: Path, monkeypatch) -> None:
        """Dry-run must not call _load_backend at all."""

        def _boom():
            raise AssertionError("_load_backend must not be called during dry-run")

        monkeypatch.setattr(_trainer, "_load_backend", _boom)
        # Should not raise — backend is never touched.
        run_training(_config(tmp_path), dry_run=True)

    def test_dry_run_out_of_scope_returns_plan_without_raising(self, tmp_path: Path) -> None:
        config = _config(tmp_path, model="unsloth/Qwen3-72B", method="full")
        plan = run_training(config, dry_run=True)
        assert plan["scope"]["out_of_scope"] is True
        assert plan["scope"]["warning"]  # non-empty warning string


# ---------------------------------------------------------------------------
# 2. Lazy-import discipline (no top-level torch/unsloth/trl)
# ---------------------------------------------------------------------------


class TestLazyImportDiscipline:
    def test_no_module_level_heavy_imports(self) -> None:
        source = inspect.getsource(_trainer)
        tree = ast.parse(source)
        heavy = {"torch", "unsloth", "trl"}
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (roots & heavy), f"heavy import at module level: {roots & heavy}"

    def test_importing_trainer_does_not_load_torch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        code = (
            "import sloth.tune._trainer; import sys; "
            "assert 'torch' not in sys.modules, 'torch imported at module top'; "
            "assert 'unsloth' not in sys.modules, 'unsloth imported at module top'; "
            "assert 'trl' not in sys.modules, 'trl imported at module top'; "
            "print('PASS')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        assert result.returncode == 0, (
            f"Expected returncode 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# 3. Missing backend -> CliError(code=2) + install hint
# ---------------------------------------------------------------------------


class TestMissingBackend:
    def test_import_error_becomes_cli_error_code_2(self, tmp_path: Path, monkeypatch) -> None:
        def _raise():
            raise ImportError("No module named 'unsloth'")

        monkeypatch.setattr(_trainer, "_load_backend", _raise)
        with pytest.raises(CliError) as exc_info:
            run_training(_config(tmp_path), dry_run=False)
        assert exc_info.value.code == 2

    def test_cli_error_carries_install_hint(self, tmp_path: Path, monkeypatch) -> None:
        def _raise():
            raise ImportError("No module named 'unsloth'")

        monkeypatch.setattr(_trainer, "_load_backend", _raise)
        with pytest.raises(CliError) as exc_info:
            run_training(_config(tmp_path), dry_run=False)
        assert "uv tool install unsloth-cli" in exc_info.value.remediation


# ---------------------------------------------------------------------------
# 4. Real path flow with injected fakes (no GPU)
# ---------------------------------------------------------------------------


class TestRealFlowWithFakes:
    def test_full_flow_invokes_backend_and_writes_metadata(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend, events = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)

        result = run_training(config, dry_run=False)

        # model loaded with the configured base model
        assert events["from_pretrained"], "FastLanguageModel.from_pretrained not called"
        assert events["from_pretrained"][0]["model_name"] == config.model
        # LoRA applied with the configured rank
        assert events["get_peft"], "get_peft_model not called"
        assert events["get_peft"][0]["r"] == config.lora_r
        # trainer ran
        assert events["trained"] == [True]
        # adapter + tokenizer saved
        saved_what = {what for what, _ in events["saved"]}
        assert saved_what == {"model", "tokenizer"}
        # metadata written next to the adapter output
        meta_path = Path(config.output) / "training_metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["model"] == config.model
        assert data["method"] == "qlora"
        # result dict echoes the plan plus a status
        assert result["status"] == "trained"
        assert result["adapter_dir"] == str(Path(config.output))
        assert result["metadata_path"] == str(meta_path)


# ---------------------------------------------------------------------------
# 5. Non-dry-run out-of-scope -> hard refusal CliError(code=1)
# ---------------------------------------------------------------------------


class TestOutOfScopeRefusal:
    def test_real_run_out_of_scope_raises_code_1(self, tmp_path: Path, monkeypatch) -> None:
        def _boom():
            raise AssertionError("backend must not load for an out-of-scope request")

        monkeypatch.setattr(_trainer, "_load_backend", _boom)
        config = _config(tmp_path, model="unsloth/Qwen3-72B", method="full")
        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert exc_info.value.code == 1
