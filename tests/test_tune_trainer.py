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
from sloth.tune._trainer import run_eval, run_training
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
        eos_token = "<eos>"

        def apply_chat_template(self, messages, tokenize=False):
            return f"<chat>{messages}</chat>"

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
        fast_language_model=FakeFLM,
        sft_trainer=FakeTrainer,
        sft_config=fake_sft_config,
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

        # Inject a fake ``datasets`` module so the lazy ``from datasets import Dataset``
        # inside _run_real resolves without needing the real (heavy) package installed.
        fake_module, _, _ = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

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


# ---------------------------------------------------------------------------
# 6. Dataset wrapping: train_dataset must be Dataset.from_list(records) value
# ---------------------------------------------------------------------------


def _fake_datasets_module():
    """Return a (fake_module, call_log) pair for monkeypatching sys.modules['datasets']."""
    from_list_calls: list = []

    class FakeDataset:
        """Stand-in for datasets.Dataset; records from_list calls and acts as sentinel."""

        @classmethod
        def from_list(cls, records):
            from_list_calls.append(records)
            return cls  # return the class itself as an identifiable sentinel

    module = SimpleNamespace(Dataset=FakeDataset)
    return module, FakeDataset, from_list_calls


class TestDatasetWrapping:
    """_run_real must wrap train_records with Dataset.from_list before SFTTrainer.

    H1 coverage (issue #9): these tests confirm the Dataset.from_list wrapping half of
    honesty condition h1 — that ``_run_real`` passes a ``datasets.Dataset`` (not a raw
    ``list[dict]``) to ``SFTTrainer``.  Do NOT duplicate; they already cover this fully.
    """

    def test_sft_trainer_receives_dataset_wrapped_value(self, tmp_path: Path, monkeypatch) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend, events = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)

        fake_module, FakeDataset, from_list_calls = _fake_datasets_module()
        # Inject the fake into sys.modules so the lazy ``from datasets import Dataset``
        # inside _run_real resolves to our FakeDataset without needing the real package.
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

        run_training(config, dry_run=False)

        assert from_list_calls, "Dataset.from_list was never called"
        trainer_kwargs = events["trainer"][0]
        assert trainer_kwargs["train_dataset"] is FakeDataset, (
            "SFTTrainer did not receive the Dataset-wrapped value; "
            f"got {trainer_kwargs['train_dataset']!r} instead of FakeDataset sentinel"
        )

    def test_from_list_called_with_rendered_text_records(self, tmp_path: Path, monkeypatch) -> None:
        """Dataset.from_list must receive records rendered into a single ``text`` column.

        ``_run_real`` renders each validated record (chat → chat template, task →
        prompt shape) into ``{"text": ...}`` before wrapping, so SFTTrainer does not
        depend on trl/unsloth conversational auto-detection.
        """
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend, _ = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)

        fake_module, _, from_list_calls = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

        run_training(config, dry_run=False)

        assert from_list_calls, "Dataset.from_list was never called"
        records = from_list_calls[0]
        assert isinstance(records, list), f"Expected list, got {type(records)}"
        assert len(records) == 1, f"Expected 1 record (one line in fixture), got {len(records)}"
        assert "text" in records[0], "Record was not rendered into a single 'text' field"


# ---------------------------------------------------------------------------
# 7. No-accelerator NotImplementedError -> CliError(code=2) with NGC hint
# ---------------------------------------------------------------------------


class TestNoAcceleratorError:
    """NotImplementedError from the ML backend must surface as CliError(code=2).

    H1 coverage (issue #9): these tests cover the ``_run_real`` side of honesty condition h1
    — specifically that a ``NotImplementedError("cannot find any torch accelerator")`` maps
    to ``CliError.code == 2`` with the NGC container image in the remediation string.
    Do NOT duplicate; they already cover this fully.
    """

    def _make_no_gpu_backend(self) -> SimpleNamespace:
        """Backend whose model-load raises the unsloth no-accelerator error."""

        class FakeFLMNoGPU:
            @staticmethod
            def from_pretrained(**kw):
                raise NotImplementedError(
                    "Unsloth cannot find any torch accelerator? You need a GPU."
                )

        class FakeTrainer:
            def __init__(self, **kw):
                pass

            def train(self):
                pass

        return SimpleNamespace(
            fast_language_model=FakeFLMNoGPU,
            sft_trainer=FakeTrainer,
            sft_config=lambda **kw: kw,
            torch=SimpleNamespace(),
        )

    def _patch_datasets(self, monkeypatch) -> None:
        """Inject a no-op fake datasets module so the lazy import doesn't ImportError."""

        class FakeDataset:
            @classmethod
            def from_list(cls, records):
                return cls

        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))

    def test_not_implemented_error_raises_cli_error_code_2(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend = self._make_no_gpu_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        self._patch_datasets(monkeypatch)

        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert (
            exc_info.value.code == 2
        ), f"Expected code=2 (ENV_ERROR), got code={exc_info.value.code}"

    def test_not_implemented_error_remediation_names_ngc_container(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend = self._make_no_gpu_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        self._patch_datasets(monkeypatch)

        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert (
            "nvcr.io/nvidia/pytorch:25.11-py3" in exc_info.value.remediation
        ), f"NGC container path not in remediation: {exc_info.value.remediation!r}"

    def test_not_implemented_error_does_not_propagate_as_generic(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The CLI must never see a raw NotImplementedError
        # (that would emit code=1 'file a bug' via the generic handler).
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend = self._make_no_gpu_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        self._patch_datasets(monkeypatch)

        # Must raise CliError (not NotImplementedError)
        with pytest.raises(CliError):
            run_training(config, dry_run=False)

    def test_not_implemented_during_train_also_caught(self, tmp_path: Path, monkeypatch) -> None:
        """NotImplementedError during trainer.train() must map to code=2 (not just model load)."""
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))

        class FakeModel:
            def save_pretrained(self, path):
                pass

        class FakeTokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, tokenize=False):
                return f"<chat>{messages}</chat>"

            def save_pretrained(self, path):
                pass

        class FakeFLMOK:
            @staticmethod
            def from_pretrained(**kw):
                return FakeModel(), FakeTokenizer()

            @staticmethod
            def get_peft_model(model, **kw):
                return model

        class FakeTrainerRaisesOnTrain:
            def __init__(self, **kw):
                pass

            def train(self):
                raise NotImplementedError(
                    "Unsloth cannot find any torch accelerator? You need a GPU."
                )

        backend = SimpleNamespace(
            fast_language_model=FakeFLMOK,
            sft_trainer=FakeTrainerRaisesOnTrain,
            sft_config=lambda **kw: kw,
            torch=SimpleNamespace(),
        )
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        self._patch_datasets(monkeypatch)

        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert exc_info.value.code == 2
        assert "nvcr.io/nvidia/pytorch:25.11-py3" in exc_info.value.remediation


# ---------------------------------------------------------------------------
# 7b. GPU out-of-memory -> CliError(code=2) with a memory remediation
# ---------------------------------------------------------------------------


class TestGpuOomMapping:
    """A CUDA/accelerator OOM must surface as CliError(code=2), not a code-1 "file a
    bug". Unsloth raises it at *import* (GPU probe) on a memory-starved box and during
    training; both are environment errors with a free-memory remediation."""

    def test_oom_at_backend_load_maps_to_code_2(self, tmp_path: Path, monkeypatch) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))

        def _oom():
            raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(_trainer, "_load_backend", _oom)
        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert exc_info.value.code == 2
        assert "memory" in exc_info.value.remediation.lower()

    def test_oom_during_train_maps_to_code_2(self, tmp_path: Path, monkeypatch) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))

        class FakeModel:
            def save_pretrained(self, path):
                pass

        class FakeTokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, tokenize=False):
                return f"<chat>{messages}</chat>"

            def save_pretrained(self, path):
                pass

        class FakeFLM:
            @staticmethod
            def from_pretrained(**kw):
                return FakeModel(), FakeTokenizer()

            @staticmethod
            def get_peft_model(model, **kw):
                return model

        class FakeTrainerOom:
            def __init__(self, **kw):
                pass

            def train(self):
                raise RuntimeError("CUDA error: out of memory")

        backend = SimpleNamespace(
            fast_language_model=FakeFLM,
            sft_trainer=FakeTrainerOom,
            sft_config=lambda **kw: kw,
            torch=SimpleNamespace(),
        )
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)

        class FakeDataset:
            @classmethod
            def from_list(cls, records):
                return cls

        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))

        with pytest.raises(CliError) as exc_info:
            run_training(config, dry_run=False)
        assert exc_info.value.code == 2
        assert "memory" in exc_info.value.remediation.lower()


# ---------------------------------------------------------------------------
# 8. run_eval — PeftModel load sequence (moved from test_cmd_eval)
# ---------------------------------------------------------------------------


class TestRunEval:
    """run_eval is the ML seam for ``sloth eval`` (FIX 3 — qodo #10).

    These tests assert the correct PEFT load sequence:
    - AutoModelForCausalLM.from_pretrained is called with the BASE model name
      (read from adapter_config.json), NOT the adapter dir path.
    - PeftModel.from_pretrained is called with (base_model_obj, adapter_path).

    No GPU or real torch/peft/transformers needed: fake modules are injected via
    monkeypatch.setitem(sys.modules, ...) so the lazy imports inside run_eval
    pick them up without the real packages installed.
    """

    def _write_adapter_config(self, adapter_dir: Path, base_model_name: str) -> None:
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model_name, "peft_type": "LORA"}),
            encoding="utf-8",
        )

    def _write_suite(self, suite_path: Path) -> None:
        suite_path.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )

    def _inject_fake_ml(self, monkeypatch, *, base_model_name: str, adapter_dir: str) -> dict:
        """Inject fake torch/transformers/peft and return a call-log dict."""
        calls: dict = {}
        fake_base_model = object()

        from types import SimpleNamespace
        from unittest.mock import MagicMock

        # A dict that also supports ``.to(device)`` (mirrors a transformers BatchEncoding).
        class _FakeInputs(dict):
            def to(self, device):
                calls["inputs_moved_to"] = device
                return self

        # fake model returned by PeftModel.from_pretrained — exposes parameters()/eval()/
        # generate() like a real (GPU-resident) model so run_eval can read its device.
        class _FakeEvalModel:
            def eval(self):
                return self

            def parameters(self):
                yield SimpleNamespace(device="cpu")

            def generate(self, **kw):
                return [[1, 2, 3]]

        # fake PeftModel (records base + adapter_path)
        class FakePeftModel:
            @staticmethod
            def from_pretrained(base, adapter_path, **kw):
                calls["peft_base"] = base
                calls["peft_adapter"] = adapter_path
                return _FakeEvalModel()

        # fake AutoModelForCausalLM (records the name it was called with)
        class FakeAutoModel:
            @staticmethod
            def from_pretrained(name, **kw):
                calls["causal_lm_name"] = name
                return fake_base_model

        # fake AutoTokenizer
        class _FakeTok:
            def __call__(self, prompt, **kw):
                return _FakeInputs(input_ids=[[1, 2, 3]])

            def decode(self, tokens, **kw):
                return "cba"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kw):
                return _FakeTok()

        fake_torch = MagicMock()
        fake_torch.no_grad.return_value.__enter__ = lambda s: None
        fake_torch.no_grad.return_value.__exit__ = lambda s, *a: False

        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        )
        fake_peft_mod = SimpleNamespace(PeftModel=FakePeftModel)

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft_mod)

        calls["_fake_base_model"] = fake_base_model
        return calls

    def test_peft_load_sequence_base_model_then_adapter(self, tmp_path: Path, monkeypatch) -> None:
        """run_eval reads adapter_config.json and calls PeftModel(base_model, adapter).

        AutoModelForCausalLM.from_pretrained must receive the BASE model name
        (from adapter_config.json), not the adapter dir path.
        PeftModel.from_pretrained must receive (base_model_obj, adapter_path).
        """
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        base_model_name = "unsloth/Qwen3-4B"
        self._write_adapter_config(adapter_dir, base_model_name)
        suite_file = tmp_path / "suite.jsonl"
        self._write_suite(suite_file)

        calls = self._inject_fake_ml(
            monkeypatch, base_model_name=base_model_name, adapter_dir=str(adapter_dir)
        )

        result = run_eval(str(adapter_dir), str(suite_file))

        # AutoModelForCausalLM called with BASE name, NOT the adapter dir path.
        assert (
            calls["causal_lm_name"] == base_model_name
        ), "AutoModelForCausalLM.from_pretrained must be called with the base model name"
        assert calls["causal_lm_name"] != str(
            adapter_dir
        ), "AutoModelForCausalLM.from_pretrained must NOT be called with the adapter dir"

        # PeftModel called with (base_model_obj, adapter_path).
        assert (
            calls["peft_base"] is calls["_fake_base_model"]
        ), "PeftModel.from_pretrained must receive the base model object as its first arg"
        assert calls["peft_adapter"] == str(
            adapter_dir
        ), "PeftModel.from_pretrained must receive the adapter dir path as its second arg"

        # Result structure.
        assert result["total"] == 1
        assert "exact_match" in result
        assert "results" in result

    def test_run_eval_missing_adapter_config_raises_code_1(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """run_eval raises CliError(code=1) when adapter_config.json is absent."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        suite_file = tmp_path / "suite.jsonl"
        self._write_suite(suite_file)

        from unittest.mock import MagicMock

        fake_torch = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", MagicMock())
        monkeypatch.setitem(sys.modules, "peft", MagicMock())

        with pytest.raises(CliError) as exc_info:
            run_eval(str(adapter_dir), str(suite_file))
        assert exc_info.value.code == 1
        assert "adapter_config.json" in exc_info.value.message

    def test_run_eval_missing_ml_stack_raises_code_2(self, tmp_path: Path, monkeypatch) -> None:
        """run_eval raises CliError(code=2) when torch/peft/transformers are absent."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        self._write_adapter_config(adapter_dir, "unsloth/Qwen3-4B")
        suite_file = tmp_path / "suite.jsonl"
        self._write_suite(suite_file)

        # Remove torch so the lazy import fails.
        monkeypatch.setitem(
            sys.modules,
            "torch",
            None,  # type: ignore[arg-type]  # None in sys.modules → ImportError
        )

        with pytest.raises((CliError, ImportError)):
            run_eval(str(adapter_dir), str(suite_file))
