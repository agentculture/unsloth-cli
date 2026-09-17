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
from sloth.tune import _trainer, metrics
from sloth.tune._trainer import run_eval, run_training
from sloth.tune.config import RunConfig
from sloth.tune.presets import PRESETS

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


def _make_fake_backend(log_history: list | None = None) -> tuple[SimpleNamespace, dict]:
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
            # Mirrors transformers' ``Trainer.state.log_history`` — the source
            # _run_real reads loss history from (NOT the TrainOutput return).
            self.state = SimpleNamespace(log_history=list(log_history or []))

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

    def test_plan_carries_null_target_modules_when_unset(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        plan = run_training(config, dry_run=True)
        assert plan["hyperparameters"]["target_modules"] is None

    def test_plan_carries_resolved_target_modules_for_preset(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.target_modules = "preset:lfm2"
        plan = run_training(config, dry_run=True)
        assert plan["hyperparameters"]["target_modules"] == PRESETS["lfm2"]

    def test_plan_carries_literal_target_modules_list_unchanged(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.target_modules = ["q_proj", "k_proj"]
        plan = run_training(config, dry_run=True)
        assert plan["hyperparameters"]["target_modules"] == ["q_proj", "k_proj"]

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

    def test_get_peft_model_omits_target_modules_when_unset(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend, events = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        fake_module, _, _ = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

        run_training(config, dry_run=False)

        assert "target_modules" not in events["get_peft"][0]

    def test_get_peft_model_receives_resolved_preset_target_modules(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = _config(tmp_path)
        config.target_modules = "preset:lfm2"
        _write_chat_dataset(Path(config.dataset))
        backend, events = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        fake_module, _, _ = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

        result = run_training(config, dry_run=False)

        assert events["get_peft"][0]["target_modules"] == PRESETS["lfm2"]

        meta_path = Path(result["metadata_path"])
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["hyperparameters"]["target_modules"] == PRESETS["lfm2"]

    def test_metadata_records_dataset_path_export_reader_expects(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The real-flow writes training_metadata.json['dataset']['path'] —
        the same key export.py's ``_training_dataset`` helper reads."""
        config = _config(tmp_path)
        _write_chat_dataset(Path(config.dataset))
        backend, _ = _make_fake_backend()
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        fake_module, _, _ = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)

        result = run_training(config, dry_run=False)

        meta_path = Path(result["metadata_path"])
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["dataset"]["path"] == config.dataset


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
# Fake tokenizer / model kit for the eval tests
#
# These stand in for transformers' tensors closely enough to exercise the REAL
# slicing arithmetic: ``input_ids``/``attention_mask`` are 2-D objects with
# ``.shape`` and row indexing, rows support ``.sum()`` and slicing, and the fake
# tokenizer *actually decodes the token slice it is handed* — so a wrong prompt
# offset shows up as a prediction that still contains its own prompt, rather
# than being hidden behind a canned decode() return value.
# ---------------------------------------------------------------------------


class _FakeRow(list):
    """A 1-D tensor stand-in: a list that also answers ``.sum()`` and slices."""

    def sum(self) -> int:
        return sum(self)

    def __getitem__(self, key):  # type: ignore[override]
        item = list.__getitem__(self, key)
        return _FakeRow(item) if isinstance(key, slice) else item


class _FakeBatch:
    """A 2-D tensor stand-in: ``.shape`` plus row indexing."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [_FakeRow(r) for r in rows]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> _FakeRow:
        return self.rows[index]


class _FakeInputs(dict):
    """A BatchEncoding stand-in: a dict that also supports ``.to(device)``."""

    def to(self, _device):  # noqa: D102 - trivial
        return self


class _FakeTokenizer:
    """Word-level tokenizer over a growing vocabulary; decodes what it is given."""

    def __init__(self, *, pad_token: str | None = None, eos_token: str | None = "</s>") -> None:
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = "right"
        self.chat_template_calls: list[dict] = []
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._ids:
            token_id = len(self._ids) + 1
            self._ids[word] = token_id
            self._words[token_id] = word
        return self._ids[word]

    def encode(self, text: str) -> list[int]:
        return [self._id(word) for word in text.split()]

    def __call__(self, text, return_tensors: str = "pt", padding: bool = False) -> _FakeInputs:
        texts = [text] if isinstance(text, str) else list(text)
        sequences = [self.encode(t) for t in texts]
        masks = [[1] * len(s) for s in sequences]
        if padding:
            assert self.pad_token is not None, "padding requested without a pad token"
            pad_id = self._id(self.pad_token)
            width = max(len(s) for s in sequences)
            padded, padded_masks = [], []
            for seq, mask in zip(sequences, masks):
                gap = width - len(seq)
                if self.padding_side == "left":
                    padded.append([pad_id] * gap + seq)
                    padded_masks.append([0] * gap + mask)
                else:
                    padded.append(seq + [pad_id] * gap)
                    padded_masks.append(mask + [0] * gap)
            sequences, masks = padded, padded_masks
        return _FakeInputs(input_ids=_FakeBatch(sequences), attention_mask=_FakeBatch(masks))

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render *messages* the way a real chat template would (recording the call)."""
        self.chat_template_calls.append(
            {"messages": messages, "add_generation_prompt": add_generation_prompt}
        )
        rendered = " ".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered = f"{rendered} <assistant>"
        return rendered

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        words = [self._words[int(t)] for t in tokens]
        if skip_special_tokens:
            words = [w for w in words if w not in {self.pad_token, self.eos_token}]
        return " ".join(words)


class _FakeEvalModel:
    """Appends the configured answer's tokens to the (padded) prompt row."""

    def __init__(self, tokenizer: _FakeTokenizer, answers: dict[str, str]) -> None:
        self.tokenizer = tokenizer
        self.answers = answers
        self.batch_widths: list[int] = []
        self.forward_calls: list[str] = []
        self.forbid_generate = False
        self.losses: list[float] = [0.0]

    def eval(self):  # noqa: D102 - trivial
        return self

    def parameters(self):  # noqa: D102 - trivial
        yield SimpleNamespace(device="cpu")

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        """A labelled forward pass — what run_perplexity must use (never generate())."""
        assert labels is not None, "run_perplexity must pass labels= for a scored pass"
        self.forward_calls.append(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))
        index = min(len(self.forward_calls) - 1, len(self.losses) - 1)
        return SimpleNamespace(loss=self.losses[index])

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens: int = 16):
        if self.forbid_generate:
            raise AssertionError("generate() must not be called on the perplexity path")
        self.batch_widths.append(len(input_ids))
        rows = []
        for i in range(len(input_ids)):
            row = list(input_ids[i])
            mask = list(attention_mask[i]) if attention_mask is not None else [1] * len(row)
            prompt = self.tokenizer.decode(
                _FakeRow([t for t, m in zip(row, mask) if m]), skip_special_tokens=False
            )
            answer = next(
                (a for key, a in self.answers.items() if key in prompt),
                "UNKNOWN",
            )
            rows.append(row + self.tokenizer.encode(answer))
        return _FakeBatch(rows)


def _install_fake_ml(
    monkeypatch, *, tokenizer: _FakeTokenizer, model: _FakeEvalModel, calls: dict | None = None
) -> dict:
    """Inject fake torch/transformers/peft modules built around *tokenizer*/*model*."""
    log: dict = calls if calls is not None else {}
    base_model = object()
    log["_fake_base_model"] = base_model

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kw):
            log["causal_lm_name"] = name
            return base_model

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter_path, **kw):
            log["peft_base"] = base
            log["peft_adapter"] = adapter_path
            return model

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            return tokenizer

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = SimpleNamespace(no_grad=_FakeNoGrad)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=_FakeAutoModel, AutoTokenizer=_FakeAutoTokenizer),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=_FakePeftModel))
    return log


def _write_task_suite(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write a task-schema JSONL suite from ``(task, input, expected_output)`` rows."""
    path.write_text(
        "".join(
            json.dumps({"task": t, "input": i, "expected_output": e}) + "\n" for t, i, e in rows
        ),
        encoding="utf-8",
    )
    return path


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
        """Inject fake torch/transformers/peft and return a call-log dict.

        The fake tokenizer decodes the **actual token slice** run_eval hands it
        (see ``_FakeTokenizer``), so a prediction that still carried its prompt
        would be visible in the scored output instead of being masked by a canned
        decode() answer.
        """
        tokenizer = _FakeTokenizer()
        model = _FakeEvalModel(tokenizer, {"abc": "cba"})
        calls = _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)
        calls["_tokenizer"] = tokenizer
        calls["_model"] = model
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


# ---------------------------------------------------------------------------
# 9. sloth.tune.metrics — pure-stdlib scoring (exact match + token F1)
# ---------------------------------------------------------------------------


class TestMetrics:
    """The scoring core both eval seams share. Pure stdlib: no torch, no transformers."""

    def test_token_f1_partial_overlap_is_two_thirds(self) -> None:
        """f1('a b c', 'a b d') == 0.667 to three decimals (2 of 3 tokens shared)."""
        assert round(metrics.token_f1("a b c", "a b d"), 3) == 0.667

    def test_token_f1_identical_and_disjoint(self) -> None:
        assert metrics.token_f1("hello world", "hello world") == 1.0
        assert metrics.token_f1("hello", "world") == 0.0

    def test_token_f1_is_case_and_punctuation_insensitive(self) -> None:
        assert metrics.token_f1("Hello, world!", "hello world") == 1.0

    def test_token_f1_empty_sides(self) -> None:
        """Two blanks match; exactly one blank does not."""
        assert metrics.token_f1("", "") == 1.0
        assert metrics.token_f1("", "a") == 0.0
        assert metrics.token_f1("a", "") == 0.0

    def test_token_f1_counts_repeats_as_a_multiset(self) -> None:
        """Repeating a token does not earn extra credit for it."""
        assert round(metrics.token_f1("a a a a", "a"), 3) == 0.4

    def test_summarize_reports_totals_and_mean_f1(self) -> None:
        records = [
            {"task": "t", "input": "i", "expected_output": "a b c"},
            {"task": "t", "input": "j", "expected_output": "x y z"},
        ]
        scored = metrics.score_records(records, ["a b c", "x y q"])
        summary = metrics.summarize(scored)
        assert summary["total"] == 2
        assert summary["exact_match"] == 1
        assert summary["exact_match_pct"] == 50.0
        # (1.0 + 2/3) / 2
        assert round(summary["f1"], 3) == 0.833

    def test_score_records_indices_continue_across_files(self) -> None:
        records = [{"task": "t", "input": "i", "expected_output": "a"}]
        scored = metrics.score_records(records, ["a"], start_index=7, source="s.jsonl")
        assert scored[0]["index"] == 7
        assert scored[0]["file"] == "s.jsonl"

    def test_metrics_imports_only_stdlib(self) -> None:
        """Every top-level import of metrics.py resolves to a stdlib module."""
        source = Path(_trainer.__file__).with_name("metrics.py").read_text(encoding="utf-8")
        roots: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        non_stdlib = {r for r in roots if r not in sys.stdlib_module_names}
        assert not non_stdlib, f"metrics.py imports non-stdlib modules: {sorted(non_stdlib)}"


# ---------------------------------------------------------------------------
# 10. run_eval — batched generation, per-file scoring, eval.json
# ---------------------------------------------------------------------------


def _adapter_with_config(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-4B", "peft_type": "LORA"}),
        encoding="utf-8",
    )
    return adapter


class TestRunEvalBatching:
    """Batched, left-padded generation must slice each row's prompt off correctly."""

    def test_predictions_exclude_their_own_prompt(self, tmp_path: Path, monkeypatch) -> None:
        """Three prompts of DIFFERENT lengths, one batch: no prediction keeps its prompt.

        This is the regression that left padding makes subtle — every row is
        ``pad* + prompt + continuation``, so the continuation starts at the shared
        PADDED width, not at the row's own token count.
        """
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(
            tmp_path / "suite.jsonl",
            [
                ("reverse", "abc", "cba"),
                ("reverse", "a much longer input here", "erehtupni"),
                ("reverse", "mid length input", "tupnidim"),
            ],
        )
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(
            tokenizer,
            {
                "abc": "cba",
                "a much longer input here": "erehtupni",
                "mid length input": "tupnidim",
            },
        )
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)

        result = run_eval(str(adapter), suite_paths=[suite], batch_size=8)

        assert model.batch_widths == [3], "all three prompts must go through ONE generate() call"
        assert tokenizer.padding_side == "left"
        predictions = [r["prediction"] for r in result["results"]]
        assert predictions == ["cba", "erehtupni", "tupnidim"]
        for entry in result["results"]:
            prompt = f"Task: {entry['task']}\nInput: {entry['input']}\nOutput:"
            assert not entry["prediction"].startswith(prompt)
            for word in prompt.split():
                assert word not in entry["prediction"].split()
        assert result["exact_match"] == 3

    def test_batches_are_chunked_at_batch_size(self, tmp_path: Path, monkeypatch) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(
            tmp_path / "suite.jsonl",
            [("echo", f"w{i}", f"w{i}") for i in range(5)],
        )
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {f"w{i}": f"w{i}" for i in range(5)})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)

        result = run_eval(str(adapter), suite_paths=[suite], batch_size=2)
        assert model.batch_widths == [2, 2, 1]
        assert result["exact_match"] == 5

    def test_eos_is_adopted_as_the_pad_token(self, tmp_path: Path, monkeypatch) -> None:
        """pad_token=None with eos set ⇒ eos becomes the pad token, batching proceeds."""
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(
            tmp_path / "suite.jsonl", [("echo", "a", "a"), ("echo", "b b b", "b")]
        )
        tokenizer = _FakeTokenizer(pad_token=None, eos_token="<eos>")
        model = _FakeEvalModel(tokenizer, {"Input: a": "a", "Input: b b b": "b"})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)

        result = run_eval(str(adapter), suite_paths=[suite], batch_size=4)
        assert tokenizer.pad_token == "<eos>"
        assert model.batch_widths == [2]
        assert [r["prediction"] for r in result["results"]] == ["a", "b"]

    def test_no_pad_and_no_eos_falls_back_to_batch_size_one(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Unpaddable tokenizer ⇒ batch size 1 plus a stderr diagnostic, same scores."""
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(
            tmp_path / "suite.jsonl", [("echo", "a", "a"), ("echo", "b b b", "b")]
        )
        tokenizer = _FakeTokenizer(pad_token=None, eos_token=None)
        model = _FakeEvalModel(tokenizer, {"Input: a": "a", "Input: b b b": "b"})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)

        result = run_eval(str(adapter), suite_paths=[suite], batch_size=8)

        captured = capsys.readouterr()
        assert model.batch_widths == [1, 1]
        assert "batch size 1" in captured.err
        assert captured.out == "", "diagnostics must never reach stdout"
        assert [r["prediction"] for r in result["results"]] == ["a", "b"]


class TestRunEvalSuiteShape:
    """Per-file entries + the aggregate, and the eval.json written next to the adapter."""

    def _run(self, tmp_path: Path, monkeypatch, suites: list[Path], **kwargs):
        adapter = _adapter_with_config(tmp_path)
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {"abc": "cba", "xyz": "zyx"})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)
        return adapter, run_eval(str(adapter), suite_paths=suites, **kwargs)

    def test_signature_accepts_the_cli_keywords(self) -> None:
        """``eval.py::_call_eval_seam`` introspects these exact parameter names."""
        params = inspect.signature(run_eval).parameters
        assert "suite_paths" in params
        assert "quant" in params
        assert "batch_size" in params
        assert list(params)[0] == "adapter_path"

    def test_single_suite_keeps_the_legacy_positional_call(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``run_eval(target, suite_path)`` — the fallback call — still works."""
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {"abc": "cba"})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)

        result = run_eval(str(adapter), str(suite))
        assert result["total"] == 1
        assert result["files"][0]["path"] == str(suite)

    def test_per_file_entries_plus_aggregate(self, tmp_path: Path, monkeypatch) -> None:
        good = _write_task_suite(tmp_path / "good.jsonl", [("reverse", "abc", "cba")])
        bad = _write_task_suite(tmp_path / "bad.jsonl", [("reverse", "xyz", "nope")])
        _adapter, result = self._run(tmp_path, monkeypatch, [good, bad])

        assert [f["path"] for f in result["files"]] == [str(good), str(bad)]
        assert result["files"][0]["exact_match"] == 1
        assert result["files"][0]["exact_match_pct"] == 100.0
        assert result["files"][0]["f1"] == 1.0
        assert result["files"][1]["exact_match"] == 0
        assert result["files"][1]["f1"] == 0.0
        # Aggregate over both files, at the TOP level (what the CLI prints today).
        assert result["total"] == 2
        assert result["exact_match"] == 1
        assert result["exact_match_pct"] == 50.0
        assert result["f1"] == 0.5
        assert [r["index"] for r in result["results"]] == [0, 1]
        assert len(result["results"]) == 2

    def test_eval_json_is_written_into_the_adapter_dir(self, tmp_path: Path, monkeypatch) -> None:
        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        adapter, result = self._run(tmp_path, monkeypatch, [suite])

        written = json.loads((adapter / "eval.json").read_text(encoding="utf-8"))
        assert written["suite_paths"] == [str(suite)]
        assert written["target"] == "adapter"
        assert written["written_at"].startswith("20")
        for key, value in result.items():
            assert written[key] == value

    def test_eval_json_is_overwritten_on_re_run(self, tmp_path: Path, monkeypatch) -> None:
        first = _write_task_suite(tmp_path / "one.jsonl", [("reverse", "abc", "cba")])
        adapter, _ = self._run(tmp_path, monkeypatch, [first])
        second = _write_task_suite(
            tmp_path / "two.jsonl", [("reverse", "abc", "cba"), ("reverse", "xyz", "zyx")]
        )
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {"abc": "cba", "xyz": "zyx"})
        _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)
        run_eval(str(adapter), suite_paths=[second])

        written = json.loads((adapter / "eval.json").read_text(encoding="utf-8"))
        assert written["suite_paths"] == [str(second)]
        assert written["total"] == 2


class TestPaddedPromptWidth:
    """The continuation offset under left padding — the subtle bit, tested directly."""

    def _inputs(self, rows: list[list[int]], masks: list[list[int]]) -> _FakeInputs:
        return _FakeInputs(input_ids=_FakeBatch(rows), attention_mask=_FakeBatch(masks))

    def test_offset_is_the_shared_padded_width_not_the_row_token_count(self) -> None:
        """Row 0 is left-padded: its own token count (2) is NOT where generation starts."""
        inputs = self._inputs([[9, 9, 1, 2], [3, 4, 5, 6]], [[0, 0, 1, 1], [1, 1, 1, 1]])
        assert _trainer._padded_prompt_width(inputs, 0) == 4
        assert _trainer._padded_prompt_width(inputs, 1) == 4

    def test_offset_without_an_attention_mask_is_the_input_width(self) -> None:
        inputs = _FakeInputs(input_ids=_FakeBatch([[1, 2, 3]]))
        assert _trainer._padded_prompt_width(inputs, 0) == 3


def test_resolve_suite_paths_prefers_the_list(tmp_path: Path) -> None:
    resolved = _trainer.resolve_suite_paths("legacy.jsonl", ["a.jsonl", "b.jsonl"])
    assert [str(p) for p in resolved] == ["a.jsonl", "b.jsonl"]
    assert [str(p) for p in _trainer.resolve_suite_paths("legacy.jsonl", None)] == ["legacy.jsonl"]


def test_resolve_suite_paths_without_any_suite_is_a_user_error() -> None:
    with pytest.raises(CliError) as exc_info:
        _trainer.resolve_suite_paths(None, None)
    assert exc_info.value.code == 1
    assert exc_info.value.remediation


# ---------------------------------------------------------------------------
# 11. t6 — eval loop: latency, token counts, chat rendering, per-schema scoring,
#     perplexity, and the recorded base precision.
# ---------------------------------------------------------------------------


class _Clock:
    """A deterministic ``perf_counter`` stand-in advancing *step* seconds per call."""

    def __init__(self, step: float = 0.5) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _freeze_clock(monkeypatch, step: float = 0.5) -> _Clock:
    clock = _Clock(step)
    monkeypatch.setattr(_trainer.time, "perf_counter", clock)
    return clock


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _eval_fakes(monkeypatch, answers: dict[str, str], *, pad_token: str | None = "<pad>"):
    tokenizer = _FakeTokenizer(pad_token=pad_token)
    model = _FakeEvalModel(tokenizer, answers)
    _install_fake_ml(monkeypatch, tokenizer=tokenizer, model=model)
    return tokenizer, model


class TestGenerationTiming:
    """``_generate_predictions`` reports per-row generated tokens and latency."""

    def test_returns_tokens_and_per_row_latency(self, tmp_path: Path, monkeypatch) -> None:
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {"abc": "cba", "xyz": "z y x"})
        _freeze_clock(monkeypatch, step=0.5)

        run = _trainer._generate_predictions(
            SimpleNamespace(no_grad=lambda: _NullContext()),
            model,
            tokenizer,
            ["abc", "xyz"],
            batch_size=8,
            max_new_tokens=8,
            device="cpu",
        )

        assert run.predictions == ["cba", "z y x"]
        assert run.generated_tokens == [1, 3]
        # One batch of two rows taking 0.5s ⇒ 250 ms attributed to each row.
        assert run.latency_ms == [250.0, 250.0]
        assert run.generate_seconds == pytest.approx(0.5)

    def test_unbatched_path_also_reports_timing(self, monkeypatch) -> None:
        tokenizer = _FakeTokenizer(pad_token=None, eos_token=None)
        model = _FakeEvalModel(tokenizer, {"abc": "cba"})
        _freeze_clock(monkeypatch, step=0.25)

        run = _trainer._generate_predictions(
            SimpleNamespace(no_grad=lambda: _NullContext()),
            model,
            tokenizer,
            ["abc"],
            batch_size=8,
            max_new_tokens=8,
            device="cpu",
        )
        assert run.predictions == ["cba"]
        assert run.generated_tokens == [1]
        assert run.latency_ms == [250.0]


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestRunEvalTiming:
    """run_eval writes the per-row and suite-level timing keys into its result."""

    def test_per_row_and_suite_level_keys(self, tmp_path: Path, monkeypatch) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(
            tmp_path / "s.jsonl", [("reverse", "abc", "cba"), ("reverse", "xyz", "zyx")]
        )
        _eval_fakes(monkeypatch, {"abc": "cba", "xyz": "zyx"})
        _freeze_clock(monkeypatch, step=0.5)

        result = run_eval(str(adapter), suite_paths=[suite], batch_size=4, base_load_in_4bit=True)

        for row in result["results"]:
            assert row["generated_tokens"] == 1
            assert row["latency_ms"] == 250.0
        assert result["median_latency_ms"] == 250.0
        # 2 generated tokens over 0.5 s of generate wall time.
        assert result["tokens_per_s"] == pytest.approx(4.0)
        assert result["batch_size"] == 4
        assert result["base_load_in_4bit"] is True
        assert result["files"][0]["median_latency_ms"] == 250.0
        assert result["files"][0]["tokens_per_s"] == pytest.approx(4.0)

    def test_base_load_in_4bit_defaults_to_none(self, tmp_path: Path, monkeypatch) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        _eval_fakes(monkeypatch, {"abc": "cba"})
        result = run_eval(str(adapter), suite_paths=[suite])
        assert result["base_load_in_4bit"] is None

    def test_suite_keyed_eval_json_records_batch_and_precision(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(tmp_path / "my_suite.jsonl", [("reverse", "abc", "cba")])
        _eval_fakes(monkeypatch, {"abc": "cba"})

        run_eval(str(adapter), suite_paths=[suite], batch_size=3, base_load_in_4bit=False)

        written = json.loads((adapter / "eval" / "my-suite.json").read_text(encoding="utf-8"))
        assert written["schema_version"] == metrics.SCHEMA_VERSION
        assert written["suite"] == "my-suite"
        assert written["batch_size"] == 3
        assert written["base_load_in_4bit"] is False
        assert written["target"] == "adapter"
        assert written["median_latency_ms"] >= 0.0
        # The legacy flat eval.json is still written (sloth summarize reads it).
        assert (adapter / "eval.json").is_file()


class TestChatSuiteEval:
    """A chat-schema suite is rendered through the tokenizer's chat template."""

    def test_chat_rows_use_add_generation_prompt(self, tmp_path: Path, monkeypatch) -> None:
        suite = _write_jsonl(
            tmp_path / "chat.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "ping"},
                        {"role": "assistant", "content": "pong"},
                    ]
                }
            ],
        )
        adapter = _adapter_with_config(tmp_path)
        tokenizer, _model = _eval_fakes(monkeypatch, {"<user>ping": "pong"})

        result = run_eval(str(adapter), suite_paths=[suite])

        assert tokenizer.chat_template_calls, "the tokenizer chat template must be used"
        call = tokenizer.chat_template_calls[0]
        assert call["add_generation_prompt"] is True
        assert [m["role"] for m in call["messages"]] == [
            "user"
        ], "the final assistant turn is the expected output, not part of the prompt"
        assert result["results"][0]["prediction"] == "pong"
        assert result["results"][0]["expected_output"] == "pong"
        assert result["exact_match"] == 1

    def test_eval_prompt_falls_back_without_a_tokenizer(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "pong"},
            ]
        }
        rendered = _trainer.eval_prompt(record)
        assert "user: ping" in rendered
        assert "pong" not in rendered


class TestPerSchemaScoring:
    """instruction / structured / toolcall suites get their own per-row flag + compliance."""

    def test_instruction_constraints(self, tmp_path: Path, monkeypatch) -> None:
        suite = _write_jsonl(
            tmp_path / "instr.jsonl",
            [
                {
                    "task": "short",
                    "input": "aaa",
                    "expected_output": "ok",
                    "constraints": [{"max_words": 1}],
                },
                {
                    "task": "short",
                    "input": "bbb",
                    "expected_output": "ok",
                    "constraints": [{"max_words": 1}],
                },
            ],
        )
        adapter = _adapter_with_config(tmp_path)
        _eval_fakes(monkeypatch, {"aaa": "ok", "bbb": "way too long"})

        result = run_eval(str(adapter), suite_paths=[suite])

        assert [r["constraints_passed"] for r in result["results"]] == [True, False]
        assert result["compliance_pct"] == 50.0
        assert result["files"][0]["compliance_pct"] == 50.0

    def test_structured_json_validity(self, tmp_path: Path, monkeypatch) -> None:
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
        suite = _write_jsonl(
            tmp_path / "struct.jsonl",
            [
                {"task": "emit", "input": "good", "json_schema": schema},
                {"task": "emit", "input": "bad", "json_schema": schema},
            ],
        )
        adapter = _adapter_with_config(tmp_path)
        _eval_fakes(monkeypatch, {"good": '{"a": "x"}', "bad": "not json"})

        result = run_eval(str(adapter), suite_paths=[suite])

        assert [r["json_valid"] for r in result["results"]] == [True, False]
        assert result["compliance_pct"] == 50.0

    def test_toolcall_matching(self, tmp_path: Path, monkeypatch) -> None:
        expected = {"name": "get_weather", "arguments": {"city": "Paris"}}
        suite = _write_jsonl(
            tmp_path / "tools.jsonl",
            [
                {"task": "call", "input": "paris", "expected_tool_call": expected},
                {"task": "call", "input": "berlin", "expected_tool_call": expected},
            ],
        )
        adapter = _adapter_with_config(tmp_path)
        good = '<tool_call> {"name": "get_weather", "arguments": {"city": "Paris"}} </tool_call>'
        _eval_fakes(monkeypatch, {"paris": good, "berlin": "no call at all"})

        result = run_eval(str(adapter), suite_paths=[suite])

        assert [r["tool_call_matched"] for r in result["results"]] == [True, False]
        assert result["compliance_pct"] == 50.0

    def test_undetectable_tool_call_family_is_a_user_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        expected = {"name": "t", "arguments": {}}
        suite = _write_jsonl(
            tmp_path / "tools.jsonl",
            [{"task": "call", "input": "x", "expected_tool_call": expected}],
        )
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "some/unknown-model"}), encoding="utf-8"
        )
        _eval_fakes(monkeypatch, {})

        with pytest.raises(CliError) as exc_info:
            run_eval(str(adapter), suite_paths=[suite])
        assert exc_info.value.code == 1
        assert "tool_call_family" in exc_info.value.remediation

    def test_explicit_tool_call_family_overrides_detection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        expected = {"name": "t", "arguments": {}}
        suite = _write_jsonl(
            tmp_path / "tools.jsonl",
            [{"task": "call", "input": "x", "expected_tool_call": expected}],
        )
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "some/unknown-model"}), encoding="utf-8"
        )
        _eval_fakes(monkeypatch, {"x": '<tool_call> {"name": "t", "arguments": {}} </tool_call>'})

        result = run_eval(str(adapter), suite_paths=[suite], tool_call_family="qwen3")
        assert result["results"][0]["tool_call_matched"] is True
        assert result["compliance_pct"] == 100.0


class TestPerplexity:
    """``run_perplexity`` scores a labelled forward pass — never ``generate()``."""

    def test_returns_exp_of_mean_token_nll(self, tmp_path: Path, monkeypatch) -> None:
        import math

        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {})
        model.forbid_generate = True
        model.losses = [2.0]
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=_NullContext))

        value = _trainer.run_perplexity(
            model, tokenizer, [json.loads(line) for line in suite.read_text().splitlines()]
        )

        assert value == pytest.approx(math.exp(2.0))
        assert model.forward_calls, "a labelled forward pass must have happened"
        assert (
            "cba" in model.forward_calls[0]
        ), "the expected output must be part of the scored text"

    def test_accepts_a_suite_path(self, tmp_path: Path, monkeypatch) -> None:
        import math

        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        tokenizer = _FakeTokenizer(pad_token="<pad>")
        model = _FakeEvalModel(tokenizer, {})
        model.forbid_generate = True
        model.losses = [1.0]
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=_NullContext))

        assert _trainer.run_perplexity(model, tokenizer, suite) == pytest.approx(math.exp(1.0))

    def test_run_eval_records_perplexity_when_asked(self, tmp_path: Path, monkeypatch) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        _tokenizer, model = _eval_fakes(monkeypatch, {"abc": "cba"})
        model.losses = [0.0]

        result = run_eval(str(adapter), suite_paths=[suite], perplexity=True)

        assert result["perplexity"] == pytest.approx(1.0)
        assert result["files"][0]["perplexity"] == pytest.approx(1.0)

    def test_perplexity_is_absent_by_default(self, tmp_path: Path, monkeypatch) -> None:
        adapter = _adapter_with_config(tmp_path)
        suite = _write_task_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        _eval_fakes(monkeypatch, {"abc": "cba"})
        result = run_eval(str(adapter), suite_paths=[suite])
        assert "perplexity" not in result


def test_run_eval_signature_keeps_backward_compatible_keywords() -> None:
    params = inspect.signature(run_eval).parameters
    for name in ("perplexity", "tool_call_family", "base_load_in_4bit"):
        assert params[name].default is None or params[name].default is False


# ---------------------------------------------------------------------------
# Training-time eval: holdout split -> eval_dataset + loss history (t7)
# ---------------------------------------------------------------------------


def _write_task_dataset(path: Path, count: int = 10) -> Path:
    """Write *count* task-schema rows — enough for a non-degenerate holdout split."""
    path.write_text(
        "".join(
            json.dumps({"task": "echo", "input": f"in-{i}", "expected_output": f"out-{i}"}) + "\n"
            for i in range(count)
        ),
        encoding="utf-8",
    )
    return path


class TestTrainingTimeEval:
    """``[eval] holdout_fraction > 0`` wires an eval_dataset into SFTTrainer."""

    def _config_with_eval(
        self, tmp_path: Path, *, fraction: float = 0.2, seed: int = 7, eval_steps: int = 5
    ) -> RunConfig:
        from sloth.tune.config import EvalConfig

        config = _config(tmp_path, method="lora")
        config.eval = EvalConfig(holdout_fraction=fraction, seed=seed, eval_steps=eval_steps)
        return config

    def _run(self, tmp_path, monkeypatch, config, log_history=None):
        backend, events = _make_fake_backend(log_history)
        monkeypatch.setattr(_trainer, "_load_backend", lambda: backend)
        fake_module, _, from_list_calls = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", fake_module)
        result = run_training(config, dry_run=False)
        return result, events, from_list_calls

    def test_sft_config_and_trainer_receive_eval_kwargs(self, tmp_path: Path, monkeypatch) -> None:
        config = self._config_with_eval(tmp_path)
        _write_task_dataset(Path(config.dataset))

        _, events, from_list_calls = self._run(tmp_path, monkeypatch, config)

        sft_kwargs = events["sft_config"][0]
        assert sft_kwargs["eval_strategy"] == "steps"
        assert sft_kwargs["eval_steps"] == 5
        trainer_kwargs = events["trainer"][0]
        assert "eval_dataset" in trainer_kwargs
        assert trainer_kwargs["eval_dataset"] is not None
        # train rows + holdout rows were each wrapped via Dataset.from_list
        assert len(from_list_calls) == 2
        assert len(from_list_calls[0]) == 8
        assert len(from_list_calls[1]) == 2

    def test_no_eval_kwargs_when_holdout_fraction_zero(self, tmp_path: Path, monkeypatch) -> None:
        config = self._config_with_eval(tmp_path, fraction=0.0)
        _write_task_dataset(Path(config.dataset))

        _, events, from_list_calls = self._run(tmp_path, monkeypatch, config)

        assert "eval_strategy" not in events["sft_config"][0]
        assert "eval_steps" not in events["sft_config"][0]
        assert "eval_dataset" not in events["trainer"][0]
        assert len(from_list_calls) == 1

    def test_no_eval_kwargs_when_eval_section_absent(self, tmp_path: Path, monkeypatch) -> None:
        config = _config(tmp_path, method="lora")
        assert config.eval is None
        _write_task_dataset(Path(config.dataset))

        _, events, _ = self._run(tmp_path, monkeypatch, config)

        assert "eval_strategy" not in events["sft_config"][0]
        assert "eval_dataset" not in events["trainer"][0]

    def test_eval_steps_zero_defaults_to_quarter_of_max_steps(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        config = self._config_with_eval(tmp_path, eval_steps=0)
        config.max_steps = 40
        _write_task_dataset(Path(config.dataset))

        result, events, _ = self._run(tmp_path, monkeypatch, config)

        assert events["sft_config"][0]["eval_steps"] == 10
        data = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert data["holdout"]["eval_steps"] == 10

    def test_eval_steps_defaults_to_at_least_one(self, tmp_path: Path, monkeypatch) -> None:
        config = self._config_with_eval(tmp_path, eval_steps=0)
        config.max_steps = 2
        _write_task_dataset(Path(config.dataset))

        _, events, _ = self._run(tmp_path, monkeypatch, config)

        assert events["sft_config"][0]["eval_steps"] == 1

    def test_metadata_records_holdout_and_loss_history(self, tmp_path: Path, monkeypatch) -> None:
        config = self._config_with_eval(tmp_path)
        _write_task_dataset(Path(config.dataset))
        log_history = [
            {"loss": 2.0, "step": 1},
            {"loss": 1.5, "step": 5},
            {"eval_loss": 1.7, "step": 5},
        ]

        result, _, _ = self._run(tmp_path, monkeypatch, config, log_history=log_history)

        data = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        holdout = data["holdout"]
        assert holdout["fraction"] == 0.2
        assert holdout["seed"] == 7
        assert holdout["eval_steps"] == 5
        assert holdout["train_count"] == 8
        assert holdout["holdout_count"] == 2
        assert holdout["train_path"].endswith("train.train.jsonl")
        assert holdout["holdout_path"].endswith("train.holdout.jsonl")
        assert data["loss_history"] == [
            {"step": 1, "train_loss": 2.0, "eval_loss": None},
            {"step": 5, "train_loss": 1.5, "eval_loss": 1.7},
        ]
        assert data["final_train_loss"] == 1.5
        assert data["final_eval_loss"] == 1.7
        # the recorded dataset stays the ORIGINAL path, not the split train file
        assert data["dataset"]["path"] == config.dataset

    def test_loss_history_recorded_without_a_holdout(self, tmp_path: Path, monkeypatch) -> None:
        """No ``[eval]`` section still records the training loss curve."""
        config = _config(tmp_path, method="lora")
        _write_task_dataset(Path(config.dataset))

        result, _, _ = self._run(
            tmp_path, monkeypatch, config, log_history=[{"loss": 0.5, "step": 1}]
        )

        data = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert data["loss_history"] == [{"step": 1, "train_loss": 0.5, "eval_loss": None}]
        assert data["final_train_loss"] == 0.5
        assert "holdout" not in data

    def test_chat_dataset_split_is_reproducible(self, tmp_path: Path, monkeypatch) -> None:
        """A chat source splits too (rows are rendered to task rows by split_holdout)."""
        config = self._config_with_eval(tmp_path)
        Path(config.dataset).write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"q-{i}"},
                            {"role": "assistant", "content": f"a-{i}"},
                        ]
                    }
                )
                + "\n"
                for i in range(10)
            ),
            encoding="utf-8",
        )

        result, _, from_list_calls = self._run(tmp_path, monkeypatch, config)

        assert len(from_list_calls) == 2
        data = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert data["holdout"]["train_count"] == 8
        assert data["holdout"]["holdout_count"] == 2

    def test_hf_dataset_skips_split_with_diagnostic(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """An ``hf:`` dataset has no local file to split — skip it, loudly."""
        config = self._config_with_eval(tmp_path)
        config.dataset = "hf:acme/demo:train"
        config.dataset_map = {"task": "task", "input": "input", "expected_output": "output"}
        monkeypatch.setattr(
            _trainer,
            "load_external_records",
            lambda spec, mapping: [
                {"task": "echo", "input": "a", "expected_output": "b"},
            ],
        )

        result, events, from_list_calls = self._run(tmp_path, monkeypatch, config)

        assert len(from_list_calls) == 1
        assert "eval_dataset" not in events["trainer"][0]
        captured = capsys.readouterr()
        assert "holdout" in captured.err.lower()
        data = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
        assert "holdout" not in data
