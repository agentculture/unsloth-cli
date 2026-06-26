# unsloth-cli runs real LoRA/QLoRA fine-tuning on NVIDIA DGX Spark by orchestrating NVIDIA's official NGC PyTorch container, instead of carrying torch/unsloth as pip dependencies

> unsloth-cli runs real LoRA/QLoRA fine-tuning on NVIDIA DGX Spark by orchestrating NVIDIA's official NGC PyTorch container, instead of carrying torch/unsloth as pip dependencies

## Audience

- Spark/DGX operators and AgentCulture mesh agents who run the fine-tuning verbs on GB10 (Blackwell, aarch64); plus everyone who installs unsloth-cli only for the import-light introspection verbs

## Before → After

- Before: torch+unsloth are base [project].dependencies, so on aarch64 'uv sync' installs CPU-only torch (2.10.0+cpu); the real train/eval/export path aborts at model-load with 'cannot find any torch accelerator' and has never run on Spark — and the heavy deps also make a plain 'uv tool install' fail wherever the wheels don't resolve
- After: the real train/eval/export verbs execute the GPU work inside NVIDIA's official NGC container (nvcr.io/nvidia/pytorch:25.11-py3) where a Blackwell-compatible torch already lives, with the unsloth dep layer added via 'uv pip install --system --no-deps unsloth unsloth_zoo bitsandbytes' (uv, never pip); the same dataset+config reproduces a real run on Spark

## Why it matters

- fine-tuning is the repo's reason to exist; today it provably does not run on its target hardware. Orchestrating NVIDIA's official, supported path is the only sanctioned way to get a working Blackwell torch, and it restores universal installability of the introspection CLI

## Requirements

- host-side GPU-free preflight runs BEFORE any docker launch: dataset-schema + TOML-config + scope-guard validation happen in-process (pure stdlib), so a bad dataset/config/out-of-scope target fails fast without paying container startup
  - honesty: running 'sloth train' with an invalid dataset or out-of-scope model returns the user/scope error WITHOUT docker being invoked (no image pull, no container launch) — assertable in a test
- all docker orchestration lives in a new pure-stdlib sloth/tune/container.py (subprocess only, imports no torch); the existing _trainer.py real path becomes the in-container entrypoint invoked inside the container
  - honesty: container.py imports and its preflight runs on a machine with NO torch installed (import-light preserved); the in-container entrypoint is the only code path that imports torch/unsloth
- preflight failure (docker missing, image unpullable, or the nvidia GPU runtime unavailable) exits CliError code=2 with a remediation naming the NGC image + nvidia-container-toolkit requirement — never code=1 'file a bug'; and any in-container no-accelerator NotImplementedError is also mapped to code=2 with the NGC hint
  - honesty: on a host with no docker or no GPU runtime, 'sloth train' exits code 2 and the hint names nvcr.io/nvidia/pytorch:25.11-py3 — asserted by a test that stubs preflight
- a pytest 'gpu' marker is registered and a real-backend smoke test (tiny train -> eval -> export) carries @pytest.mark.gpu + skipif(no CUDA); it is skipped cleanly on CPU-only CI and exercises the real path when a CUDA device is present
  - honesty: the gpu-marked smoke test is collected-but-skipped under CPU-only pytest (suite stays green) and runs the real backend when torch.cuda.is_available()

## Honesty conditions

- a real LoRA/QLoRA run completes >0 training steps on GB10 via the orchestrated container — demonstrated on hardware, not just --dry-run
- both audiences are served: GPU users get a working real run; introspection-only installers get a successful import-light install with no torch pulled
- reproducible today: 'uv sync' on aarch64 yields a '+cpu' torch and the real path raises 'cannot find any torch accelerator'
- removing the heavy base deps restores 'uv tool install unsloth-cli' on arches where the torch/unsloth wheels don't resolve
- the build adds no torch/unsloth wheel vendoring and no second container image; the large-dense-model scope guard stays unchanged
- an operator runs train->eval->export on GB10 and gets adapter + metadata + eval scores — captured by the gpu smoke test
- asserted by a unit test: the no-accelerator / no-docker path returns exit code 2 with the NGC hint
- asserted: a clean install resolves without torch/unsloth and 'sloth whoami' runs
- the same dataset+config reproduces a real run; the container image + invocation are pinned and deterministic

## Success signals

- a real 'sloth train' LoRA/QLoRA run on GB10 completes >0 steps with loss recorded, writes the adapter + training metadata; 'sloth eval' then 'sloth export' complete on that adapter — the full loop runs on Spark
- with no CUDA torch/accelerator, 'sloth train' exits CliError code=2 with a remediation naming the NGC container path — not code=1 'file a bug'
- a fresh 'uv tool install unsloth-cli' / 'uv sync' succeeds on aarch64 WITHOUT pulling torch+unsloth, and the introspection verbs still run import-light

## Scope / boundaries

- not vendoring/building torch or unsloth wheels ourselves; not full fine-tuning of large dense models (LoRA/QLoRA only, scope guard unchanged); not a general container runtime — only the one pinned NGC image and the documented dependency layer

## Decisions

- remove torch+unsloth from [project].dependencies (reverse the #6 base-dep decision); the GPU stack comes from the container layer, never from pip resolution
- sloth train/eval/export auto-launch the NGC container from the host: docker run --gpus all (with NVIDIA's --ulimit memlock=-1 --ulimit stack=67108864) nvcr.io/nvidia/pytorch:25.11-py3, install the dep layer inside, run the job against a bind-mounted workdir, stream logs, collect outputs. The CLI never runs the GPU path in its own process
- in scope for this work: fix train passing list[dict] to SFTTrainer (wrap as datasets.Dataset.from_list) and fix eval loading a PEFT adapter dir via AutoModelForCausalLM (use PeftModel.from_pretrained(base, adapter) reading base_model_name_or_path from adapter_config.json)
- --dry-run stays fully host-side and GPU/docker-free (unchanged): imports no torch, launches no container, and now also prints the resolved docker image + command it WOULD run
- Pin NVIDIA's official Spark recipe but install with UV, never pip: image nvcr.io/nvidia/pytorch:25.11-py3; inside the container install the dep layer via 'uv pip install --system' (uv bootstrapped with the astral standalone installer) — transformers, peft, hf_transfer, datasets==4.3.0, trl==0.26.1, and 'unsloth unsloth_zoo bitsandbytes' with --no-deps. No 'pip install' runs anywhere
- unsloth-cli carries NO torch/unsloth dependency at all — not a base dep, not an optional extra; the unsloth install is dropped completely and the GPU stack exists ONLY inside the container. sloth reaches the container by bind-mounting the checkout and running 'python -m sloth' against the mounted source (no install step, pip-free)

## Hard questions

- risk: DGX Spark UMA can OOM even within memory capacity; mitigation: document NVIDIA's 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"' flush and keep conservative batch_size defaults
- risk: installing via 'uv pip install --system' diverges from NVIDIA's pip-tested recipe; must be validated on GB10 hardware that uv resolves the identical working set (unsloth --no-deps + unsloth_zoo + bitsandbytes + trl==0.26.1 against the container's preinstalled torch) and that bootstrapping uv inside the container is reliable

## Open / follow-up

- image + dep pins will age: nvcr.io/nvidia/pytorch:25.11-py3, datasets==4.3.0, trl==0.26.1 need a documented bump/verify path
