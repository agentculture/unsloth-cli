#!/usr/bin/env bash
# unsloth-cli-guide — surface unsloth-cli's own live teaching output.
#
# Thin wrapper over the CLI's introspection verbs so the guide shows *real*,
# current output (not a stale copy). CLI resolution is portable: an installed
# `sloth` on PATH, else `uv run sloth` from a checkout.
#
# Usage: guide.sh <topic> [args...]
#   overview            sloth overview            (descriptive snapshot)
#   learn               sloth learn               (the self-teaching prompt)
#   verbs               sloth cli overview        (the CLI surface itself)
#   explain <path...>   sloth explain <path...>   (markdown docs for a verb/noun)
#   doctor              sloth doctor              (agent-identity invariants)
#   finetune            the fine-tuning quickstart + `sloth train --help`
#
# Every underlying verb supports --json; pass it through, e.g.
#   guide.sh learn --json
set -uo pipefail

# Resolve the CLI: prefer an installed `sloth`, else `uv run sloth` from a checkout.
if command -v sloth >/dev/null 2>&1; then
  SLOTH=(sloth)
elif command -v uv >/dev/null 2>&1; then
  SLOTH=(uv run sloth)
else
  echo "error: could not find 'sloth' on PATH or 'uv' to run it" >&2
  echo "hint: install with 'uv tool install unsloth-cli', or run from a checkout with uv installed" >&2
  exit 2
fi

topic="${1:-overview}"
shift || true

case "$topic" in
  overview) "${SLOTH[@]}" overview "$@" ;;
  learn)    "${SLOTH[@]}" learn "$@" ;;
  verbs)    "${SLOTH[@]}" cli overview "$@" ;;
  explain)  "${SLOTH[@]}" explain "$@" ;;
  doctor)   "${SLOTH[@]}" doctor "$@" ;;
  finetune)
    cat <<'EOF'
Fine-tuning quickstart (LoRA/QLoRA adapters; GPU work runs in the NGC container):

  # 1. Plan only — GPU-free, runs anywhere; prints the resolved plan + docker command
  sloth train --config examples/qlora-smoke.toml --dry-run

  # 2. Real run — LoRA/QLoRA adapter job inside the NGC container
  sloth train --config examples/qlora-smoke.toml

  # 3. Evaluate the adapter against a local task-schema suite (no network)
  sloth eval --adapter runs/qlora-smoke --suite examples/eval-suite.jsonl

  # 4. Export to a standard PEFT/safetensors layout
  sloth export --adapter runs/qlora-smoke --output runs/qlora-smoke-export

The /finetune skill runs steps 1–4 as one loop. See docs/fine-tuning.md,
docs/dgx-spark.md (Spark prereqs + gotchas), and docs/benchmarks.md.

`sloth train --help`:
EOF
    "${SLOTH[@]}" train --help 2>&1 || true
    ;;
  *)
    echo "unknown topic: $topic" >&2
    echo "topics: overview | learn | verbs | explain <path...> | doctor | finetune" >&2
    exit 1
    ;;
esac
