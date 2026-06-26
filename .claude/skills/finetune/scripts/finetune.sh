#!/usr/bin/env bash
# finetune.sh — drive the validate → train → eval → export loop (/finetune skill).
#
# Portable wrapper around unsloth-cli's fine-tuning verbs. It resolves the
# `sloth` CLI (installed console script first, then `uv run sloth` from the
# repo checkout), orchestrates the loop, and propagates every exit code and
# `error:`/`hint:` line verbatim.
#
# Usage:
#   finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]
#   finetune.sh <verb> [args...]   # thin pass-through to `sloth <verb>`
#   finetune.sh help

set -euo pipefail

# ── resolve the sloth CLI (installed tool first, then dev checkout) ───────────
SLOTH=()
resolve_sloth() {
    if command -v sloth >/dev/null 2>&1; then
        SLOTH=(sloth)
        return 0
    fi
    # Dev fallback: inside an unsloth-cli checkout, run via uv.
    local dir
    dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    while [ -n "$dir" ] && [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ] \
            && grep -q '^name = "unsloth-cli"' "$dir/pyproject.toml" 2>/dev/null; then
            if command -v uv >/dev/null 2>&1; then
                SLOTH=(uv run --project "$dir" sloth)
                return 0
            fi
            break
        fi
        dir=$(dirname "$dir")
    done
    printf 'error: sloth CLI not found.\n' >&2
    printf 'hint: install it with: uv tool install unsloth-cli; the console script is sloth.\n' >&2
    return 1
}

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
finetune.sh — drive the validate → train → eval → export loop for unsloth-cli.

Usage:
  finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]
  finetune.sh <verb> [args...]
  finetune.sh help

Commands:
  run          Orchestrated end-to-end loop (see below).
  help         Show this message.
  <verb> ...   Thin pass-through to `sloth <verb>` — use for individual steps
               or any other sloth verb (whoami, doctor, explain, ...).

run flags:
  --config <run.toml>    TOML describing model, dataset, output, method. (required)
  --suite <suite.jsonl>  Task-schema JSONL eval suite. (required)
  --dry-run              Validate + resolve the plan only; no GPU, no torch import.
  --json                 Forward --json to every sloth call (machine-readable output).

Loop steps (run without --dry-run):
  step 1/4  sloth train --config <c> --dry-run   validate + plan (GPU-free, always)
  step 2/4  sloth train --config <c>             real training (GPU required)
  step 3/4  sloth eval  --adapter <out> --suite  eval the adapter
  step 4/4  sloth export --adapter <out>         export to safetensors

With --dry-run: only step 1 runs; exits 0 on success, surfacing the resolved plan.
The loop stops on the first non-zero exit code, forwarding the CLI's error:/hint:.

Requirements:
  dry-run  — stdlib Python 3.11+, no GPU, no torch.
  real run — the tuning stack ships with unsloth-cli (uv tool install unsloth-cli) + a CUDA GPU.

Examples:
  # Smoke-check a config without a GPU:
  finetune.sh run --config run.toml --suite eval.jsonl --dry-run

  # Full end-to-end run with JSON output:
  finetune.sh run --config run.toml --suite eval.jsonl --json

  # Drive eval alone (pass-through):
  finetune.sh eval --adapter adapters/my-lora --suite eval.jsonl --json
EOF
}

# ── orchestrated loop ──────────────────────────────────────────────────────────
cmd_run() {
    local config="" suite="" dry_run=false json_flag=false

    while [ $# -gt 0 ]; do
        case "$1" in
            --config)
                if [ $# -lt 2 ]; then
                    printf 'error: --config requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]\n' >&2
                    exit 1
                fi
                config="$2"; shift 2 ;;
            --suite)
                if [ $# -lt 2 ]; then
                    printf 'error: --suite requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]\n' >&2
                    exit 1
                fi
                suite="$2"; shift 2 ;;
            --dry-run)
                dry_run=true; shift ;;
            --json)
                json_flag=true; shift ;;
            -h | --help)
                usage; exit 0 ;;
            *)
                printf 'error: unknown flag for run: %s\n' "$1" >&2
                printf 'hint: run `finetune.sh help` for usage.\n' >&2
                exit 1 ;;
        esac
    done

    if [ -z "$config" ]; then
        printf 'error: --config is required.\n' >&2
        printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]\n' >&2
        exit 1
    fi
    if [ -z "$suite" ]; then
        printf 'error: --suite is required.\n' >&2
        printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl> [--dry-run] [--json]\n' >&2
        exit 1
    fi

    # Build the optional --json array once; pass it to every sloth call.
    local json_arg=()
    if $json_flag; then
        json_arg=(--json)
    fi

    # --dry-run: step 1 only — validate + resolve plan, no GPU.
    if $dry_run; then
        printf 'step 1/1  validate + plan (dry-run, GPU-free)\n' >&2
        "${SLOTH[@]}" train --config "$config" --dry-run "${json_arg[@]}"
        return $?
    fi

    # Real run — step 1: dry-run first (validate + capture plan JSON for adapter dir).
    printf 'step 1/4  validate + plan (dry-run)\n' >&2
    local plan_json
    plan_json=$("${SLOTH[@]}" train --config "$config" --dry-run --json) || {
        local rc=$?
        # error:/hint: already printed to stderr by the CLI.
        exit $rc
    }

    # Extract the adapter output directory from the plan JSON via python3.
    # The plan's "output" field matches the [run] output key in the TOML and
    # is where the trainer writes the adapter (and where eval/export expect it).
    local adapter_dir
    adapter_dir=$(printf '%s' "$plan_json" | python3 -c \
        "import sys, json; d = json.load(sys.stdin); print(d['output'])" 2>/dev/null) || {
        printf 'error: could not extract "output" from the training plan JSON.\n' >&2
        printf 'hint: check that `sloth train --dry-run --json` emits a valid JSON plan with an "output" key.\n' >&2
        exit 1
    }

    if [ -z "$adapter_dir" ]; then
        printf 'error: training plan JSON has an empty "output" field.\n' >&2
        printf 'hint: set `output = "<path>"` in the [run] section of %s.\n' "$config" >&2
        exit 1
    fi

    # Step 2: Real training (GPU required).
    printf 'step 2/4  train (real run — GPU + tuning stack required)\n' >&2
    "${SLOTH[@]}" train --config "$config" "${json_arg[@]}" || exit $?

    # Step 3: Eval.
    printf 'step 3/4  eval\n' >&2
    "${SLOTH[@]}" eval --adapter "$adapter_dir" --suite "$suite" "${json_arg[@]}" || exit $?

    # Step 4: Export to safetensors.
    printf 'step 4/4  export → safetensors\n' >&2
    "${SLOTH[@]}" export --adapter "$adapter_dir" --format safetensors "${json_arg[@]}" || exit $?

    printf 'done: adapter at %s\n' "$adapter_dir" >&2
}

# ── main dispatch ──────────────────────────────────────────────────────────────
case "${1:-}" in
    help | --help | -h)
        usage
        exit 0
        ;;
    "")
        printf 'error: no command given.\n' >&2
        printf 'hint: run `finetune.sh help` for usage, or `finetune.sh run --config <toml> --suite <jsonl> --dry-run` to validate a config.\n' >&2
        exit 1
        ;;
    run)
        shift
        resolve_sloth || exit 2
        cmd_run "$@"
        ;;
    *)
        # Thin pass-through: `finetune.sh train ...` → `sloth train ...` etc.
        resolve_sloth || exit 2
        "${SLOTH[@]}" "$@"
        ;;
esac
