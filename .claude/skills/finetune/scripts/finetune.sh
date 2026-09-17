#!/usr/bin/env bash
# finetune.sh — drive the validate → train → eval → export loop (/finetune skill).
#
# Portable wrapper around unsloth-cli's fine-tuning verbs. It resolves the
# `sloth` CLI (`SLOTH_BIN` override first, then `uv run --project <dir> sloth`
# from a walked-up unsloth-cli checkout, then the installed console script on
# `PATH`), orchestrates the loop, and propagates every exit code and
# `error:`/`hint:` line verbatim.
#
# Usage:
#   finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--suite ...]
#                    [--dry-run] [--json] [--batch-size N] [--perplexity]
#                    [--export-format FMT] [--quant LIST] [--base REF]
#   finetune.sh <verb> [args...]   # thin pass-through to `sloth <verb>`
#   finetune.sh help

set -euo pipefail

# ── resolve the sloth CLI (SLOTH_BIN override → dev checkout → PATH) ──────────
SLOTH=()
resolve_sloth() {
    if [ -n "${SLOTH_BIN:-}" ]; then
        SLOTH=("$SLOTH_BIN")
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
    if command -v sloth >/dev/null 2>&1; then
        SLOTH=(sloth)
        return 0
    fi
    printf 'error: sloth CLI not found.\n' >&2
    printf 'hint: install it with: uv tool install unsloth-cli; the console script is sloth.\n' >&2
    return 1
}

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
finetune.sh — drive the validate → train → eval → export loop for unsloth-cli.

Usage:
  finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--suite ...]
                   [--dry-run] [--json] [--batch-size <n>] [--perplexity]
                   [--export-format <fmt>] [--quant <list>] [--base <ref>]
  finetune.sh <verb> [args...]
  finetune.sh help

Commands:
  run          Orchestrated end-to-end loop (see below).
  help         Show this message.
  <verb> ...   Thin pass-through to `sloth <verb>` — use for individual steps
               or any other sloth verb (whoami, doctor, explain, ...).

run flags:
  --config <run.toml>    TOML describing model, dataset, output, method. (required)
  --suite <suite.jsonl | dir>  Task-schema JSONL eval suite. Repeatable — pass
                          --suite more than once to score several named suites
                          in the eval step. (required, at least one)
  --dry-run              Validate + resolve the plan only; no GPU, no torch import.
  --json                 Forward --json to every sloth call (machine-readable output).
  --batch-size <n>       Forwarded to `sloth eval --batch-size <n>` (generation
                          batch size for the eval step).
  --perplexity           Forwarded to `sloth eval --perplexity` (also compute
                          held-out perplexity/loss during eval).
  --export-format <fmt>  Format for the export step: safetensors, merged-16bit,
                          merged-4bit, gguf, awq, nvfp4 (default: safetensors).
                          Forwarded to `sloth export --format <fmt>`.
  --export-output <dir>  Destination for a container-format export (default:
                          <adapter>-<format> next to the adapter; safetensors stays in place).
  --quant <list>         Comma-separated ggml quantizations for --export-format gguf
                          (e.g. q4_k_m,q8_0). Forwarded to `sloth export --quant <list>`.
  --base <ref>           Adapter-vs-base-model comparison ref (a Hugging Face
                          repo id or a local model directory). When given, adds
                          step 5: `sloth compare --base <ref> <adapter_dir>`.

Loop steps (run without --dry-run):
  step 1/N  sloth train --config <c> --dry-run              validate + plan (GPU-free, always)
  step 2/N  sloth train --config <c>                        real training (GPU required)
  step 3/N  sloth eval  --adapter <out> --suite ... [...]   eval the adapter
  step 4/N  sloth export --adapter <out> --format <fmt>     export (safetensors: host; else container)
  step 5/5  sloth compare --base <ref> <out> [--config <c>] adapter-vs-base-model
                                                             comparison, gated on [eval.thresholds]
                                                             (only when --base is given; N is 5)

N is 5 when --base is given, 4 otherwise (step 5 is skipped, not printed, without --base).

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

  # Score several named suites, with perplexity and a larger eval batch size:
  finetune.sh run --config run.toml --suite regression.jsonl --suite toolcalls.jsonl \\
      --batch-size 16 --perplexity

  # Full run exporting a gguf instead of safetensors:
  finetune.sh run --config run.toml --suite eval.jsonl --export-format gguf --quant q4_k_m

  # Full run, then compare the trained adapter against its base model (step 5):
  finetune.sh run --config run.toml --suite eval.jsonl --base unsloth/Qwen3-4B --json

  # Drive eval alone (pass-through):
  finetune.sh eval --adapter adapters/my-lora --suite eval.jsonl --json
EOF
}

# ── orchestrated loop ──────────────────────────────────────────────────────────
cmd_run() {
    local config="" dry_run=false json_flag=false
    local export_format="safetensors" quant="" export_output=""
    local batch_size="" perplexity=false base=""
    local -a suites=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --config)
                if [ $# -lt 2 ]; then
                    printf 'error: --config requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--dry-run] [--json]\n' >&2
                    exit 1
                fi
                config="$2"; shift 2 ;;
            --suite)
                if [ $# -lt 2 ]; then
                    printf 'error: --suite requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--dry-run] [--json]\n' >&2
                    exit 1
                fi
                suites+=("$2"); shift 2 ;;
            --batch-size)
                if [ $# -lt 2 ]; then
                    printf 'error: --batch-size requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> --batch-size <n>\n' >&2
                    exit 1
                fi
                batch_size="$2"; shift 2 ;;
            --perplexity)
                perplexity=true; shift ;;
            --base)
                if [ $# -lt 2 ]; then
                    printf 'error: --base requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> --base <hf-id-or-dir>\n' >&2
                    exit 1
                fi
                base="$2"; shift 2 ;;
            --export-format)
                if [ $# -lt 2 ]; then
                    printf 'error: --export-format requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> --export-format <fmt>\n' >&2
                    exit 1
                fi
                export_format="$2"; shift 2 ;;
            --export-output)
                if [ -z "${2:-}" ]; then
                    printf 'error: --export-output requires an argument.\n' >&2
                    printf 'hint: finetune.sh run ... --export-format gguf --export-output <dir>\n' >&2
                    exit 1
                fi
                export_output="$2"; shift 2 ;;
            --quant)
                if [ $# -lt 2 ]; then
                    printf 'error: --quant requires an argument.\n' >&2
                    printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> --export-format gguf --quant <list>\n' >&2
                    exit 1
                fi
                quant="$2"; shift 2 ;;
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
        printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--dry-run] [--json]\n' >&2
        exit 1
    fi
    if [ "${#suites[@]}" -eq 0 ]; then
        printf 'error: --suite is required.\n' >&2
        printf 'hint: finetune.sh run --config <run.toml> --suite <suite.jsonl | dir> [--dry-run] [--json]\n' >&2
        exit 1
    fi

    # Build the optional --json array once; pass it to every sloth call.
    local json_arg=()
    if $json_flag; then
        json_arg=(--json)
    fi

    # Build the repeatable --suite array (one --suite flag per entry) once;
    # forwarded to the eval step.
    local suite_arg=()
    local s
    for s in "${suites[@]}"; do
        suite_arg+=(--suite "$s")
    done

    # Total step count: 5 when --base is given (adds the compare step), 4 otherwise.
    local total_steps=4
    if [ -n "$base" ]; then
        total_steps=5
    fi

    # --dry-run: step 1 only — validate + resolve plan, no GPU.
    if $dry_run; then
        printf 'step 1/1  validate + plan (dry-run, GPU-free)\n' >&2
        "${SLOTH[@]}" train --config "$config" --dry-run "${json_arg[@]}"
        return $?
    fi

    # Real run — step 1: dry-run first (validate + capture plan JSON for adapter dir).
    printf 'step 1/%d  validate + plan (dry-run)\n' "$total_steps" >&2
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
    printf 'step 2/%d  train (real run — GPU + tuning stack required)\n' "$total_steps" >&2
    "${SLOTH[@]}" train --config "$config" "${json_arg[@]}" || exit $?

    # Step 3: Eval. --suite is repeatable (suite_arg holds one --suite per entry);
    # --batch-size and --perplexity are forwarded only when given.
    local eval_arg=()
    if [ -n "$batch_size" ]; then
        eval_arg+=(--batch-size "$batch_size")
    fi
    if $perplexity; then
        eval_arg+=(--perplexity)
    fi
    printf 'step 3/%d  eval\n' "$total_steps" >&2
    "${SLOTH[@]}" eval --adapter "$adapter_dir" "${suite_arg[@]}" \
        "${eval_arg[@]}" "${json_arg[@]}" || exit $?

    # Step 4: Export. --export-format defaults to safetensors (host, no GPU);
    # any other format runs inside the NGC container. --quant is forwarded
    # only when given — sloth export applies it to --format gguf only.
    local export_arg=()
    if [ -n "$quant" ]; then
        export_arg=(--quant "$quant")
    fi
    # Container formats write a NEW model directory and require --output (sloth
    # export refuses to guess). Default: <adapter>-<format> next to the adapter,
    # overridable with --export-output. safetensors keeps its in-place default.
    if [ "$export_format" != "safetensors" ]; then
        if [ -z "$export_output" ]; then
            export_output="${adapter_dir%/}-${export_format}"
        fi
        export_arg+=(--output "$export_output")
    elif [ -n "$export_output" ]; then
        export_arg+=(--output "$export_output")
    fi
    printf 'step 4/%d  export → %s\n' "$total_steps" "$export_format" >&2
    "${SLOTH[@]}" export --adapter "$adapter_dir" --format "$export_format" \
        "${export_arg[@]}" "${json_arg[@]}" || exit $?

    # Step 5: adapter-vs-base-model compare (only when --base is given).
    if [ -n "$base" ]; then
        local compare_arg=()
        if [ -n "$config" ]; then
            compare_arg+=(--config "$config")
        fi
        printf 'step 5/5  compare --base %s\n' "$base" >&2
        "${SLOTH[@]}" compare --base "$base" "$adapter_dir" \
            "${compare_arg[@]}" "${json_arg[@]}" || exit $?
    fi

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
