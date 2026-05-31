"""Markdown catalog for ``unsloth-cli explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("unsloth-cli",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# unsloth-cli

A clonable template for AgentCulture mesh agents. It carries an agent-first CLI
(cited from the teken `python-cli` reference), a mesh identity (`culture.yaml` +
`CLAUDE.md`), the canonical guildmaster skill kit under `.claude/skills/`, and a
buildable/deployable package baseline. Clone it, rename the package, edit
`culture.yaml`, and you have a new agent.

## Verbs

- `unsloth-cli whoami` — identity probe from `culture.yaml`.
- `unsloth-cli learn` — structured self-teaching prompt.
- `unsloth-cli explain <path>` — markdown docs for any noun/verb.
- `unsloth-cli overview` — descriptive snapshot of the agent.
- `unsloth-cli doctor` — check the agent-identity invariants.
- `unsloth-cli cli overview` — describe the CLI surface.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `unsloth-cli explain whoami`
- `unsloth-cli explain doctor`
"""

_WHOAMI = """\
# unsloth-cli whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    unsloth-cli whoami
    unsloth-cli whoami --json
"""

_LEARN = """\
# unsloth-cli learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    unsloth-cli learn
    unsloth-cli learn --json
"""

_EXPLAIN = """\
# unsloth-cli explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    unsloth-cli explain unsloth-cli
    unsloth-cli explain whoami
    unsloth-cli explain --json <path>
"""

_OVERVIEW = """\
# unsloth-cli overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    unsloth-cli overview
    unsloth-cli overview --json
"""

_DOCTOR = """\
# unsloth-cli doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`claude` → `CLAUDE.md`), plus a
skills-present check. Exits 1 when unhealthy.

## Usage

    unsloth-cli doctor
    unsloth-cli doctor --json
"""

_CLI = """\
# unsloth-cli cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    unsloth-cli cli overview
    unsloth-cli cli overview --json
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("unsloth-cli",): _ROOT,
    # The console-script / package name is `sloth` (the dist name is
    # `unsloth-cli`). The agent-first rubric's `explain_self` check runs
    # `explain <script-name>`, i.e. `explain sloth`, so alias it to the root.
    ("sloth",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
}
