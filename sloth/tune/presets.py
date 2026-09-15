"""Named ``target_modules`` presets for fine-tune run-configs.

Pure stdlib — no torch/unsloth imports. Presets are resolved by *name* at
``load_config`` time via ``preset:<name>`` and the resolved regex string is
what the trainer eventually hands to PEFT/Unsloth.

Why a regex string and not a name list: a live probe against LFM2.5-1.2B
showed Unsloth folds a plain list of target-module names into a tag regex
that does not include ``conv`` — an 8-name list adapted 0 conv modules. Only
the regex *string* form reached all 92 LFM2.5-1.2B target modules. Presets
must therefore be regex strings, not lists.
"""

from __future__ import annotations

PRESETS: dict[str, str] = {
    "lfm2": (
        r"model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|"
        r"conv\.(in|out)_proj|feed_forward\.w[123])"
    ),
}
"""Name -> regex string. ``resolve_target_modules`` looks values up here."""


def resolve_target_modules(value: list[str] | str | None) -> list[str] | str | None:
    """Resolve a ``target_modules`` config value to its final form.

    - ``None`` passes through unchanged.
    - A ``list[str]`` passes through unchanged.
    - A string of the form ``"preset:<name>"`` resolves to ``PRESETS[<name>]``.
    - Any other string (a literal regex) passes through unchanged.

    Unknown preset names are the caller's responsibility to reject (see
    ``sloth.tune.config.load_config``, which validates at load time); this
    function itself simply returns the value unresolved if the name is not
    known, so callers that need a hard failure must check membership first.
    """
    if isinstance(value, str) and value.startswith("preset:"):
        name = value[len("preset:") :]
        return PRESETS.get(name, value)
    return value
