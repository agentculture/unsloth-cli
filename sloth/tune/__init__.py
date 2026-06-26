"""LoRA/QLoRA adapter fine-tuning core for unsloth-cli (the ``sloth`` tuning verbs).

This package holds the dependency-free core — dataset validation, run config,
training metadata, and the model-scope guard. The only module that touches
torch/unsloth is :mod:`sloth.tune._trainer`, which **lazy-imports** the heavy
stack inside its run function, never at module top level, so importing the
``sloth`` package stays torch-free and the introspection verbs keep fast startup.
"""
