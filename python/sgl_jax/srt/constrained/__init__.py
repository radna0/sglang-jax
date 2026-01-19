"""Constrained decoding with grammar backends."""

from sgl_jax.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)

__all__ = [
    "BaseGrammarBackend",
    "BaseGrammarObject",
]

# Optional dependency: llguidance-backed constrained decoding.
try:  # pragma: no cover
    from sgl_jax.srt.constrained.llguidance_backend import GuidanceBackend, GuidanceGrammar

    __all__.extend(["GuidanceBackend", "GuidanceGrammar"])
except ModuleNotFoundError:
    # Normal deployments that don't use constrained decoding should not require
    # llguidance to be installed.
    pass
