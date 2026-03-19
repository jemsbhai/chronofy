"""Subjective Logic extensions for Chronofy.

Provides opinion-based temporal validity using Jøsang's Subjective Logic
framework, enriching the scalar decay model with full (b, d, u, a)
opinions that distinguish evidential direction from epistemic uncertainty.

Requires: pip install chronofy[sl]  (depends on jsonld-ex)

Core exports:
- OpinionDecayFunction: DecayFunction that returns SL Opinions
- OpinionConfig: per-fact-type configuration (half_life, base_rate, etc.)
"""

try:
    from jsonld_ex.confidence_algebra import Opinion  # noqa: F401

    from chronofy.sl.opinion_decay import OpinionConfig, OpinionDecayFunction

    __all__ = [
        "OpinionDecayFunction",
        "OpinionConfig",
        "Opinion",
    ]

except ImportError as e:
    raise ImportError(
        "chronofy.sl requires jsonld-ex. Install via: pip install chronofy[sl]"
    ) from e
