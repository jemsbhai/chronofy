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

    from chronofy.sl.conflict import ConflictDetector, ConflictReport
    from chronofy.sl.fusion import FusionReport, TemporalEvidenceFusion
    from chronofy.sl.opinion_decay import OpinionConfig, OpinionDecayFunction
    from chronofy.sl.opinion_scorer import (
        OpinionScoredFact,
        OpinionScorer,
        OpinionScoringStrategy,
        ProjectedMultiplicative,
        UncertaintyAwareBlend,
        UncertaintyPenalized,
    )
    from chronofy.sl.stl_opinion import OpinionSTLResult, OpinionSTLVerifier
    from chronofy.sl.trust import TrustEntry, TrustProfile, TrustWeightedDecay
    from chronofy.sl.pipeline import GroupedFusionResult, SLPipeline, SLPipelineResult
    from chronofy.sl.opinion_graph import OpinionRuleGraph

    __all__ = [
        "OpinionDecayFunction",
        "OpinionConfig",
        "Opinion",
        "ConflictDetector",
        "ConflictReport",
        "TemporalEvidenceFusion",
        "FusionReport",
        "OpinionScorer",
        "OpinionScoredFact",
        "OpinionScoringStrategy",
        "ProjectedMultiplicative",
        "UncertaintyPenalized",
        "UncertaintyAwareBlend",
        "OpinionSTLVerifier",
        "OpinionSTLResult",
        "TrustEntry",
        "TrustProfile",
        "TrustWeightedDecay",
        "SLPipeline",
        "SLPipelineResult",
        "GroupedFusionResult",
        "OpinionRuleGraph",
    ]

except ImportError as e:
    raise ImportError(
        "chronofy.sl requires jsonld-ex. Install via: pip install chronofy[sl]"
    ) from e
