"""Chronofy: Temporal validity framework for time-aware AI systems."""

__version__ = "0.1.8"

from chronofy.models import TemporalFact, ReasoningStep, ReasoningTrace
from chronofy.decay.base import DecayFunction
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.half_life import HalfLifeDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.power_law import PowerLawDecay
from chronofy.decay.weibull import WeibullDecay
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.embedding.base import TemporalEncoder
from chronofy.embedding.sinusoidal import SinusoidalEncoder
from chronofy.embedding.embedder import TemporalEmbedder
from chronofy.verification.stl import STLVerifier, STLResult
from chronofy.pipeline import ChronofyPipeline
from chronofy.analysis.corpus_stats import AgeStats, CorpusStats, ValidityStats
from chronofy.plugins import (
    PluginValidationError,
    validate_decay_function,
    validate_estimation_method,
    validate_scoring_strategy,
)
from chronofy.scoring.temporal_scorer import (
    ScoringStrategy,
    ScoredFact,
    MultiplicativeScoring,
    HarmonicScoring,
    WeightedBlendScoring,
    PowerScoring,
    TemporalScorer,
)
from chronofy.comparison.decay_comparison import (
    DecayComparison,
    DecayComparisonResult,
)
from chronofy.analysis.beta_estimator import (
    BetaEstimateResult,
    BetaEstimator,
    EstimationMethod,
    MLEBernoulli,
    MomentMatching,
    EnsembleMethod,
)

__all__ = [
    "TemporalFact",
    "ReasoningStep",
    "ReasoningTrace",
    "DecayFunction",
    "ExponentialDecay",
    "HalfLifeDecay",
    "LinearDecay",
    "PowerLawDecay",
    "WeibullDecay",
    "TemporalEncoder",
    "SinusoidalEncoder",
    "TemporalEmbedder",
    "EpistemicFilter",
    "STLVerifier",
    "STLResult",
    "ChronofyPipeline",
    "CorpusStats",
    "AgeStats",
    "ValidityStats",
    "BetaEstimateResult",
    "BetaEstimator",
    "EstimationMethod",
    "MLEBernoulli",
    "MomentMatching",
    "EnsembleMethod",
    "PluginValidationError",
    "validate_decay_function",
    "validate_estimation_method",
    "validate_scoring_strategy",
    "ScoringStrategy",
    "ScoredFact",
    "MultiplicativeScoring",
    "HarmonicScoring",
    "WeightedBlendScoring",
    "PowerScoring",
    "TemporalScorer",
    "DecayComparison",
    "DecayComparisonResult",
]
