"""Chronofy: Temporal validity framework for time-aware AI systems."""

__version__ = "0.1.2"

from chronofy.models import TemporalFact, ReasoningStep, ReasoningTrace
from chronofy.decay.base import DecayFunction
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.half_life import HalfLifeDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.power_law import PowerLawDecay
from chronofy.decay.weibull import WeibullDecay
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.verification.stl import STLVerifier, STLResult
from chronofy.pipeline import ChronofyPipeline
from chronofy.analysis.corpus_stats import AgeStats, CorpusStats, ValidityStats

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
    "EpistemicFilter",
    "STLVerifier",
    "STLResult",
    "ChronofyPipeline",
    "CorpusStats",
    "AgeStats",
    "ValidityStats",
]
