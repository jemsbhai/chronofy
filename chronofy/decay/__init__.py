"""Decay function abstractions and implementations.

Chronofy provides a pluggable menu of temporal decay backends:

  - ExponentialDecay: β-parameterized, Bayesian-optimal under OU dynamics
  - HalfLifeDecay:   half-life parameterized (equivalent to exponential)
  - LinearDecay:     simplest model, hard expiry at 1/α
  - PowerLawDecay:   heavy-tailed, from citation/memory literature
  - WeibullDecay:    generalized exponential, from survival analysis
"""

from chronofy.decay.base import DecayFunction
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.half_life import HalfLifeDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.power_law import PowerLawDecay
from chronofy.decay.weibull import WeibullDecay

__all__ = [
    "DecayFunction",
    "ExponentialDecay",
    "HalfLifeDecay",
    "LinearDecay",
    "PowerLawDecay",
    "WeibullDecay",
]
