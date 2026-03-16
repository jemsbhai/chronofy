"""Decay function abstractions and registry."""

from chronofy.decay.base import DecayFunction
from chronofy.decay.exponential import ExponentialDecay

__all__ = ["DecayFunction", "ExponentialDecay"]
