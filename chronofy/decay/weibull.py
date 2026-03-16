"""Weibull decay function.

Generalized exponential from reliability engineering / survival analysis:

    V(Δt) = q_e · exp(-(Δt / λ_j)^k_j)

where λ_j is the scale parameter and k_j is the shape parameter.

Special cases:
  - k = 1: recovers standard exponential decay (with β = 1/λ)
  - k > 1: accelerating obsolescence (wear-out regime)
  - k < 1: decelerating obsolescence (infant mortality / burn-in)

This directly implements the hazard-based surrogate from the ASEV analysis:
the Weibull hazard h(a) = (k/λ)(a/λ)^(k-1) gives a Weibull survival
function, which is exactly the decay applied here.
"""

from __future__ import annotations

import math
from datetime import datetime

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


class WeibullDecay(DecayFunction):
    """Weibull temporal decay: V(e, T_q) = q_e · exp(-(Δt/λ)^k).

    Args:
        scale: Mapping from fact_type → scale parameter λ (characteristic life).
        shape: Mapping from fact_type → shape parameter k.
        default_scale: Fallback λ for unknown fact types.
        default_shape: Fallback k for unknown fact types.
        time_unit: Unit for Δt computation. One of "days", "hours", "seconds".
    """

    def __init__(
        self,
        scale: dict[str, float] | None = None,
        shape: dict[str, float] | None = None,
        default_scale: float = 7.0,
        default_shape: float = 1.0,
        time_unit: str = "days",
    ) -> None:
        self._scale = scale or {}
        self._shape = shape or {}
        self._default_scale = default_scale
        self._default_shape = default_shape
        self._time_divisor = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[time_unit]

    def _get_scale(self, fact_type: str) -> float:
        return self._scale.get(fact_type, self._default_scale)

    def _get_shape(self, fact_type: str) -> float:
        return self._shape.get(fact_type, self._default_shape)

    def _age_in_units(self, fact: TemporalFact, query_time: datetime) -> float:
        delta_seconds = (query_time - fact.timestamp).total_seconds()
        return max(delta_seconds / self._time_divisor, 0.0)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        lam = self._get_scale(fact.fact_type)
        k = self._get_shape(fact.fact_type)
        age = self._age_in_units(fact, query_time)
        return fact.source_quality * math.exp(-((age / lam) ** k))

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Return equivalent β = 1/λ only when k = 1 (exponential case)."""
        k = self._get_shape(fact_type)
        if abs(k - 1.0) < 1e-9:
            return 1.0 / self._get_scale(fact_type)
        return None

    def __repr__(self) -> str:
        types = ", ".join(
            f"{k}(λ={self._get_scale(k):.1f},k={self._get_shape(k):.1f})"
            for k in sorted(set(list(self._scale.keys()) + list(self._shape.keys())))
        )
        return f"WeibullDecay({types})"
