"""Half-life decay function.

Standard model from pharmacokinetics and radiometric dating:

    V(Δt) = q_e · 0.5^(Δt / h_j)

where h_j is the half-life for fact type j — the time for validity
to drop to 50%.

Mathematically equivalent to exponential decay with β = ln(2)/h,
but parameterized in terms of half-life which is more intuitive
in many domains (e.g., "medication effects halve every 6 hours").
"""

from __future__ import annotations

import math
from datetime import datetime

from chronofy.decay._validation import (
    validate_parameter,
    validate_parameter_map,
    validate_time_unit,
)
from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


class HalfLifeDecay(DecayFunction):
    """Half-life temporal decay: V(e, T_q) = q_e · 0.5^(Δt / h_j).

    Args:
        half_life: Mapping from fact_type → half-life in the configured time unit.
        default_half_life: Fallback half-life for unknown fact types.
        time_unit: Unit for Δt computation. One of "days", "hours", "seconds".
    """

    def __init__(
        self,
        half_life: dict[str, float] | None = None,
        default_half_life: float = 7.0,
        time_unit: str = "days",
    ) -> None:
        self._half_life = validate_parameter_map(
            half_life, "half_life", positive=True
        )
        self._default_half_life = validate_parameter(
            default_half_life, "default_half_life", positive=True
        )
        self._time_divisor = validate_time_unit(time_unit)

    def _get_half_life(self, fact_type: str) -> float:
        return self._half_life.get(fact_type, self._default_half_life)

    def _age_in_units(self, fact: TemporalFact, query_time: datetime) -> float:
        delta_seconds = (query_time - fact.timestamp).total_seconds()
        return max(delta_seconds / self._time_divisor, 0.0)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        h = self._get_half_life(fact.fact_type)
        age = self._age_in_units(fact, query_time)
        return float(fact.source_quality * (0.5 ** (age / h)))

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Return equivalent β = ln(2)/h."""
        h = self._get_half_life(fact_type)
        return math.log(2) / h

    def __repr__(self) -> str:
        types = ", ".join(f"{k}={v:.1f}" for k, v in sorted(self._half_life.items()))
        return f"HalfLifeDecay({types})"
