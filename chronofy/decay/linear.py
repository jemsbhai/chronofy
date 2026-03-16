"""Linear decay function.

Simplest possible decay model — validity drops linearly to zero:

    V(Δt) = q_e · max(0, 1 - α_j · Δt)

where α_j is the decay rate for fact type j. Evidence expires
completely at age 1/α_j.

Useful as a baseline and in domains with hard expiry deadlines
(e.g., regulatory compliance windows, prescription validity).
"""

from __future__ import annotations

from datetime import datetime

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


class LinearDecay(DecayFunction):
    """Linear temporal decay: V(e, T_q) = q_e · max(0, 1 - α_j · Δt).

    Args:
        rate: Mapping from fact_type → decay rate α (validity/time_unit).
        default_rate: Fallback rate for unknown fact types.
        time_unit: Unit for Δt computation. One of "days", "hours", "seconds".
    """

    def __init__(
        self,
        rate: dict[str, float] | None = None,
        default_rate: float = 0.1,
        time_unit: str = "days",
    ) -> None:
        self._rate = rate or {}
        self._default_rate = default_rate
        self._time_divisor = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[time_unit]

    def _get_rate(self, fact_type: str) -> float:
        return self._rate.get(fact_type, self._default_rate)

    def _age_in_units(self, fact: TemporalFact, query_time: datetime) -> float:
        delta_seconds = (query_time - fact.timestamp).total_seconds()
        return max(delta_seconds / self._time_divisor, 0.0)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        alpha = self._get_rate(fact.fact_type)
        age = self._age_in_units(fact, query_time)
        return fact.source_quality * max(0.0, 1.0 - alpha * age)

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Linear decay has no equivalent β."""
        return None

    def expiry_time(self, fact_type: str) -> float:
        """Return 1/α — the age at which validity hits zero (with q=1)."""
        alpha = self._get_rate(fact_type)
        return 1.0 / alpha if alpha > 0 else float("inf")

    def __repr__(self) -> str:
        types = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self._rate.items()))
        return f"LinearDecay({types})"
