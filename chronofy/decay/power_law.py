"""Power-law decay function.

Heavy-tailed decay model from citation and memory literature:

    V(Δt) = q_e · (1 + Δt)^(-α_j)

where α_j is the exponent for fact type j. The +1 ensures V(0) = 1.

Power-law decay has a heavier tail than exponential — old evidence
retains more residual value. This is appropriate for domains where
information decays quickly initially but retains long-term relevance
(e.g., scientific citations, legal precedents).
"""

from __future__ import annotations

from datetime import datetime

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


class PowerLawDecay(DecayFunction):
    """Power-law temporal decay: V(e, T_q) = q_e · (1 + Δt)^(-α_j).

    Args:
        exponent: Mapping from fact_type → exponent α.
        default_exponent: Fallback exponent for unknown fact types.
        time_unit: Unit for Δt computation. One of "days", "hours", "seconds".
    """

    def __init__(
        self,
        exponent: dict[str, float] | None = None,
        default_exponent: float = 1.0,
        time_unit: str = "days",
    ) -> None:
        self._exponent = exponent or {}
        self._default_exponent = default_exponent
        self._time_divisor = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[time_unit]

    def _get_exponent(self, fact_type: str) -> float:
        return self._exponent.get(fact_type, self._default_exponent)

    def _age_in_units(self, fact: TemporalFact, query_time: datetime) -> float:
        delta_seconds = (query_time - fact.timestamp).total_seconds()
        return max(delta_seconds / self._time_divisor, 0.0)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        alpha = self._get_exponent(fact.fact_type)
        age = self._age_in_units(fact, query_time)
        return fact.source_quality * ((1.0 + age) ** (-alpha))

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Power-law has no equivalent β."""
        return None

    def __repr__(self) -> str:
        types = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self._exponent.items()))
        return f"PowerLawDecay({types})"
