"""Exponential decay function for temporal validity.

Implements the core decay equation from the TLDA framework:

    V(e, T_q) = q_e · exp(-β_j · (T_q - t_e))

where:
    q_e:  source reliability weight ∈ (0, 1]
    β_j:  learnable decay coefficient for fact type j
    T_q:  query timestamp
    t_e:  observation timestamp of the evidence

Decision-theoretic grounding (Proposition 1):
    Under an Ornstein-Uhlenbeck latent process dθ = -κ(θ-μ)dt + σdW,
    the information content of a measurement decays as exp(-2κ·Δt).
    Therefore, the optimal β_j = 2κ_j where κ_j is the mean-reversion rate.

    This means β is NOT an arbitrary hyperparameter — it approximates
    twice the mean-reversion rate of the underlying latent process.
"""

from __future__ import annotations

import math
from datetime import datetime

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact

# Default β values grounded in clinical domain knowledge.
# These correspond to approximate 2κ values for each fact category.
DEFAULT_BETA: dict[str, float] = {
    "vital_sign": 5.0,        # κ ≈ 2.5 — physiology shifts in hours/days
    "lab_result": 2.0,        # κ ≈ 1.0 — labs shift in days/weeks
    "medication": 1.0,        # κ ≈ 0.5 — regimens change over weeks
    "diagnosis": 0.5,         # κ ≈ 0.25 — diagnoses evolve over months
    "chronic_condition": 0.01, # κ ≈ 0.005 — stable over years
    "demographic": 0.0,       # κ = 0 — invariant (blood type, genetics)
    "general": 0.5,           # Conservative default
}


class ExponentialDecay(DecayFunction):
    """Exponential temporal decay: V(e, T_q) = q_e · exp(-β_j · Δt).

    The exponential is the Bayesian-optimal decay function when the latent
    state follows an Ornstein-Uhlenbeck process (Proposition 1).

    Args:
        beta: Mapping from fact_type → decay coefficient.
              Missing types fall back to default_beta.
        default_beta: Fallback β for unknown fact types.
        time_unit: Unit for Δt computation. One of "days", "hours", "seconds".

    Example:
        >>> decay = ExponentialDecay(beta={"vital_sign": 5.0, "demographic": 0.0})
        >>> fact = TemporalFact(content="K+ = 4.1", timestamp=yesterday, fact_type="vital_sign")
        >>> decay.compute(fact, datetime.now())  # High validity — 1 day old, high β
        0.006737...
    """

    def __init__(
        self,
        beta: dict[str, float] | None = None,
        default_beta: float = 0.5,
        time_unit: str = "days",
    ) -> None:
        self._beta = {**DEFAULT_BETA, **(beta or {})}
        self._default_beta = default_beta
        self._time_divisor = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[time_unit]

    def _get_beta(self, fact_type: str) -> float:
        return self._beta.get(fact_type, self._default_beta)

    def _age_in_units(self, fact: TemporalFact, query_time: datetime) -> float:
        delta_seconds = (query_time - fact.timestamp).total_seconds()
        return max(delta_seconds / self._time_divisor, 0.0)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        """Compute temporal validity for a single fact.

        Returns q_e · exp(-β_j · Δt) where Δt is in the configured time unit.
        """
        beta = self._get_beta(fact.fact_type)
        age = self._age_in_units(fact, query_time)

        # Temporal invariance guarantee: when β = 0, decay is always 1.0
        if beta == 0.0:
            return fact.source_quality

        return fact.source_quality * math.exp(-beta * age)

    def compute_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[float]:
        """Compute validity scores for a batch of facts."""
        return [self.compute(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Return the β coefficient for a given fact type."""
        return self._get_beta(fact_type)

    def half_life(self, fact_type: str) -> float | None:
        """Return the half-life in the configured time unit.

        The half-life is ln(2)/β — the time for validity to drop to 50%.
        Returns None for invariant fact types (β = 0).
        """
        beta = self._get_beta(fact_type)
        if beta <= 0.0:
            return None
        return math.log(2) / beta

    @staticmethod
    def from_mean_reversion_rate(kappa: dict[str, float], **kwargs: object) -> "ExponentialDecay":
        """Construct from mean-reversion rates κ, using β = 2κ (Proposition 1).

        This is the theoretically grounded constructor: if you know the
        mean-reversion rate of the latent process governing each fact type,
        the optimal decay coefficient is exactly twice that rate.

        Args:
            kappa: Mapping from fact_type → mean-reversion rate κ.

        Example:
            >>> decay = ExponentialDecay.from_mean_reversion_rate({
            ...     "vital_sign": 2.5,   # Fast mean-reversion
            ...     "demographic": 0.0,   # No mean-reversion (invariant)
            ... })
        """
        beta = {k: 2.0 * v for k, v in kappa.items()}
        return ExponentialDecay(beta=beta, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        types = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self._beta.items()) if v > 0)
        return f"ExponentialDecay({types})"
