"""Base protocol for decay functions.

The pluggable decay backend is a key architectural decision informed by
the ASEV (Age-Sensitive Epistemic Value) analysis: different latent processes
require different decay profiles. The exponential is the Bayesian-optimal
surrogate under Gaussian (Ornstein-Uhlenbeck) dynamics, but Weibull,
power-law, skew-normal, and non-parametric alternatives are needed for
more complex processes.

All decay functions implement the same protocol, enabling Chronofy to
swap backends without changing the retrieval or verification layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from chronofy.models import TemporalFact


class DecayFunction(ABC):
    """Protocol for temporal decay functions.

    A decay function maps (fact, query_time) → validity score in [0, 1],
    where 1.0 means fully valid and 0.0 means completely obsolete.

    The score should be interpreted as a surrogate for the epistemic value
    V(e, t_q) — how much this evidence still improves present-state accuracy.
    Under Gaussian latent dynamics, this reduces to exp(-2κ · Δt) where κ is
    the mean-reversion rate (Proposition 1 in the Chronofy paper).
    """

    @abstractmethod
    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        """Compute the temporal validity score for a fact at query time.

        Args:
            fact: The evidence item with timestamp and metadata.
            query_time: The time at which the query is issued.

        Returns:
            A float in [0, 1] representing temporal validity.
            Incorporates both temporal decay and source reliability (q_e).
        """
        ...

    @abstractmethod
    def compute_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[float]:
        """Compute validity scores for a batch of facts.

        Default implementation iterates; subclasses may vectorize.
        """
        ...

    def get_beta(self, fact_type: str) -> float | None:
        """Return the decay coefficient for a fact type, if applicable.

        Not all decay functions use a scalar β (e.g., GP-based decay).
        Returns None if not applicable.
        """
        return None
