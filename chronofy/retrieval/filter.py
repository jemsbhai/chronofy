"""Epistemic filter for temporal evidence retrieval.

Implements the τ-threshold filtering mechanism from TLDA Layer 2.
During retrieval, facts whose temporal validity score falls below
the configurable threshold τ are structurally excluded from the
candidate context window delivered to the LLM.

This guarantees the LLM never receives evidence below the minimum
quality threshold (Property 1: Retrieval Completeness).
"""

from __future__ import annotations

from datetime import datetime

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


class EpistemicFilter:
    """Filters temporally expired facts from a candidate evidence set.

    The epistemic filter computes the temporal validity score for each
    fact and retains only those above the threshold τ. This is the
    core mechanism that prevents stale evidence from entering the
    LLM context window.

    Args:
        decay_fn: The decay function to use for computing validity scores.
        threshold: Minimum validity score τ ∈ [0, 1] for a fact to be retained.
            Higher values are more aggressive (exclude more facts).
            Lower values are more permissive.

    Example:
        >>> from chronofy import ExponentialDecay, EpistemicFilter, TemporalFact
        >>> from datetime import datetime, timedelta
        >>> decay = ExponentialDecay(beta={"vital_sign": 5.0})
        >>> ef = EpistemicFilter(decay_fn=decay, threshold=0.1)
        >>> facts = [
        ...     TemporalFact(content="K+ = 4.1", timestamp=datetime.now() - timedelta(days=1),
        ...                  fact_type="vital_sign"),
        ...     TemporalFact(content="K+ = 3.2", timestamp=datetime.now() - timedelta(days=180),
        ...                  fact_type="vital_sign"),
        ... ]
        >>> valid = ef.filter(facts, datetime.now())
        >>> len(valid)  # Only yesterday's reading survives
        1
    """

    def __init__(self, decay_fn: DecayFunction, threshold: float = 0.1) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self._decay_fn = decay_fn
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The current validity threshold τ."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._threshold = value

    def score(self, fact: TemporalFact, query_time: datetime) -> float:
        """Compute the temporal validity score for a single fact."""
        return self._decay_fn.compute(fact, query_time)

    def score_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[tuple[TemporalFact, float]]:
        """Score all facts and return (fact, score) pairs sorted by score descending."""
        scores = self._decay_fn.compute_batch(facts, query_time)
        paired = list(zip(facts, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired

    def filter(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[TemporalFact]:
        """Return only facts whose validity score meets or exceeds τ.

        This implements the Retrieval Completeness property: the returned
        set excludes all evidence items e satisfying
            q_e · c(tr_j) · exp(-β_j · Δt) < τ

        Facts are returned in descending order of validity score.
        """
        scored = self.score_batch(facts, query_time)
        return [fact for fact, s in scored if s >= self._threshold]

    def filter_with_scores(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[tuple[TemporalFact, float]]:
        """Like filter(), but returns (fact, score) pairs."""
        scored = self.score_batch(facts, query_time)
        return [(fact, s) for fact, s in scored if s >= self._threshold]

    def partition(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> tuple[list[TemporalFact], list[TemporalFact]]:
        """Split facts into (valid, expired) based on threshold τ.

        Useful for diagnostics: the expired set tells you what was
        filtered out and may need re-acquisition.
        """
        scored = self.score_batch(facts, query_time)
        valid = [fact for fact, s in scored if s >= self._threshold]
        expired = [fact for fact, s in scored if s < self._threshold]
        return valid, expired

    def needs_reacquisition(
        self, facts: list[TemporalFact], query_time: datetime, min_required: int = 1
    ) -> bool:
        """Check whether the valid evidence set is insufficient.

        Returns True if fewer than min_required facts survive filtering,
        signaling that the Sequential Exploration Decay trigger should fire.
        """
        valid = self.filter(facts, query_time)
        return len(valid) < min_required

    def __repr__(self) -> str:
        return f"EpistemicFilter(decay_fn={self._decay_fn!r}, threshold={self._threshold})"
