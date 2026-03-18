"""Temporal retrieval scoring — combine semantic similarity with temporal validity.

Retrieval-augmented generation requires ranking candidate facts by both how
semantically relevant they are to the query and how temporally fresh they are.
This module formalises that combination via a pluggable ScoringStrategy.

Architecture:
    ScoringStrategy (ABC)              ← one method: score(sim, val) → float
      ├── MultiplicativeScoring        ← sim × val  (default; RAG standard)
      ├── HarmonicScoring              ← 2·sim·val / (sim + val)
      ├── WeightedBlendScoring(α)      ← α·sim + (1-α)·val
      └── PowerScoring(α)              ← sim^α · val^(1-α)

    ScoredFact                         ← frozen result carrier
    TemporalScorer(decay_fn, strategy) ← orchestrator

Design principles:
  - ScoringStrategy is a single-method ABC: easy to subclass, easy to test.
  - TemporalScorer owns no decay logic — it delegates to the injected DecayFunction.
  - ScoredFact is frozen: scores are immutable once computed.
  - All strategies are stateless or carry only hyperparameters.
  - validate_scoring_strategy() in plugins.py smoke-tests custom strategies.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


# ---------------------------------------------------------------------------
# ScoredFact — immutable result carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredFact:
    """The result of scoring a single fact against a query.

    Carries the original fact alongside all three score components so
    callers can inspect or log any of them independently.

    Attributes:
        fact:            The original TemporalFact.
        similarity:      Semantic similarity score ∈ [0, 1] (caller-supplied).
        validity:        Temporal validity score ∈ [0, 1] (from decay function).
        combined_score:  Final ranking score produced by the ScoringStrategy.
    """

    fact: TemporalFact
    similarity: float
    validity: float
    combined_score: float


# ---------------------------------------------------------------------------
# ScoringStrategy ABC
# ---------------------------------------------------------------------------


class ScoringStrategy(ABC):
    """Abstract base for combining semantic similarity and temporal validity.

    Implement this interface to plug in any scoring formula. The only
    requirement is that ``score()`` maps two floats in [0, 1] to a float
    in [0, 1].

    Use ``validate_scoring_strategy()`` from ``chronofy.plugins`` to
    smoke-test a custom implementation before deploying it.
    """

    @abstractmethod
    def score(self, similarity: float, validity: float) -> float:
        """Combine similarity and validity into a single ranking score.

        Args:
            similarity: Semantic similarity ∈ [0, 1].
            validity:   Temporal validity ∈ [0, 1].

        Returns:
            Combined score ∈ [0, 1].
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class MultiplicativeScoring(ScoringStrategy):
    """Multiplicative combination: score = similarity × validity.

    This is the standard temporal RAG baseline and the default strategy.
    It is aggressive: a low validity score strongly suppresses even highly
    relevant facts, and vice versa. Appropriate when both dimensions are
    independently necessary — a stale fact is not worth retrieving regardless
    of its semantic match.

    Example:
        sim=0.9, val=0.1 → 0.09  (stale fact heavily penalised)
        sim=0.9, val=0.9 → 0.81  (fresh, relevant fact promoted)
    """

    def score(self, similarity: float, validity: float) -> float:
        return float(similarity * validity)


class HarmonicScoring(ScoringStrategy):
    """Harmonic mean: score = 2·sim·val / (sim + val).

    Analogous to the F1 score: emphasises balance between the two
    dimensions. A fact must be both reasonably similar AND reasonably
    fresh to score well — neither extreme suppresses the other as
    aggressively as the multiplicative strategy.

    Properties:
        - Symmetric: score(a, b) == score(b, a)
        - score(x, x) == x
        - score ≤ min(sim, val)  (always bounded by the weaker dimension)
        - Returns 0.0 when either input is 0.0

    Example:
        sim=0.8, val=0.5 → 2·0.8·0.5 / 1.3 ≈ 0.615
        sim=0.7, val=0.7 → 0.7
    """

    def score(self, similarity: float, validity: float) -> float:
        denom = similarity + validity
        if denom == 0.0:
            return 0.0
        return float(2.0 * similarity * validity / denom)


class WeightedBlendScoring(ScoringStrategy):
    """Weighted arithmetic mean: score = α·similarity + (1-α)·validity.

    Allows explicit control over the trade-off between semantic relevance
    and temporal freshness. When α=1.0 the score is pure similarity (vanilla
    RAG); when α=0.0 it is pure validity (rank by freshness only).

    Args:
        alpha: Weight assigned to similarity ∈ [0.0, 1.0].
               Weight assigned to validity is (1 - alpha).

    Raises:
        ValueError: If alpha is outside [0.0, 1.0].

    Example:
        alpha=0.7, sim=0.8, val=0.6 → 0.7·0.8 + 0.3·0.6 = 0.74
        alpha=0.5, sim=0.6, val=0.4 → 0.50  (arithmetic mean)
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0.0, 1.0], got {alpha}. "
                f"alpha=1.0 weights similarity only; alpha=0.0 weights validity only."
            )
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def score(self, similarity: float, validity: float) -> float:
        return float(self._alpha * similarity + (1.0 - self._alpha) * validity)

    def __repr__(self) -> str:
        return f"WeightedBlendScoring(alpha={self._alpha})"


class PowerScoring(ScoringStrategy):
    """Geometric / power-mean blend: score = similarity^α · validity^(1-α).

    The geometric mean (α=0.5) is the multiplicative analogue of the
    arithmetic mean. Compared to MultiplicativeScoring, PowerScoring
    with α<1 is less aggressive when one dimension is very low —
    useful when you want temporal freshness to matter but not dominate.

    Args:
        alpha: Exponent for similarity ∈ [0.0, 1.0].
               Exponent for validity is (1 - alpha).

    Raises:
        ValueError: If alpha is outside [0.0, 1.0].

    Example:
        alpha=0.5: score = sqrt(sim · val)   (geometric mean)
        alpha=0.7, sim=0.8, val=0.6 → 0.8^0.7 · 0.6^0.3 ≈ 0.737
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0.0, 1.0], got {alpha}. "
                f"alpha=1.0 gives pure similarity; alpha=0.0 gives pure validity."
            )
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def score(self, similarity: float, validity: float) -> float:
        # Handle zero bases explicitly to avoid 0^0 ambiguity
        if similarity <= 0.0 or validity <= 0.0:
            return 0.0
        return float((similarity ** self._alpha) * (validity ** (1.0 - self._alpha)))

    def __repr__(self) -> str:
        return f"PowerScoring(alpha={self._alpha})"


# ---------------------------------------------------------------------------
# TemporalScorer — orchestrator
# ---------------------------------------------------------------------------


class TemporalScorer:
    """Scores and ranks TemporalFacts by combining semantic similarity and
    temporal validity via a pluggable ScoringStrategy.

    The scorer owns no decay logic: it delegates validity computation to
    the injected ``DecayFunction``, and no scoring logic: it delegates
    combination to the injected ``ScoringStrategy``.

    Args:
        decay_fn: Any DecayFunction instance (ExponentialDecay, custom, etc.)
        strategy: Any ScoringStrategy instance. Defaults to MultiplicativeScoring.

    Example:
        >>> from chronofy import ExponentialDecay
        >>> from chronofy.scoring.temporal_scorer import TemporalScorer, HarmonicScoring
        >>>
        >>> decay = ExponentialDecay(beta={"news": 0.001})
        >>> scorer = TemporalScorer(decay_fn=decay, strategy=HarmonicScoring())
        >>>
        >>> ranked = scorer.rank(facts, similarities, query_time=datetime.now(), top_k=5)
        >>> for sf in ranked:
        ...     print(f"[{sf.combined_score:.3f}] sim={sf.similarity:.2f}  "
        ...           f"val={sf.validity:.3f}  {sf.fact.content[:60]}")
    """

    def __init__(
        self,
        decay_fn: DecayFunction,
        strategy: ScoringStrategy | None = None,
    ) -> None:
        self._decay_fn = decay_fn
        self._strategy = strategy if strategy is not None else MultiplicativeScoring()

    @property
    def strategy(self) -> ScoringStrategy:
        """The active ScoringStrategy."""
        return self._strategy

    @property
    def decay_fn(self) -> DecayFunction:
        """The active DecayFunction."""
        return self._decay_fn

    def score_fact(
        self,
        fact: TemporalFact,
        similarity: float,
        query_time: datetime,
    ) -> ScoredFact:
        """Score a single fact.

        Args:
            fact:        The evidence item to score.
            similarity:  Semantic similarity score ∈ [0, 1] (caller-supplied).
            query_time:  Reference time for validity computation.

        Returns:
            ScoredFact with validity, and combined_score populated.
        """
        validity = self._decay_fn.compute(fact, query_time)
        combined = self._strategy.score(similarity, validity)
        return ScoredFact(
            fact=fact,
            similarity=similarity,
            validity=validity,
            combined_score=combined,
        )

    def score_facts(
        self,
        facts: list[TemporalFact],
        similarities: list[float],
        query_time: datetime,
    ) -> list[ScoredFact]:
        """Score a batch of facts.

        Args:
            facts:        The evidence items to score.
            similarities: Parallel list of semantic similarity scores ∈ [0, 1].
            query_time:   Reference time for validity computation.

        Returns:
            List of ScoredFact in the same order as the input.

        Raises:
            ValueError: If facts and similarities have different lengths.
        """
        if len(facts) != len(similarities):
            raise ValueError(
                f"facts and similarities must have equal length, "
                f"got {len(facts)} and {len(similarities)}."
            )
        if not facts:
            return []

        validities = self._decay_fn.compute_batch(facts, query_time)
        return [
            ScoredFact(
                fact=fact,
                similarity=sim,
                validity=val,
                combined_score=self._strategy.score(sim, val),
            )
            for fact, sim, val in zip(facts, similarities, validities)
        ]

    def rank(
        self,
        facts: list[TemporalFact],
        similarities: list[float],
        query_time: datetime,
        top_k: int | None = None,
    ) -> list[ScoredFact]:
        """Score and rank facts by combined score, descending.

        Args:
            facts:        The evidence items to rank.
            similarities: Parallel list of semantic similarity scores ∈ [0, 1].
            query_time:   Reference time for validity computation.
            top_k:        Return only the top-k results. None returns all.
                          Must be a positive integer if specified.

        Returns:
            List of ScoredFact sorted by combined_score descending.
            Length is min(top_k, len(facts)) when top_k is specified.

        Raises:
            ValueError: If top_k is specified and is not a positive integer.
            ValueError: If facts and similarities have different lengths.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(
                f"top_k must be a positive integer, got {top_k}."
            )

        scored = self.score_facts(facts, similarities, query_time)
        if not scored:
            return []

        # Sort descending by combined_score; Python sort is stable so ties
        # preserve the original input order.
        ranked = sorted(scored, key=lambda sf: sf.combined_score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def __repr__(self) -> str:
        return (
            f"TemporalScorer("
            f"decay_fn={self._decay_fn!r}, "
            f"strategy={self._strategy!r})"
        )
