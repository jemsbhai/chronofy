"""Opinion-aware temporal retrieval scoring.

TemporalScorer equivalent that works with full Subjective Logic Opinions
instead of scalar validity, enabling scoring strategies that can inspect
uncertainty directly.

Key advantage over scalar TemporalScorer:
    The scalar scorer combines sim × validity where validity is a single
    float that conflates "low evidence" with "stale evidence." The Opinion
    scorer exposes the full (b, d, u, a) tuple, enabling strategies that
    treat uncertain evidence differently from negative evidence.

    Example: Two facts both have P(ω) = 0.5, but one has (b=0.45, d=0.05,
    u=0.50) — moderate belief with uncertainty — and the other has
    (b=0.0, d=0.0, u=1.0) — total ignorance. UncertaintyPenalized
    scores the first higher because it has some evidence; the scalar
    scorer treats them identically.

Architecture:
    OpinionScoringStrategy (ABC)
        ├── ProjectedMultiplicative   sim × P(ω)  (scalar-equivalent default)
        ├── UncertaintyPenalized      sim × P(ω) × (1 - u)
        └── UncertaintyAwareBlend     α·sim + β·P(ω) + (1-α-β)·(1-u)

    OpinionScoredFact (frozen dataclass)
        fact, similarity, validity_opinion, validity, combined_score

    OpinionScorer(decay_fn, strategy)
        .score_fact()   → OpinionScoredFact
        .score_facts()  → list[OpinionScoredFact]
        .rank()         → list[OpinionScoredFact] sorted descending

Backward compatibility:
    OpinionScorer accepts BOTH OpinionDecayFunction and plain DecayFunction.
    With plain DecayFunction, scalar outputs are wrapped into dogmatic
    Opinions (u=0). ProjectedMultiplicative then produces identical scores
    to the scalar TemporalScorer with MultiplicativeScoring.

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §3.2 (Projected Probability), §10.4 (Opinion Aging).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


# ═══════════════════════════════════════════════════════════════════
# OpinionScoredFact — immutable result carrier
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpinionScoredFact:
    """Result of opinion-aware scoring for a single fact.

    Carries the full Opinion alongside scalar compatibility fields
    so consumers can inspect uncertainty, belief, and disbelief
    independently.

    Attributes:
        fact:             The original TemporalFact.
        similarity:       Semantic similarity score ∈ [0, 1] (caller-supplied).
        validity_opinion: Full SL Opinion from temporal decay.
        validity:         Scalar P(validity_opinion) for backward compat.
        combined_score:   Final ranking score from the OpinionScoringStrategy.
    """

    fact: TemporalFact
    similarity: float
    validity_opinion: Opinion
    validity: float
    combined_score: float


# ═══════════════════════════════════════════════════════════════════
# OpinionScoringStrategy ABC
# ═══════════════════════════════════════════════════════════════════


class OpinionScoringStrategy(ABC):
    """Abstract base for combining similarity with an Opinion-valued validity.

    Unlike scalar ScoringStrategy (which receives two floats), this
    interface receives the full Opinion, enabling strategies that
    inspect uncertainty, belief, and disbelief directly.
    """

    @abstractmethod
    def score(self, similarity: float, validity_opinion: Opinion) -> float:
        """Combine similarity and validity Opinion into a ranking score.

        Args:
            similarity:       Semantic similarity ∈ [0, 1].
            validity_opinion: Temporal validity as a full SL Opinion.

        Returns:
            Combined score (typically in [0, 1] but not strictly required).
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# Built-in strategies
# ═══════════════════════════════════════════════════════════════════


class ProjectedMultiplicative(OpinionScoringStrategy):
    """score = sim × P(ω).

    Scalar-equivalent default. The Opinion passes through but only its
    projected probability is used. Produces identical scores to the
    scalar TemporalScorer with MultiplicativeScoring.

    Use when: you want Opinion tracking for diagnostics but don't need
    uncertainty to affect the ranking itself.
    """

    def score(self, similarity: float, validity_opinion: Opinion) -> float:
        return float(similarity * validity_opinion.projected_probability())

    def __repr__(self) -> str:
        return "ProjectedMultiplicative()"


class UncertaintyPenalized(OpinionScoringStrategy):
    """score = sim × P(ω) × (1 - u).

    Adds an explicit penalty for uncertain evidence. Two opinions with
    the same projected probability but different uncertainty will produce
    different scores — the more certain one ranks higher.

    Properties:
        - Vacuous opinion (u=1): score = 0 regardless of similarity
        - Dogmatic opinion (u=0): reduces to sim × P(ω) = sim × b
        - Monotonically decreasing in u (given fixed sim and P)

    Use when: you want to actively suppress stale evidence that has
    decayed toward the uninformative base rate, preferring fresh evidence
    with lower uncertainty even if its projected probability is similar.
    """

    def score(self, similarity: float, validity_opinion: Opinion) -> float:
        p = validity_opinion.projected_probability()
        certainty = 1.0 - validity_opinion.uncertainty
        return float(similarity * p * certainty)

    def __repr__(self) -> str:
        return "UncertaintyPenalized()"


class UncertaintyAwareBlend(OpinionScoringStrategy):
    """score = α·sim + β·P(ω) + γ·(1-u), where γ = 1-α-β.

    Three-way weighted blend giving explicit control over similarity,
    projected validity, and certainty independently.

    Args:
        alpha: Weight for similarity ∈ [0, 1]. Default 0.4.
        beta:  Weight for P(ω) ∈ [0, 1]. Default 0.4.
        gamma = 1 - alpha - beta: weight for (1 - u).

    Constraint: alpha + beta ≤ 1, both non-negative.

    Use when: you want fine-grained control over how much each
    dimension contributes to the ranking.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.4) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if alpha + beta > 1.0 + 1e-9:
            raise ValueError(
                f"alpha + beta must be ≤ 1.0, got {alpha} + {beta} = {alpha + beta}"
            )
        self._alpha = alpha
        self._beta = beta
        self._gamma = max(0.0, 1.0 - alpha - beta)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    def score(self, similarity: float, validity_opinion: Opinion) -> float:
        p = validity_opinion.projected_probability()
        certainty = 1.0 - validity_opinion.uncertainty
        return float(
            self._alpha * similarity
            + self._beta * p
            + self._gamma * certainty
        )

    def __repr__(self) -> str:
        return (
            f"UncertaintyAwareBlend("
            f"α={self._alpha}, β={self._beta}, γ={self._gamma:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════
# OpinionScorer — orchestrator
# ═══════════════════════════════════════════════════════════════════


class OpinionScorer:
    """Scores and ranks TemporalFacts using full SL Opinions.

    Delegates validity computation to the injected decay function and
    combination to the injected OpinionScoringStrategy.

    Accepts both OpinionDecayFunction (full SL path) and plain
    DecayFunction (scalar outputs wrapped into dogmatic Opinions).

    Args:
        decay_fn: OpinionDecayFunction or plain DecayFunction.
        strategy: OpinionScoringStrategy instance.
            Defaults to ProjectedMultiplicative.

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        scorer = OpinionScorer(decay_fn=odf, strategy=UncertaintyPenalized())
        ranked = scorer.rank(facts, similarities, query_time, top_k=5)
        for sf in ranked:
            print(f"[{sf.combined_score:.3f}] u={sf.validity_opinion.uncertainty:.2f}")
    """

    def __init__(
        self,
        decay_fn: DecayFunction,
        strategy: OpinionScoringStrategy | None = None,
    ) -> None:
        self._decay_fn = decay_fn
        self._strategy = strategy if strategy is not None else ProjectedMultiplicative()
        self._is_opinion_aware = isinstance(decay_fn, OpinionDecayFunction)

    @property
    def strategy(self) -> OpinionScoringStrategy:
        return self._strategy

    @property
    def decay_fn(self) -> DecayFunction:
        return self._decay_fn

    def _compute_opinion(
        self, fact: TemporalFact, query_time: datetime
    ) -> Opinion:
        """Compute validity Opinion for a fact.

        Uses compute_opinion() for OpinionDecayFunction, or wraps
        scalar compute() into a dogmatic Opinion for plain DecayFunction.
        """
        if self._is_opinion_aware:
            return self._decay_fn.compute_opinion(fact, query_time)  # type: ignore[union-attr]
        else:
            scalar = self._decay_fn.compute(fact, query_time)
            return Opinion.from_confidence(scalar, uncertainty=0.0)

    def score_fact(
        self,
        fact: TemporalFact,
        similarity: float,
        query_time: datetime,
    ) -> OpinionScoredFact:
        """Score a single fact.

        Args:
            fact:       The evidence item to score.
            similarity: Semantic similarity ∈ [0, 1] (caller-supplied).
            query_time: Reference time for validity computation.

        Returns:
            OpinionScoredFact with full Opinion and combined score.
        """
        opinion = self._compute_opinion(fact, query_time)
        validity = opinion.projected_probability()
        combined = self._strategy.score(similarity, opinion)
        return OpinionScoredFact(
            fact=fact,
            similarity=similarity,
            validity_opinion=opinion,
            validity=validity,
            combined_score=combined,
        )

    def score_facts(
        self,
        facts: list[TemporalFact],
        similarities: list[float],
        query_time: datetime,
    ) -> list[OpinionScoredFact]:
        """Score a batch of facts.

        Args:
            facts:        Evidence items to score.
            similarities: Parallel list of similarity scores ∈ [0, 1].
            query_time:   Reference time for validity computation.

        Returns:
            List of OpinionScoredFact in the same order as input.

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

        return [
            self.score_fact(fact, sim, query_time)
            for fact, sim in zip(facts, similarities)
        ]

    def rank(
        self,
        facts: list[TemporalFact],
        similarities: list[float],
        query_time: datetime,
        top_k: int | None = None,
    ) -> list[OpinionScoredFact]:
        """Score and rank facts by combined score, descending.

        Args:
            facts:        Evidence items to rank.
            similarities: Parallel list of similarity scores ∈ [0, 1].
            query_time:   Reference time for validity computation.
            top_k:        Return only the top-k results. None returns all.

        Returns:
            List of OpinionScoredFact sorted by combined_score descending.

        Raises:
            ValueError: If top_k is not a positive integer.
            ValueError: If facts and similarities have different lengths.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        scored = self.score_facts(facts, similarities, query_time)
        if not scored:
            return []

        ranked = sorted(scored, key=lambda sf: sf.combined_score, reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def __repr__(self) -> str:
        mode = "opinion" if self._is_opinion_aware else "scalar-wrapped"
        return (
            f"OpinionScorer("
            f"mode={mode!r}, "
            f"strategy={self._strategy!r})"
        )
