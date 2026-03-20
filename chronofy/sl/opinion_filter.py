"""SL-aware epistemic filter with dual-threshold and three-way partition.

Extends the scalar EpistemicFilter with full Subjective Logic Opinions,
enabling dual-threshold filtering (p_min + u_max) and a diagnostic
three-way partition (valid / stale / uncertain) that maps to different
re-acquisition strategies.

Key advantage over scalar EpistemicFilter:
    The scalar filter produces a binary valid/expired split. It cannot
    distinguish WHY a fact was filtered:
        - Stale: P(ω') < p_min because evidence decayed to ignorance
          (high u). Action: re-acquire the same source type.
        - Uncertain: P(ω') ≥ p_min (base_rate props it up) but u > u_max,
          meaning the system is ignorant rather than confident.
          Action: need more or better sources.

    This three-way diagnostic maps to different re-acquisition strategies,
    enabling smarter Sequential Exploration Decay triggers.

Mathematical basis:
    Given a decayed Opinion ω' = (b', d', u', a):
        P(ω') = b' + a · u'       (projected probability)

    As evidence decays (Jøsang §10.4):
        b' → 0, d' → 0, u' → 1   (reverts to ignorance)
        P(ω') → a                  (approaches base_rate)

    With base_rate a = 0.5:
        - p_min = 0.6 filters facts whose P dropped below threshold
        - u_max = 0.5 additionally catches facts that pass P (because
          a · u props it up) but have unacceptably high uncertainty

Default p_min:
    Defaults to 0.6 (above SL base_rate of 0.5), consistent with
    SLPipeline.default(). A threshold ≤ base_rate would never filter
    anything because fully decayed opinions still have P ≈ base_rate.

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §10.4 (Opinion Aging), §3.2 (Projected Probability).
    Paper §III-C (Epistemic Filtering), Property 1 (Retrieval Completeness).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from jsonld_ex.confidence_algebra import Opinion

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


# ═══════════════════════════════════════════════════════════════════
# Partition result
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpinionPartitionResult:
    """Three-way partition of facts by temporal validity and uncertainty.

    Each category carries (TemporalFact, Opinion) pairs so downstream
    consumers can inspect the full SL Opinion for each fact.

    Categories:
        valid: P(ω') ≥ p_min AND u ≤ u_max.
            Fresh, sufficient evidence. Include in LLM context.
        stale: P(ω') < p_min.
            Evidence decayed below threshold. Re-acquire same source type.
        uncertain: P(ω') ≥ p_min BUT u > u_max.
            Projected probability is propped up by base_rate, but the
            system is ignorant. Need more or better sources.

    When u_max = 1.0 (default, disabled), the uncertain category is
    always empty and the partition reduces to the scalar binary split.
    """

    valid: list[tuple[TemporalFact, Opinion]]
    stale: list[tuple[TemporalFact, Opinion]]
    uncertain: list[tuple[TemporalFact, Opinion]]


# ═══════════════════════════════════════════════════════════════════
# Filter
# ═══════════════════════════════════════════════════════════════════


class OpinionEpistemicFilter:
    """SL-aware epistemic filter with dual-threshold and three-way partition.

    Computes full SL Opinions for each fact via OpinionDecayFunction,
    then filters by projected probability (p_min) and optionally by
    uncertainty ceiling (u_max).

    Args:
        decay_fn: OpinionDecayFunction for computing temporally-decayed
            Opinions. Must be an OpinionDecayFunction, not a plain
            DecayFunction (raises TypeError otherwise).
        p_min: Minimum projected probability P(ω') for a fact to be
            retained. Default 0.6 (above SL base_rate of 0.5).
        u_max: Maximum allowed uncertainty for a fact to be retained.
            Default 1.0 (disabled — all uncertainty levels accepted).
            Set lower to filter high-uncertainty facts that pass p_min
            only because base_rate props up P(ω').

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        result = ef.partition(facts, query_time)
        for fact, opinion in result.stale:
            print(f"Re-acquire: {fact.content} (u={opinion.uncertainty:.2f})")
    """

    def __init__(
        self,
        decay_fn: OpinionDecayFunction,
        p_min: float = 0.6,
        u_max: float = 1.0,
    ) -> None:
        if not isinstance(decay_fn, OpinionDecayFunction):
            raise TypeError(
                f"OpinionEpistemicFilter requires an OpinionDecayFunction, "
                f"got {type(decay_fn).__name__}. Use the scalar EpistemicFilter "
                f"for plain DecayFunction instances."
            )
        if not 0.0 <= p_min <= 1.0:
            raise ValueError(f"p_min must be in [0, 1], got {p_min}")
        if not 0.0 <= u_max <= 1.0:
            raise ValueError(f"u_max must be in [0, 1], got {u_max}")

        self._decay_fn = decay_fn
        self._p_min = p_min
        self._u_max = u_max

    @property
    def p_min(self) -> float:
        """Minimum projected probability threshold."""
        return self._p_min

    @p_min.setter
    def p_min(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p_min must be in [0, 1], got {value}")
        self._p_min = value

    @property
    def u_max(self) -> float:
        """Maximum uncertainty threshold."""
        return self._u_max

    @u_max.setter
    def u_max(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"u_max must be in [0, 1], got {value}")
        self._u_max = value

    # ───────────────────────────────────────────────────────────
    # Scoring
    # ───────────────────────────────────────────────────────────

    def score_opinion(self, fact: TemporalFact, query_time: datetime) -> Opinion:
        """Compute the temporally-decayed Opinion for a single fact."""
        return self._decay_fn.compute_opinion(fact, query_time)

    def score_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[tuple[TemporalFact, Opinion]]:
        """Score all facts and return (fact, Opinion) pairs sorted by P(ω') descending."""
        opinions = self._decay_fn.compute_opinion_batch(facts, query_time)
        paired = list(zip(facts, opinions))
        paired.sort(key=lambda x: x[1].projected_probability(), reverse=True)
        return paired

    # ───────────────────────────────────────────────────────────
    # Filtering
    # ───────────────────────────────────────────────────────────

    def _passes(self, opinion: Opinion) -> bool:
        """Check whether an Opinion passes both thresholds."""
        return (
            opinion.projected_probability() >= self._p_min
            and opinion.uncertainty <= self._u_max
        )

    def filter(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[TemporalFact]:
        """Return only facts passing both p_min and u_max thresholds.

        Facts are returned in descending order of projected probability.
        Backward compatible with EpistemicFilter.filter() signature.
        """
        scored = self.score_batch(facts, query_time)
        return [fact for fact, op in scored if self._passes(op)]

    def filter_with_opinions(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[tuple[TemporalFact, Opinion]]:
        """Like filter(), but returns (fact, Opinion) pairs."""
        scored = self.score_batch(facts, query_time)
        return [(fact, op) for fact, op in scored if self._passes(op)]

    # ───────────────────────────────────────────────────────────
    # Three-way partition
    # ───────────────────────────────────────────────────────────

    def partition(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> OpinionPartitionResult:
        """Split facts into valid / stale / uncertain categories.

        Categories:
            valid: P(ω') ≥ p_min AND u ≤ u_max
            stale: P(ω') < p_min (regardless of u)
            uncertain: P(ω') ≥ p_min BUT u > u_max

        Every input fact appears in exactly one category.
        Each category is sorted by P(ω') descending.
        """
        scored = self.score_batch(facts, query_time)

        valid: list[tuple[TemporalFact, Opinion]] = []
        stale: list[tuple[TemporalFact, Opinion]] = []
        uncertain: list[tuple[TemporalFact, Opinion]] = []

        for fact, op in scored:
            prob = op.projected_probability()
            if prob < self._p_min:
                stale.append((fact, op))
            elif op.uncertainty > self._u_max:
                uncertain.append((fact, op))
            else:
                valid.append((fact, op))

        return OpinionPartitionResult(
            valid=valid,
            stale=stale,
            uncertain=uncertain,
        )

    # ───────────────────────────────────────────────────────────
    # Re-acquisition
    # ───────────────────────────────────────────────────────────

    def needs_reacquisition(
        self, facts: list[TemporalFact], query_time: datetime, min_required: int = 1
    ) -> bool:
        """Check whether the valid evidence set is insufficient.

        Returns True if fewer than min_required facts pass both
        thresholds, signaling the Sequential Exploration Decay trigger.
        """
        valid = self.filter(facts, query_time)
        return len(valid) < min_required

    def __repr__(self) -> str:
        return (
            f"OpinionEpistemicFilter("
            f"p_min={self._p_min}, "
            f"u_max={self._u_max}, "
            f"decay_fn={self._decay_fn!r})"
        )
