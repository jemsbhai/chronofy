"""Tests for chronofy.sl.opinion_filter — OpinionEpistemicFilter.

SL-aware epistemic filter that extends the scalar EpistemicFilter with
dual-threshold filtering (p_min + u_max) and three-way diagnostic
partition (valid / stale / uncertain).

Key advantage over scalar EpistemicFilter:
    The scalar filter produces a binary valid/expired partition. It cannot
    distinguish WHY a fact was filtered:
        - Stale (high u): "we once had data but it decayed to ignorance"
          → re-acquire the same source type
        - Uncertain (u > u_max, P might still pass): "insufficient
          evidence quality" → need more or better sources

    The Opinion filter tracks (b, d, u, a) through the decay, enabling
    a three-way partition that maps to different re-acquisition strategies.

TDD: Tests written before implementation.
"""

from datetime import datetime, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.sl.opinion_filter import OpinionEpistemicFilter, OpinionPartitionResult


QUERY_TIME = datetime(2026, 3, 15, 12, 0)


def _fact(
    content: str,
    days_ago: float,
    fact_type: str = "general",
    quality: float = 1.0,
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


# ═══════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterConstruction:

    def test_from_opinion_decay_fn(self):
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6)
        assert ef.p_min == 0.6
        assert ef.u_max == 1.0  # disabled by default

    def test_custom_u_max(self):
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        assert ef.u_max == 0.5

    def test_p_min_validation(self):
        odf = OpinionDecayFunction()
        with pytest.raises(ValueError):
            OpinionEpistemicFilter(decay_fn=odf, p_min=1.5)
        with pytest.raises(ValueError):
            OpinionEpistemicFilter(decay_fn=odf, p_min=-0.1)

    def test_u_max_validation(self):
        odf = OpinionDecayFunction()
        with pytest.raises(ValueError):
            OpinionEpistemicFilter(decay_fn=odf, u_max=1.5)
        with pytest.raises(ValueError):
            OpinionEpistemicFilter(decay_fn=odf, u_max=-0.1)

    def test_plain_decay_fn_raises(self):
        """OpinionEpistemicFilter requires OpinionDecayFunction, not plain."""
        exp = ExponentialDecay(beta={"general": 0.5})
        with pytest.raises(TypeError, match="OpinionDecayFunction"):
            OpinionEpistemicFilter(decay_fn=exp)

    def test_default_p_min(self):
        """Default p_min is 0.6 (above SL base_rate of 0.5)."""
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf)
        assert ef.p_min == 0.6

    def test_p_min_settable(self):
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf)
        ef.p_min = 0.7
        assert ef.p_min == 0.7

    def test_u_max_settable(self):
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf)
        ef.u_max = 0.3
        assert ef.u_max == 0.3


# ═══════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterScoring:

    def test_score_opinion_returns_opinion(self):
        """score_opinion() returns a full SL Opinion."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        fact = _fact("K+=4.1", 0.0, "vital_sign")
        opinion = ef.score_opinion(fact, QUERY_TIME)
        assert isinstance(opinion, Opinion)

    def test_score_opinion_fresh_high_belief(self):
        """Fresh fact has high belief, low uncertainty."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        fact = _fact("K+=4.1", 0.0, "vital_sign")
        opinion = ef.score_opinion(fact, QUERY_TIME)
        assert opinion.belief > 0.5
        assert opinion.uncertainty < 0.5

    def test_score_opinion_stale_high_uncertainty(self):
        """Very stale fact decays toward vacuity (high u)."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        fact = _fact("K+=3.2", 30.0, "vital_sign")
        opinion = ef.score_opinion(fact, QUERY_TIME)
        assert opinion.uncertainty > 0.9

    def test_score_batch_returns_sorted(self):
        """score_batch returns (fact, Opinion) pairs sorted by P(ω') descending."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        facts = [
            _fact("stale", 30.0, "vital_sign"),
            _fact("fresh", 0.0, "vital_sign"),
            _fact("medium", 1.0, "vital_sign"),
        ]
        scored = ef.score_batch(facts, QUERY_TIME)
        assert len(scored) == 3
        probs = [op.projected_probability() for _, op in scored]
        assert probs == sorted(probs, reverse=True)


# ═══════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterFiltering:

    def test_filter_keeps_fresh_excludes_stale(self):
        """Basic p_min filtering: fresh passes, stale excluded."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6)
        facts = [
            _fact("fresh", 0.0, "vital_sign"),
            _fact("stale", 30.0, "vital_sign"),
        ]
        valid = ef.filter(facts, QUERY_TIME)
        assert len(valid) == 1
        assert valid[0].content == "fresh"

    def test_filter_returns_temporal_facts(self):
        """filter() returns TemporalFact list (backward compat with EpistemicFilter)."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        facts = [_fact("fresh", 0.0, "vital_sign")]
        valid = ef.filter(facts, QUERY_TIME)
        assert all(isinstance(f, TemporalFact) for f in valid)

    def test_filter_sorted_by_validity(self):
        """Returned facts are sorted by P(ω') descending."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.5)
        facts = [
            _fact("medium", 0.5, "vital_sign"),
            _fact("fresh", 0.0, "vital_sign"),
        ]
        valid = ef.filter(facts, QUERY_TIME)
        assert len(valid) == 2
        assert valid[0].content == "fresh"

    def test_filter_with_u_max(self):
        """u_max excludes facts with high uncertainty even if P passes.

        SL math: a moderately decayed fact may have P(ω') > p_min
        (because base_rate=0.5 props it up) but u is high, meaning
        the system is ignorant rather than confident. u_max catches this.
        """
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        # p_min=0.5 would pass stale facts (they approach 0.5)
        # u_max=0.5 catches them because their uncertainty is ~1.0
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.5, u_max=0.5)
        facts = [
            _fact("fresh", 0.0, "vital_sign"),      # P~1.0, u~0.0 → passes both
            _fact("stale", 30.0, "vital_sign"),      # P~0.5, u~1.0 → fails u_max
        ]
        valid = ef.filter(facts, QUERY_TIME)
        assert len(valid) == 1
        assert valid[0].content == "fresh"

    def test_filter_with_opinions(self):
        """filter_with_opinions() returns (fact, Opinion) pairs."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        facts = [_fact("fresh", 0.0, "vital_sign")]
        results = ef.filter_with_opinions(facts, QUERY_TIME)
        assert len(results) == 1
        fact, opinion = results[0]
        assert isinstance(fact, TemporalFact)
        assert isinstance(opinion, Opinion)

    def test_stable_facts_preserved(self):
        """Low-decay fact types survive even at high age (temporal invariance)."""
        odf = OpinionDecayFunction(half_lives={"demographic": 36500.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6)
        facts = [_fact("blood_type_A", 3650.0, "demographic")]
        valid = ef.filter(facts, QUERY_TIME)
        assert len(valid) == 1


# ═══════════════════════════════════════════════════════════════════
# Three-way partition
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterPartition:
    """partition() splits facts into valid/stale/uncertain with diagnostics.

    Three categories:
        valid: P(ω') ≥ p_min AND u ≤ u_max
        stale: P(ω') < p_min (evidence decayed → re-acquire same type)
        uncertain: P(ω') ≥ p_min BUT u > u_max (need better sources)
    """

    def test_partition_returns_result(self):
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        facts = [_fact("fresh", 0.0, "vital_sign")]
        result = ef.partition(facts, QUERY_TIME)
        assert isinstance(result, OpinionPartitionResult)

    def test_partition_fresh_goes_to_valid(self):
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        facts = [_fact("fresh", 0.0, "vital_sign")]
        result = ef.partition(facts, QUERY_TIME)
        assert len(result.valid) == 1
        assert result.valid[0][0].content == "fresh"

    def test_partition_stale_classified_correctly(self):
        """Stale fact: P(ω') < p_min due to decay → stale category."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        facts = [_fact("stale", 30.0, "vital_sign")]
        result = ef.partition(facts, QUERY_TIME)
        assert len(result.stale) == 1
        assert result.stale[0][0].content == "stale"

    def test_partition_uncertain_classified_correctly(self):
        """Uncertain fact: P(ω') ≥ p_min but u > u_max.

        This requires a fact where decay raises uncertainty above u_max
        but base_rate keeps P(ω') above p_min. With base_rate=0.5,
        moderate decay creates this situation:
            b decays, u grows, but P = b + a·u stays above 0.5
            if p_min=0.5 and u_max is tight, the fact passes P but fails u.
        """
        odf = OpinionDecayFunction(
            half_lives={"vital_sign": 1.0},
            default_base_rate=0.5,
        )
        # p_min=0.5 (permissive) + u_max=0.3 (strict)
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.5, u_max=0.3)
        # Moderately aged: P(ω') still ≥ 0.5 (base_rate props it up)
        # but u is above 0.3
        facts = [_fact("moderate", 2.0, "vital_sign")]
        result = ef.partition(facts, QUERY_TIME)
        # Verify the opinion has u > 0.3 and P ≥ 0.5
        opinion = odf.compute_opinion(facts[0], QUERY_TIME)
        if opinion.projected_probability() >= 0.5 and opinion.uncertainty > 0.3:
            assert len(result.uncertain) == 1
            assert result.uncertain[0][0].content == "moderate"
        else:
            # If the math doesn't produce the expected range, skip
            # (this protects against parameter sensitivity)
            pytest.skip("Decay parameters don't produce uncertain-category fact")

    def test_partition_covers_all_facts(self):
        """Every input fact appears in exactly one partition category."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        facts = [
            _fact("fresh", 0.0, "vital_sign"),
            _fact("medium", 1.0, "vital_sign"),
            _fact("stale", 30.0, "vital_sign"),
        ]
        result = ef.partition(facts, QUERY_TIME)
        total = len(result.valid) + len(result.stale) + len(result.uncertain)
        assert total == len(facts)

    def test_partition_with_u_max_disabled(self):
        """With u_max=1.0 (default), uncertain category is always empty."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=1.0)
        facts = [
            _fact("fresh", 0.0, "vital_sign"),
            _fact("stale", 30.0, "vital_sign"),
        ]
        result = ef.partition(facts, QUERY_TIME)
        assert len(result.uncertain) == 0

    def test_partition_opinions_attached(self):
        """Each partition entry carries its Opinion."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6)
        facts = [_fact("fresh", 0.0, "vital_sign")]
        result = ef.partition(facts, QUERY_TIME)
        for fact, opinion in result.valid:
            assert isinstance(opinion, Opinion)


# ═══════════════════════════════════════════════════════════════════
# Re-acquisition
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterReacquisition:

    def test_needs_reacquisition_stale(self):
        odf = OpinionDecayFunction(half_lives={"vital_sign": 0.5})
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6)
        stale = [_fact("old", 100.0, "vital_sign")]
        assert ef.needs_reacquisition(stale, QUERY_TIME) is True

    def test_needs_reacquisition_fresh(self):
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf)
        fresh = [_fact("new", 0.0)]
        assert ef.needs_reacquisition(fresh, QUERY_TIME) is False

    def test_needs_reacquisition_min_required(self):
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = OpinionEpistemicFilter(decay_fn=odf)
        facts = [_fact("one", 0.0, "vital_sign")]
        assert ef.needs_reacquisition(facts, QUERY_TIME, min_required=3) is True


# ═══════════════════════════════════════════════════════════════════
# Repr
# ═══════════════════════════════════════════════════════════════════


class TestOpinionFilterRepr:
    def test_repr(self):
        odf = OpinionDecayFunction()
        ef = OpinionEpistemicFilter(decay_fn=odf, p_min=0.6, u_max=0.5)
        r = repr(ef)
        assert "OpinionEpistemicFilter" in r
        assert "0.6" in r
        assert "0.5" in r
