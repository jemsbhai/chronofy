"""Tests for chronofy.retrieval.filter — EpistemicFilter.

Validates the τ-threshold filtering mechanism (TLDA Layer 2)
and the Retrieval Completeness property: the LLM never receives
evidence below the minimum validity threshold.
"""

from datetime import datetime, timedelta

import pytest

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.retrieval.filter import EpistemicFilter


QUERY_TIME = datetime(2026, 3, 15)


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


@pytest.fixture
def decay() -> ExponentialDecay:
    return ExponentialDecay(beta={"vital_sign": 5.0, "chronic": 0.01, "general": 0.5})


@pytest.fixture
def ef(decay: ExponentialDecay) -> EpistemicFilter:
    return EpistemicFilter(decay_fn=decay, threshold=0.1)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_threshold(self, decay: ExponentialDecay):
        ef = EpistemicFilter(decay_fn=decay, threshold=0.5)
        assert ef.threshold == 0.5

    def test_threshold_boundary_zero(self, decay: ExponentialDecay):
        ef = EpistemicFilter(decay_fn=decay, threshold=0.0)
        assert ef.threshold == 0.0

    def test_threshold_boundary_one(self, decay: ExponentialDecay):
        ef = EpistemicFilter(decay_fn=decay, threshold=1.0)
        assert ef.threshold == 1.0

    def test_invalid_threshold_high(self, decay: ExponentialDecay):
        with pytest.raises(ValueError):
            EpistemicFilter(decay_fn=decay, threshold=1.5)

    def test_invalid_threshold_negative(self, decay: ExponentialDecay):
        with pytest.raises(ValueError):
            EpistemicFilter(decay_fn=decay, threshold=-0.1)

    def test_threshold_setter(self, ef: EpistemicFilter):
        ef.threshold = 0.5
        assert ef.threshold == 0.5

    def test_threshold_setter_invalid(self, ef: EpistemicFilter):
        with pytest.raises(ValueError):
            ef.threshold = 2.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_score_single_fact(self, ef: EpistemicFilter):
        fact = _fact("test", 1.0, "general")
        score = ef.score(fact, QUERY_TIME)
        assert 0.0 <= score <= 1.0

    def test_score_batch_sorted_descending(self, ef: EpistemicFilter):
        """score_batch returns pairs sorted by score descending."""
        facts = [
            _fact("old", 10.0, "general"),
            _fact("new", 0.5, "general"),
            _fact("mid", 3.0, "general"),
        ]
        scored = ef.score_batch(facts, QUERY_TIME)
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_score_batch_preserves_all(self, ef: EpistemicFilter):
        """score_batch returns all facts, not just valid ones."""
        facts = [_fact("a", 0.0), _fact("b", 100.0)]
        scored = ef.score_batch(facts, QUERY_TIME)
        assert len(scored) == 2


# ---------------------------------------------------------------------------
# Filtering — Retrieval Completeness property
# ---------------------------------------------------------------------------

class TestFilter:
    def test_fresh_facts_pass(self, ef: EpistemicFilter):
        """Very recent facts should survive filtering."""
        facts = [_fact("fresh", 0.0, "general")]
        result = ef.filter(facts, QUERY_TIME)
        assert len(result) == 1

    def test_stale_facts_filtered(self, ef: EpistemicFilter):
        """Very old facts with high β should be filtered out."""
        # vital_sign β=5.0, at 1 day: exp(-5) ≈ 0.007 < 0.1
        facts = [_fact("stale_vitals", 1.0, "vital_sign")]
        result = ef.filter(facts, QUERY_TIME)
        assert len(result) == 0

    def test_mixed_filtering(self, ef: EpistemicFilter):
        """Mix of fresh and stale — only fresh survive."""
        facts = [
            _fact("yesterday_vital", 1.0, "vital_sign"),   # exp(-5) < 0.1 → OUT
            _fact("today_vital", 0.0, "vital_sign"),        # exp(0) = 1.0  → IN
            _fact("old_chronic", 365.0, "chronic"),          # exp(-0.01*365) ≈ 0.026 < 0.1 → OUT
            _fact("recent_chronic", 1.0, "chronic"),         # exp(-0.01) ≈ 0.99 → IN
        ]
        result = ef.filter(facts, QUERY_TIME)
        contents = [f.content for f in result]
        assert "today_vital" in contents
        assert "recent_chronic" in contents
        assert "yesterday_vital" not in contents

    def test_filter_returns_sorted(self, ef: EpistemicFilter):
        """Filtered results are in descending validity order."""
        facts = [
            _fact("a", 0.5, "general"),
            _fact("b", 0.1, "general"),
            _fact("c", 0.3, "general"),
        ]
        result = ef.filter(facts, QUERY_TIME)
        # All should pass (general β=0.5, ages < 1 day → scores > 0.1)
        # Verify sorted by freshness (= highest score first)
        ages = [f.age_at(QUERY_TIME) for f in result]
        # Younger = higher score = should come first
        assert ages == sorted(ages)

    def test_filter_with_scores(self, ef: EpistemicFilter):
        """filter_with_scores returns (fact, score) tuples."""
        facts = [_fact("x", 0.0, "general")]
        result = ef.filter_with_scores(facts, QUERY_TIME)
        assert len(result) == 1
        fact, score = result[0]
        assert fact.content == "x"
        assert score == pytest.approx(1.0)

    def test_empty_input(self, ef: EpistemicFilter):
        assert ef.filter([], QUERY_TIME) == []

    def test_quality_affects_filtering(self, ef: EpistemicFilter):
        """Low source quality can push a fact below threshold."""
        # general β=0.5, age=2 days → exp(-1.0) ≈ 0.368 → with q=1.0, passes τ=0.1
        # But with q=0.2: 0.2 * 0.368 ≈ 0.074 < 0.1 → filtered out
        high_q = _fact("good_source", 2.0, "general", quality=1.0)
        low_q = _fact("bad_source", 2.0, "general", quality=0.2)
        result = ef.filter([high_q, low_q], QUERY_TIME)
        contents = [f.content for f in result]
        assert "good_source" in contents
        assert "bad_source" not in contents


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_partition_splits_correctly(self, ef: EpistemicFilter):
        """partition() returns (valid, expired) with no overlap and no loss."""
        facts = [
            _fact("fresh", 0.0, "general"),
            _fact("stale", 100.0, "general"),
        ]
        valid, expired = ef.partition(facts, QUERY_TIME)
        assert len(valid) + len(expired) == len(facts)
        assert len(valid) == 1
        assert valid[0].content == "fresh"
        assert expired[0].content == "stale"

    def test_partition_all_valid(self, ef: EpistemicFilter):
        facts = [_fact("a", 0.0), _fact("b", 0.0)]
        valid, expired = ef.partition(facts, QUERY_TIME)
        assert len(valid) == 2
        assert len(expired) == 0

    def test_partition_all_expired(self, ef: EpistemicFilter):
        facts = [_fact("a", 100.0, "vital_sign"), _fact("b", 100.0, "vital_sign")]
        valid, expired = ef.partition(facts, QUERY_TIME)
        assert len(valid) == 0
        assert len(expired) == 2


# ---------------------------------------------------------------------------
# Sequential Exploration Decay trigger
# ---------------------------------------------------------------------------

class TestReacquisition:
    def test_sufficient_evidence(self, ef: EpistemicFilter):
        """Not triggered when enough valid facts exist."""
        facts = [_fact("a", 0.0), _fact("b", 0.0)]
        assert ef.needs_reacquisition(facts, QUERY_TIME, min_required=1) is False

    def test_insufficient_evidence(self, ef: EpistemicFilter):
        """Triggered when all facts are stale."""
        facts = [_fact("a", 100.0, "vital_sign"), _fact("b", 100.0, "vital_sign")]
        assert ef.needs_reacquisition(facts, QUERY_TIME, min_required=1) is True

    def test_min_required_respected(self, ef: EpistemicFilter):
        """Triggered when valid < min_required even if some valid facts exist."""
        facts = [_fact("a", 0.0, "general")]  # 1 valid fact
        assert ef.needs_reacquisition(facts, QUERY_TIME, min_required=2) is True
        assert ef.needs_reacquisition(facts, QUERY_TIME, min_required=1) is False

    def test_empty_always_triggers(self, ef: EpistemicFilter):
        assert ef.needs_reacquisition([], QUERY_TIME, min_required=1) is True


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self, ef: EpistemicFilter):
        r = repr(ef)
        assert "EpistemicFilter" in r
        assert "0.1" in r
