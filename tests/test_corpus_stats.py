"""Tests for chronofy.analysis.corpus_stats.CorpusStats.

TDD: these tests are written BEFORE the implementation.
All tests must fail (red) before any implementation is written.

Test coverage plan:
  - Empty corpus edge case
  - Single fact
  - Age stats correctness (known values)
  - Validity stats correctness (known decay values)
  - Staleness rate at various thresholds
  - Effective density
  - Temporal span
  - Coverage gaps (simple, multiple, none, edge cases)
  - Summary dict completeness
  - Invariant facts (beta=0, validity=1.0 always)
  - All-stale corpus
  - Works with different decay backends (exponential, linear)
  - CorpusStats is re-usable (calling methods twice gives same result)
  - Threshold boundary conditions (fact exactly at threshold)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from chronofy.analysis.corpus_stats import AgeStats, CorpusStats, ValidityStats
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.linear import LinearDecay
from chronofy.models import TemporalFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2024, 1, 1, 12, 0, 0)


def make_fact(
    age_days: float,
    fact_type: str = "general",
    source_quality: float = 1.0,
) -> TemporalFact:
    """Create a TemporalFact with a known age at QUERY_TIME."""
    ts = QUERY_TIME - timedelta(days=age_days)
    return TemporalFact(
        content=f"fact aged {age_days}d",
        timestamp=ts,
        fact_type=fact_type,
        source_quality=source_quality,
    )


def decay_0() -> ExponentialDecay:
    """Zero-decay (all facts stay valid forever)."""
    return ExponentialDecay(beta={"general": 0.0}, default_beta=0.0)


def decay_fast() -> ExponentialDecay:
    """Fast exponential decay: β=1.0/day for 'general'."""
    return ExponentialDecay(beta={"general": 1.0}, default_beta=1.0)


# ---------------------------------------------------------------------------
# 1. Empty corpus
# ---------------------------------------------------------------------------


class TestEmptyCorpus:
    def test_fact_count_zero(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.fact_count == 0

    def test_temporal_span_zero(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.temporal_span_days == 0.0

    def test_effective_density_zero(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.effective_density == 0.0

    def test_staleness_rate_zero(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.staleness_rate(threshold=0.5) == 0.0

    def test_coverage_gaps_empty(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.coverage_gaps(min_gap_days=1) == []

    def test_age_stats_none(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.age_stats is None

    def test_validity_stats_none(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.validity_stats is None

    def test_summary_has_required_keys(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        summary = stats.summary()
        required_keys = {
            "fact_count",
            "temporal_span_days",
            "effective_density",
            "age_stats",
            "validity_stats",
        }
        assert required_keys.issubset(summary.keys())


# ---------------------------------------------------------------------------
# 2. Single fact
# ---------------------------------------------------------------------------


class TestSingleFact:
    def setup_method(self):
        self.fact = make_fact(age_days=10.0)
        self.stats = CorpusStats(
            facts=[self.fact], query_time=QUERY_TIME, decay_fn=decay_fast()
        )

    def test_fact_count(self):
        assert self.stats.fact_count == 1

    def test_temporal_span_zero(self):
        # Single fact: span between oldest and newest is 0
        assert self.stats.temporal_span_days == 0.0

    def test_age_stats_all_equal(self):
        age_stats = self.stats.age_stats
        assert age_stats is not None
        assert math.isclose(age_stats.mean, 10.0, rel_tol=1e-6)
        assert math.isclose(age_stats.median, 10.0, rel_tol=1e-6)
        assert math.isclose(age_stats.min, 10.0, rel_tol=1e-6)
        assert math.isclose(age_stats.max, 10.0, rel_tol=1e-6)
        assert math.isclose(age_stats.std, 0.0, abs_tol=1e-9)

    def test_validity_stats_correct(self):
        # β=1.0, age=10 days → exp(-10) ≈ 0.0000454
        expected = math.exp(-10.0)
        vstats = self.stats.validity_stats
        assert vstats is not None
        assert math.isclose(vstats.mean, expected, rel_tol=1e-6)
        assert math.isclose(vstats.min, expected, rel_tol=1e-6)
        assert math.isclose(vstats.max, expected, rel_tol=1e-6)

    def test_effective_density_matches_validity(self):
        expected = math.exp(-10.0)
        assert math.isclose(self.stats.effective_density, expected, rel_tol=1e-6)

    def test_staleness_rate_above_threshold(self):
        # validity ≈ 0.0000454 < 0.5 → stale
        assert self.stats.staleness_rate(threshold=0.5) == 1.0

    def test_staleness_rate_below_threshold(self):
        # validity ≈ 0.0000454 > 0.0 → not stale
        assert self.stats.staleness_rate(threshold=0.0) == 0.0

    def test_coverage_gaps_single_fact(self):
        # Only one timestamp, no gaps possible
        assert self.stats.coverage_gaps(min_gap_days=1) == []


# ---------------------------------------------------------------------------
# 3. Age stats with known values
# ---------------------------------------------------------------------------


class TestAgeStats:
    def setup_method(self):
        # ages: 0, 2, 4, 6, 8 days
        self.facts = [make_fact(age_days=float(a)) for a in [0, 2, 4, 6, 8]]
        self.stats = CorpusStats(
            facts=self.facts, query_time=QUERY_TIME, decay_fn=decay_0()
        )

    def test_fact_count(self):
        assert self.stats.fact_count == 5

    def test_age_min(self):
        assert math.isclose(self.stats.age_stats.min, 0.0, abs_tol=1e-6)

    def test_age_max(self):
        assert math.isclose(self.stats.age_stats.max, 8.0, rel_tol=1e-6)

    def test_age_mean(self):
        # (0+2+4+6+8)/5 = 4.0
        assert math.isclose(self.stats.age_stats.mean, 4.0, rel_tol=1e-6)

    def test_age_median(self):
        assert math.isclose(self.stats.age_stats.median, 4.0, rel_tol=1e-6)

    def test_age_std(self):
        # std of [0,2,4,6,8] = sqrt(8) ≈ 2.828
        expected_std = math.sqrt(8.0)
        assert math.isclose(self.stats.age_stats.std, expected_std, rel_tol=1e-5)

    def test_temporal_span(self):
        # max_age - min_age = 8 - 0 = 8
        assert math.isclose(self.stats.temporal_span_days, 8.0, rel_tol=1e-6)

    def test_age_stats_percentile_25(self):
        assert self.stats.age_stats.p25 is not None
        assert self.stats.age_stats.p25 >= 0.0
        assert self.stats.age_stats.p25 <= 4.0

    def test_age_stats_percentile_75(self):
        assert self.stats.age_stats.p75 is not None
        assert self.stats.age_stats.p75 >= 4.0
        assert self.stats.age_stats.p75 <= 8.0


# ---------------------------------------------------------------------------
# 4. Validity stats with known decay values
# ---------------------------------------------------------------------------


class TestValidityStats:
    def setup_method(self):
        # β=1.0: validities = exp(-0), exp(-1), exp(-2), exp(-3), exp(-4)
        self.ages = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.expected_validities = [math.exp(-a) for a in self.ages]
        self.facts = [make_fact(age_days=a) for a in self.ages]
        self.stats = CorpusStats(
            facts=self.facts, query_time=QUERY_TIME, decay_fn=decay_fast()
        )

    def test_validity_min(self):
        assert math.isclose(
            self.stats.validity_stats.min, min(self.expected_validities), rel_tol=1e-6
        )

    def test_validity_max(self):
        assert math.isclose(
            self.stats.validity_stats.max, max(self.expected_validities), rel_tol=1e-6
        )

    def test_validity_mean(self):
        expected_mean = sum(self.expected_validities) / len(self.expected_validities)
        assert math.isclose(
            self.stats.validity_stats.mean, expected_mean, rel_tol=1e-6
        )

    def test_effective_density_matches_mean(self):
        expected_mean = sum(self.expected_validities) / len(self.expected_validities)
        assert math.isclose(self.stats.effective_density, expected_mean, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 5. Staleness rate
# ---------------------------------------------------------------------------


class TestStalenessRate:
    def setup_method(self):
        # β=1: age=0 → v=1.0, age=1 → v≈0.368, age=2 → v≈0.135
        self.facts = [make_fact(age_days=a) for a in [0.0, 1.0, 2.0]]
        self.stats = CorpusStats(
            facts=self.facts, query_time=QUERY_TIME, decay_fn=decay_fast()
        )

    def test_all_stale_high_threshold(self):
        # threshold=1.1 → all below → rate=1.0
        assert self.stats.staleness_rate(threshold=1.1) == 1.0

    def test_none_stale_zero_threshold(self):
        # threshold=0.0 → none below → rate=0.0
        assert self.stats.staleness_rate(threshold=0.0) == 0.0

    def test_partial_staleness(self):
        # threshold=0.5 → only age=0 (v=1.0) survives → 2/3 stale
        rate = self.stats.staleness_rate(threshold=0.5)
        assert math.isclose(rate, 2 / 3, rel_tol=1e-6)

    def test_threshold_boundary(self):
        # Fact at exactly exp(-1) ≈ 0.3679 — boundary fact is NOT stale
        # (validity >= threshold means NOT stale; strict < means stale)
        threshold = math.exp(-1.0)
        rate = self.stats.staleness_rate(threshold=threshold)
        # age=0 (v=1.0 >= threshold) and age=1 (v=exp(-1) >= threshold) are fresh
        # age=2 (v=exp(-2) < threshold) is stale → 1/3 stale
        assert math.isclose(rate, 1 / 3, rel_tol=1e-6)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            self.stats.staleness_rate(threshold=-0.1)

    def test_threshold_above_one_returns_all_stale(self):
        # threshold=1.5 — validity scores are always <= 1.0, so all facts are stale
        assert self.stats.staleness_rate(threshold=1.5) == 1.0


# ---------------------------------------------------------------------------
# 6. Coverage gaps
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_no_gaps_dense_corpus(self):
        # Facts every day for 5 days — no gap >= 3 days
        facts = [make_fact(age_days=float(d)) for d in range(6)]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.coverage_gaps(min_gap_days=3) == []

    def test_single_gap(self):
        # Facts at day 0 and day 10 — one gap of 10 days
        facts = [make_fact(age_days=0.0), make_fact(age_days=10.0)]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        gaps = stats.coverage_gaps(min_gap_days=5)
        assert len(gaps) == 1
        start, end = gaps[0]
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        gap_days = (end - start).total_seconds() / 86400
        assert math.isclose(gap_days, 10.0, rel_tol=1e-5)

    def test_multiple_gaps(self):
        # Facts at days 0, 20, 40 — two gaps of 20 days each
        facts = [make_fact(age_days=float(d)) for d in [0, 20, 40]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        gaps = stats.coverage_gaps(min_gap_days=5)
        assert len(gaps) == 2

    def test_gap_below_min_threshold_excluded(self):
        # Facts at days 0, 3, 10 — gap of 3 days between 0 and 3, gap of 7 between 3 and 10
        facts = [make_fact(age_days=float(d)) for d in [0, 3, 10]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        gaps = stats.coverage_gaps(min_gap_days=5)
        # Only the 7-day gap qualifies
        assert len(gaps) == 1

    def test_coverage_gaps_are_sorted_chronologically(self):
        # Gaps should be returned chronologically (oldest gap first)
        facts = [make_fact(age_days=float(d)) for d in [0, 30, 60]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        gaps = stats.coverage_gaps(min_gap_days=5)
        assert len(gaps) == 2
        # First gap should be older (larger age) than second
        start0, _ = gaps[0]
        start1, _ = gaps[1]
        assert start0 < start1

    def test_duplicate_timestamps_no_false_gaps(self):
        # Two facts with identical timestamps should not produce a zero-length gap
        ts = QUERY_TIME - timedelta(days=5)
        f1 = TemporalFact(content="a", timestamp=ts, fact_type="general")
        f2 = TemporalFact(content="b", timestamp=ts, fact_type="general")
        stats = CorpusStats(facts=[f1, f2], query_time=QUERY_TIME, decay_fn=decay_0())
        assert stats.coverage_gaps(min_gap_days=1) == []

    def test_invalid_min_gap_raises(self):
        stats = CorpusStats(
            facts=[make_fact(1.0)], query_time=QUERY_TIME, decay_fn=decay_0()
        )
        with pytest.raises(ValueError):
            stats.coverage_gaps(min_gap_days=0)


# ---------------------------------------------------------------------------
# 7. Invariant facts (beta=0)
# ---------------------------------------------------------------------------


class TestInvariantFacts:
    def test_all_facts_valid_forever(self):
        # Zero decay: all facts should have validity = source_quality = 1.0
        facts = [make_fact(age_days=float(d)) for d in [0, 100, 1000, 10000]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_0())
        assert math.isclose(stats.validity_stats.min, 1.0, rel_tol=1e-9)
        assert math.isclose(stats.validity_stats.max, 1.0, rel_tol=1e-9)
        assert math.isclose(stats.effective_density, 1.0, rel_tol=1e-9)
        assert stats.staleness_rate(threshold=0.5) == 0.0


# ---------------------------------------------------------------------------
# 8. Source quality affects validity
# ---------------------------------------------------------------------------


class TestSourceQuality:
    def test_low_quality_reduces_validity(self):
        # age=0, β=0 → validity = source_quality
        fact_high = TemporalFact(
            content="high", timestamp=QUERY_TIME, fact_type="general", source_quality=1.0
        )
        fact_low = TemporalFact(
            content="low", timestamp=QUERY_TIME, fact_type="general", source_quality=0.5
        )
        stats = CorpusStats(
            facts=[fact_high, fact_low], query_time=QUERY_TIME, decay_fn=decay_0()
        )
        assert math.isclose(stats.validity_stats.min, 0.5, rel_tol=1e-9)
        assert math.isclose(stats.validity_stats.max, 1.0, rel_tol=1e-9)
        assert math.isclose(stats.effective_density, 0.75, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 9. Works with a different decay backend (LinearDecay)
# ---------------------------------------------------------------------------


class TestDifferentDecayBackend:
    def test_linear_decay_accepted(self):
        # CorpusStats must accept any DecayFunction, not just ExponentialDecay
        facts = [make_fact(age_days=1.0), make_fact(age_days=2.0)]
        linear = LinearDecay(rate={"general": 0.1}, default_rate=0.1)
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=linear)
        # Just verify it runs and produces plausible values
        assert 0.0 <= stats.effective_density <= 1.0
        assert stats.validity_stats is not None


# ---------------------------------------------------------------------------
# 10. Summary dict
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys_nonempty_corpus(self):
        facts = [make_fact(age_days=float(d)) for d in [0, 5, 10]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_fast())
        summary = stats.summary()
        required = {
            "fact_count",
            "temporal_span_days",
            "effective_density",
            "age_stats",
            "validity_stats",
        }
        assert required.issubset(summary.keys())

    def test_summary_age_stats_is_dict(self):
        facts = [make_fact(age_days=float(d)) for d in [0, 5, 10]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_fast())
        summary = stats.summary()
        assert isinstance(summary["age_stats"], dict)

    def test_summary_validity_stats_is_dict(self):
        facts = [make_fact(age_days=float(d)) for d in [0, 5, 10]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_fast())
        summary = stats.summary()
        assert isinstance(summary["validity_stats"], dict)

    def test_summary_empty_corpus_age_stats_none(self):
        stats = CorpusStats(facts=[], query_time=QUERY_TIME, decay_fn=decay_0())
        summary = stats.summary()
        assert summary["age_stats"] is None
        assert summary["validity_stats"] is None


# ---------------------------------------------------------------------------
# 11. Idempotency: calling methods twice gives same result
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_calling_methods_twice_is_stable(self):
        facts = [make_fact(age_days=float(d)) for d in [1, 5, 10]]
        stats = CorpusStats(facts=facts, query_time=QUERY_TIME, decay_fn=decay_fast())
        assert stats.effective_density == stats.effective_density
        assert stats.staleness_rate(0.5) == stats.staleness_rate(0.5)
        assert stats.coverage_gaps(5) == stats.coverage_gaps(5)
        assert stats.temporal_span_days == stats.temporal_span_days
