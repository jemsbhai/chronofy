"""Tests for chronofy.sl.fusion — Temporal evidence fusion.

Validates multi-source evidence fusion via decay → fuse pipeline:

- TemporalEvidenceFusion decays each fact to an Opinion, then fuses
- Cumulative fusion for independent sources (reduces uncertainty)
- Averaging fusion for correlated sources (prevents double-counting)
- Byzantine filtering removes adversarial/outlier evidence
- Diminishing returns: fusing redundant evidence has decreasing marginal gain
  (ASEV Axiom 6: redundancy / diminishing returns)
- FusionReport carries full diagnostics

References:
    Jøsang (2016) §12.3 (cumulative), §12.5 (averaging), §10.4 (aging)
    ASEV Axiom 6: concavity of mutual information under repeated observation
"""

from datetime import datetime, timedelta

import pytest

jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction, OpinionConfig
from chronofy.sl.fusion import (
    TemporalEvidenceFusion,
    FusionReport,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2026, 3, 19)


def _fact(
    content: str = "K+ = 4.1",
    days_ago: float = 0.0,
    fact_type: str = "general",
    quality: float = 1.0,
    source: str | None = None,
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
        source=source,
    )


@pytest.fixture
def odf() -> OpinionDecayFunction:
    return OpinionDecayFunction(
        half_lives={"general": 7.0, "vital_sign": 1.0},
        construction="confidence",
        base_uncertainty=0.1,
    )


@pytest.fixture
def fuser(odf) -> TemporalEvidenceFusion:
    return TemporalEvidenceFusion(decay_fn=odf)


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestConstruction:
    """TemporalEvidenceFusion setup and configuration."""

    def test_default_construction(self, odf):
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        assert fuser.fusion_method == "cumulative"

    def test_averaging_mode(self, odf):
        fuser = TemporalEvidenceFusion(decay_fn=odf, fusion_method="averaging")
        assert fuser.fusion_method == "averaging"

    def test_invalid_method_raises(self, odf):
        with pytest.raises(ValueError, match="fusion_method"):
            TemporalEvidenceFusion(decay_fn=odf, fusion_method="invalid")


# ===========================================================================
# 2. Basic Fusion
# ===========================================================================


class TestBasicFusion:
    """Core decay → fuse pipeline."""

    def test_single_fact_returns_its_opinion(self, fuser):
        """Fusing one item should return the decayed opinion as-is."""
        facts = [_fact(days_ago=3)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert isinstance(report, FusionReport)
        assert isinstance(report.fused_opinion, Opinion)
        assert len(report.decayed_opinions) == 1
        # Single-item fusion: fused == decayed
        assert report.fused_opinion.belief == pytest.approx(
            report.decayed_opinions[0].belief, abs=1e-9
        )

    def test_two_facts_returns_opinion(self, fuser):
        facts = [_fact(days_ago=1), _fact(days_ago=2)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert isinstance(report.fused_opinion, Opinion)
        assert len(report.decayed_opinions) == 2

    def test_many_facts_returns_opinion(self, fuser):
        facts = [_fact(days_ago=i) for i in range(10)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert isinstance(report.fused_opinion, Opinion)
        assert len(report.decayed_opinions) == 10

    def test_empty_facts_raises(self, fuser):
        with pytest.raises(ValueError, match="at least one"):
            fuser.fuse([], QUERY_TIME)

    def test_additivity_preserved_in_fused(self, fuser):
        """b + d + u = 1 for the fused opinion."""
        facts = [_fact(days_ago=1), _fact(days_ago=5), _fact(days_ago=10)]
        report = fuser.fuse(facts, QUERY_TIME)
        o = report.fused_opinion
        assert o.belief + o.disbelief + o.uncertainty == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 3. Cumulative Fusion — Independent Sources
# ===========================================================================


class TestCumulativeFusion:
    """Cumulative fusion (⊕) for independent evidence sources."""

    def test_cumulative_reduces_uncertainty(self, odf):
        """Key SL property: independent evidence reduces uncertainty."""
        fuser = TemporalEvidenceFusion(decay_fn=odf, fusion_method="cumulative")
        one_fact = [_fact(days_ago=2)]
        two_facts = [_fact(days_ago=2), _fact(days_ago=2, content="K+ = 4.2")]

        report_one = fuser.fuse(one_fact, QUERY_TIME)
        report_two = fuser.fuse(two_facts, QUERY_TIME)

        assert report_two.fused_opinion.uncertainty < report_one.fused_opinion.uncertainty

    def test_cumulative_fresh_evidence_dominates(self, odf):
        """Newer evidence contributes more after decay (less uncertainty)."""
        fuser = TemporalEvidenceFusion(decay_fn=odf, fusion_method="cumulative")
        facts = [
            _fact(days_ago=1, content="fresh"),   # recent
            _fact(days_ago=30, content="stale"),   # old
        ]
        report = fuser.fuse(facts, QUERY_TIME)
        # The fused opinion should be influenced more by the fresh reading
        # since the stale one has higher uncertainty after decay
        fresh_only = fuser.fuse([facts[0]], QUERY_TIME)
        stale_only = fuser.fuse([facts[1]], QUERY_TIME)
        # Fused belief should be closer to fresh than to stale
        fused_b = report.fused_opinion.belief
        fresh_b = fresh_only.fused_opinion.belief
        stale_b = stale_only.fused_opinion.belief
        assert abs(fused_b - fresh_b) < abs(fused_b - stale_b)


# ===========================================================================
# 4. Averaging Fusion — Correlated Sources
# ===========================================================================


class TestAveragingFusion:
    """Averaging fusion (⊘) for correlated/dependent evidence sources."""

    def test_averaging_does_not_overclaim(self, odf):
        """Averaging fusion should NOT reduce uncertainty as much as cumulative."""
        cumulative = TemporalEvidenceFusion(decay_fn=odf, fusion_method="cumulative")
        averaging = TemporalEvidenceFusion(decay_fn=odf, fusion_method="averaging")

        facts = [_fact(days_ago=2), _fact(days_ago=3)]
        report_cum = cumulative.fuse(facts, QUERY_TIME)
        report_avg = averaging.fuse(facts, QUERY_TIME)

        # Averaging should preserve more uncertainty than cumulative
        assert report_avg.fused_opinion.uncertainty >= report_cum.fused_opinion.uncertainty

    def test_averaging_idempotent_property(self, odf):
        """SL property: A ⊘ A = A (idempotent — same evidence twice doesn't help)."""
        fuser = TemporalEvidenceFusion(decay_fn=odf, fusion_method="averaging")
        fact = _fact(days_ago=3, quality=0.8)
        one = fuser.fuse([fact], QUERY_TIME)
        two = fuser.fuse([fact, fact], QUERY_TIME)
        # Fused opinion of identical items should ≈ single item
        assert one.fused_opinion.belief == pytest.approx(
            two.fused_opinion.belief, abs=0.01
        )
        assert one.fused_opinion.uncertainty == pytest.approx(
            two.fused_opinion.uncertainty, abs=0.01
        )


# ===========================================================================
# 5. Diminishing Returns (ASEV Axiom 6)
# ===========================================================================


class TestDiminishingReturns:
    """Cumulative fusion exhibits concavity of marginal information gain."""

    def test_marginal_uncertainty_reduction_decreases(self, odf):
        """Adding the Nth observation reduces uncertainty less than the (N-1)th.

        This is the formal manifestation of ASEV Axiom 6:
        the marginal ASEV of a new item falls when similar evidence
        is already present.
        """
        fuser = TemporalEvidenceFusion(decay_fn=odf, fusion_method="cumulative")

        # Build up a list of identical-age facts
        base_fact = _fact(days_ago=2, quality=0.8)
        uncertainties = []
        for n in range(1, 6):
            report = fuser.fuse([base_fact] * n, QUERY_TIME)
            uncertainties.append(report.fused_opinion.uncertainty)

        # Compute marginal reductions
        marginal_reductions = [
            uncertainties[i] - uncertainties[i + 1]
            for i in range(len(uncertainties) - 1)
        ]
        # Each marginal reduction should be <= the previous
        for i in range(len(marginal_reductions) - 1):
            assert marginal_reductions[i] >= marginal_reductions[i + 1] - 1e-9


# ===========================================================================
# 6. Byzantine Filtering
# ===========================================================================


class TestByzantineFusion:
    """Decay → Byzantine filter → fuse pipeline."""

    def test_byzantine_removes_outlier(self, odf):
        """An adversarial reading wildly different from the majority should be filtered."""
        fuser = TemporalEvidenceFusion(
            decay_fn=odf, fusion_method="cumulative", byzantine=True,
        )
        honest = [
            _fact(days_ago=1, quality=0.9, content="K+ = 4.1"),
            _fact(days_ago=1, quality=0.85, content="K+ = 4.0"),
            _fact(days_ago=2, quality=0.9, content="K+ = 4.2"),
        ]
        adversarial = [
            _fact(days_ago=1, quality=0.1, content="K+ = FAKE"),
        ]
        report = fuser.fuse(honest + adversarial, QUERY_TIME)
        assert report.removed_count >= 1

    def test_byzantine_disabled_by_default(self, odf):
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        facts = [_fact(days_ago=1, quality=0.9), _fact(days_ago=1, quality=0.1)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert report.removed_count == 0

    def test_byzantine_all_honest_none_removed(self, odf):
        fuser = TemporalEvidenceFusion(
            decay_fn=odf, fusion_method="cumulative", byzantine=True,
        )
        facts = [
            _fact(days_ago=1, quality=0.9),
            _fact(days_ago=2, quality=0.85),
            _fact(days_ago=1, quality=0.88),
        ]
        report = fuser.fuse(facts, QUERY_TIME)
        assert report.removed_count == 0


# ===========================================================================
# 7. FusionReport
# ===========================================================================


class TestFusionReport:
    """FusionReport carries full diagnostics."""

    def test_report_has_fused_opinion(self, fuser):
        report = fuser.fuse([_fact(days_ago=1)], QUERY_TIME)
        assert isinstance(report.fused_opinion, Opinion)

    def test_report_has_decayed_opinions(self, fuser):
        facts = [_fact(days_ago=1), _fact(days_ago=5)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert len(report.decayed_opinions) == 2
        assert all(isinstance(o, Opinion) for o in report.decayed_opinions)

    def test_report_has_projected_probability(self, fuser):
        report = fuser.fuse([_fact(days_ago=1)], QUERY_TIME)
        assert 0.0 <= report.projected_probability <= 1.0

    def test_projected_probability_matches_fused(self, fuser):
        report = fuser.fuse([_fact(days_ago=3)], QUERY_TIME)
        assert report.projected_probability == pytest.approx(
            report.fused_opinion.projected_probability(), abs=1e-9
        )

    def test_report_has_source_count(self, fuser):
        facts = [_fact(days_ago=i) for i in range(4)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert report.source_count == 4

    def test_report_has_removed_count(self, fuser):
        report = fuser.fuse([_fact(days_ago=1)], QUERY_TIME)
        assert report.removed_count == 0

    def test_report_has_fusion_method(self, fuser):
        report = fuser.fuse([_fact(days_ago=1)], QUERY_TIME)
        assert report.fusion_method == "cumulative"

    def test_report_has_query_time(self, fuser):
        report = fuser.fuse([_fact(days_ago=1)], QUERY_TIME)
        assert report.query_time == QUERY_TIME


# ===========================================================================
# 8. Integration with OpinionDecayFunction
# ===========================================================================


class TestIntegration:
    """Fusion interoperates correctly with both construction modes."""

    def test_evidence_mode_fusion(self):
        odf = OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="evidence",
        )
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        facts = [_fact(days_ago=1, quality=0.9), _fact(days_ago=2, quality=0.8)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert isinstance(report.fused_opinion, Opinion)
        assert report.fused_opinion.uncertainty < 1.0

    def test_confidence_mode_fusion(self):
        odf = OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="confidence",
            base_uncertainty=0.1,
        )
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        facts = [_fact(days_ago=1, quality=0.9), _fact(days_ago=2, quality=0.8)]
        report = fuser.fuse(facts, QUERY_TIME)
        assert isinstance(report.fused_opinion, Opinion)

    def test_mixed_fact_types(self):
        odf = OpinionDecayFunction(
            half_lives={"vital_sign": 1.0, "chronic": 365.0},
            construction="confidence",
            base_uncertainty=0.1,
        )
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        facts = [
            _fact(days_ago=3, fact_type="vital_sign", quality=0.9),
            _fact(days_ago=3, fact_type="chronic", quality=0.8),
        ]
        report = fuser.fuse(facts, QUERY_TIME)
        # Vital sign decays faster, so it contributes less
        assert isinstance(report.fused_opinion, Opinion)

    def test_fuse_to_scalar(self):
        """Convenience: projected_probability gives a drop-in scalar."""
        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        facts = [_fact(days_ago=1), _fact(days_ago=2)]
        report = fuser.fuse(facts, QUERY_TIME)
        scalar = report.projected_probability
        assert isinstance(scalar, float)
        assert 0.0 <= scalar <= 1.0
