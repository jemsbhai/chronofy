"""Tests for chronofy.sl.conflict — Pairwise conflict detection in evidence sets.

Validates that ConflictDetector correctly identifies contradictory evidence
before fusion or LLM consumption, providing:

- Pairwise conflict matrix (symmetric, via Jøsang §12.3.4)
- Discord scores: per-opinion mean conflict against all peers
- Cohesion score: aggregate group agreement (1.0 - max discord)
- Conflict pairs: (i, j, score) tuples above a configurable threshold
- Full integration with OpinionDecayFunction (decay → detect pipeline)

Scientific invariants tested:
- Conflict matrix symmetry: con(A, B) = con(B, A)
- Vacuous opinions have zero pairwise conflict (no evidence to disagree)
- Agreeing opinions have zero/near-zero conflict
- Opposing opinions (b_A·d_B + d_A·b_B) have maximal conflict
- Cohesion score is 1.0 for a perfectly agreeing group
- Temporal decay increases uncertainty, which reduces conflict magnitude
- Single-opinion sets are trivially cohesive (cohesion = 1.0)

References:
    Jøsang (2016) §12.3.4 (Pairwise Conflict), §12.3.4 (Discord).
    ASEV Evaluation, Axiom 6: Redundancy / Diminishing Returns —
        conflict detection is a prerequisite for safe fusion.
"""

from datetime import datetime, timedelta

import pytest

jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion, pairwise_conflict

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction, OpinionConfig
from chronofy.sl.conflict import (
    ConflictDetector,
    ConflictReport,
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


def _opinion(b: float, d: float, u: float, a: float = 0.5) -> Opinion:
    """Shorthand for Opinion construction."""
    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


@pytest.fixture
def odf() -> OpinionDecayFunction:
    """Standard OpinionDecayFunction for tests."""
    return OpinionDecayFunction(
        half_lives={"general": 7.0, "vital_sign": 1.0},
        construction="confidence",
        base_uncertainty=0.1,
    )


@pytest.fixture
def detector(odf) -> ConflictDetector:
    """Default ConflictDetector."""
    return ConflictDetector(decay_fn=odf)


# ===========================================================================
# 1. Construction and Validation
# ===========================================================================


class TestConstruction:
    """ConflictDetector setup and configuration."""

    def test_default_construction(self, odf):
        det = ConflictDetector(decay_fn=odf)
        assert det is not None

    def test_custom_threshold(self, odf):
        det = ConflictDetector(decay_fn=odf, default_threshold=0.3)
        assert det._default_threshold == 0.3

    def test_threshold_must_be_positive(self, odf):
        with pytest.raises(ValueError, match="threshold"):
            ConflictDetector(decay_fn=odf, default_threshold=-0.1)

    def test_threshold_must_be_at_most_one(self, odf):
        with pytest.raises(ValueError, match="threshold"):
            ConflictDetector(decay_fn=odf, default_threshold=1.5)

    def test_repr(self, detector):
        r = repr(detector)
        assert "ConflictDetector" in r


# ===========================================================================
# 2. ConflictReport Structure
# ===========================================================================


class TestConflictReport:
    """ConflictReport carries correct diagnostics."""

    def test_report_from_two_agreeing_facts(self, detector):
        facts = [_fact("K+ = 4.1"), _fact("K+ = 4.2")]
        report = detector.detect(facts, QUERY_TIME)

        assert isinstance(report, ConflictReport)
        assert report.source_count == 2
        assert len(report.decayed_opinions) == 2
        assert len(report.conflict_matrix) == 2
        assert len(report.conflict_matrix[0]) == 2
        assert len(report.discord_scores) == 2

    def test_report_threshold_stored(self, detector):
        facts = [_fact("a"), _fact("b")]
        report = detector.detect(facts, QUERY_TIME, threshold=0.25)
        assert report.threshold == 0.25

    def test_report_uses_default_threshold(self, odf):
        det = ConflictDetector(decay_fn=odf, default_threshold=0.42)
        facts = [_fact("a"), _fact("b")]
        report = det.detect(facts, QUERY_TIME)
        assert report.threshold == 0.42

    def test_report_explicit_threshold_overrides_default(self, odf):
        det = ConflictDetector(decay_fn=odf, default_threshold=0.42)
        facts = [_fact("a"), _fact("b")]
        report = det.detect(facts, QUERY_TIME, threshold=0.1)
        assert report.threshold == 0.1

    def test_cohesion_score_in_zero_one(self, detector):
        facts = [_fact("a"), _fact("b"), _fact("c")]
        report = detector.detect(facts, QUERY_TIME)
        assert 0.0 <= report.cohesion_score <= 1.0


# ===========================================================================
# 3. Single-Fact Edge Case
# ===========================================================================


class TestSingleFact:
    """A single fact is trivially non-conflicting."""

    def test_single_fact_cohesion_is_one(self, detector):
        report = detector.detect([_fact("K+ = 4.1")], QUERY_TIME)
        assert report.cohesion_score == 1.0

    def test_single_fact_no_conflict_pairs(self, detector):
        report = detector.detect([_fact("K+ = 4.1")], QUERY_TIME)
        assert report.conflict_pairs == []

    def test_single_fact_matrix_is_1x1_zero(self, detector):
        report = detector.detect([_fact("K+ = 4.1")], QUERY_TIME)
        assert report.conflict_matrix == [[0.0]]

    def test_single_fact_discord_is_zero(self, detector):
        report = detector.detect([_fact("K+ = 4.1")], QUERY_TIME)
        assert report.discord_scores == [0.0]


# ===========================================================================
# 4. Empty Input
# ===========================================================================


class TestEmptyInput:
    """Empty fact list raises ValueError."""

    def test_detect_empty_raises(self, detector):
        with pytest.raises(ValueError, match="at least one"):
            detector.detect([], QUERY_TIME)

    def test_detect_from_opinions_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ConflictDetector.detect_from_opinions([])


# ===========================================================================
# 5. Agreeing Opinions — Low Conflict
# ===========================================================================


class TestAgreement:
    """When all sources agree, conflict should be near zero."""

    def test_identical_high_quality_fresh_facts(self, detector):
        """Identical facts: same content, same quality, same time.

        With base_uncertainty=0.1, quality=0.9 maps to (b=0.81, d=0.09, u=0.10).
        Even identical opinions have nonzero pairwise conflict because d > 0:
            con = b·d + d·b = 2 × 0.81 × 0.09 = 0.1458
        This is a real SL property: any opinion with both b > 0 and d > 0
        has nonzero conflict with copies of itself. Cohesion ≈ 0.854.
        """
        facts = [_fact("K+ = 4.1", quality=0.9) for _ in range(3)]
        report = detector.detect(facts, QUERY_TIME)
        assert report.cohesion_score > 0.80
        assert report.conflict_pairs == []

    def test_opinions_with_same_direction(self):
        """All opinions point positive (high belief) → near-zero conflict."""
        opinions = [
            _opinion(0.7, 0.1, 0.2),
            _opinion(0.8, 0.05, 0.15),
            _opinion(0.65, 0.1, 0.25),
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.15)
        # All believe the same thing — conflict should be low
        for score in report.discord_scores:
            assert score < 0.15

    def test_dogmatic_agreement(self):
        """Two dogmatic opinions that agree: b=1, d=0 → zero conflict."""
        opinions = [
            _opinion(1.0, 0.0, 0.0),
            _opinion(1.0, 0.0, 0.0),
        ]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.conflict_matrix[0][1] == 0.0
        assert report.cohesion_score == 1.0


# ===========================================================================
# 6. Disagreeing Opinions — High Conflict
# ===========================================================================


class TestDisagreement:
    """Opposing opinions should produce high conflict scores."""

    def test_dogmatic_opposition(self):
        """One fully believes, the other fully disbelieves → conflict = 1.0."""
        opinions = [
            _opinion(1.0, 0.0, 0.0),
            _opinion(0.0, 1.0, 0.0),
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.15)
        assert report.conflict_matrix[0][1] == pytest.approx(1.0)
        assert report.conflict_matrix[1][0] == pytest.approx(1.0)
        assert report.cohesion_score == pytest.approx(0.0)
        assert len(report.conflict_pairs) == 1

    def test_strong_opposition_with_uncertainty(self):
        """Strong disagreement moderated by uncertainty."""
        opinions = [
            _opinion(0.7, 0.1, 0.2),
            _opinion(0.1, 0.7, 0.2),
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.1)
        # con = 0.7*0.7 + 0.1*0.1 = 0.49 + 0.01 = 0.50
        assert report.conflict_matrix[0][1] == pytest.approx(0.50)
        assert len(report.conflict_pairs) == 1

    def test_conflict_pair_tuple_format(self):
        """Conflict pairs carry (i, j, score) with i < j."""
        opinions = [
            _opinion(0.8, 0.1, 0.1),
            _opinion(0.1, 0.8, 0.1),
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.1)
        assert len(report.conflict_pairs) == 1
        i, j, score = report.conflict_pairs[0]
        assert i < j  # canonical ordering
        assert score == pytest.approx(report.conflict_matrix[i][j])


# ===========================================================================
# 7. Mixed Scenarios — One Outlier
# ===========================================================================


class TestMixedScenarios:
    """Group of agreeing opinions plus one outlier."""

    def test_one_outlier_among_three(self):
        """Two agree, one disagrees → outlier has highest discord."""
        agreeing_1 = _opinion(0.8, 0.05, 0.15)
        agreeing_2 = _opinion(0.75, 0.05, 0.20)
        outlier = _opinion(0.05, 0.8, 0.15)

        report = ConflictDetector.detect_from_opinions(
            [agreeing_1, agreeing_2, outlier], threshold=0.1
        )
        # Outlier (index 2) should have the highest discord
        assert report.discord_scores[2] > report.discord_scores[0]
        assert report.discord_scores[2] > report.discord_scores[1]
        # Conflict pairs should include (0, 2) and (1, 2) but not (0, 1)
        flagged_pairs = {(i, j) for i, j, _ in report.conflict_pairs}
        assert (0, 2) in flagged_pairs
        assert (1, 2) in flagged_pairs

    def test_cohesion_lower_with_outlier(self):
        """Adding an outlier lowers cohesion."""
        agreeing = [_opinion(0.8, 0.05, 0.15), _opinion(0.75, 0.1, 0.15)]
        report_clean = ConflictDetector.detect_from_opinions(agreeing)

        with_outlier = agreeing + [_opinion(0.05, 0.85, 0.10)]
        report_dirty = ConflictDetector.detect_from_opinions(with_outlier)

        assert report_dirty.cohesion_score < report_clean.cohesion_score


# ===========================================================================
# 8. Vacuous Opinions — Zero Conflict (SL Mathematical Invariant)
# ===========================================================================


class TestVacuousOpinions:
    """Vacuous opinions (u=1) have zero pairwise conflict.

    This is a mathematical invariant of Jøsang's conflict formula:
        con(A, B) = b_A·d_B + d_A·b_B
    When b = d = 0 (vacuous), all terms are zero.
    """

    def test_two_vacuous_opinions(self):
        vacuous = _opinion(0.0, 0.0, 1.0)
        report = ConflictDetector.detect_from_opinions([vacuous, vacuous])
        assert report.conflict_matrix[0][1] == 0.0
        assert report.cohesion_score == 1.0

    def test_vacuous_vs_dogmatic(self):
        """A vacuous opinion cannot conflict with anything — it carries no evidence."""
        vacuous = _opinion(0.0, 0.0, 1.0)
        dogmatic = _opinion(1.0, 0.0, 0.0)
        report = ConflictDetector.detect_from_opinions([vacuous, dogmatic])
        assert report.conflict_matrix[0][1] == pytest.approx(0.0)

    def test_highly_uncertain_facts_low_conflict(self, odf):
        """Very old facts (high uncertainty after decay) should have low conflict,
        even if their original content was opposing."""
        det = ConflictDetector(decay_fn=odf)
        # Two facts from 100 half-lives ago — both essentially vacuous after decay
        old_high = _fact("K+ = 5.5", days_ago=700.0, quality=0.9)
        old_low = _fact("K+ = 2.8", days_ago=700.0, quality=0.1)
        report = det.detect([old_high, old_low], QUERY_TIME)
        # After massive decay, both opinions are near-vacuous
        for op in report.decayed_opinions:
            assert op.uncertainty > 0.9
        assert report.cohesion_score > 0.9


# ===========================================================================
# 9. Conflict Matrix Symmetry (Mathematical Invariant)
# ===========================================================================


class TestSymmetry:
    """Conflict matrix must be symmetric: con(A, B) = con(B, A).

    This follows directly from the formula con = b_A·d_B + d_A·b_B,
    which is symmetric by construction.
    """

    def test_symmetry_two_opinions(self):
        opinions = [_opinion(0.6, 0.2, 0.2), _opinion(0.3, 0.5, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.conflict_matrix[0][1] == pytest.approx(
            report.conflict_matrix[1][0]
        )

    def test_symmetry_four_opinions(self):
        opinions = [
            _opinion(0.7, 0.1, 0.2),
            _opinion(0.2, 0.6, 0.2),
            _opinion(0.5, 0.3, 0.2),
            _opinion(0.1, 0.1, 0.8),
        ]
        report = ConflictDetector.detect_from_opinions(opinions)
        n = len(opinions)
        for i in range(n):
            for j in range(n):
                assert report.conflict_matrix[i][j] == pytest.approx(
                    report.conflict_matrix[j][i]
                ), f"Asymmetry at ({i}, {j})"

    def test_diagonal_is_zero(self):
        """An opinion cannot conflict with itself: con(A, A) = 0 is NOT
        guaranteed — b·d + d·b = 2bd ≥ 0, and is zero only when b=0 or d=0.
        However, for a valid single opinion, this represents internal conflict
        (self-contradiction), not pairwise disagreement. The diagonal should
        store self-conflict = b·d + d·b = 2·b·d for diagnostic purposes,
        OR be set to 0 by convention (since pairwise conflict is between
        distinct opinions).

        We choose 0 by convention: the diagonal of the pairwise conflict
        matrix represents "conflict with self" which is undefined for
        pairwise conflict. Self-conflict is captured by conflict_metric().
        """
        opinions = [_opinion(0.5, 0.3, 0.2), _opinion(0.4, 0.4, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.conflict_matrix[0][0] == 0.0
        assert report.conflict_matrix[1][1] == 0.0


# ===========================================================================
# 10. Threshold Sensitivity
# ===========================================================================


class TestThresholdSensitivity:
    """Varying threshold changes which pairs are flagged."""

    def test_low_threshold_catches_more(self):
        opinions = [
            _opinion(0.6, 0.2, 0.2),
            _opinion(0.2, 0.5, 0.3),
            _opinion(0.5, 0.3, 0.2),
        ]
        report_strict = ConflictDetector.detect_from_opinions(opinions, threshold=0.5)
        report_loose = ConflictDetector.detect_from_opinions(opinions, threshold=0.05)
        assert len(report_loose.conflict_pairs) >= len(report_strict.conflict_pairs)

    def test_threshold_zero_flags_all_nonzero(self):
        """Threshold of 0.0 should flag every pair with any conflict > 0."""
        opinions = [
            _opinion(0.6, 0.2, 0.2),
            _opinion(0.5, 0.3, 0.2),
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.0)
        # Both have nonzero b and d, so con > 0
        assert len(report.conflict_pairs) == 1

    def test_threshold_one_flags_nothing(self):
        """Threshold of 1.0 — only perfect opposition would be flagged,
        but conflict ≤ 1.0 so even (1,0,0) vs (0,1,0) at con=1.0 is
        not strictly greater. Use > threshold check."""
        opinions = [
            _opinion(0.8, 0.1, 0.1),
            _opinion(0.1, 0.8, 0.1),
        ]
        # con = 0.8*0.8 + 0.1*0.1 = 0.65 < 1.0
        report = ConflictDetector.detect_from_opinions(opinions, threshold=1.0)
        assert len(report.conflict_pairs) == 0


# ===========================================================================
# 11. Temporal Decay Interaction
# ===========================================================================


class TestTemporalDecayInteraction:
    """Conflict detection after temporal decay via OpinionDecayFunction."""

    def test_fresh_opposing_facts_high_conflict(self, detector):
        """Fresh facts with opposing quality scores → high conflict."""
        high_q = _fact("K+ = 5.5", days_ago=0.0, quality=0.95)
        low_q = _fact("K+ = 2.8", days_ago=0.0, quality=0.05)
        report = detector.detect([high_q, low_q], QUERY_TIME)
        # Fresh, high quality vs fresh, low quality → one believes, one disbelieves
        assert report.conflict_matrix[0][1] > 0.1

    def test_decay_reduces_conflict(self, detector):
        """As facts age, their opinions become more uncertain,
        reducing pairwise conflict magnitude."""
        high_q = _fact("K+ = 5.5", quality=0.95)
        low_q = _fact("K+ = 2.8", quality=0.05)

        # Fresh
        report_fresh = detector.detect(
            [_fact("a", days_ago=0.0, quality=0.95),
             _fact("b", days_ago=0.0, quality=0.05)],
            QUERY_TIME,
        )

        # Old (14 days = 2 half-lives for general)
        report_old = detector.detect(
            [_fact("a", days_ago=14.0, quality=0.95),
             _fact("b", days_ago=14.0, quality=0.05)],
            QUERY_TIME,
        )

        assert report_old.conflict_matrix[0][1] < report_fresh.conflict_matrix[0][1]

    def test_mixed_age_facts(self, detector):
        """Fresh fact vs old fact: old fact has decayed toward vacuous,
        so conflict is lower than two fresh opposing facts."""
        fresh_high = _fact("a", days_ago=0.0, quality=0.95)
        old_low = _fact("b", days_ago=30.0, quality=0.05)
        report = detector.detect([fresh_high, old_low], QUERY_TIME)
        # Old fact is near-vacuous → low conflict
        assert report.decayed_opinions[1].uncertainty > report.decayed_opinions[0].uncertainty


# ===========================================================================
# 12. detect_from_opinions (Static Path)
# ===========================================================================


class TestDetectFromOpinions:
    """Static method for pre-decayed opinions."""

    def test_returns_conflict_report(self):
        opinions = [_opinion(0.7, 0.1, 0.2), _opinion(0.6, 0.2, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert isinstance(report, ConflictReport)
        assert report.source_count == 2

    def test_opinions_are_stored_unchanged(self):
        """detect_from_opinions should store opinions as-is (no decay)."""
        opinions = [_opinion(0.7, 0.1, 0.2), _opinion(0.6, 0.2, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions)
        for orig, stored in zip(opinions, report.decayed_opinions):
            assert orig.belief == stored.belief
            assert orig.disbelief == stored.disbelief
            assert orig.uncertainty == stored.uncertainty

    def test_default_threshold(self):
        opinions = [_opinion(0.7, 0.1, 0.2), _opinion(0.1, 0.7, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.threshold == 0.15  # default

    def test_custom_threshold(self):
        opinions = [_opinion(0.7, 0.1, 0.2), _opinion(0.1, 0.7, 0.2)]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.5)
        assert report.threshold == 0.5


# ===========================================================================
# 13. Conflict Pairs Ordering
# ===========================================================================


class TestConflictPairsOrdering:
    """Conflict pairs should be sorted by score descending for usability."""

    def test_pairs_sorted_descending(self):
        """If multiple pairs exceed threshold, highest conflict first."""
        opinions = [
            _opinion(0.8, 0.1, 0.1),   # 0: positive
            _opinion(0.1, 0.8, 0.1),   # 1: negative (opposes 0 strongly)
            _opinion(0.5, 0.3, 0.2),   # 2: moderate (weaker conflict with 1)
        ]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.05)
        if len(report.conflict_pairs) >= 2:
            scores = [s for _, _, s in report.conflict_pairs]
            assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 14. Mathematical Correctness — Cross-Validation with jsonld-ex
# ===========================================================================


class TestCrossValidation:
    """Verify that ConflictDetector produces identical results to
    calling jsonld_ex.confidence_algebra.pairwise_conflict directly."""

    def test_matrix_matches_jex_pairwise(self):
        opinions = [
            _opinion(0.7, 0.1, 0.2),
            _opinion(0.3, 0.5, 0.2),
            _opinion(0.1, 0.2, 0.7),
        ]
        report = ConflictDetector.detect_from_opinions(opinions)
        n = len(opinions)
        for i in range(n):
            for j in range(i + 1, n):
                expected = pairwise_conflict(opinions[i], opinions[j])
                assert report.conflict_matrix[i][j] == pytest.approx(expected)
                assert report.conflict_matrix[j][i] == pytest.approx(expected)


# ===========================================================================
# 15. Large Group Behavior
# ===========================================================================


class TestLargeGroup:
    """Behavior with many opinions."""

    def test_ten_agreeing_opinions(self):
        opinions = [_opinion(0.7 + i * 0.02, 0.1, 0.2 - i * 0.02) for i in range(10)]
        report = ConflictDetector.detect_from_opinions(opinions, threshold=0.15)
        assert report.cohesion_score > 0.8
        assert report.source_count == 10

    def test_conflict_matrix_size(self):
        n = 5
        opinions = [_opinion(0.6, 0.2, 0.2) for _ in range(n)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert len(report.conflict_matrix) == n
        for row in report.conflict_matrix:
            assert len(row) == n


# ===========================================================================
# 16. Internal Conflict (conflict_metric) Exposure
# ===========================================================================


class TestInternalConflict:
    """ConflictReport should expose per-opinion internal conflict scores.

    Internal conflict (Jøsang §12.3.4, conflict_metric) measures how
    self-contradictory a SINGLE opinion is (high b AND high d), which
    is distinct from pairwise conflict between two opinions. This is
    useful for detecting opinions that resulted from fusing disagreeing
    sources.
    """

    def test_report_has_internal_conflicts(self):
        """ConflictReport carries per-opinion internal conflict scores."""
        opinions = [
            _opinion(0.45, 0.45, 0.10),  # high internal conflict
            _opinion(0.8, 0.05, 0.15),   # low internal conflict
        ]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert hasattr(report, "internal_conflicts")
        assert len(report.internal_conflicts) == 2
        # First opinion has high internal conflict
        assert report.internal_conflicts[0] > report.internal_conflicts[1]

    def test_vacuous_opinion_zero_internal_conflict(self):
        """Vacuous opinion: ignorance, not contradiction → zero internal conflict."""
        opinions = [_opinion(0.0, 0.0, 1.0)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.internal_conflicts[0] == pytest.approx(0.0)

    def test_dogmatic_belief_zero_internal_conflict(self):
        """Dogmatic belief (b=1, d=0) has zero internal conflict."""
        opinions = [_opinion(1.0, 0.0, 0.0)]
        report = ConflictDetector.detect_from_opinions(opinions)
        assert report.internal_conflicts[0] == pytest.approx(0.0)
