"""Tests for chronofy.sl.stl_opinion — Opinion-aware STL verifier.

Extends the scalar STLVerifier to track full (b, d, u, a) Opinions per
reasoning step, enabling richer weakest-link bounds and nuanced
re-acquisition diagnostics.

Key differences from scalar STLVerifier:
    - Per-step validity is an Opinion, not a float
    - Weakest-link bound is an Opinion (can inspect b, d, u separately)
    - Re-acquisition diagnostics distinguish:
        * Decayed evidence (high u) → re-acquire same source type
        * Negative evidence (high d, low u) → real negative finding
        * Insufficient evidence (u ≈ 1) → need a different source entirely
    - Scalar robustness ρ preserved for backward compatibility

Scientific invariants tested:
    - Robustness ρ matches scalar STLVerifier when using same decay fn
    - Per-step opinion projected probability matches scalar step_validity
    - Weakest-link opinion's projected probability matches scalar bound
    - Empty trace / no-fact steps handled consistently
    - Temporal decay increases uncertainty in step opinions
    - Freshness ordering preserved: newer facts → lower uncertainty

References:
    Jøsang (2016) §10.4 (Opinion Aging), §3.2 (Projected Probability).
    Paper Theorem 1 (Decay Propagation Bound).
    Paper Eq. 3 (validity signal), Eq. 4 (STL formula), Eq. 5 (robustness).
"""

from datetime import datetime, timedelta

import pytest

jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction, OpinionConfig
from chronofy.sl.stl_opinion import (
    OpinionSTLResult,
    OpinionSTLVerifier,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2026, 3, 19, 12, 0)


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


def _step(index: int, facts: list[TemporalFact], content: str = "") -> ReasoningStep:
    return ReasoningStep(
        step_index=index,
        content=content or f"Step {index}",
        facts_used=facts,
    )


def _trace(steps: list[ReasoningStep]) -> ReasoningTrace:
    return ReasoningTrace(steps=steps, query_time=QUERY_TIME)


@pytest.fixture
def odf() -> OpinionDecayFunction:
    return OpinionDecayFunction(
        half_lives={"general": 7.0, "vital_sign": 1.0, "chronic": 69.3},
        construction="confidence",
        base_uncertainty=0.1,
    )


@pytest.fixture
def verifier(odf) -> OpinionSTLVerifier:
    return OpinionSTLVerifier(decay_fn=odf, threshold=0.5)


# ===========================================================================
# 1. Construction and Validation
# ===========================================================================


class TestConstruction:
    """OpinionSTLVerifier setup and configuration."""

    def test_default_construction(self, odf):
        v = OpinionSTLVerifier(decay_fn=odf)
        assert v is not None
        assert v.threshold == 0.5  # default

    def test_custom_threshold(self, odf):
        v = OpinionSTLVerifier(decay_fn=odf, threshold=0.3)
        assert v.threshold == 0.3

    def test_threshold_bounds(self, odf):
        with pytest.raises(ValueError, match="threshold"):
            OpinionSTLVerifier(decay_fn=odf, threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            OpinionSTLVerifier(decay_fn=odf, threshold=1.5)

    def test_repr(self, verifier):
        r = repr(verifier)
        assert "OpinionSTLVerifier" in r

    def test_accepts_plain_decay_function(self):
        """OpinionSTLVerifier should accept a plain DecayFunction,
        wrapping scalar outputs into opinions for compatibility."""
        plain_decay = ExponentialDecay(beta={"general": 0.5})
        v = OpinionSTLVerifier(decay_fn=plain_decay, threshold=0.5)
        assert v is not None


# ===========================================================================
# 2. OpinionSTLResult Structure
# ===========================================================================


class TestResultStructure:
    """OpinionSTLResult carries correct fields."""

    def test_result_has_step_opinions(self, verifier):
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result, OpinionSTLResult)
        assert len(result.step_opinions) == 1
        assert isinstance(result.step_opinions[0], Opinion)

    def test_result_has_robustness(self, verifier):
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result.robustness, float)

    def test_result_has_satisfied_bool(self, verifier):
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result.satisfied, bool)

    def test_result_has_weakest_step_opinion(self, verifier):
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result.weakest_step_opinion, Opinion)

    def test_result_has_weakest_link_opinion(self, verifier):
        """The global weakest-link opinion across ALL facts in ALL steps."""
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result.weakest_link_opinion, Opinion)

    def test_result_has_output_confidence_bound(self, verifier):
        """Scalar bound for backward compat: P(weakest_link_opinion)."""
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert isinstance(result.output_confidence_bound, float)
        assert 0.0 <= result.output_confidence_bound <= 1.0

    def test_result_has_step_validity_scalars(self, verifier):
        """Scalar step validity for backward compat with STLVerifier."""
        trace = _trace([_step(0, [_fact("a")]), _step(1, [_fact("b")])])
        result = verifier.verify(trace)
        assert len(result.step_validity) == 2
        for v in result.step_validity:
            assert isinstance(v, float)

    def test_result_has_weakest_step_index(self, verifier):
        trace = _trace([_step(0, [_fact("a")])])
        result = verifier.verify(trace)
        assert result.weakest_step_index == 0

    def test_result_has_weakest_fact(self, verifier):
        f = _fact("a")
        trace = _trace([_step(0, [f])])
        result = verifier.verify(trace)
        assert result.weakest_fact is not None


# ===========================================================================
# 3. Empty Trace
# ===========================================================================


class TestEmptyTrace:
    """Empty trace is trivially satisfied."""

    def test_empty_trace_satisfied(self, verifier):
        trace = _trace([])
        result = verifier.verify(trace)
        assert result.satisfied is True
        assert result.robustness == 0.0
        assert result.step_opinions == []
        assert result.step_validity == []

    def test_empty_trace_bound_is_one(self, verifier):
        trace = _trace([])
        result = verifier.verify(trace)
        assert result.output_confidence_bound == 1.0

    def test_empty_trace_no_weakest(self, verifier):
        trace = _trace([])
        result = verifier.verify(trace)
        assert result.weakest_step_index is None
        assert result.weakest_fact is None
        assert result.weakest_step_opinion is None
        assert result.weakest_link_opinion is None


# ===========================================================================
# 4. Steps With No Facts (Vacuous Truth)
# ===========================================================================


class TestNoFactSteps:
    """Steps using no facts have vacuous truth (validity = 1.0)."""

    def test_no_facts_step_opinion_is_dogmatic_belief(self, verifier):
        """A step with no facts: nothing to decay → full validity.
        The opinion should reflect certainty: b=1, d=0, u=0 (or
        projected probability = 1.0)."""
        trace = _trace([_step(0, [])])
        result = verifier.verify(trace)
        assert result.step_validity[0] == 1.0
        assert result.step_opinions[0].projected_probability() == pytest.approx(1.0)

    def test_mixed_no_fact_and_fact_steps(self, verifier):
        """Step with facts is the weakest link, not the no-fact step."""
        old_fact = _fact("old", days_ago=14.0)
        trace = _trace([
            _step(0, []),           # vacuous: validity = 1.0
            _step(1, [old_fact]),   # decayed: validity < 1.0
        ])
        result = verifier.verify(trace)
        assert result.weakest_step_index == 1
        assert result.step_validity[0] > result.step_validity[1]


# ===========================================================================
# 5. Fresh Facts — High Validity
# ===========================================================================


class TestFreshFacts:
    """Fresh, high-quality facts should satisfy the threshold easily."""

    def test_fresh_fact_satisfied(self, verifier):
        trace = _trace([_step(0, [_fact("K+ = 4.1", days_ago=0.0, quality=0.95)])])
        result = verifier.verify(trace)
        assert result.satisfied is True
        assert result.robustness > 0.0

    def test_fresh_fact_low_uncertainty(self, verifier):
        """Fresh fact opinion should have low uncertainty."""
        trace = _trace([_step(0, [_fact("K+ = 4.1", days_ago=0.0, quality=0.95)])])
        result = verifier.verify(trace)
        # base_uncertainty=0.1 in fixture, so u = 0.1 for fresh fact
        assert result.step_opinions[0].uncertainty <= 0.15


# ===========================================================================
# 6. Stale Facts — High Uncertainty, Low Validity
# ===========================================================================


class TestStaleFacts:
    """Stale facts should have high uncertainty and potentially violate threshold."""

    def test_very_old_fact_violates(self, odf):
        """A fact from 30 days ago with 7-day half-life is very decayed.

        With quality=1.0 and base_rate=0.5, the projected probability
        approaches a=0.5 from above as u→1 (P = b' + 0.5·u' ≥ 0.5).
        So threshold=0.5 is never strictly violated for d=0 opinions.
        Use a higher threshold (0.6) to verify the mechanism.
        """
        v = OpinionSTLVerifier(decay_fn=odf, threshold=0.6)
        trace = _trace([_step(0, [_fact("old", days_ago=30.0)])])
        result = v.verify(trace)
        # P(ω') ≈ 0.523 < 0.6 threshold
        assert result.step_validity[0] < 0.6
        assert result.satisfied is False
        assert result.robustness < 0.0

    def test_old_fact_high_uncertainty(self, verifier):
        """Decayed opinion should have high uncertainty."""
        trace = _trace([_step(0, [_fact("old", days_ago=14.0)])])
        result = verifier.verify(trace)
        assert result.step_opinions[0].uncertainty > 0.5

    def test_decay_increases_uncertainty_monotonically(self, verifier):
        """Older facts → higher uncertainty in step opinion."""
        facts = [_fact(f"f{i}", days_ago=float(i * 3)) for i in range(5)]
        trace = _trace([_step(i, [f]) for i, f in enumerate(facts)])
        result = verifier.verify(trace)
        uncertainties = [op.uncertainty for op in result.step_opinions]
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] <= uncertainties[i + 1] + 1e-9


# ===========================================================================
# 7. Weakest-Link Bound (Theorem 1)
# ===========================================================================


class TestWeakestLinkBound:
    """The output confidence bound equals the projected probability of
    the weakest fact's opinion across all steps."""

    def test_bound_is_global_minimum(self, verifier):
        """Multiple steps, multiple facts — bound is the global min."""
        fresh = _fact("fresh", days_ago=0.0, quality=0.9)
        medium = _fact("medium", days_ago=7.0, quality=0.8)
        stale = _fact("stale", days_ago=21.0, quality=0.7)

        trace = _trace([
            _step(0, [fresh, medium]),
            _step(1, [stale]),
        ])
        result = verifier.verify(trace)

        # Stale is the weakest — bound should match its projected probability
        stale_opinion = verifier._compute_fact_opinion(stale, QUERY_TIME)
        assert result.output_confidence_bound == pytest.approx(
            stale_opinion.projected_probability(), abs=1e-6
        )

    def test_bound_equals_weakest_link_projected_probability(self, verifier):
        """Scalar bound should equal P(weakest_link_opinion)."""
        trace = _trace([_step(0, [_fact("a", days_ago=10.0)])])
        result = verifier.verify(trace)
        assert result.output_confidence_bound == pytest.approx(
            result.weakest_link_opinion.projected_probability(), abs=1e-9
        )

    def test_weakest_fact_tracked(self, verifier):
        """The result should identify which fact is the weakest."""
        stale = _fact("stale", days_ago=20.0)
        fresh = _fact("fresh", days_ago=0.0)
        trace = _trace([_step(0, [fresh, stale])])
        result = verifier.verify(trace)
        assert result.weakest_fact.content == "stale"


# ===========================================================================
# 8. Multi-Step Traces
# ===========================================================================


class TestMultiStep:
    """Multi-step reasoning chains."""

    def test_three_steps_weakest_identified(self, verifier):
        trace = _trace([
            _step(0, [_fact("a", days_ago=1.0)]),
            _step(1, [_fact("b", days_ago=10.0)]),
            _step(2, [_fact("c", days_ago=3.0)]),
        ])
        result = verifier.verify(trace)
        assert result.weakest_step_index == 1
        assert len(result.step_opinions) == 3
        assert len(result.step_validity) == 3

    def test_robustness_is_min_margin(self, verifier):
        """ρ = min_i(P(ω_i) - γ)."""
        trace = _trace([
            _step(0, [_fact("a", days_ago=0.0)]),
            _step(1, [_fact("b", days_ago=14.0)]),
        ])
        result = verifier.verify(trace)
        expected_rho = min(v - verifier.threshold for v in result.step_validity)
        assert result.robustness == pytest.approx(expected_rho, abs=1e-9)


# ===========================================================================
# 9. Step With Multiple Facts — Min Opinion
# ===========================================================================


class TestMultiFactStep:
    """A step using multiple facts has validity = min over facts."""

    def test_step_opinion_is_weakest_fact(self, verifier):
        """Per-step opinion should be the opinion with lowest projected probability."""
        fresh = _fact("fresh", days_ago=0.0, quality=0.9)
        stale = _fact("stale", days_ago=14.0, quality=0.5)
        trace = _trace([_step(0, [fresh, stale])])
        result = verifier.verify(trace)

        # Step opinion should match the stale fact's opinion (lower P)
        stale_opinion = verifier._compute_fact_opinion(stale, QUERY_TIME)
        assert result.step_opinions[0].projected_probability() == pytest.approx(
            stale_opinion.projected_probability(), abs=1e-6
        )

    def test_step_validity_is_min_projected_probability(self, verifier):
        """Scalar step validity = P(step_opinion)."""
        trace = _trace([_step(0, [
            _fact("a", days_ago=0.0, quality=0.8),
            _fact("b", days_ago=7.0, quality=0.5),
        ])])
        result = verifier.verify(trace)
        assert result.step_validity[0] == pytest.approx(
            result.step_opinions[0].projected_probability(), abs=1e-9
        )


# ===========================================================================
# 10. Backward Compatibility — Scalar Consistency
# ===========================================================================


class TestScalarConsistency:
    """Scalar outputs (robustness, step_validity, bound) should be consistent
    with the scalar STLVerifier when using the same underlying decay fn."""

    def test_step_validity_matches_projected_probability(self, verifier):
        """step_validity[i] == step_opinions[i].projected_probability()."""
        trace = _trace([
            _step(0, [_fact("a", days_ago=3.0)]),
            _step(1, [_fact("b", days_ago=10.0)]),
        ])
        result = verifier.verify(trace)
        for sv, op in zip(result.step_validity, result.step_opinions):
            assert sv == pytest.approx(op.projected_probability(), abs=1e-9)


# ===========================================================================
# 11. Re-Acquisition Diagnostics
# ===========================================================================


class TestReAcquisitionDiagnostics:
    """The Opinion-aware verifier should enable distinguishing TYPES of
    failure, not just detecting failure."""

    def test_decayed_evidence_has_high_uncertainty(self, odf):
        """When an old fact causes violation, the weakest-link opinion
        has high uncertainty — meaning 'we once had evidence but it decayed.'

        With quality=1.0, d=0, base_rate=0.5: P → 0.5 as u → 1.
        Use threshold=0.6 so the violation is detectable.
        """
        v = OpinionSTLVerifier(decay_fn=odf, threshold=0.6)
        trace = _trace([_step(0, [_fact("old reading", days_ago=30.0)])])
        result = v.verify(trace)
        assert not result.satisfied
        assert result.weakest_link_opinion.uncertainty > 0.8

    def test_low_quality_fresh_has_low_uncertainty(self, verifier):
        """A fresh but low-quality fact: belief is low but uncertainty is
        still bounded by base_uncertainty. This is a real negative finding,
        not a stale one."""
        trace = _trace([_step(0, [_fact("low q", days_ago=0.0, quality=0.1)])])
        result = verifier.verify(trace)
        # Uncertainty is just base_uncertainty (0.1 in fixture)
        assert result.weakest_link_opinion.uncertainty <= 0.15
        # Low quality → disbelief is high
        assert result.weakest_link_opinion.disbelief > result.weakest_link_opinion.belief

    def test_can_distinguish_stale_from_negative(self, verifier):
        """Key diagnostic property: given two traces that both violate threshold,
        one due to staleness (high u) and one due to negative evidence (low b, low u),
        the opinions are distinguishable."""
        # Stale: high quality but very old
        stale_trace = _trace([_step(0, [_fact("stale", days_ago=30.0, quality=0.9)])])
        stale_result = verifier.verify(stale_trace)

        # Negative: fresh but low quality
        neg_trace = _trace([_step(0, [_fact("neg", days_ago=0.0, quality=0.1)])])
        neg_result = verifier.verify(neg_trace)

        # Both may violate or not — key is the opinions differ structurally
        stale_op = stale_result.weakest_link_opinion
        neg_op = neg_result.weakest_link_opinion

        # Stale has much higher uncertainty
        assert stale_op.uncertainty > neg_op.uncertainty + 0.3


# ===========================================================================
# 12. PlainDecayFunction Fallback
# ===========================================================================


class TestPlainDecayFallback:
    """OpinionSTLVerifier accepts a plain DecayFunction (not Opinion-aware)
    and wraps scalar outputs into opinions for compatibility."""

    def test_plain_decay_produces_valid_result(self):
        plain = ExponentialDecay(beta={"general": 0.5})
        v = OpinionSTLVerifier(decay_fn=plain, threshold=0.5)
        trace = _trace([_step(0, [_fact("a", days_ago=3.0)])])
        result = v.verify(trace)
        assert isinstance(result, OpinionSTLResult)
        assert isinstance(result.step_opinions[0], Opinion)

    def test_plain_decay_scalar_matches(self):
        """When using plain DecayFunction, scalar step_validity should
        match DecayFunction.compute() directly."""
        plain = ExponentialDecay(beta={"general": 0.5})
        v = OpinionSTLVerifier(decay_fn=plain, threshold=0.5)
        f = _fact("a", days_ago=5.0)
        trace = _trace([_step(0, [f])])
        result = v.verify(trace)

        expected_scalar = plain.compute(f, QUERY_TIME)
        assert result.step_validity[0] == pytest.approx(expected_scalar, abs=1e-6)

    def test_plain_decay_robustness_matches_scalar_verifier(self):
        """Robustness from OpinionSTLVerifier with plain decay should match
        the scalar STLVerifier's robustness."""
        from chronofy.verification.stl import STLVerifier

        plain = ExponentialDecay(beta={"general": 0.5, "vital_sign": 5.0})
        scalar_v = STLVerifier(decay_fn=plain, threshold=0.5)
        opinion_v = OpinionSTLVerifier(decay_fn=plain, threshold=0.5)

        facts = [
            _fact("a", days_ago=1.0),
            _fact("b", days_ago=5.0, fact_type="vital_sign"),
        ]
        trace = _trace([_step(0, [facts[0]]), _step(1, [facts[1]])])

        scalar_result = scalar_v.verify(trace)
        opinion_result = opinion_v.verify(trace)

        assert opinion_result.robustness == pytest.approx(
            scalar_result.robustness, abs=1e-6
        )


# ===========================================================================
# 13. Fact-Type Sensitivity
# ===========================================================================


class TestFactTypeSensitivity:
    """Different fact types decay at different rates → different opinions."""

    def test_vital_sign_decays_faster(self, verifier):
        """Vital sign (half_life=1d) decays faster than general (half_life=7d)."""
        vital = _fact("HR = 72", days_ago=3.0, fact_type="vital_sign")
        general = _fact("diagnosis", days_ago=3.0, fact_type="general")

        trace = _trace([_step(0, [vital]), _step(1, [general])])
        result = verifier.verify(trace)

        # Vital at 3 days (3 half-lives) → much more uncertain than general (< 1 half-life)
        assert result.step_opinions[0].uncertainty > result.step_opinions[1].uncertainty

    def test_chronic_condition_preserves_value(self, verifier):
        """Chronic condition (half_life=69.3d) barely decays over 7 days."""
        chronic = _fact("diabetes", days_ago=7.0, fact_type="chronic")
        trace = _trace([_step(0, [chronic])])
        result = verifier.verify(trace)
        # After 7 days with 69.3-day half-life, barely decayed
        assert result.step_opinions[0].uncertainty < 0.2
        assert result.satisfied is True


# ===========================================================================
# 14. Robustness Method Shortcut
# ===========================================================================


class TestRobustnessShortcut:
    """Convenience method returning just ρ."""

    def test_robustness_method_matches_verify(self, verifier):
        trace = _trace([_step(0, [_fact("a", days_ago=5.0)])])
        rho = verifier.robustness(trace)
        result = verifier.verify(trace)
        assert rho == pytest.approx(result.robustness, abs=1e-12)
