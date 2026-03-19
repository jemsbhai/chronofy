"""Tests for chronofy.sl.trust — Trust-discounted temporal evidence.

Validates source reliability modeling via Jøsang's trust discount
operator (§14.3), absorbing ASEV Axioms 3-4:

  Axiom 3 (Reliability Monotonicity): higher-quality measurements have
    weakly greater ASEV at the same age.
  Axiom 4 (Blackwell Monotonicity): if observation channel A Blackwell-
    dominates B, then A has at least as much ASEV.

Core idea: instead of source_quality being a scalar multiplier,
the source's reliability IS an Opinion about that source. The evidence
Opinion is trust-discounted through it.

Trust discount (Jøsang §14.3):
    b_Ax = b_AB · b_Bx
    d_Ax = b_AB · d_Bx
    u_Ax = d_AB + u_AB + b_AB · u_Bx

Properties:
    Full trust (b=1):    adopts evidence opinion unchanged
    Zero trust (b=0):    result is vacuous (total uncertainty)
    Partial trust:       dilutes evidence toward uncertainty

References:
    Jøsang (2016) §14.3 (Trust Discount), §14.5 (Trust Networks)
    ASEV Evaluation, Axioms 3-4
"""

from datetime import datetime, timedelta

import pytest

jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.sl.trust import (
    TrustProfile,
    TrustWeightedDecay,
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


# ===========================================================================
# 1. TrustProfile
# ===========================================================================


class TestTrustProfile:
    """Maps source identifiers to trust Opinions."""

    def test_construction_empty(self):
        tp = TrustProfile()
        assert len(tp) == 0

    def test_set_and_get_trust(self):
        tp = TrustProfile()
        trust = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        tp.set_trust("hospital_lab", trust)
        assert tp.get_trust("hospital_lab") == trust

    def test_get_unknown_source_returns_default(self):
        tp = TrustProfile(default_trust=Opinion(belief=0.5, disbelief=0.0, uncertainty=0.5))
        result = tp.get_trust("unknown_source")
        assert result.belief == pytest.approx(0.5)
        assert result.uncertainty == pytest.approx(0.5)

    def test_default_trust_is_vacuous(self):
        """When no default is set, unknown sources get vacuous opinion."""
        tp = TrustProfile()
        result = tp.get_trust("unknown_source")
        assert result.uncertainty == pytest.approx(1.0)
        assert result.belief == pytest.approx(0.0)

    def test_from_scalars(self):
        """Convenience: build from scalar trust scores."""
        tp = TrustProfile.from_scalars(
            {"hospital_lab": 0.95, "patient_report": 0.6},
            uncertainty=0.1,
        )
        lab = tp.get_trust("hospital_lab")
        patient = tp.get_trust("patient_report")
        assert lab.belief > patient.belief

    def test_from_scalars_preserves_order(self):
        tp = TrustProfile.from_scalars(
            {"a": 0.9, "b": 0.5, "c": 0.1},
        )
        a = tp.get_trust("a")
        b = tp.get_trust("b")
        c = tp.get_trust("c")
        assert a.belief > b.belief > c.belief

    def test_len(self):
        tp = TrustProfile()
        tp.set_trust("a", Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1))
        tp.set_trust("b", Opinion(belief=0.5, disbelief=0.1, uncertainty=0.4))
        assert len(tp) == 2

    def test_contains(self):
        tp = TrustProfile()
        tp.set_trust("lab", Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1))
        assert "lab" in tp
        assert "unknown" not in tp

    def test_sources_property(self):
        tp = TrustProfile()
        tp.set_trust("a", Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1))
        tp.set_trust("b", Opinion(belief=0.5, disbelief=0.0, uncertainty=0.5))
        assert tp.sources == {"a", "b"}


# ===========================================================================
# 2. TrustWeightedDecay — Construction
# ===========================================================================


class TestTrustWeightedDecayConstruction:
    """TrustWeightedDecay wraps OpinionDecayFunction + TrustProfile."""

    def test_construction(self):
        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        tp = TrustProfile.from_scalars({"lab": 0.95})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)
        assert twd.trust_profile is tp

    def test_is_not_decay_function_subclass(self):
        """TrustWeightedDecay is a higher-level component, not a DecayFunction.
        It composes OpinionDecayFunction rather than extending DecayFunction."""
        from chronofy.decay.base import DecayFunction
        assert not issubclass(TrustWeightedDecay, DecayFunction)


# ===========================================================================
# 3. Trust Discount Application
# ===========================================================================


class TestTrustDiscount:
    """Trust discount applied after temporal decay."""

    @pytest.fixture
    def odf(self) -> OpinionDecayFunction:
        return OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="confidence",
            base_uncertainty=0.1,
        )

    @pytest.fixture
    def trust_profile(self) -> TrustProfile:
        return TrustProfile.from_scalars(
            {"hospital_lab": 0.95, "patient_report": 0.5, "unreliable": 0.05},
            uncertainty=0.05,
        )

    @pytest.fixture
    def twd(self, odf, trust_profile) -> TrustWeightedDecay:
        return TrustWeightedDecay(decay_fn=odf, trust_profile=trust_profile)

    def test_discount_returns_opinion(self, twd):
        fact = _fact(days_ago=3, source="hospital_lab")
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        assert isinstance(opinion, Opinion)

    def test_discount_preserves_additivity(self, twd):
        fact = _fact(days_ago=3, source="hospital_lab")
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        total = opinion.belief + opinion.disbelief + opinion.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_compute_returns_scalar(self, twd):
        fact = _fact(days_ago=3, source="hospital_lab")
        scalar = twd.compute(fact, QUERY_TIME)
        assert isinstance(scalar, float)
        assert 0.0 <= scalar <= 1.0

    def test_compute_matches_opinion_projection(self, twd):
        fact = _fact(days_ago=3, source="hospital_lab")
        scalar = twd.compute(fact, QUERY_TIME)
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        assert scalar == pytest.approx(opinion.projected_probability(), abs=1e-9)

    # --- ASEV Axiom 3: Reliability Monotonicity ---

    def test_higher_trust_yields_higher_belief(self, twd):
        """ASEV Axiom 3: higher-quality source → higher ASEV at same age."""
        fact_trusted = _fact(days_ago=3, source="hospital_lab", quality=0.9)
        fact_untrusted = _fact(days_ago=3, source="patient_report", quality=0.9)

        op_trusted = twd.compute_opinion(fact_trusted, QUERY_TIME)
        op_untrusted = twd.compute_opinion(fact_untrusted, QUERY_TIME)

        assert op_trusted.belief > op_untrusted.belief

    def test_higher_trust_yields_lower_uncertainty(self, twd):
        """Trusted sources retain more information after discount."""
        fact_trusted = _fact(days_ago=3, source="hospital_lab")
        fact_untrusted = _fact(days_ago=3, source="unreliable")

        op_trusted = twd.compute_opinion(fact_trusted, QUERY_TIME)
        op_untrusted = twd.compute_opinion(fact_untrusted, QUERY_TIME)

        assert op_trusted.uncertainty < op_untrusted.uncertainty

    def test_higher_trust_yields_higher_projected_probability(self, twd):
        """Scalar projection also respects reliability ordering."""
        fact_trusted = _fact(days_ago=3, source="hospital_lab")
        fact_untrusted = _fact(days_ago=3, source="unreliable")

        p_trusted = twd.compute(fact_trusted, QUERY_TIME)
        p_untrusted = twd.compute(fact_untrusted, QUERY_TIME)

        assert p_trusted > p_untrusted

    # --- Full trust / zero trust boundary behavior ---

    def test_full_trust_preserves_opinion(self, odf):
        """Full trust (b=1) should adopt evidence opinion unchanged."""
        tp = TrustProfile()
        tp.set_trust("perfect", Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0))
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact = _fact(days_ago=3, source="perfect")
        discounted = twd.compute_opinion(fact, QUERY_TIME)
        raw = odf.compute_opinion(fact, QUERY_TIME)

        assert discounted.belief == pytest.approx(raw.belief, abs=1e-9)
        assert discounted.disbelief == pytest.approx(raw.disbelief, abs=1e-9)
        assert discounted.uncertainty == pytest.approx(raw.uncertainty, abs=1e-9)

    def test_zero_trust_yields_vacuous(self, odf):
        """Zero trust (b=0, d=1) should yield near-vacuous opinion."""
        tp = TrustProfile()
        tp.set_trust("liar", Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0))
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact = _fact(days_ago=3, source="liar")
        discounted = twd.compute_opinion(fact, QUERY_TIME)

        # With zero trust belief, discounted belief should be 0
        assert discounted.belief == pytest.approx(0.0, abs=1e-9)
        assert discounted.disbelief == pytest.approx(0.0, abs=1e-9)
        # All mass goes to uncertainty
        assert discounted.uncertainty == pytest.approx(1.0, abs=1e-9)

    # --- Unknown source handling ---

    def test_unknown_source_uses_default(self, twd):
        """Facts with no source or unknown source use default trust."""
        fact = _fact(days_ago=3, source="unknown_sensor")
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        # Default trust is vacuous → high uncertainty after discount
        assert opinion.uncertainty > 0.5

    def test_no_source_uses_default(self, twd):
        """Facts with source=None use default trust."""
        fact = _fact(days_ago=3, source=None)
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        assert isinstance(opinion, Opinion)

    # --- Batch operations ---

    def test_compute_batch(self, twd):
        facts = [
            _fact(days_ago=1, source="hospital_lab"),
            _fact(days_ago=5, source="patient_report"),
            _fact(days_ago=10, source="unreliable"),
        ]
        scalars = twd.compute_batch(facts, QUERY_TIME)
        assert len(scalars) == 3
        assert all(isinstance(s, float) for s in scalars)
        assert all(0.0 <= s <= 1.0 for s in scalars)

    def test_compute_opinion_batch(self, twd):
        facts = [
            _fact(days_ago=1, source="hospital_lab"),
            _fact(days_ago=5, source="patient_report"),
        ]
        opinions = twd.compute_opinion_batch(facts, QUERY_TIME)
        assert len(opinions) == 2
        assert all(isinstance(o, Opinion) for o in opinions)


# ===========================================================================
# 4. Interaction between Trust and Temporal Decay
# ===========================================================================


class TestTrustTemporalInteraction:
    """Trust discount composes correctly with temporal decay."""

    @pytest.fixture
    def odf(self) -> OpinionDecayFunction:
        return OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="confidence",
            base_uncertainty=0.1,
        )

    def test_trust_and_age_both_reduce_value(self, odf):
        """Both low trust AND old age should compound to lower value."""
        tp = TrustProfile.from_scalars({"good": 0.9, "bad": 0.3})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        # Good source, fresh → highest value
        good_fresh = twd.compute(_fact(days_ago=1, source="good"), QUERY_TIME)
        # Good source, old → medium
        good_old = twd.compute(_fact(days_ago=30, source="good"), QUERY_TIME)
        # Bad source, fresh → medium
        bad_fresh = twd.compute(_fact(days_ago=1, source="bad"), QUERY_TIME)
        # Bad source, old → lowest
        bad_old = twd.compute(_fact(days_ago=30, source="bad"), QUERY_TIME)

        assert good_fresh > good_old
        assert good_fresh > bad_fresh
        assert bad_old < good_old
        assert bad_old < bad_fresh

    def test_trust_discount_after_decay_order(self, odf):
        """Pipeline: first decay by age, then discount by trust."""
        tp = TrustProfile.from_scalars({"src": 0.8}, uncertainty=0.1)
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact = _fact(days_ago=5, source="src")

        # Raw decayed (no trust discount)
        raw = odf.compute_opinion(fact, QUERY_TIME)
        # Trust-discounted
        discounted = twd.compute_opinion(fact, QUERY_TIME)

        # Discount should weaken the opinion (more uncertainty)
        assert discounted.uncertainty >= raw.uncertainty - 1e-9

    def test_monotonic_trust_ordering_across_ages(self, odf):
        """At any fixed age, trust ordering is preserved (Axiom 3)."""
        tp = TrustProfile.from_scalars({"high": 0.95, "low": 0.3})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        for days in [0, 1, 7, 30]:
            high_val = twd.compute(_fact(days_ago=days, source="high"), QUERY_TIME)
            low_val = twd.compute(_fact(days_ago=days, source="low"), QUERY_TIME)
            assert high_val >= low_val, f"Failed at days_ago={days}"


# ===========================================================================
# 5. Integration with Fusion
# ===========================================================================


class TestTrustFusionIntegration:
    """TrustWeightedDecay opinions can be fused."""

    def test_trust_discounted_opinions_fuseable(self):
        from chronofy.sl.fusion import TemporalEvidenceFusion

        odf = OpinionDecayFunction(half_lives={"general": 7.0}, base_uncertainty=0.1)
        tp = TrustProfile.from_scalars({"lab": 0.95, "self_report": 0.5})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        facts = [
            _fact(days_ago=1, source="lab", quality=0.9),
            _fact(days_ago=1, source="self_report", quality=0.8),
        ]

        # Get trust-discounted opinions
        opinions = twd.compute_opinion_batch(facts, QUERY_TIME)

        # These should be valid for fusion
        from jsonld_ex.confidence_algebra import cumulative_fuse
        fused = cumulative_fuse(*opinions)
        assert isinstance(fused, Opinion)
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)
