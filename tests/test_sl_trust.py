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
    TrustEntry,
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
# 1b. TrustEntry
# ===========================================================================


class TestTrustEntry:
    """TrustEntry bundles a trust opinion with temporal metadata."""

    def test_construction_minimal(self):
        entry = TrustEntry(opinion=Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1))
        assert entry.opinion.belief == pytest.approx(0.9)
        assert entry.timestamp is None
        assert entry.half_life is None

    def test_construction_with_timestamp(self):
        ts = datetime(2025, 6, 1)
        entry = TrustEntry(
            opinion=Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
            timestamp=ts,
            half_life=180.0,  # 6-month half-life in days
        )
        assert entry.timestamp == ts
        assert entry.half_life == 180.0

    def test_static_entry_has_no_temporal_info(self):
        entry = TrustEntry(opinion=Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1))
        assert not entry.is_temporal

    def test_temporal_entry_has_temporal_info(self):
        entry = TrustEntry(
            opinion=Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            timestamp=datetime(2025, 1, 1),
            half_life=90.0,
        )
        assert entry.is_temporal


# ===========================================================================
# 1c. Trust Decay — Trust opinions age over time
# ===========================================================================


class TestTrustDecay:
    """Trust itself decays: a lab certification from 5 years ago is less
    trustworthy than one from yesterday."""

    def test_set_trust_with_timestamp(self):
        """TrustProfile.set_trust() accepts optional temporal metadata."""
        tp = TrustProfile()
        tp.set_trust(
            "lab",
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
            timestamp=datetime(2025, 6, 1),
            half_life=365.0,
        )
        assert "lab" in tp

    def test_static_trust_unchanged_with_query_time(self):
        """Static trust (no timestamp) is unaffected by query_time."""
        tp = TrustProfile()
        tp.set_trust("lab", Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1))
        result_no_qt = tp.get_trust("lab")
        result_with_qt = tp.get_trust("lab", query_time=QUERY_TIME)
        assert result_no_qt.belief == pytest.approx(result_with_qt.belief)

    def test_fresh_trust_barely_decayed(self):
        """Trust established recently should barely decay."""
        tp = TrustProfile()
        original = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        tp.set_trust(
            "lab", original,
            timestamp=QUERY_TIME - timedelta(days=1),
            half_life=365.0,
        )
        decayed = tp.get_trust("lab", query_time=QUERY_TIME)
        assert decayed.belief > 0.85  # barely changed
        assert decayed.uncertainty < 0.15

    def test_old_trust_heavily_decayed(self):
        """Trust established long ago should decay significantly."""
        tp = TrustProfile()
        original = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        tp.set_trust(
            "old_lab", original,
            timestamp=QUERY_TIME - timedelta(days=730),  # 2 years ago
            half_life=180.0,  # 6-month half-life
        )
        decayed = tp.get_trust("old_lab", query_time=QUERY_TIME)
        # After ~4 half-lives, belief should be heavily reduced
        assert decayed.belief < 0.1
        assert decayed.uncertainty > 0.8

    def test_trust_decay_monotonic_with_age(self):
        """Older trust establishment → more decayed trust opinion."""
        tp = TrustProfile()
        original = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)

        for source, days_ago in [("recent", 7), ("medium", 90), ("old", 365)]:
            tp.set_trust(
                source, original,
                timestamp=QUERY_TIME - timedelta(days=days_ago),
                half_life=180.0,
            )

        recent = tp.get_trust("recent", query_time=QUERY_TIME)
        medium = tp.get_trust("medium", query_time=QUERY_TIME)
        old = tp.get_trust("old", query_time=QUERY_TIME)

        assert recent.belief > medium.belief > old.belief
        assert recent.uncertainty < medium.uncertainty < old.uncertainty

    def test_trust_decay_preserves_bd_ratio(self):
        """Trust decay preserves evidence direction (b/d ratio)."""
        tp = TrustProfile()
        original = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        tp.set_trust(
            "src", original,
            timestamp=QUERY_TIME - timedelta(days=90),
            half_life=180.0,
        )
        decayed = tp.get_trust("src", query_time=QUERY_TIME)
        # b/d ratio should be preserved
        if decayed.disbelief > 1e-9:
            original_ratio = original.belief / original.disbelief
            decayed_ratio = decayed.belief / decayed.disbelief
            assert original_ratio == pytest.approx(decayed_ratio, rel=0.01)

    def test_trust_decay_preserves_additivity(self):
        """b + d + u = 1 after trust decay."""
        tp = TrustProfile()
        tp.set_trust(
            "src",
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            timestamp=QUERY_TIME - timedelta(days=60),
            half_life=90.0,
        )
        decayed = tp.get_trust("src", query_time=QUERY_TIME)
        total = decayed.belief + decayed.disbelief + decayed.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_no_query_time_returns_raw_opinion(self):
        """Without query_time, temporal entries return their raw opinion."""
        tp = TrustProfile()
        original = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        tp.set_trust(
            "lab", original,
            timestamp=QUERY_TIME - timedelta(days=365),
            half_life=180.0,
        )
        raw = tp.get_trust("lab")  # no query_time
        assert raw.belief == pytest.approx(original.belief)

    def test_configurable_decay_function(self):
        """Users can plug in different decay functions for trust aging."""
        from jsonld_ex.confidence_decay import linear_decay

        tp = TrustProfile(trust_decay_fn=linear_decay)
        original = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        tp.set_trust(
            "src", original,
            timestamp=QUERY_TIME - timedelta(days=90),
            half_life=180.0,
        )
        decayed = tp.get_trust("src", query_time=QUERY_TIME)
        # Linear decay should give different result than exponential
        assert isinstance(decayed, Opinion)
        assert decayed.belief < original.belief

    def test_configurable_decay_step_function(self):
        """Step decay: trust is fully valid before half_life, zero after."""
        from jsonld_ex.confidence_decay import step_decay

        tp = TrustProfile(trust_decay_fn=step_decay)
        original = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)

        # Before half-life: trust unchanged
        tp.set_trust(
            "fresh", original,
            timestamp=QUERY_TIME - timedelta(days=30),
            half_life=90.0,
        )
        fresh = tp.get_trust("fresh", query_time=QUERY_TIME)
        assert fresh.belief == pytest.approx(original.belief)

        # After half-life: trust fully decayed
        tp.set_trust(
            "stale", original,
            timestamp=QUERY_TIME - timedelta(days=180),
            half_life=90.0,
        )
        stale = tp.get_trust("stale", query_time=QUERY_TIME)
        assert stale.belief == pytest.approx(0.0, abs=1e-9)
        assert stale.uncertainty == pytest.approx(1.0, abs=1e-9)

    def test_from_scalars_with_timestamps(self):
        """Convenience: build temporal trust entries from scalars."""
        tp = TrustProfile.from_scalars(
            {"lab": 0.95, "clinic": 0.8},
            uncertainty=0.05,
            timestamps={
                "lab": QUERY_TIME - timedelta(days=30),
                "clinic": QUERY_TIME - timedelta(days=365),
            },
            half_lives={"lab": 365.0, "clinic": 180.0},
        )
        lab = tp.get_trust("lab", query_time=QUERY_TIME)
        clinic = tp.get_trust("clinic", query_time=QUERY_TIME)
        # Lab: recent, long half-life → barely decayed
        # Clinic: old, short half-life → heavily decayed
        assert lab.belief > clinic.belief


# ===========================================================================
# 1d. Trust Decay in TrustWeightedDecay Pipeline
# ===========================================================================


class TestTrustDecayInPipeline:
    """TrustWeightedDecay applies trust decay before trust discount."""

    @pytest.fixture
    def odf(self) -> OpinionDecayFunction:
        return OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="confidence",
            base_uncertainty=0.1,
        )

    def test_pipeline_decays_trust_then_discounts(self, odf):
        """Full pipeline: decay evidence → decay trust → trust discount."""
        tp = TrustProfile()
        tp.set_trust(
            "old_lab",
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
            timestamp=QUERY_TIME - timedelta(days=365),
            half_life=180.0,
        )
        tp.set_trust(
            "new_lab",
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
            timestamp=QUERY_TIME - timedelta(days=7),
            half_life=180.0,
        )
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact_old = _fact(days_ago=3, source="old_lab")
        fact_new = _fact(days_ago=3, source="new_lab")

        op_old = twd.compute_opinion(fact_old, QUERY_TIME)
        op_new = twd.compute_opinion(fact_new, QUERY_TIME)

        # Same evidence age, but old_lab's trust has decayed more
        assert op_new.belief > op_old.belief

    def test_pipeline_static_trust_unchanged(self, odf):
        """Static trust entries (no timestamp) behave as before."""
        tp = TrustProfile.from_scalars({"lab": 0.9})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact = _fact(days_ago=3, source="lab")
        opinion = twd.compute_opinion(fact, QUERY_TIME)
        assert isinstance(opinion, Opinion)
        assert opinion.belief > 0


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
