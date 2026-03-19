"""Tests for chronofy.sl.opinion_decay — Opinion-based temporal decay.

Validates the integration of Jøsang's Subjective Logic opinion aging
operator (§10.4) with Chronofy's DecayFunction protocol, ensuring:

- OpinionDecayFunction satisfies DecayFunction ABC (backward compatible)
- Initial opinions can be constructed via from_evidence() or from_confidence()
- Temporal decay migrates mass into uncertainty (Jøsang aging invariant)
- The b/d ratio is preserved during decay (evidence direction unchanged)
- compute() returns projected probability P(ω') for backward compatibility
- compute_opinion() returns full Opinion for SL-aware consumers
- Base rates are configurable per fact_type
- Half-lives are configurable per fact_type (parallel to β in ExponentialDecay)
"""

from datetime import datetime, timedelta

import pytest

# Guard: skip entire module if jsonld-ex not installed
jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction, OpinionConfig


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2026, 3, 19)


def _fact(
    content: str = "K+ = 4.1",
    days_ago: float = 0.0,
    fact_type: str = "general",
    quality: float = 1.0,
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


# ===========================================================================
# 1. ABC Compliance
# ===========================================================================


class TestABCCompliance:
    """OpinionDecayFunction must satisfy the DecayFunction protocol."""

    def test_is_subclass_of_decay_function(self):
        assert issubclass(OpinionDecayFunction, DecayFunction)

    def test_compute_returns_float(self):
        odf = OpinionDecayFunction()
        result = odf.compute(_fact(days_ago=1), QUERY_TIME)
        assert isinstance(result, float)

    def test_compute_batch_returns_list_of_floats(self):
        odf = OpinionDecayFunction()
        facts = [_fact(days_ago=i) for i in range(3)]
        results = odf.compute_batch(facts, QUERY_TIME)
        assert isinstance(results, list)
        assert all(isinstance(r, float) for r in results)
        assert len(results) == 3

    def test_compute_returns_value_in_unit_interval(self):
        odf = OpinionDecayFunction()
        for days in [0, 1, 7, 30, 365]:
            v = odf.compute(_fact(days_ago=days), QUERY_TIME)
            assert 0.0 <= v <= 1.0


# ===========================================================================
# 2. Opinion Construction
# ===========================================================================


class TestOpinionConstruction:
    """Initial opinions from TemporalFact before any temporal decay."""

    def test_from_confidence_mode(self):
        """source_quality maps to projected probability via from_confidence."""
        odf = OpinionDecayFunction(construction="confidence")
        opinion = odf.compute_opinion(_fact(days_ago=0, quality=0.9), QUERY_TIME)
        assert isinstance(opinion, Opinion)
        # Fresh fact: no decay, projected prob ≈ source_quality
        assert opinion.projected_probability() == pytest.approx(0.9, abs=0.05)

    def test_from_evidence_mode(self):
        """source_quality maps to evidence counts via from_evidence."""
        odf = OpinionDecayFunction(construction="evidence")
        opinion = odf.compute_opinion(_fact(days_ago=0, quality=0.9), QUERY_TIME)
        assert isinstance(opinion, Opinion)
        # Fresh fact: should have high belief, low uncertainty
        assert opinion.belief > 0.5
        assert opinion.uncertainty > 0  # evidence-based always has some u

    def test_from_confidence_preserves_quality_at_zero_age(self):
        odf = OpinionDecayFunction(construction="confidence")
        for q in [0.1, 0.5, 0.8, 1.0]:
            opinion = odf.compute_opinion(_fact(days_ago=0, quality=q), QUERY_TIME)
            assert opinion.projected_probability() == pytest.approx(q, abs=0.05)

    def test_from_evidence_higher_quality_means_higher_belief(self):
        odf = OpinionDecayFunction(construction="evidence")
        low = odf.compute_opinion(_fact(days_ago=0, quality=0.3), QUERY_TIME)
        high = odf.compute_opinion(_fact(days_ago=0, quality=0.9), QUERY_TIME)
        assert high.belief > low.belief

    def test_from_evidence_higher_quality_means_lower_uncertainty(self):
        odf = OpinionDecayFunction(construction="evidence")
        low = odf.compute_opinion(_fact(days_ago=0, quality=0.3), QUERY_TIME)
        high = odf.compute_opinion(_fact(days_ago=0, quality=0.9), QUERY_TIME)
        assert high.uncertainty < low.uncertainty

    def test_custom_base_uncertainty_in_confidence_mode(self):
        """Users can set how much prior uncertainty to inject."""
        odf = OpinionDecayFunction(construction="confidence", base_uncertainty=0.3)
        opinion = odf.compute_opinion(_fact(days_ago=0, quality=0.8), QUERY_TIME)
        # With base_uncertainty=0.3, even fresh facts have u >= 0.3
        assert opinion.uncertainty >= 0.3 - 1e-9

    def test_invalid_construction_mode_raises(self):
        with pytest.raises(ValueError, match="construction"):
            OpinionDecayFunction(construction="invalid")


# ===========================================================================
# 3. Temporal Decay (Jøsang Aging Operator)
# ===========================================================================


class TestTemporalDecay:
    """Decay migrates mass into uncertainty, preserving b/d ratio."""

    @pytest.fixture
    def odf(self) -> OpinionDecayFunction:
        return OpinionDecayFunction(
            half_lives={"general": 7.0},  # 7-day half-life
            construction="confidence",
            base_uncertainty=0.1,
        )

    def test_fresh_fact_has_low_uncertainty(self, odf):
        opinion = odf.compute_opinion(_fact(days_ago=0), QUERY_TIME)
        assert opinion.uncertainty <= 0.15  # Close to base_uncertainty

    def test_old_fact_has_high_uncertainty(self, odf):
        opinion = odf.compute_opinion(_fact(days_ago=30), QUERY_TIME)
        assert opinion.uncertainty > 0.5

    def test_very_old_fact_approaches_vacuous(self, odf):
        opinion = odf.compute_opinion(_fact(days_ago=365), QUERY_TIME)
        assert opinion.uncertainty > 0.95

    def test_uncertainty_monotonically_increases_with_age(self, odf):
        ages = [0, 1, 3, 7, 14, 30, 90]
        opinions = [odf.compute_opinion(_fact(days_ago=a), QUERY_TIME) for a in ages]
        uncertainties = [o.uncertainty for o in opinions]
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] <= uncertainties[i + 1] + 1e-9

    def test_belief_monotonically_decreases_with_age(self, odf):
        ages = [0, 1, 3, 7, 14, 30, 90]
        opinions = [odf.compute_opinion(_fact(days_ago=a), QUERY_TIME) for a in ages]
        beliefs = [o.belief for o in opinions]
        for i in range(len(beliefs) - 1):
            assert beliefs[i] >= beliefs[i + 1] - 1e-9

    def test_bd_ratio_preserved_during_decay(self, odf):
        """Key SL invariant: we forget HOW MUCH evidence, not WHICH DIRECTION."""
        # Use a fact with both belief and disbelief
        odf_evidence = OpinionDecayFunction(
            half_lives={"general": 7.0},
            construction="evidence",
        )
        fresh = odf_evidence.compute_opinion(_fact(days_ago=0, quality=0.7), QUERY_TIME)
        aged = odf_evidence.compute_opinion(_fact(days_ago=14, quality=0.7), QUERY_TIME)

        # Both should have some belief and potentially some disbelief
        if fresh.disbelief > 1e-9 and aged.disbelief > 1e-9:
            ratio_fresh = fresh.belief / fresh.disbelief
            ratio_aged = aged.belief / aged.disbelief
            assert ratio_fresh == pytest.approx(ratio_aged, rel=0.01)

    def test_additivity_preserved(self, odf):
        """b + d + u = 1 always."""
        for days in [0, 1, 7, 30, 365]:
            opinion = odf.compute_opinion(_fact(days_ago=days), QUERY_TIME)
            total = opinion.belief + opinion.disbelief + opinion.uncertainty
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_projected_probability_decreases_with_age(self, odf):
        """P(ω') = b' + a·u' should generally decrease as evidence decays."""
        fresh = odf.compute_opinion(_fact(days_ago=0), QUERY_TIME)
        old = odf.compute_opinion(_fact(days_ago=30), QUERY_TIME)
        # For high initial belief (quality=1.0) with a=0.5 base rate,
        # P decreases as belief migrates to uncertainty
        assert fresh.projected_probability() >= old.projected_probability()


# ===========================================================================
# 4. Backward Compatibility — compute() vs compute_opinion()
# ===========================================================================


class TestBackwardCompatibility:
    """compute() must return the projected probability of compute_opinion()."""

    def test_compute_equals_projected_probability(self):
        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        fact = _fact(days_ago=5)
        scalar = odf.compute(fact, QUERY_TIME)
        opinion = odf.compute_opinion(fact, QUERY_TIME)
        assert scalar == pytest.approx(opinion.projected_probability(), abs=1e-9)

    def test_compute_batch_matches_individual(self):
        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        facts = [_fact(days_ago=i * 3) for i in range(5)]
        batch = odf.compute_batch(facts, QUERY_TIME)
        individual = [odf.compute(f, QUERY_TIME) for f in facts]
        for b, i in zip(batch, individual):
            assert b == pytest.approx(i, abs=1e-9)


# ===========================================================================
# 5. Configurable Per-Fact-Type Parameters
# ===========================================================================


class TestPerFactTypeConfig:
    """Half-lives and base rates should be configurable per fact_type."""

    def test_different_half_lives_per_type(self):
        odf = OpinionDecayFunction(
            half_lives={"vital_sign": 1.0, "chronic": 365.0},
            default_half_life=7.0,
        )
        vital = odf.compute_opinion(
            _fact(days_ago=3, fact_type="vital_sign"), QUERY_TIME
        )
        chronic = odf.compute_opinion(
            _fact(days_ago=3, fact_type="chronic"), QUERY_TIME
        )
        # Vital signs decay much faster
        assert vital.uncertainty > chronic.uncertainty

    def test_default_half_life_for_unknown_type(self):
        odf = OpinionDecayFunction(
            half_lives={"vital_sign": 1.0},
            default_half_life=7.0,
        )
        opinion = odf.compute_opinion(
            _fact(days_ago=7, fact_type="unknown_type"), QUERY_TIME
        )
        # At exactly 1 half-life, belief should be ~half
        assert opinion.belief < 0.6

    def test_different_base_rates_per_type(self):
        odf = OpinionDecayFunction(
            base_rates={"vital_sign": 0.5, "demographic": 0.9},
            default_base_rate=0.5,
        )
        vital = odf.compute_opinion(
            _fact(days_ago=30, fact_type="vital_sign"), QUERY_TIME
        )
        demo = odf.compute_opinion(
            _fact(days_ago=30, fact_type="demographic"), QUERY_TIME
        )
        assert vital.base_rate == pytest.approx(0.5)
        assert demo.base_rate == pytest.approx(0.9)

    def test_default_base_rate(self):
        odf = OpinionDecayFunction(default_base_rate=0.6)
        opinion = odf.compute_opinion(_fact(days_ago=0), QUERY_TIME)
        assert opinion.base_rate == pytest.approx(0.6)


# ===========================================================================
# 6. OpinionConfig
# ===========================================================================


class TestOpinionConfig:
    """OpinionConfig data class for per-fact-type configuration."""

    def test_config_stores_half_life(self):
        cfg = OpinionConfig(half_life=3.0)
        assert cfg.half_life == 3.0

    def test_config_stores_base_rate(self):
        cfg = OpinionConfig(half_life=3.0, base_rate=0.7)
        assert cfg.base_rate == 0.7

    def test_config_defaults(self):
        cfg = OpinionConfig(half_life=7.0)
        assert cfg.base_rate == 0.5
        assert cfg.base_uncertainty == 0.0

    def test_odf_accepts_config_dict(self):
        """Per-type config via dict of OpinionConfig objects."""
        configs = {
            "vital_sign": OpinionConfig(half_life=1.0, base_rate=0.5),
            "demographic": OpinionConfig(half_life=3650.0, base_rate=0.9),
        }
        odf = OpinionDecayFunction(configs=configs)
        vital = odf.compute_opinion(
            _fact(days_ago=3, fact_type="vital_sign"), QUERY_TIME
        )
        demo = odf.compute_opinion(
            _fact(days_ago=3, fact_type="demographic"), QUERY_TIME
        )
        assert vital.uncertainty > demo.uncertainty
        assert vital.base_rate == pytest.approx(0.5)
        assert demo.base_rate == pytest.approx(0.9)


# ===========================================================================
# 7. Integration with Existing Chronofy Components
# ===========================================================================


class TestIntegration:
    """OpinionDecayFunction should work as drop-in for any DecayFunction consumer."""

    def test_works_with_epistemic_filter(self):
        from chronofy.retrieval.filter import EpistemicFilter

        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        ef = EpistemicFilter(decay_fn=odf, threshold=0.3)

        facts = [
            _fact(content="fresh", days_ago=1),
            _fact(content="old", days_ago=30),
        ]
        valid = ef.filter(facts, QUERY_TIME)
        # Fresh fact should survive, old one likely filtered
        assert any("fresh" in f.content for f in valid)

    def test_works_with_temporal_scorer(self):
        from chronofy.scoring.temporal_scorer import TemporalScorer

        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        scorer = TemporalScorer(decay_fn=odf)
        facts = [_fact(days_ago=3)]
        similarities = [0.9]
        results = scorer.score_facts(facts, similarities, QUERY_TIME)
        assert len(results) == 1
        assert isinstance(results[0].combined_score, float)
        assert 0.0 <= results[0].validity <= 1.0

    def test_works_with_stl_verifier(self):
        from chronofy.verification.stl import STLVerifier
        from chronofy.models import ReasoningStep, ReasoningTrace

        odf = OpinionDecayFunction(half_lives={"general": 7.0})
        verifier = STLVerifier(decay_fn=odf, threshold=0.1)

        trace = ReasoningTrace(
            steps=[
                ReasoningStep(
                    step_index=0,
                    content="Check potassium",
                    facts_used=[_fact(days_ago=1)],
                )
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.robustness > 0  # Fresh fact, should pass
