"""Tests for chronofy.decay.exponential — ExponentialDecay.

Validates the core decay equation V(e, T_q) = q_e · exp(-β_j · Δt)
and its decision-theoretic grounding (Proposition 1: β = 2κ).
"""

import math
from datetime import datetime, timedelta

import pytest

from chronofy.decay.exponential import ExponentialDecay, DEFAULT_BETA
from chronofy.models import TemporalFact


def _fact(days_ago: float, fact_type: str = "general", quality: float = 1.0) -> TemporalFact:
    """Helper: create a fact that is `days_ago` old relative to 2026-03-15."""
    return TemporalFact(
        content=f"test_{fact_type}",
        timestamp=datetime(2026, 3, 15) - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


QUERY_TIME = datetime(2026, 3, 15)


class TestExponentialDecayBasics:
    """Core decay computation tests."""

    def test_zero_age_gives_full_validity(self):
        """A fact observed right now should have validity = q_e."""
        decay = ExponentialDecay()
        fact = _fact(0.0, "general")
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(1.0)

    def test_zero_age_with_quality(self):
        """At zero age, validity = q_e (just the quality weight)."""
        decay = ExponentialDecay()
        fact = _fact(0.0, "general", quality=0.7)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(0.7)

    def test_positive_age_decays(self):
        """Older facts have strictly lower validity (for β > 0)."""
        decay = ExponentialDecay(beta={"general": 1.0})
        v_new = decay.compute(_fact(0.0), QUERY_TIME)
        v_old = decay.compute(_fact(1.0), QUERY_TIME)
        assert v_new > v_old

    def test_exponential_formula(self):
        """Verify the exact formula: q_e · exp(-β · Δt)."""
        beta_val = 2.0
        days_ago = 3.0
        quality = 0.9
        decay = ExponentialDecay(beta={"lab_result": beta_val})
        fact = _fact(days_ago, "lab_result", quality)
        expected = quality * math.exp(-beta_val * days_ago)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)

    def test_high_beta_rapid_decay(self):
        """Vital signs (high β) should decay very fast."""
        decay = ExponentialDecay()  # Uses DEFAULT_BETA
        fact = _fact(1.0, "vital_sign")  # 1 day old, β=5.0
        expected = math.exp(-5.0 * 1.0)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)
        assert decay.compute(fact, QUERY_TIME) < 0.01  # Nearly zero after 1 day

    def test_low_beta_slow_decay(self):
        """Chronic conditions (low β) should decay very slowly."""
        decay = ExponentialDecay()
        fact = _fact(365.0, "chronic_condition")  # 1 year old, β=0.01
        expected = math.exp(-0.01 * 365.0)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)
        assert decay.compute(fact, QUERY_TIME) > 0.02  # Still has some validity


class TestTemporalInvariance:
    """Property 2: Temporal Invariance Guarantee.

    When β_j = 0 (e.g., demographics, blood type), decay is always 1.0
    regardless of age. This is the formal temporal invariance null axiom.
    """

    def test_demographic_invariance(self):
        """Demographics (β=0) never decay."""
        decay = ExponentialDecay()
        fact = _fact(3650.0, "demographic")  # 10 years old
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(1.0)

    def test_custom_zero_beta(self):
        """Custom fact type with β=0 is invariant."""
        decay = ExponentialDecay(beta={"blood_type": 0.0})
        fact = _fact(10000.0, "blood_type")
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(1.0)

    def test_invariance_with_quality(self):
        """Invariant facts still carry their quality weight."""
        decay = ExponentialDecay()
        fact = _fact(3650.0, "demographic", quality=0.8)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(0.8)


class TestProposition1:
    """Proposition 1: β = 2κ — decision-theoretic grounding.

    Under an Ornstein-Uhlenbeck latent process with mean-reversion rate κ,
    the information content decays as exp(-2κ · Δt). Therefore β_j = 2κ_j.
    """

    def test_from_mean_reversion_rate(self):
        """Construct from κ values; verify β = 2κ."""
        kappa = {"vital_sign": 2.5, "lab_result": 1.0, "demographic": 0.0}
        decay = ExponentialDecay.from_mean_reversion_rate(kappa)
        assert decay.get_beta("vital_sign") == pytest.approx(5.0)
        assert decay.get_beta("lab_result") == pytest.approx(2.0)
        assert decay.get_beta("demographic") == pytest.approx(0.0)

    def test_from_kappa_matches_direct(self):
        """from_mean_reversion_rate(κ) should produce same results as beta=2κ."""
        kappa = {"test": 1.5}
        decay_kappa = ExponentialDecay.from_mean_reversion_rate(kappa)
        decay_beta = ExponentialDecay(beta={"test": 3.0})
        fact = _fact(2.0, "test")
        assert decay_kappa.compute(fact, QUERY_TIME) == pytest.approx(
            decay_beta.compute(fact, QUERY_TIME)
        )


class TestHalfLife:
    """Half-life = ln(2)/β — the time for validity to drop to 50%."""

    def test_half_life_formula(self):
        decay = ExponentialDecay(beta={"test": 2.0})
        assert decay.half_life("test") == pytest.approx(math.log(2) / 2.0)

    def test_half_life_invariant_returns_none(self):
        """Invariant facts (β=0) have no half-life."""
        decay = ExponentialDecay()
        assert decay.half_life("demographic") is None

    def test_validity_at_half_life(self):
        """At the half-life, validity should be ~0.5 (with q=1)."""
        beta_val = 3.0
        decay = ExponentialDecay(beta={"test": beta_val})
        hl = decay.half_life("test")
        assert hl is not None
        fact = _fact(hl, "test")
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(0.5, rel=1e-6)


class TestBatchAndDefaults:
    """Batch computation and default β behavior."""

    def test_compute_batch(self):
        """Batch computation returns one score per fact."""
        decay = ExponentialDecay(beta={"general": 1.0})
        facts = [_fact(0.0), _fact(1.0), _fact(2.0)]
        scores = decay.compute_batch(facts, QUERY_TIME)
        assert len(scores) == 3
        # Scores should be monotonically decreasing
        assert scores[0] > scores[1] > scores[2]

    def test_default_beta_for_unknown_type(self):
        """Unknown fact types use the default_beta."""
        decay = ExponentialDecay(default_beta=0.75)
        fact = _fact(1.0, "unknown_type")
        expected = math.exp(-0.75 * 1.0)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)

    def test_custom_beta_overrides_default(self):
        """Explicit beta dict overrides DEFAULT_BETA for that type."""
        decay = ExponentialDecay(beta={"vital_sign": 99.0})
        assert decay.get_beta("vital_sign") == pytest.approx(99.0)

    def test_get_beta(self):
        decay = ExponentialDecay()
        assert decay.get_beta("vital_sign") == pytest.approx(DEFAULT_BETA["vital_sign"])


class TestTimeUnit:
    """Configurable time unit (days, hours, seconds)."""

    def test_hours_unit(self):
        """With time_unit='hours', Δt is measured in hours."""
        decay = ExponentialDecay(beta={"test": 1.0}, time_unit="hours")
        fact = TemporalFact(
            content="test",
            timestamp=QUERY_TIME - timedelta(hours=2),
            fact_type="test",
        )
        expected = math.exp(-1.0 * 2.0)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)

    def test_seconds_unit(self):
        """With time_unit='seconds', Δt is measured in seconds."""
        decay = ExponentialDecay(beta={"test": 0.01}, time_unit="seconds")
        fact = TemporalFact(
            content="test",
            timestamp=QUERY_TIME - timedelta(seconds=100),
            fact_type="test",
        )
        expected = math.exp(-0.01 * 100.0)
        assert decay.compute(fact, QUERY_TIME) == pytest.approx(expected)


class TestRepr:
    """String representation."""

    def test_repr_includes_nonzero_betas(self):
        decay = ExponentialDecay(beta={"fast": 5.0, "slow": 0.1, "none": 0.0})
        r = repr(decay)
        assert "ExponentialDecay" in r
        assert "fast" in r
        assert "slow" in r
        # β=0 types should be excluded
        assert "none" not in r
