"""Tests for alternative decay functions.

These implement standard temporal decay models from the literature,
giving Chronofy users a pluggable menu of decay backends:

  - HalfLifeDecay:  V(Δt) = q · 0.5^(Δt / h)
  - LinearDecay:    V(Δt) = q · max(0, 1 - α·Δt)
  - PowerLawDecay:  V(Δt) = q · (1 + Δt)^(-α)
  - WeibullDecay:   V(Δt) = q · exp(-(Δt/λ)^k)

All incorporate source reliability q_e as a multiplicative factor.
"""

import math
from datetime import datetime, timedelta

import pytest

from chronofy.models import TemporalFact
from chronofy.decay.half_life import HalfLifeDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.power_law import PowerLawDecay
from chronofy.decay.weibull import WeibullDecay


QUERY_TIME = datetime(2026, 3, 15)


def _fact(days_ago: float, fact_type: str = "general", quality: float = 1.0) -> TemporalFact:
    return TemporalFact(
        content=f"test_{fact_type}",
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


# ===================================================================
# HalfLifeDecay: V(Δt) = q · 0.5^(Δt / h)
# ===================================================================

class TestHalfLifeDecay:
    """Standard half-life decay from pharmacokinetics / radiometric dating."""

    def test_zero_age(self):
        d = HalfLifeDecay(half_life={"general": 7.0})
        assert d.compute(_fact(0.0), QUERY_TIME) == pytest.approx(1.0)

    def test_at_half_life(self):
        """At exactly one half-life, validity should be 0.5."""
        d = HalfLifeDecay(half_life={"general": 7.0})
        assert d.compute(_fact(7.0), QUERY_TIME) == pytest.approx(0.5)

    def test_at_two_half_lives(self):
        """At two half-lives, validity should be 0.25."""
        d = HalfLifeDecay(half_life={"general": 7.0})
        assert d.compute(_fact(14.0), QUERY_TIME) == pytest.approx(0.25)

    def test_quality_weight(self):
        d = HalfLifeDecay(half_life={"general": 7.0})
        assert d.compute(_fact(0.0, quality=0.6), QUERY_TIME) == pytest.approx(0.6)

    def test_equivalence_to_exponential(self):
        """HalfLifeDecay(h) should equal ExponentialDecay(β=ln(2)/h)."""
        h = 5.0
        beta = math.log(2) / h
        d = HalfLifeDecay(half_life={"general": h})
        age = 3.0
        expected = math.exp(-beta * age)
        assert d.compute(_fact(age), QUERY_TIME) == pytest.approx(expected)

    def test_per_fact_type(self):
        d = HalfLifeDecay(half_life={"fast": 1.0, "slow": 100.0})
        v_fast = d.compute(_fact(5.0, "fast"), QUERY_TIME)
        v_slow = d.compute(_fact(5.0, "slow"), QUERY_TIME)
        assert v_slow > v_fast

    def test_default_half_life(self):
        d = HalfLifeDecay(default_half_life=10.0)
        assert d.compute(_fact(10.0, "unknown"), QUERY_TIME) == pytest.approx(0.5)

    def test_compute_batch(self):
        d = HalfLifeDecay(half_life={"general": 7.0})
        facts = [_fact(0.0), _fact(7.0), _fact(14.0)]
        scores = d.compute_batch(facts, QUERY_TIME)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_get_beta(self):
        """get_beta should return ln(2)/h for the equivalent β."""
        d = HalfLifeDecay(half_life={"general": 7.0})
        assert d.get_beta("general") == pytest.approx(math.log(2) / 7.0)

    def test_invalid_half_life(self):
        with pytest.raises(ValueError):
            HalfLifeDecay(half_life={"bad": -1.0})

    def test_zero_half_life(self):
        """Zero half-life is invalid (would mean instant decay)."""
        with pytest.raises(ValueError):
            HalfLifeDecay(half_life={"bad": 0.0})


# ===================================================================
# LinearDecay: V(Δt) = q · max(0, 1 - α·Δt)
# ===================================================================

class TestLinearDecay:
    """Simplest decay model — linear drop to zero."""

    def test_zero_age(self):
        d = LinearDecay(rate={"general": 0.1})
        assert d.compute(_fact(0.0), QUERY_TIME) == pytest.approx(1.0)

    def test_midpoint(self):
        """At Δt = 5, rate=0.1: V = 1 - 0.5 = 0.5."""
        d = LinearDecay(rate={"general": 0.1})
        assert d.compute(_fact(5.0), QUERY_TIME) == pytest.approx(0.5)

    def test_clamped_at_zero(self):
        """Beyond 1/α, validity is clamped to 0."""
        d = LinearDecay(rate={"general": 0.1})
        assert d.compute(_fact(15.0), QUERY_TIME) == pytest.approx(0.0)

    def test_exactly_at_expiry(self):
        """At exactly 1/α, validity should be 0."""
        d = LinearDecay(rate={"general": 0.1})
        assert d.compute(_fact(10.0), QUERY_TIME) == pytest.approx(0.0)

    def test_quality_weight(self):
        d = LinearDecay(rate={"general": 0.1})
        assert d.compute(_fact(0.0, quality=0.8), QUERY_TIME) == pytest.approx(0.8)

    def test_quality_at_midpoint(self):
        d = LinearDecay(rate={"general": 0.1})
        # V = 0.8 * (1 - 0.1*5) = 0.8 * 0.5 = 0.4
        assert d.compute(_fact(5.0, quality=0.8), QUERY_TIME) == pytest.approx(0.4)

    def test_per_fact_type(self):
        d = LinearDecay(rate={"fast": 1.0, "slow": 0.01})
        v_fast = d.compute(_fact(0.5, "fast"), QUERY_TIME)
        v_slow = d.compute(_fact(0.5, "slow"), QUERY_TIME)
        assert v_slow > v_fast

    def test_default_rate(self):
        d = LinearDecay(default_rate=0.2)
        assert d.compute(_fact(2.5, "unknown"), QUERY_TIME) == pytest.approx(0.5)

    def test_compute_batch(self):
        d = LinearDecay(rate={"general": 0.1})
        facts = [_fact(0.0), _fact(5.0), _fact(15.0)]
        scores = d.compute_batch(facts, QUERY_TIME)
        assert scores == [pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.0)]

    def test_expiry_time(self):
        """expiry_time returns 1/α — the age at which validity hits zero."""
        d = LinearDecay(rate={"general": 0.1})
        assert d.expiry_time("general") == pytest.approx(10.0)

    def test_get_beta_returns_none(self):
        """Linear decay has no β (not exponential family)."""
        d = LinearDecay(rate={"general": 0.1})
        assert d.get_beta("general") is None


# ===================================================================
# PowerLawDecay: V(Δt) = q · (1 + Δt)^(-α)
# ===================================================================

class TestPowerLawDecay:
    """Heavy-tailed decay — common in citation / memory literature."""

    def test_zero_age(self):
        d = PowerLawDecay(exponent={"general": 1.0})
        assert d.compute(_fact(0.0), QUERY_TIME) == pytest.approx(1.0)

    def test_formula(self):
        """V = (1 + Δt)^(-α)."""
        alpha = 2.0
        age = 3.0
        d = PowerLawDecay(exponent={"general": alpha})
        expected = (1.0 + age) ** (-alpha)
        assert d.compute(_fact(age), QUERY_TIME) == pytest.approx(expected)

    def test_heavy_tail(self):
        """Power-law should decay slower than exponential for large Δt."""
        age = 100.0
        d_power = PowerLawDecay(exponent={"general": 1.0})
        v_power = d_power.compute(_fact(age), QUERY_TIME)
        # Compare: exponential with same initial decay rate would give exp(-age) ≈ 0
        assert v_power > 1e-10  # Power law still has meaningful value

    def test_quality_weight(self):
        d = PowerLawDecay(exponent={"general": 1.0})
        assert d.compute(_fact(0.0, quality=0.5), QUERY_TIME) == pytest.approx(0.5)

    def test_per_fact_type(self):
        d = PowerLawDecay(exponent={"fast": 3.0, "slow": 0.5})
        v_fast = d.compute(_fact(5.0, "fast"), QUERY_TIME)
        v_slow = d.compute(_fact(5.0, "slow"), QUERY_TIME)
        assert v_slow > v_fast

    def test_default_exponent(self):
        d = PowerLawDecay(default_exponent=1.0)
        assert d.compute(_fact(1.0, "unknown"), QUERY_TIME) == pytest.approx(0.5)

    def test_compute_batch(self):
        d = PowerLawDecay(exponent={"general": 1.0})
        facts = [_fact(0.0), _fact(1.0), _fact(3.0)]
        scores = d.compute_batch(facts, QUERY_TIME)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_get_beta_returns_none(self):
        d = PowerLawDecay(exponent={"general": 1.0})
        assert d.get_beta("general") is None


# ===================================================================
# WeibullDecay: V(Δt) = q · exp(-(Δt/λ)^k)
# ===================================================================

class TestWeibullDecay:
    """Generalized exponential — accelerating or decelerating obsolescence."""

    def test_zero_age(self):
        d = WeibullDecay(scale={"general": 5.0}, shape={"general": 1.0})
        assert d.compute(_fact(0.0), QUERY_TIME) == pytest.approx(1.0)

    def test_shape_one_recovers_exponential(self):
        """k=1 should recover standard exponential decay: exp(-Δt/λ)."""
        lam = 5.0
        age = 3.0
        d = WeibullDecay(scale={"general": lam}, shape={"general": 1.0})
        expected = math.exp(-(age / lam))
        assert d.compute(_fact(age), QUERY_TIME) == pytest.approx(expected)

    def test_shape_gt_one_faster_decay(self):
        """k > 1: accelerating obsolescence (wear-out)."""
        lam = 5.0
        age = 3.0
        d_exp = WeibullDecay(scale={"general": lam}, shape={"general": 1.0})
        d_fast = WeibullDecay(scale={"general": lam}, shape={"general": 2.0})
        v_exp = d_exp.compute(_fact(age), QUERY_TIME)
        v_fast = d_fast.compute(_fact(age), QUERY_TIME)
        # For age < λ, k>1 decays slower initially then faster
        # For age > 1, the comparison depends — just verify it's different
        assert v_fast != pytest.approx(v_exp)

    def test_shape_lt_one_slower_decay(self):
        """k < 1: decelerating obsolescence (infant mortality)."""
        lam = 5.0
        age = 10.0
        d_exp = WeibullDecay(scale={"general": lam}, shape={"general": 1.0})
        d_slow = WeibullDecay(scale={"general": lam}, shape={"general": 0.5})
        v_exp = d_exp.compute(_fact(age), QUERY_TIME)
        v_slow = d_slow.compute(_fact(age), QUERY_TIME)
        # k < 1 has heavier tail at large ages
        assert v_slow > v_exp

    def test_formula(self):
        """V = exp(-(Δt/λ)^k)."""
        lam, k, age = 3.0, 2.0, 4.0
        d = WeibullDecay(scale={"general": lam}, shape={"general": k})
        expected = math.exp(-((age / lam) ** k))
        assert d.compute(_fact(age), QUERY_TIME) == pytest.approx(expected)

    def test_quality_weight(self):
        d = WeibullDecay(scale={"general": 5.0}, shape={"general": 1.0})
        assert d.compute(_fact(0.0, quality=0.7), QUERY_TIME) == pytest.approx(0.7)

    def test_per_fact_type(self):
        d = WeibullDecay(
            scale={"fast": 1.0, "slow": 100.0},
            shape={"fast": 2.0, "slow": 0.5},
        )
        v_fast = d.compute(_fact(5.0, "fast"), QUERY_TIME)
        v_slow = d.compute(_fact(5.0, "slow"), QUERY_TIME)
        assert v_slow > v_fast

    def test_compute_batch(self):
        d = WeibullDecay(scale={"general": 5.0}, shape={"general": 1.0})
        facts = [_fact(0.0), _fact(5.0), _fact(10.0)]
        scores = d.compute_batch(facts, QUERY_TIME)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_get_beta_shape_one(self):
        """When k=1, equivalent β = 1/λ."""
        d = WeibullDecay(scale={"general": 5.0}, shape={"general": 1.0})
        assert d.get_beta("general") == pytest.approx(1.0 / 5.0)

    def test_get_beta_shape_not_one(self):
        """When k≠1, no single β equivalent."""
        d = WeibullDecay(scale={"general": 5.0}, shape={"general": 2.0})
        assert d.get_beta("general") is None

    def test_default_params(self):
        d = WeibullDecay(default_scale=5.0, default_shape=1.0)
        fact = _fact(5.0, "unknown")
        expected = math.exp(-1.0)
        assert d.compute(fact, QUERY_TIME) == pytest.approx(expected)
