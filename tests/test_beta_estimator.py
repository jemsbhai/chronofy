"""Tests for chronofy.analysis.beta_estimator.BetaEstimator.

TDD: written BEFORE the implementation.

Scientific grounding:
  Under the model P(valid | age, β) = exp(-β · age), the MLE for β
  given observations {(a_i, v_i)} maximises:
      L(β) = Σ v_i · log(exp(-β·a_i)) + (1-v_i) · log(1 - exp(-β·a_i))
           = -β · Σ v_i·a_i + Σ (1-v_i) · log(1 - exp(-β·a_i))

Test coverage plan:
  - Known analytical special cases (all valid → β→0, all invalid → β large)
  - Single observation
  - Monotone recovery: older facts are more likely invalid
  - Result fields are complete and consistent
  - fit_corpus groups by fact_type correctly
  - fit_corpus with parallel label list
  - to_decay produces ExponentialDecay with fitted betas
  - Edge cases: all valid, all invalid, single type, empty type
  - Beta is always non-negative
  - Half-life is consistent with beta
  - n_obs and n_valid counts are correct
  - Raises on empty input
  - Raises on mismatched ages/labels
  - Beta=0 case (invariant fact type): all valid regardless of age
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from chronofy.analysis.beta_estimator import (
    BetaEstimateResult,
    BetaEstimator,
    EstimationMethod,
    MLEBernoulli,
    MomentMatching,
    EnsembleMethod,
)
from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2024, 6, 1)


def make_fact(age_days: float, fact_type: str = "general") -> TemporalFact:
    ts = QUERY_TIME - timedelta(days=age_days)
    return TemporalFact(
        content=f"fact aged {age_days}d",
        timestamp=ts,
        fact_type=fact_type,
    )


# ---------------------------------------------------------------------------
# 1. BetaEstimateResult structure
# ---------------------------------------------------------------------------


class TestBetaEstimateResult:
    def test_result_has_required_fields(self):
        result = BetaEstimateResult(
            beta=0.5,
            half_life=math.log(2) / 0.5,
            log_likelihood=-3.0,
            n_obs=10,
            n_valid=7,
            converged=True,
            fact_type="general",
        )
        assert result.beta == 0.5
        assert result.n_obs == 10
        assert result.n_valid == 7
        assert result.converged is True
        assert result.fact_type == "general"

    def test_half_life_consistent_with_beta(self):
        beta = 1.0
        result = BetaEstimateResult(
            beta=beta,
            half_life=math.log(2) / beta,
            log_likelihood=-1.0,
            n_obs=5,
            n_valid=3,
            converged=True,
            fact_type="test",
        )
        assert math.isclose(result.half_life, math.log(2), rel_tol=1e-9)

    def test_infinite_half_life_when_beta_zero(self):
        result = BetaEstimateResult(
            beta=0.0,
            half_life=float("inf"),
            log_likelihood=0.0,
            n_obs=5,
            n_valid=5,
            converged=True,
            fact_type="invariant",
        )
        assert result.half_life == float("inf")

    def test_to_dict_has_all_keys(self):
        result = BetaEstimateResult(
            beta=0.3,
            half_life=math.log(2) / 0.3,
            log_likelihood=-2.0,
            n_obs=8,
            n_valid=5,
            converged=True,
            fact_type="lab",
        )
        d = result.to_dict()
        assert {"beta", "half_life", "log_likelihood", "n_obs", "n_valid",
                "converged", "fact_type"}.issubset(d.keys())


# ---------------------------------------------------------------------------
# 2. fit() — basic behaviour
# ---------------------------------------------------------------------------


class TestFitBasic:
    def setup_method(self):
        self.estimator = BetaEstimator()

    def test_raises_on_empty_ages(self):
        with pytest.raises(ValueError):
            self.estimator.fit(ages=[], valid=[])

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            self.estimator.fit(ages=[1.0, 2.0], valid=[True])

    def test_beta_non_negative(self):
        result = self.estimator.fit(ages=[1.0, 5.0, 10.0], valid=[True, False, False])
        assert result.beta >= 0.0

    def test_n_obs_correct(self):
        result = self.estimator.fit(ages=[1.0, 2.0, 3.0], valid=[True, True, False])
        assert result.n_obs == 3

    def test_n_valid_correct(self):
        result = self.estimator.fit(ages=[1.0, 2.0, 3.0], valid=[True, True, False])
        assert result.n_valid == 2

    def test_returns_beta_estimate_result(self):
        result = self.estimator.fit(ages=[1.0, 5.0], valid=[True, False])
        assert isinstance(result, BetaEstimateResult)

    def test_fact_type_stored(self):
        result = self.estimator.fit(ages=[1.0], valid=[True], fact_type="vital_sign")
        assert result.fact_type == "vital_sign"

    def test_default_fact_type(self):
        result = self.estimator.fit(ages=[1.0], valid=[True])
        assert result.fact_type == "general"


# ---------------------------------------------------------------------------
# 3. fit() — analytical special cases
# ---------------------------------------------------------------------------


class TestFitAnalytical:
    def setup_method(self):
        self.estimator = BetaEstimator()

    def test_all_valid_gives_low_beta(self):
        # If everything is still valid, the process decays slowly → small β
        ages = [1.0, 5.0, 10.0, 30.0, 100.0]
        valid = [True] * len(ages)
        result = self.estimator.fit(ages=ages, valid=valid)
        assert result.beta < 0.1

    def test_all_invalid_gives_high_beta(self):
        # If everything is invalid even at age 1, process decays fast → large β
        ages = [1.0, 1.0, 1.0, 1.0, 1.0]
        valid = [False] * len(ages)
        result = self.estimator.fit(ages=ages, valid=valid)
        assert result.beta > 1.0

    def test_older_invalid_gives_higher_beta_than_younger_invalid(self):
        # Scenario A: goes invalid at age 1 → fast decay
        result_fast = self.estimator.fit(ages=[0.5, 1.0], valid=[True, False])
        # Scenario B: goes invalid at age 100 → slow decay
        result_slow = self.estimator.fit(ages=[50.0, 100.0], valid=[True, False])
        assert result_fast.beta > result_slow.beta

    def test_half_life_consistent_with_fitted_beta(self):
        ages = [1.0, 2.0, 5.0, 10.0]
        valid = [True, True, False, False]
        result = self.estimator.fit(ages=ages, valid=valid)
        if result.beta > 0:
            expected_half_life = math.log(2) / result.beta
            assert math.isclose(result.half_life, expected_half_life, rel_tol=1e-6)
        else:
            assert result.half_life == float("inf")

    def test_single_valid_observation(self):
        # Just one observation — should not crash, beta should be low
        result = self.estimator.fit(ages=[5.0], valid=[True])
        assert result.beta >= 0.0
        assert result.n_obs == 1

    def test_single_invalid_observation(self):
        result = self.estimator.fit(ages=[5.0], valid=[False])
        assert result.beta >= 0.0
        assert result.n_obs == 1


# ---------------------------------------------------------------------------
# 4. fit_corpus() — grouping by fact_type
# ---------------------------------------------------------------------------


class TestFitCorpus:
    def setup_method(self):
        self.estimator = BetaEstimator()

    def _make_corpus_and_labels(self):
        """Two types: 'fast' (goes invalid quickly) and 'slow' (stays valid long)."""
        facts = [
            # fast: invalid by day 2
            make_fact(0.5, "fast"),
            make_fact(1.0, "fast"),
            make_fact(2.0, "fast"),
            make_fact(5.0, "fast"),
            # slow: valid even at day 100
            make_fact(1.0, "slow"),
            make_fact(10.0, "slow"),
            make_fact(50.0, "slow"),
            make_fact(100.0, "slow"),
        ]
        labels = [True, True, False, False, True, True, True, False]
        return facts, labels

    def test_returns_dict_keyed_by_fact_type(self):
        facts, labels = self._make_corpus_and_labels()
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        assert isinstance(results, dict)
        assert "fast" in results
        assert "slow" in results

    def test_fast_type_has_higher_beta_than_slow(self):
        facts, labels = self._make_corpus_and_labels()
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        assert results["fast"].beta > results["slow"].beta

    def test_n_obs_per_type_correct(self):
        facts, labels = self._make_corpus_and_labels()
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        assert results["fast"].n_obs == 4
        assert results["slow"].n_obs == 4

    def test_raises_on_mismatched_lengths(self):
        facts = [make_fact(1.0, "general"), make_fact(2.0, "general")]
        with pytest.raises(ValueError):
            self.estimator.fit_corpus(facts=facts, labels=[True],
                                      query_time=QUERY_TIME)

    def test_raises_on_empty_facts(self):
        with pytest.raises(ValueError):
            self.estimator.fit_corpus(facts=[], labels=[], query_time=QUERY_TIME)

    def test_single_fact_type(self):
        facts = [make_fact(float(d), "lab") for d in [1, 5, 10, 30]]
        labels = [True, True, False, False]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        assert "lab" in results
        assert len(results) == 1

    def test_fact_type_stored_in_result(self):
        facts = [make_fact(1.0, "vital"), make_fact(5.0, "vital")]
        labels = [True, False]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        assert results["vital"].fact_type == "vital"


# ---------------------------------------------------------------------------
# 5. to_decay() — produces a working ExponentialDecay
# ---------------------------------------------------------------------------


class TestToDecay:
    def setup_method(self):
        self.estimator = BetaEstimator()

    def test_returns_exponential_decay(self):
        facts = [make_fact(1.0, "lab"), make_fact(5.0, "lab")]
        labels = [True, False]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        decay = self.estimator.to_decay(results)
        assert isinstance(decay, ExponentialDecay)

    def test_decay_uses_fitted_beta(self):
        facts = [make_fact(1.0, "lab"), make_fact(5.0, "lab")]
        labels = [True, False]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        decay = self.estimator.to_decay(results)
        fitted_beta = results["lab"].beta
        assert math.isclose(decay.get_beta("lab"), fitted_beta, rel_tol=1e-9)

    def test_decay_produces_valid_scores(self):
        facts = [make_fact(float(d), "general") for d in [1, 5, 10, 30]]
        labels = [True, True, False, False]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        decay = self.estimator.to_decay(results)
        # Score for a fresh fact should be in (0, 1]
        fresh = make_fact(0.0, "general")
        score = decay.compute(fresh, QUERY_TIME)
        assert 0.0 < score <= 1.0

    def test_decay_respects_temporal_invariance_for_zero_beta(self):
        # If all observations are valid, beta → 0 → decay → 1.0 always
        facts = [make_fact(float(d), "demographic") for d in [1, 100, 1000]]
        labels = [True, True, True]
        results = self.estimator.fit_corpus(facts=facts, labels=labels,
                                            query_time=QUERY_TIME)
        decay = self.estimator.to_decay(results)
        old_fact = make_fact(500.0, "demographic")
        score = decay.compute(old_fact, QUERY_TIME)
        # With beta~0, validity should remain close to 1.0
        assert score > 0.9


# ---------------------------------------------------------------------------
# 6. EstimationMethod protocol
# ---------------------------------------------------------------------------


class TestEstimationMethodProtocol:
    def test_mle_bernoulli_is_estimation_method(self):
        assert isinstance(MLEBernoulli(), EstimationMethod)

    def test_moment_matching_is_estimation_method(self):
        assert isinstance(MomentMatching(), EstimationMethod)

    def test_ensemble_is_estimation_method(self):
        ensemble = EnsembleMethod(methods=[MLEBernoulli(), MomentMatching()])
        assert isinstance(ensemble, EstimationMethod)

    def test_custom_method_can_be_used(self):
        """Any object implementing EstimationMethod.fit() works as a backend."""
        class ConstantBeta(EstimationMethod):
            def fit(self, ages: list[float], valid: list[bool]) -> float:
                return 0.42

        estimator = BetaEstimator(method=ConstantBeta())
        result = estimator.fit(ages=[1.0, 5.0], valid=[True, False])
        assert math.isclose(result.beta, 0.42, rel_tol=1e-9)

    def test_estimation_method_has_fit_signature(self):
        """fit() must accept ages and valid lists and return a float."""
        method = MLEBernoulli()
        beta = method.fit(ages=[1.0, 2.0, 5.0], valid=[True, True, False])
        assert isinstance(beta, float)
        assert beta >= 0.0


# ---------------------------------------------------------------------------
# 7. MLEBernoulli
# ---------------------------------------------------------------------------


class TestMLEBernoulli:
    def setup_method(self):
        self.method = MLEBernoulli()

    def test_returns_non_negative_beta(self):
        beta = self.method.fit(ages=[1.0, 5.0, 10.0], valid=[True, False, False])
        assert beta >= 0.0

    def test_all_valid_returns_low_beta(self):
        beta = self.method.fit(ages=[1.0, 10.0, 100.0], valid=[True, True, True])
        assert beta < 0.1

    def test_all_invalid_returns_high_beta(self):
        beta = self.method.fit(ages=[1.0, 1.0, 1.0], valid=[False, False, False])
        assert beta > 1.0

    def test_default_estimator_uses_mle_bernoulli(self):
        # BetaEstimator() with no method arg should use MLEBernoulli
        estimator = BetaEstimator()
        result = estimator.fit(ages=[1.0, 5.0, 10.0], valid=[True, True, False])
        mle_beta = self.method.fit(ages=[1.0, 5.0, 10.0], valid=[True, True, False])
        assert math.isclose(result.beta, mle_beta, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 8. MomentMatching
# ---------------------------------------------------------------------------


class TestMomentMatching:
    def setup_method(self):
        self.method = MomentMatching()

    def test_returns_non_negative_beta(self):
        beta = self.method.fit(ages=[1.0, 5.0, 10.0], valid=[True, False, False])
        assert beta >= 0.0

    def test_all_valid_returns_low_beta(self):
        beta = self.method.fit(ages=[1.0, 10.0, 100.0], valid=[True, True, True])
        assert beta < 0.1

    def test_all_invalid_returns_high_beta(self):
        beta = self.method.fit(ages=[1.0, 1.0, 1.0], valid=[False, False, False])
        assert beta > 1.0

    def test_older_invalid_gives_lower_beta_than_young_invalid(self):
        # Goes invalid at 100 days → slow decay
        beta_slow = self.method.fit(ages=[50.0, 100.0], valid=[True, False])
        # Goes invalid at 1 day → fast decay
        beta_fast = self.method.fit(ages=[0.5, 1.0], valid=[True, False])
        assert beta_fast > beta_slow


# ---------------------------------------------------------------------------
# 9. EnsembleMethod
# ---------------------------------------------------------------------------


class TestEnsembleMethod:
    def test_default_equal_weights(self):
        ensemble = EnsembleMethod(methods=[MLEBernoulli(), MomentMatching()])
        beta = ensemble.fit(ages=[1.0, 5.0, 10.0], valid=[True, True, False])
        assert beta >= 0.0

    def test_explicit_weights(self):
        ensemble = EnsembleMethod(
            methods=[MLEBernoulli(), MomentMatching()],
            weights=[0.9, 0.1],
        )
        beta = ensemble.fit(ages=[1.0, 5.0, 10.0], valid=[True, True, False])
        assert beta >= 0.0

    def test_single_method_ensemble_matches_method(self):
        mle = MLEBernoulli()
        ensemble = EnsembleMethod(methods=[mle], weights=[1.0])
        ages = [1.0, 5.0, 10.0]
        valid = [True, True, False]
        assert math.isclose(
            ensemble.fit(ages, valid), mle.fit(ages, valid), rel_tol=1e-6
        )

    def test_weights_need_not_sum_to_one(self):
        # Weights are normalised internally
        ensemble = EnsembleMethod(
            methods=[MLEBernoulli(), MomentMatching()],
            weights=[2.0, 2.0],   # will be normalised to [0.5, 0.5]
        )
        beta = ensemble.fit(ages=[1.0, 5.0], valid=[True, False])
        assert beta >= 0.0

    def test_raises_on_empty_methods(self):
        with pytest.raises(ValueError):
            EnsembleMethod(methods=[])

    def test_raises_on_weight_method_length_mismatch(self):
        with pytest.raises(ValueError):
            EnsembleMethod(
                methods=[MLEBernoulli(), MomentMatching()],
                weights=[1.0],   # wrong length
            )

    def test_raises_on_non_positive_weights(self):
        with pytest.raises(ValueError):
            EnsembleMethod(
                methods=[MLEBernoulli(), MomentMatching()],
                weights=[1.0, -0.1],
            )

    def test_ensemble_used_in_estimator(self):
        ensemble = EnsembleMethod(methods=[MLEBernoulli(), MomentMatching()])
        estimator = BetaEstimator(method=ensemble)
        result = estimator.fit(
            ages=[1.0, 5.0, 10.0, 30.0],
            valid=[True, True, False, False],
        )
        assert isinstance(result, BetaEstimateResult)
        assert result.beta >= 0.0
        assert result.converged is True
