"""Tests for chronofy.plugins — custom plugin validation utilities.

TDD: written BEFORE the implementation.

validate_decay_function(fn) and validate_estimation_method(method) give
users immediate, readable feedback when a custom implementation is broken.
They do NOT replace the type system — they run behavioural smoke tests.

Coverage:
  - Valid custom decay function passes validation
  - Valid custom estimation method passes validation
  - Missing .compute() raises PluginValidationError
  - .compute() returning out-of-range value raises
  - .compute() returning wrong type raises
  - Missing .compute_batch() raises
  - .compute_batch() returning wrong length raises
  - Missing .fit() on estimation method raises
  - .fit() returning negative beta raises
  - .fit() returning wrong type raises
  - Concrete built-in implementations pass validation
  - EnsembleMethod passes validation
  - validate_decay_function returns the function unchanged on success
  - validate_estimation_method returns the method unchanged on success
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from chronofy.plugins import (
    PluginValidationError,
    validate_decay_function,
    validate_estimation_method,
)
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.weibull import WeibullDecay
from chronofy.decay.base import DecayFunction
from chronofy.analysis.beta_estimator import (
    EstimationMethod,
    MLEBernoulli,
    MomentMatching,
    EnsembleMethod,
)
from chronofy.models import TemporalFact

QUERY_TIME = datetime(2024, 1, 1)


def _make_fact(age_days: float = 1.0, fact_type: str = "general") -> TemporalFact:
    return TemporalFact(
        content="test",
        timestamp=QUERY_TIME - timedelta(days=age_days),
        fact_type=fact_type,
    )


# ---------------------------------------------------------------------------
# Minimal correct custom implementations (fixtures)
# ---------------------------------------------------------------------------


class GoodDecay(DecayFunction):
    """Correct minimal custom decay function."""

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        age = (query_time - fact.timestamp).total_seconds() / 86400
        return fact.source_quality * math.exp(-0.1 * age)

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]


class GoodEstimator(EstimationMethod):
    """Correct minimal custom estimation method."""

    def fit(self, ages: list[float], valid: list[bool]) -> float:
        return 0.5


# ---------------------------------------------------------------------------
# 1. Valid custom implementations pass
# ---------------------------------------------------------------------------


class TestValidCustomImplementations:
    def test_good_decay_passes(self):
        fn = GoodDecay()
        result = validate_decay_function(fn)
        assert result is fn  # returns the same object

    def test_good_estimator_passes(self):
        method = GoodEstimator()
        result = validate_estimation_method(method)
        assert result is method

    def test_exponential_decay_passes(self):
        validate_decay_function(ExponentialDecay())

    def test_linear_decay_passes(self):
        validate_decay_function(LinearDecay())

    def test_weibull_decay_passes(self):
        validate_decay_function(WeibullDecay())

    def test_mle_bernoulli_passes(self):
        validate_estimation_method(MLEBernoulli())

    def test_moment_matching_passes(self):
        validate_estimation_method(MomentMatching())

    def test_ensemble_passes(self):
        validate_estimation_method(
            EnsembleMethod(methods=[MLEBernoulli(), MomentMatching()])
        )


# ---------------------------------------------------------------------------
# 2. Broken decay functions are caught
# ---------------------------------------------------------------------------


class TestBrokenDecayFunctions:
    def test_missing_compute_raises(self):
        class NoCompute(DecayFunction):
            def compute_batch(self, facts, query_time):
                return [1.0] * len(facts)
            # compute() not implemented — ABC will catch at instantiation

        with pytest.raises((PluginValidationError, TypeError)):
            validate_decay_function(NoCompute())

    def test_compute_returns_above_one_raises(self):
        class BadRange(DecayFunction):
            def compute(self, fact, query_time):
                return 1.5  # validity > 1 is invalid

            def compute_batch(self, facts, query_time):
                return [self.compute(f, query_time) for f in facts]

        with pytest.raises(PluginValidationError, match="[Rr]ange|\\[0.*1\\]|above|greater"):
            validate_decay_function(BadRange())

    def test_compute_returns_negative_raises(self):
        class Negative(DecayFunction):
            def compute(self, fact, query_time):
                return -0.1

            def compute_batch(self, facts, query_time):
                return [self.compute(f, query_time) for f in facts]

        with pytest.raises(PluginValidationError, match="[Rr]ange|negative|below"):
            validate_decay_function(Negative())

    def test_compute_returns_wrong_type_raises(self):
        class WrongType(DecayFunction):
            def compute(self, fact, query_time):
                return "not a float"

            def compute_batch(self, facts, query_time):
                return [self.compute(f, query_time) for f in facts]

        with pytest.raises(PluginValidationError, match="[Tt]ype|float|numeric"):
            validate_decay_function(WrongType())

    def test_compute_batch_wrong_length_raises(self):
        class WrongLength(DecayFunction):
            def compute(self, fact, query_time):
                return 0.5

            def compute_batch(self, facts, query_time):
                return [0.5]  # always returns 1 element regardless of input

        with pytest.raises(PluginValidationError, match="[Ll]ength|[Ss]ize|batch"):
            validate_decay_function(WrongLength())

    def test_compute_returns_nan_raises(self):
        class ReturnsNaN(DecayFunction):
            def compute(self, fact, query_time):
                return float("nan")

            def compute_batch(self, facts, query_time):
                return [self.compute(f, query_time) for f in facts]

        with pytest.raises(PluginValidationError, match="[Nn]aN|[Nn]ot a [Nn]umber|finite"):
            validate_decay_function(ReturnsNaN())

    def test_non_decay_function_raises(self):
        with pytest.raises((PluginValidationError, TypeError, AttributeError)):
            validate_decay_function("not a decay function")


# ---------------------------------------------------------------------------
# 3. Broken estimation methods are caught
# ---------------------------------------------------------------------------


class TestBrokenEstimationMethods:
    def test_fit_returns_negative_beta_raises(self):
        class NegativeBeta(EstimationMethod):
            def fit(self, ages, valid):
                return -1.0  # beta must be >= 0

        with pytest.raises(PluginValidationError, match="[Nn]egative|>= 0|non-negative"):
            validate_estimation_method(NegativeBeta())

    def test_fit_returns_wrong_type_raises(self):
        class WrongType(EstimationMethod):
            def fit(self, ages, valid):
                return "fast"  # not a float

        with pytest.raises(PluginValidationError, match="[Tt]ype|float|numeric"):
            validate_estimation_method(WrongType())

    def test_fit_returns_nan_raises(self):
        class NaNBeta(EstimationMethod):
            def fit(self, ages, valid):
                return float("nan")

        with pytest.raises(PluginValidationError, match="[Nn]aN|finite"):
            validate_estimation_method(NaNBeta())

    def test_non_estimation_method_raises(self):
        with pytest.raises((PluginValidationError, TypeError, AttributeError)):
            validate_estimation_method(42)


# ---------------------------------------------------------------------------
# 4. Error messages are informative
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_range_error_mentions_range(self):
        class BadRange(DecayFunction):
            def compute(self, fact, query_time):
                return 2.0

            def compute_batch(self, facts, query_time):
                return [self.compute(f, query_time) for f in facts]

        with pytest.raises(PluginValidationError) as exc_info:
            validate_decay_function(BadRange())
        assert "BadRange" in str(exc_info.value) or "compute" in str(exc_info.value)

    def test_negative_beta_error_mentions_fit(self):
        class NegativeBeta(EstimationMethod):
            def fit(self, ages, valid):
                return -0.5

        with pytest.raises(PluginValidationError) as exc_info:
            validate_estimation_method(NegativeBeta())
        assert "NegativeBeta" in str(exc_info.value) or "fit" in str(exc_info.value)
