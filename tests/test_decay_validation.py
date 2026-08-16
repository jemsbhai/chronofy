"""Regression tests for built-in decay configuration and power endpoints."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest

from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.half_life import HalfLifeDecay
from chronofy.decay.linear import LinearDecay
from chronofy.decay.power_law import PowerLawDecay
from chronofy.decay.weibull import WeibullDecay
from chronofy.models import TemporalFact
from chronofy.scoring.temporal_scorer import PowerScoring

QUERY_TIME = datetime(2026, 3, 15)


def _fact(days_ago: float, quality: float = 0.4) -> TemporalFact:
    return TemporalFact(
        content="configuration probe",
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type="probe",
        source_quality=quality,
    )


NONNEGATIVE_CONFIGS: list[tuple[Callable[[Any], object], str]] = [
    (lambda value: ExponentialDecay(beta={"probe": value}), "beta for fact type 'probe'"),
    (lambda value: ExponentialDecay(default_beta=value), "default_beta"),
    (lambda value: LinearDecay(rate={"probe": value}), "rate for fact type 'probe'"),
    (lambda value: LinearDecay(default_rate=value), "default_rate"),
    (
        lambda value: PowerLawDecay(exponent={"probe": value}),
        "exponent for fact type 'probe'",
    ),
    (lambda value: PowerLawDecay(default_exponent=value), "default_exponent"),
]

POSITIVE_CONFIGS: list[tuple[Callable[[Any], object], str]] = [
    (
        lambda value: HalfLifeDecay(half_life={"probe": value}),
        "half_life for fact type 'probe'",
    ),
    (lambda value: HalfLifeDecay(default_half_life=value), "default_half_life"),
    (lambda value: WeibullDecay(scale={"probe": value}), "scale for fact type 'probe'"),
    (lambda value: WeibullDecay(default_scale=value), "default_scale"),
    (lambda value: WeibullDecay(shape={"probe": value}), "shape for fact type 'probe'"),
    (lambda value: WeibullDecay(default_shape=value), "default_shape"),
]


@pytest.mark.parametrize(("factory", "name"), NONNEGATIVE_CONFIGS)
@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, -math.inf, True, "0.1", None])
def test_nonnegative_configs_reject_invalid_values(
    factory: Callable[[Any], object], name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a finite nonnegative number"):
        factory(value)


@pytest.mark.parametrize(("factory", "name"), POSITIVE_CONFIGS)
@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf, -math.inf, True, "1", None])
def test_positive_configs_reject_invalid_values(
    factory: Callable[[Any], object], name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a finite positive number"):
        factory(value)


@pytest.mark.parametrize(
    "factory",
    [
        ExponentialDecay,
        HalfLifeDecay,
        LinearDecay,
        PowerLawDecay,
        WeibullDecay,
    ],
)
def test_all_decay_classes_report_time_unit_errors_consistently(
    factory: Callable[..., object],
) -> None:
    with pytest.raises(
        ValueError,
        match="time_unit must be one of 'seconds', 'hours', or 'days', got 'weeks'",
    ):
        factory(time_unit="weeks")


@pytest.mark.parametrize(
    "decay",
    [
        ExponentialDecay(beta={"probe": 1e308}),
        HalfLifeDecay(half_life={"probe": 5e-324}),
        LinearDecay(rate={"probe": 1e308}),
        PowerLawDecay(exponent={"probe": 1e308}),
        WeibullDecay(scale={"probe": 1e-300}, shape={"probe": 1e308}),
    ],
)
@pytest.mark.parametrize("days_ago", [-1.0, 0.0, 1.0, 1000.0])
def test_valid_extreme_configs_keep_scores_finite_and_bounded(
    decay: Any, days_ago: float
) -> None:
    score = decay.compute(_fact(days_ago), QUERY_TIME)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "decay",
    [
        ExponentialDecay(beta={"probe": 0.0}),
        LinearDecay(rate={"probe": 0.0}),
        PowerLawDecay(exponent={"probe": 0.0}),
    ],
)
def test_nonnegative_decay_parameters_accept_zero(decay: Any) -> None:
    assert decay.compute(_fact(1000.0), QUERY_TIME) == pytest.approx(0.4)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, True, "0.1"])
def test_mean_reversion_rates_are_validated(value: object) -> None:
    with pytest.raises(ValueError, match="kappa for fact type 'probe'"):
        ExponentialDecay.from_mean_reversion_rate({"probe": value})  # type: ignore[dict-item]


@pytest.mark.parametrize("alpha", [math.nan, math.inf, -math.inf, True, "0.5"])
def test_power_scoring_rejects_non_finite_or_non_numeric_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha must be a finite number in"):
        PowerScoring(alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("alpha", "similarity", "validity", "expected"),
    [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.8, 0.0),
        (1.0, 0.7, 0.0, 0.7),
        (1.0, 0.7, 0.8, 0.7),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.8, 0.8),
        (0.0, 0.7, 0.0, 0.0),
        (0.0, 0.7, 0.8, 0.8),
        (0.5, 0.0, 0.8, 0.0),
        (0.5, 0.7, 0.0, 0.0),
    ],
)
def test_power_scoring_zero_base_contract(
    alpha: float, similarity: float, validity: float, expected: float
) -> None:
    assert PowerScoring(alpha).score(similarity, validity) == pytest.approx(expected)
