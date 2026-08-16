"""Shared validation helpers for built-in decay configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

_TIME_DIVISORS: dict[str, float] = {
    "seconds": 1.0,
    "hours": 3600.0,
    "days": 86400.0,
}


def validate_time_unit(time_unit: str) -> float:
    """Return the divisor for a supported time unit or raise a clear error."""
    try:
        return _TIME_DIVISORS[time_unit]
    except (KeyError, TypeError):
        raise ValueError(
            "time_unit must be one of 'seconds', 'hours', or 'days', "
            f"got {time_unit!r}."
        ) from None


def validate_parameter(value: object, name: str, *, positive: bool) -> float:
    """Validate and normalize one numeric decay parameter."""
    requirement = "positive" if positive else "nonnegative"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must be a finite {requirement} number, got {value!r}."
        )

    normalized = float(value)
    if not math.isfinite(normalized) or (normalized <= 0.0 if positive else normalized < 0.0):
        raise ValueError(
            f"{name} must be a finite {requirement} number, got {value!r}."
        )
    return normalized


def validate_parameter_map(
    values: Mapping[str, object] | None,
    name: str,
    *,
    positive: bool,
) -> dict[str, float]:
    """Validate a fact-type-to-parameter mapping and return a defensive copy."""
    requirement = "positive" if positive else "nonnegative"
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(
            f"{name} must be a mapping of fact types to finite {requirement} numbers, "
            f"got {values!r}."
        )

    return {
        fact_type: validate_parameter(
            value,
            f"{name} for fact type {fact_type!r}",
            positive=positive,
        )
        for fact_type, value in values.items()
    }
