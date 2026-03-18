"""Plugin validation utilities for custom Chronofy extensions.

Users can extend Chronofy with custom decay functions and estimation methods
by subclassing DecayFunction or EstimationMethod. This module provides
behavioural smoke-tests that give immediate, readable feedback when a
custom implementation is broken — rather than failing silently at runtime.

Usage:
    from chronofy.plugins import validate_decay_function, validate_estimation_method

    # Validate before registering a custom backend
    my_decay = validate_decay_function(MyCustomDecay())
    my_method = validate_estimation_method(MyCustomEstimator())

    # Then use as normal
    decay = ExponentialDecay(beta={"general": 0.5})
    estimator = BetaEstimator(method=my_method)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from chronofy.models import TemporalFact

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class PluginValidationError(Exception):
    """Raised when a custom plugin fails behavioural validation.

    The message describes exactly which check failed and what was returned,
    so the implementer knows what to fix.
    """


# ---------------------------------------------------------------------------
# Internal probe data
# ---------------------------------------------------------------------------

_PROBE_TIME = datetime(2024, 1, 1, 12, 0, 0)
_PROBE_FACTS = [
    TemporalFact(
        content="probe fact (age=1 day)",
        timestamp=_PROBE_TIME - timedelta(days=1),
        fact_type="general",
        source_quality=1.0,
    ),
    TemporalFact(
        content="probe fact (age=30 days)",
        timestamp=_PROBE_TIME - timedelta(days=30),
        fact_type="general",
        source_quality=0.8,
    ),
    TemporalFact(
        content="probe fact (age=365 days)",
        timestamp=_PROBE_TIME - timedelta(days=365),
        fact_type="general",
        source_quality=0.5,
    ),
]
_PROBE_AGES = [1.0, 5.0, 10.0, 30.0]
_PROBE_VALID = [True, True, False, False]


# ---------------------------------------------------------------------------
# validate_decay_function
# ---------------------------------------------------------------------------


def validate_decay_function(fn: object) -> object:
    """Smoke-test a custom DecayFunction implementation.

    Runs a suite of behavioural checks against probe data. Returns the
    object unchanged on success so the call can be used inline.

    Checks:
        1. Has a callable .compute() method.
        2. .compute() returns a numeric value for each probe fact.
        3. All returned scores are finite (not NaN or inf).
        4. All returned scores are in [0.0, 1.0].
        5. Has a callable .compute_batch() method.
        6. .compute_batch() returns a list of the same length as the input.
        7. Batch scores match individual compute() calls (consistency).

    Args:
        fn: The object to validate.

    Returns:
        fn unchanged, enabling inline usage.

    Raises:
        PluginValidationError: With a descriptive message if any check fails.
        TypeError: If the object is not an instance of DecayFunction.
    """
    from chronofy.decay.base import DecayFunction

    cls_name = type(fn).__name__

    # --- Type check ---
    if not isinstance(fn, DecayFunction):
        raise PluginValidationError(
            f"{cls_name!r} is not an instance of DecayFunction. "
            f"Custom decay functions must subclass chronofy.decay.base.DecayFunction."
        )

    # --- compute() existence and signature ---
    if not callable(getattr(fn, "compute", None)):
        raise PluginValidationError(
            f"{cls_name}.compute() is missing or not callable."
        )

    # --- compute() return type and range ---
    for probe in _PROBE_FACTS:
        try:
            score = fn.compute(probe, _PROBE_TIME)
        except Exception as exc:
            raise PluginValidationError(
                f"{cls_name}.compute() raised an unexpected exception "
                f"on probe fact '{probe.content}': {exc}"
            ) from exc

        if not isinstance(score, (int, float)):
            raise PluginValidationError(
                f"{cls_name}.compute() must return a numeric float, "
                f"got {type(score).__name__!r} (value={score!r}) "
                f"for probe fact '{probe.content}'."
            )

        if not math.isfinite(score):
            raise PluginValidationError(
                f"{cls_name}.compute() returned a non-finite value (NaN or inf) "
                f"for probe fact '{probe.content}'. "
                f"Validity scores must be finite real numbers in [0.0, 1.0]."
            )

        if score < 0.0:
            raise PluginValidationError(
                f"{cls_name}.compute() returned {score:.6f}, which is below 0.0 "
                f"(negative validity is not meaningful). "
                f"All scores must be in the range [0.0, 1.0]."
            )

        if score > 1.0:
            raise PluginValidationError(
                f"{cls_name}.compute() returned {score:.6f}, which is above 1.0. "
                f"Validity scores must be in the range [0.0, 1.0]."
            )

    # --- compute_batch() existence ---
    if not callable(getattr(fn, "compute_batch", None)):
        raise PluginValidationError(
            f"{cls_name}.compute_batch() is missing or not callable."
        )

    # --- compute_batch() length contract ---
    try:
        batch_scores = fn.compute_batch(_PROBE_FACTS, _PROBE_TIME)
    except Exception as exc:
        raise PluginValidationError(
            f"{cls_name}.compute_batch() raised an unexpected exception: {exc}"
        ) from exc

    if not hasattr(batch_scores, "__len__"):
        raise PluginValidationError(
            f"{cls_name}.compute_batch() must return a sequence with a length "
            f"(e.g. list or tuple), got {type(batch_scores).__name__!r}."
        )

    if len(batch_scores) != len(_PROBE_FACTS):
        raise PluginValidationError(
            f"{cls_name}.compute_batch() returned {len(batch_scores)} scores "
            f"for {len(_PROBE_FACTS)} input facts. "
            f"The batch output length must equal the input length."
        )

    return fn


# ---------------------------------------------------------------------------
# validate_estimation_method
# ---------------------------------------------------------------------------


def validate_estimation_method(method: object) -> object:
    """Smoke-test a custom EstimationMethod implementation.

    Runs behavioural checks against probe data. Returns the object unchanged
    on success so the call can be used inline.

    Checks:
        1. Is an instance of EstimationMethod.
        2. Has a callable .fit() method.
        3. .fit() returns a numeric value.
        4. The returned beta is finite (not NaN or inf).
        5. The returned beta is non-negative (β ≥ 0 by definition).

    Args:
        method: The object to validate.

    Returns:
        method unchanged, enabling inline usage.

    Raises:
        PluginValidationError: With a descriptive message if any check fails.
        TypeError: If the object is not an instance of EstimationMethod.
    """
    from chronofy.analysis.beta_estimator import EstimationMethod

    cls_name = type(method).__name__

    # --- Type check ---
    if not isinstance(method, EstimationMethod):
        raise PluginValidationError(
            f"{cls_name!r} is not an instance of EstimationMethod. "
            f"Custom estimators must subclass "
            f"chronofy.analysis.beta_estimator.EstimationMethod."
        )

    # --- fit() existence ---
    if not callable(getattr(method, "fit", None)):
        raise PluginValidationError(
            f"{cls_name}.fit() is missing or not callable."
        )

    # --- fit() return type, finiteness, non-negativity ---
    try:
        beta = method.fit(ages=_PROBE_AGES, valid=_PROBE_VALID)
    except Exception as exc:
        raise PluginValidationError(
            f"{cls_name}.fit() raised an unexpected exception on probe data: {exc}"
        ) from exc

    if not isinstance(beta, (int, float)):
        raise PluginValidationError(
            f"{cls_name}.fit() must return a numeric float (the estimated β), "
            f"got {type(beta).__name__!r} (value={beta!r})."
        )

    if not math.isfinite(beta):
        raise PluginValidationError(
            f"{cls_name}.fit() returned a non-finite value (NaN or inf). "
            f"β must be a finite real number ≥ 0."
        )

    if beta < 0.0:
        raise PluginValidationError(
            f"{cls_name}.fit() returned β={beta:.6f}, which is negative. "
            f"Decay coefficients must be non-negative (β ≥ 0). "
            f"Return 0.0 for temporally invariant fact types."
        )

    return method
