"""Beta decay rate estimator for temporal fact corpora.

Estimates the optimal decay coefficient β from empirical observations of
fact validity over time, using a pluggable estimation backend.

Scientific grounding:
  Under the model P(valid | age, β) = exp(-β · age), β governs how quickly
  a fact type loses epistemic value. By Proposition 1 (Chronofy paper),
  β = 2κ where κ is the mean-reversion rate of the underlying latent process.
  This module learns β empirically when κ is not known analytically.

Architecture:
  EstimationMethod (ABC)          ← swap in any technique
    ├── MLEBernoulli              ← maximum likelihood under Bernoulli model
    ├── MomentMatching            ← method of moments (fast, closed-form)
    └── EnsembleMethod            ← weighted combination of any methods

  BetaEstimator(method=...)
    ├── .fit(ages, valid)                         → BetaEstimateResult
    ├── .fit_corpus(facts, labels, query_time)    → dict[str, BetaEstimateResult]
    └── .to_decay(results)                        → ExponentialDecay
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Upper bound for β search — corresponds to a half-life of ln(2)/50 ≈ 0.014 days
_MAX_BETA: float = 50.0
# Lower bound — treat as zero (temporal invariance)
_MIN_BETA: float = 0.0
# Tolerance for golden-section search
_TOLERANCE: float = 1e-8


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BetaEstimateResult:
    """Result of fitting a decay coefficient β for a single fact type.

    Attributes:
        beta:            MLE estimate of the decay coefficient β ≥ 0.
        half_life:       ln(2)/β in days — the age at which validity halves.
                         float('inf') when β = 0 (invariant fact type).
        log_likelihood:  Log-likelihood at the fitted β (MLE only; NaN for
                         methods that do not compute a likelihood).
        n_obs:           Total number of observations used.
        n_valid:         Number of observations where the fact was still valid.
        converged:       Whether the optimisation converged normally.
        fact_type:       The fact type this result corresponds to.
    """

    beta: float
    half_life: float
    log_likelihood: float
    n_obs: int
    n_valid: int
    converged: bool
    fact_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "half_life": self.half_life,
            "log_likelihood": self.log_likelihood,
            "n_obs": self.n_obs,
            "n_valid": self.n_valid,
            "converged": self.converged,
            "fact_type": self.fact_type,
        }


# ---------------------------------------------------------------------------
# EstimationMethod ABC
# ---------------------------------------------------------------------------


class EstimationMethod(ABC):
    """Abstract base for beta estimation techniques.

    A concrete EstimationMethod takes a list of (age, valid) observations
    and returns a single scalar β ≥ 0. Any class implementing this interface
    can be passed to BetaEstimator as the ``method`` argument.

    The interface is intentionally minimal — the orchestration logic
    (grouping by fact type, building results, producing ExponentialDecay)
    all lives in BetaEstimator. Methods only need to solve the scalar
    optimisation problem.
    """

    @abstractmethod
    def fit(self, ages: list[float], valid: list[bool]) -> float:
        """Estimate β from observations.

        Args:
            ages:  List of fact ages in days at the time of observation.
            valid: Parallel list of booleans — True if the fact was still
                   correct/valid at that age.

        Returns:
            Estimated β ≥ 0.
        """
        ...


# ---------------------------------------------------------------------------
# MLEBernoulli
# ---------------------------------------------------------------------------


class MLEBernoulli(EstimationMethod):
    """Maximum likelihood estimation under the Bernoulli decay model.

    Models each observation as an independent Bernoulli trial:

        P(valid_i = 1 | age_i, β) = exp(-β · age_i)

    The log-likelihood is:

        L(β) = -β·Σ(v_i·a_i) + Σ(1-v_i)·log(1 - exp(-β·a_i))

    Maximised via golden-section search over β ∈ [0, MAX_BETA].

    Special cases handled analytically:
        - All valid:   L is monotonically decreasing in β → β* = 0.
        - All invalid: L is monotonically increasing → β* = MAX_BETA (clipped).
    """

    def fit(self, ages: list[float], valid: list[bool]) -> float:
        n_valid = sum(valid)
        n_invalid = len(valid) - n_valid

        # Analytical edge cases
        if n_invalid == 0:
            # Everything still valid — process is very stable
            return 0.0

        if n_valid == 0:
            # Everything invalid — maximise by pushing β as high as possible
            return _MAX_BETA

        # Golden-section search to maximise L(β)
        beta, _ = _golden_section_max(
            f=lambda b: _log_likelihood_bernoulli(b, ages, valid),
            lo=_MIN_BETA,
            hi=_MAX_BETA,
            tol=_TOLERANCE,
        )
        return max(0.0, beta)


def _log_likelihood_bernoulli(
    beta: float, ages: list[float], valid: list[bool]
) -> float:
    """Bernoulli log-likelihood for a given β."""
    ll = 0.0
    for a, v in zip(ages, valid):
        p = math.exp(-beta * a)
        if v:
            ll += math.log(max(p, 1e-300))
        else:
            ll += math.log(max(1.0 - p, 1e-300))
    return ll


# ---------------------------------------------------------------------------
# MomentMatching
# ---------------------------------------------------------------------------


class MomentMatching(EstimationMethod):
    """Method-of-moments estimator — fast, closed-form, no optimisation.

    Matches the empirical survival fraction to the theoretical:

        p_valid = exp(-β · mean_age)

    Solving for β:

        β = -log(p_valid) / mean_age

    This is a plug-in estimator: fast and interpretable, though slightly
    less statistically efficient than MLE for small samples. It is most
    accurate when the age distribution across observations is approximately
    uniform.
    """

    def fit(self, ages: list[float], valid: list[bool]) -> float:
        n = len(ages)
        n_valid = sum(valid)
        p_valid = n_valid / n

        # All valid → β = 0
        if p_valid >= 1.0:
            return 0.0

        mean_age = sum(ages) / n

        # All invalid or zero mean_age
        if p_valid <= 0.0 or mean_age <= 0.0:
            return _MAX_BETA

        beta = -math.log(p_valid) / mean_age
        return min(max(beta, _MIN_BETA), _MAX_BETA)


# ---------------------------------------------------------------------------
# EnsembleMethod
# ---------------------------------------------------------------------------


class EnsembleMethod(EstimationMethod):
    """Weighted average of multiple EstimationMethod backends.

    Combines estimates from several methods into a single β via a
    weighted arithmetic mean. Weights are normalised internally and
    need not sum to 1.0.

    Args:
        methods: Non-empty list of EstimationMethod instances.
        weights: Optional list of positive weights, one per method.
                 Defaults to equal weighting. Normalised internally.

    Raises:
        ValueError: If methods is empty, weights length mismatches methods,
                    or any weight is non-positive.

    Example:
        >>> ensemble = EnsembleMethod(
        ...     methods=[MLEBernoulli(), MomentMatching()],
        ...     weights=[0.7, 0.3],
        ... )
        >>> beta = ensemble.fit(ages=[1.0, 5.0, 10.0], valid=[True, True, False])
    """

    def __init__(
        self,
        methods: list[EstimationMethod],
        weights: list[float] | None = None,
    ) -> None:
        if not methods:
            raise ValueError("EnsembleMethod requires at least one method.")
        if weights is not None:
            if len(weights) != len(methods):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"methods length ({len(methods)})."
                )
            if any(w <= 0.0 for w in weights):
                raise ValueError("All weights must be strictly positive.")
            total = sum(weights)
            self._weights = [w / total for w in weights]
        else:
            k = len(methods)
            self._weights = [1.0 / k] * k

        self._methods = methods

    def fit(self, ages: list[float], valid: list[bool]) -> float:
        """Return the weighted average of each method's β estimate."""
        betas = [m.fit(ages, valid) for m in self._methods]
        return sum(w * b for w, b in zip(self._weights, betas))


# ---------------------------------------------------------------------------
# BetaEstimator
# ---------------------------------------------------------------------------


class BetaEstimator:
    """Fits decay coefficients β from empirical (age, valid) observations.

    Accepts any EstimationMethod backend, defaulting to MLEBernoulli.
    Provides three levels of API:

        fit()           — scalar fit from raw ages/valid lists
        fit_corpus()    — grouped fit across a labelled TemporalFact corpus
        to_decay()      — converts fit results into a ready-to-use ExponentialDecay

    Args:
        method: The estimation backend. Defaults to MLEBernoulli().

    Example:
        >>> estimator = BetaEstimator()                         # MLE default
        >>> estimator = BetaEstimator(method=MomentMatching())  # moment matching
        >>> estimator = BetaEstimator(                          # ensemble
        ...     method=EnsembleMethod([MLEBernoulli(), MomentMatching()])
        ... )
    """

    def __init__(self, method: EstimationMethod | None = None) -> None:
        self._method = method if method is not None else MLEBernoulli()

    def fit(
        self,
        ages: list[float],
        valid: list[bool],
        fact_type: str = "general",
    ) -> BetaEstimateResult:
        """Estimate β from a list of (age_in_days, still_valid) observations.

        Args:
            ages:      Ages in days at the time each fact was evaluated.
            valid:     Parallel booleans — True if the fact was still valid.
            fact_type: Label stored in the result (does not affect estimation).

        Returns:
            BetaEstimateResult with the fitted β and diagnostic fields.

        Raises:
            ValueError: If ages is empty or lengths of ages and valid differ.
        """
        if not ages:
            raise ValueError("ages must not be empty.")
        if len(ages) != len(valid):
            raise ValueError(
                f"ages and valid must have equal length, "
                f"got {len(ages)} and {len(valid)}."
            )

        beta = self._method.fit(ages, valid)
        beta = max(0.0, float(beta))

        half_life = math.log(2) / beta if beta > 0.0 else float("inf")
        ll = (
            _log_likelihood_bernoulli(beta, ages, valid)
            if isinstance(self._method, (MLEBernoulli, EnsembleMethod))
            else float("nan")
        )

        return BetaEstimateResult(
            beta=beta,
            half_life=half_life,
            log_likelihood=ll,
            n_obs=len(ages),
            n_valid=sum(valid),
            converged=True,
            fact_type=fact_type,
        )

    def fit_corpus(
        self,
        facts: list[TemporalFact],
        labels: list[bool],
        query_time: datetime,
    ) -> dict[str, BetaEstimateResult]:
        """Fit β independently for each fact_type in a labelled corpus.

        Args:
            facts:      The TemporalFact instances to fit over.
            labels:     Parallel booleans — True if the corresponding fact
                        was still valid at query_time.
            query_time: Reference time used to compute fact ages.

        Returns:
            Dictionary mapping fact_type → BetaEstimateResult.

        Raises:
            ValueError: If facts is empty or lengths of facts and labels differ.
        """
        if not facts:
            raise ValueError("facts must not be empty.")
        if len(facts) != len(labels):
            raise ValueError(
                f"facts and labels must have equal length, "
                f"got {len(facts)} and {len(labels)}."
            )

        # Group by fact_type
        groups: dict[str, tuple[list[float], list[bool]]] = {}
        for fact, label in zip(facts, labels):
            age = fact.age_at(query_time)
            if fact.fact_type not in groups:
                groups[fact.fact_type] = ([], [])
            groups[fact.fact_type][0].append(age)
            groups[fact.fact_type][1].append(label)

        return {
            ft: self.fit(ages=ages, valid=valids, fact_type=ft)
            for ft, (ages, valids) in groups.items()
        }

    def to_decay(
        self,
        results: dict[str, BetaEstimateResult],
        default_beta: float = 0.5,
    ) -> ExponentialDecay:
        """Convert fit results into a ready-to-use ExponentialDecay instance.

        Args:
            results:      Output of fit_corpus().
            default_beta: Fallback β for fact types not present in results.

        Returns:
            ExponentialDecay configured with the empirically fitted β values.

        Example:
            >>> results = estimator.fit_corpus(facts, labels, query_time)
            >>> decay = estimator.to_decay(results)
            >>> score = decay.compute(my_fact, datetime.now())
        """
        beta_map = {ft: r.beta for ft, r in results.items()}
        return ExponentialDecay(beta=beta_map, default_beta=default_beta)


# ---------------------------------------------------------------------------
# Internal optimisation utility
# ---------------------------------------------------------------------------


def _golden_section_max(
    f: "callable[[float], float]",
    lo: float,
    hi: float,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Golden-section search for the maximum of a unimodal function on [lo, hi].

    Returns:
        (x_max, f_max) — the maximiser and the maximum value found.
    """
    phi = (math.sqrt(5) - 1) / 2  # golden ratio conjugate ≈ 0.618

    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)

    while abs(b - a) > tol:
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)

    x_max = (a + b) / 2
    return x_max, f(x_max)
