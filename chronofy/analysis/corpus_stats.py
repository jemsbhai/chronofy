"""Temporal statistical analysis for fact corpora.

Answers: "What is the temporal health of this fact corpus at query time?"

Key concepts:
  - Age stats: distribution of wall-clock ages across facts (in days)
  - Validity stats: distribution of decay-weighted validity scores
  - Effective density: mean validity — a single-number corpus health indicator
  - Staleness rate: fraction of facts below a validity threshold
  - Temporal span: range from oldest to newest fact (in days)
  - Coverage gaps: silent periods with no facts above a minimum gap duration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


# ---------------------------------------------------------------------------
# Data classes for structured stat output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgeStats:
    """Descriptive statistics for the age distribution of a fact corpus (in days)."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p25: float
    p75: float

    def to_dict(self) -> dict[str, float]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "p25": self.p25,
            "p75": self.p75,
        }


@dataclass(frozen=True)
class ValidityStats:
    """Descriptive statistics for the validity score distribution of a fact corpus."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p25: float
    p75: float

    def to_dict(self) -> dict[str, float]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "p25": self.p25,
            "p75": self.p75,
        }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class CorpusStats:
    """Temporal health statistics for a collection of TemporalFacts.

    Computes and exposes descriptive statistics on both the age distribution
    and the decay-weighted validity distribution of a fact corpus. Results
    are computed lazily and cached on first access.

    Args:
        facts:      The collection of evidence items to analyse.
        query_time: The reference time (typically the query timestamp T_q).
        decay_fn:   The decay backend used to compute validity scores.

    Example:
        >>> decay = ExponentialDecay(beta={"lab_result": 2.0})
        >>> stats = CorpusStats(facts=my_facts, query_time=datetime.now(), decay_fn=decay)
        >>> print(f"Effective density: {stats.effective_density:.3f}")
        >>> print(f"Stale fraction:    {stats.staleness_rate(threshold=0.1):.1%}")
        >>> gaps = stats.coverage_gaps(min_gap_days=7)
        >>> print(f"Silent periods:    {len(gaps)}")
    """

    def __init__(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
        decay_fn: DecayFunction,
    ) -> None:
        self._facts = list(facts)
        self._query_time = query_time
        self._decay_fn = decay_fn

        # Lazy-computed caches
        self._ages: list[float] | None = None
        self._validities: list[float] | None = None
        self._age_stats: AgeStats | None | bool = False   # False = not yet computed
        self._validity_stats: ValidityStats | None | bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fact_count(self) -> int:
        """Total number of facts in the corpus."""
        return len(self._facts)

    @property
    def temporal_span_days(self) -> float:
        """Range from the oldest to the newest fact timestamp, in days.

        Returns 0.0 for an empty corpus or a corpus with a single unique timestamp.
        """
        ages = self._get_ages()
        if len(ages) < 2:
            return 0.0
        return max(ages) - min(ages)

    @property
    def effective_density(self) -> float:
        """Mean validity score across the corpus.

        A single-number health indicator: 1.0 means all facts are perfectly
        fresh; 0.0 means the corpus is completely stale. Returns 0.0 for
        an empty corpus.
        """
        vals = self._get_validities()
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @property
    def age_stats(self) -> AgeStats | None:
        """Descriptive statistics for the age distribution (days).

        Returns None for an empty corpus.
        """
        if self._age_stats is False:
            self._age_stats = self._compute_age_stats()
        return self._age_stats  # type: ignore[return-value]

    @property
    def validity_stats(self) -> ValidityStats | None:
        """Descriptive statistics for the validity score distribution.

        Returns None for an empty corpus.
        """
        if self._validity_stats is False:
            self._validity_stats = self._compute_validity_stats()
        return self._validity_stats  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def staleness_rate(self, threshold: float) -> float:
        """Fraction of facts with validity score strictly below *threshold*.

        A fact is considered stale when its validity score is less than
        the threshold (strict inequality — a fact exactly at the threshold
        is considered fresh).

        Args:
            threshold: Minimum acceptable validity in [0.0, 1.0].

        Returns:
            A float in [0.0, 1.0]. Returns 0.0 for an empty corpus.

        Raises:
            ValueError: If threshold is negative.
        """
        if threshold < 0.0:
            raise ValueError(
                f"Threshold must be >= 0.0, got {threshold}"
            )
        vals = self._get_validities()
        if not vals:
            return 0.0
        stale_count = sum(1 for v in vals if v < threshold)
        return stale_count / len(vals)

    def coverage_gaps(self, min_gap_days: float) -> list[tuple[datetime, datetime]]:
        """Find temporal silent periods between consecutive fact timestamps.

        A gap is a consecutive pair of fact timestamps separated by at least
        *min_gap_days* days. Gaps are returned sorted chronologically
        (oldest gap first).

        Args:
            min_gap_days: Minimum gap length to report, in days. Must be > 0.

        Returns:
            List of (start, end) datetime pairs for each qualifying gap,
            in chronological order. Returns [] for a corpus with 0 or 1 facts.

        Raises:
            ValueError: If min_gap_days <= 0.
        """
        if min_gap_days <= 0:
            raise ValueError(f"min_gap_days must be > 0, got {min_gap_days}")

        if len(self._facts) < 2:
            return []

        # Collect and sort unique timestamps chronologically (oldest first)
        timestamps = sorted(set(f.timestamp for f in self._facts))

        if len(timestamps) < 2:
            return []

        min_gap_seconds = min_gap_days * 86400.0
        gaps: list[tuple[datetime, datetime]] = []

        for i in range(len(timestamps) - 1):
            t_start = timestamps[i]
            t_end = timestamps[i + 1]
            gap_seconds = (t_end - t_start).total_seconds()
            if gap_seconds >= min_gap_seconds:
                gaps.append((t_start, t_end))

        return gaps

    def summary(self) -> dict[str, Any]:
        """Return all statistics as a flat dictionary.

        Suitable for JSON serialisation, logging, and reporting.

        Returns:
            Dictionary with keys: fact_count, temporal_span_days,
            effective_density, age_stats (dict or None),
            validity_stats (dict or None).
        """
        return {
            "fact_count": self.fact_count,
            "temporal_span_days": self.temporal_span_days,
            "effective_density": self.effective_density,
            "age_stats": self.age_stats.to_dict() if self.age_stats is not None else None,
            "validity_stats": (
                self.validity_stats.to_dict() if self.validity_stats is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ages(self) -> list[float]:
        """Compute and cache age-in-days for every fact."""
        if self._ages is None:
            self._ages = [f.age_at(self._query_time) for f in self._facts]
        return self._ages

    def _get_validities(self) -> list[float]:
        """Compute and cache validity scores for every fact."""
        if self._validities is None:
            self._validities = self._decay_fn.compute_batch(self._facts, self._query_time)
        return self._validities

    def _compute_age_stats(self) -> AgeStats | None:
        ages = self._get_ages()
        if not ages:
            return None
        return AgeStats(
            min=_min(ages),
            max=_max(ages),
            mean=_mean(ages),
            median=_percentile(ages, 50),
            std=_std(ages),
            p25=_percentile(ages, 25),
            p75=_percentile(ages, 75),
        )

    def _compute_validity_stats(self) -> ValidityStats | None:
        vals = self._get_validities()
        if not vals:
            return None
        return ValidityStats(
            min=_min(vals),
            max=_max(vals),
            mean=_mean(vals),
            median=_percentile(vals, 50),
            std=_std(vals),
            p25=_percentile(vals, 25),
            p75=_percentile(vals, 75),
        )


# ---------------------------------------------------------------------------
# Pure-Python descriptive stat helpers (no NumPy dependency)
# ---------------------------------------------------------------------------


def _min(xs: list[float]) -> float:
    return min(xs)


def _max(xs: list[float]) -> float:
    return max(xs)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    """Population standard deviation (ddof=0)."""
    if len(xs) == 1:
        return 0.0
    mu = _mean(xs)
    variance = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(variance)


def _percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    sorted_xs = sorted(xs)
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    # Map percentile p ∈ [0, 100] to a fractional index
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    frac = idx - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac
