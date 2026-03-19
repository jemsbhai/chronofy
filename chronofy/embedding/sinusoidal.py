"""Sinusoidal positional encoding for timestamps.

Maps timestamps to dense vectors using sin/cos pairs at geometrically
spaced frequencies, following the transformer positional encoding
paradigm (Vaswani et al., 2017). This encoder requires no training
and was validated in the Chronofy ablation study (Experiment 6).

The position value is the number of days elapsed since a reference date.
Frequency bands span from 1 day (fast oscillation) to max_period_days
(slow oscillation), covering both intra-week and multi-year patterns.

Output:
    For temporal_dims = 2k, the encoding of a timestamp t is:
        [sin(ω_1·d), cos(ω_1·d), sin(ω_2·d), cos(ω_2·d), ..., sin(ω_k·d), cos(ω_k·d)]
    where d = (t - reference_time) in days and ω_i are geometrically spaced.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from chronofy.embedding.base import TemporalEncoder
from chronofy.models import TemporalFact

# Sensible default: epoch-like reference for consistent encodings
_DEFAULT_REFERENCE = datetime(2020, 1, 1)


class SinusoidalEncoder(TemporalEncoder):
    """Sinusoidal positional encoding for timestamps.

    Maps each timestamp to a vector in R^t using sin/cos pairs at
    geometrically spaced frequencies. No training required.

    Args:
        temporal_dims: Output dimensionality (must be a positive even integer).
        reference_time: Epoch for computing relative day offsets.
            Defaults to 2020-01-01.
        max_period_days: The longest wavelength in days. Controls the
            slowest frequency band. Defaults to 3650 (~10 years).

    Example:
        >>> enc = SinusoidalEncoder(temporal_dims=16)
        >>> vecs = enc.encode([datetime(2025, 6, 15)])
        >>> vecs.shape
        (1, 16)
    """

    def __init__(
        self,
        temporal_dims: int = 16,
        reference_time: datetime | None = None,
        max_period_days: float = 3650.0,
    ) -> None:
        if temporal_dims <= 0:
            raise ValueError(
                f"temporal_dims must be a positive even integer, got {temporal_dims}"
            )
        if temporal_dims % 2 != 0:
            raise ValueError(
                f"temporal_dims must be even (sin/cos pairs), got {temporal_dims}"
            )
        self._temporal_dims = temporal_dims
        self._reference_time = reference_time or _DEFAULT_REFERENCE
        self._max_period_days = max_period_days

        # Precompute frequency bands: geometric spacing from 1 day to max_period
        n_pairs = temporal_dims // 2
        # Frequencies: 2π / period, where period ranges from 1 to max_period_days
        # Using the transformer convention: freq_i = 1 / (max_period ^ (2i / d))
        exponents = np.arange(n_pairs, dtype=np.float64) / n_pairs
        periods = max_period_days ** exponents  # 1 day to max_period_days
        self._angular_frequencies = 2.0 * np.pi / periods  # shape (n_pairs,)

    @property
    def temporal_dims(self) -> int:
        """The dimensionality of the temporal subspace."""
        return self._temporal_dims

    @property
    def reference_time(self) -> datetime:
        """The reference datetime used for computing day offsets."""
        return self._reference_time

    @property
    def max_period_days(self) -> float:
        """The longest wavelength in days."""
        return self._max_period_days

    def encode(
        self,
        timestamps: list[datetime],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode timestamps into sinusoidal temporal vectors.

        Args:
            timestamps: List of datetime objects.
            reference_time: Override reference time for this call.
                Uses the instance default if None.

        Returns:
            np.ndarray of shape (len(timestamps), temporal_dims).
            Values are in [-1, 1] (sin/cos range).
        """
        if len(timestamps) == 0:
            return np.empty((0, self._temporal_dims), dtype=np.float64)

        ref = reference_time or self._reference_time

        # Compute day offsets: shape (n,)
        days = np.array(
            [(t - ref).total_seconds() / 86400.0 for t in timestamps],
            dtype=np.float64,
        )

        # Outer product: (n, 1) × (1, n_pairs) → (n, n_pairs)
        angles = days[:, np.newaxis] * self._angular_frequencies[np.newaxis, :]

        # Interleave sin and cos: [sin(ω1·d), cos(ω1·d), sin(ω2·d), cos(ω2·d), ...]
        encoding = np.empty((len(timestamps), self._temporal_dims), dtype=np.float64)
        encoding[:, 0::2] = np.sin(angles)
        encoding[:, 1::2] = np.cos(angles)

        return encoding

    def encode_facts(
        self,
        facts: list[TemporalFact],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Convenience: encode timestamps extracted from TemporalFacts."""
        timestamps = [f.timestamp for f in facts]
        return self.encode(timestamps, reference_time=reference_time)

    def __repr__(self) -> str:
        return (
            f"SinusoidalEncoder(temporal_dims={self._temporal_dims}, "
            f"max_period_days={self._max_period_days})"
        )
