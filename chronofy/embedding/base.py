"""Base protocol for temporal encoders.

A temporal encoder maps timestamps to fixed-dimension vectors in R^t,
forming the temporal subspace of the combined embedding. Different
encoders can be swapped in without changing the downstream embedder
or retrieval layers.

Design principles:
  - TemporalEncoder is a single-responsibility ABC: timestamps in, vectors out.
  - The temporal_dims property is abstract so downstream code can query
    the output dimensionality without knowing the concrete encoder.
  - encode() takes a list of datetimes and returns an (n, temporal_dims) array.
  - encode_facts() is a convenience wrapper extracting timestamps from TemporalFacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from chronofy.models import TemporalFact


class TemporalEncoder(ABC):
    """Abstract base for temporal encoding strategies.

    A temporal encoder maps timestamps to dense vectors in R^t. The
    output vectors form the temporal subspace when concatenated with
    semantic embeddings by TemporalEmbedder.

    Subclasses must implement:
        - temporal_dims (property): the output dimensionality t
        - encode(timestamps, reference_time): the encoding logic

    Example subclass::

        class MyEncoder(TemporalEncoder):
            @property
            def temporal_dims(self) -> int:
                return 16

            def encode(self, timestamps, reference_time=None):
                # ... your encoding logic ...
                return np.zeros((len(timestamps), 16))
    """

    @property
    @abstractmethod
    def temporal_dims(self) -> int:
        """The dimensionality of the temporal subspace (t)."""
        ...

    @abstractmethod
    def encode(
        self,
        timestamps: list[datetime],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode a list of timestamps into temporal vectors.

        Args:
            timestamps: List of datetime objects to encode.
            reference_time: Optional reference point for relative encoding.
                If None, implementations should use their default.

        Returns:
            np.ndarray of shape (len(timestamps), temporal_dims) with
            dtype float64 (or float32).
        """
        ...

    def encode_facts(
        self,
        facts: list[TemporalFact],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Convenience: encode timestamps extracted from TemporalFacts.

        Args:
            facts: List of TemporalFact instances.
            reference_time: Optional reference point for relative encoding.

        Returns:
            np.ndarray of shape (len(facts), temporal_dims).
        """
        timestamps = [f.timestamp for f in facts]
        return self.encode(timestamps, reference_time=reference_time)
