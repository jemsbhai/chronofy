"""Temporal Embedder — orchestrator for Layer 1.

Combines temporal encoder output with caller-supplied semantic vectors
to produce the full embedding e = [e_temp ; e_sem]. The embedder does
NOT own the sentence-transformer or any semantic model; it takes
pre-computed semantic vectors and prepends the temporal subspace.

This follows the same delegation pattern as TemporalScorer (which takes
caller-supplied similarities) and keeps dependencies clean: the core
embedding module requires only numpy.

The optional temporal_weight parameter in cosine_similarity() controls
how much influence the temporal subspace has relative to the semantic
subspace during retrieval. At weight=1.0 (default), both subspaces
contribute equally per dimension. Higher weights amplify temporal
proximity; weight=0.0 ignores the temporal subspace entirely.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chronofy.embedding.base import TemporalEncoder
from chronofy.models import TemporalFact


class TemporalEmbedder:
    """Orchestrates temporal + semantic embedding concatenation.

    Given a TemporalEncoder and pre-computed semantic vectors, produces
    the combined embedding e = [e_temp ; e_sem] where e_temp ∈ R^t and
    e_sem ∈ R^{d-t}.

    Args:
        encoder: A TemporalEncoder instance (e.g. SinusoidalEncoder).
        semantic_dims: Expected positive semantic vector dimensionality.
            When provided, embed() and cosine_similarity() enforce it.

    Example:
        >>> from chronofy.embedding import SinusoidalEncoder, TemporalEmbedder
        >>> enc = SinusoidalEncoder(temporal_dims=16)
        >>> embedder = TemporalEmbedder(encoder=enc, semantic_dims=384)
        >>> combined = embedder.embed(facts, semantic_vectors)
        >>> combined.shape  # (n, 16 + 384) = (n, 400)
    """

    def __init__(
        self,
        encoder: TemporalEncoder,
        semantic_dims: int | None = None,
    ) -> None:
        if not isinstance(encoder, TemporalEncoder):
            raise TypeError(
                "encoder must be a TemporalEncoder instance, "
                f"got {type(encoder).__name__}."
            )
        temporal_dims = encoder.temporal_dims
        if (
            isinstance(temporal_dims, bool)
            or not isinstance(temporal_dims, int)
            or temporal_dims <= 0
        ):
            raise ValueError(
                "encoder.temporal_dims must be a positive integer, "
                f"got {temporal_dims!r}."
            )
        if semantic_dims is not None and (
            isinstance(semantic_dims, bool)
            or not isinstance(semantic_dims, int)
            or semantic_dims <= 0
        ):
            raise ValueError(
                "semantic_dims must be a positive integer or None, "
                f"got {semantic_dims!r}."
            )

        self._encoder = encoder
        self._temporal_dims = temporal_dims
        self._semantic_dims = semantic_dims

    @property
    def encoder(self) -> TemporalEncoder:
        """The active temporal encoder."""
        return self._encoder

    @property
    def temporal_dims(self) -> int:
        """Dimensionality of the temporal subspace."""
        return self._temporal_dims

    @property
    def semantic_dims(self) -> int | None:
        """Expected semantic dimensionality (None if not specified)."""
        return self._semantic_dims

    @property
    def total_dims(self) -> int | None:
        """Total embedding dimensionality (temporal + semantic).

        Returns None if semantic_dims was not specified at construction.
        """
        if self._semantic_dims is None:
            return None
        return self._temporal_dims + self._semantic_dims

    def embed(
        self,
        facts: list[TemporalFact],
        semantic_vectors: NDArray[Any],
        reference_time: datetime | None = None,
    ) -> NDArray[Any]:
        """Produce combined embeddings: [e_temp ; e_sem].

        Args:
            facts: List of TemporalFact instances.
            semantic_vectors: Pre-computed semantic vectors with exact shape
                ``(n, d_sem)``, where ``d_sem`` is positive.
            reference_time: Override reference time for the temporal encoder.

        Returns:
            np.ndarray of shape (n, temporal_dims + d_sem).

        Raises:
            ValueError: If dimensions, shapes, or batch lengths violate the
                embedding contract.
        """
        if not isinstance(semantic_vectors, np.ndarray):
            raise TypeError(
                "semantic_vectors must be a numpy.ndarray with shape (n, d_sem)."
            )
        if semantic_vectors.ndim != 2:
            raise ValueError(
                "semantic_vectors must have shape (n, d_sem); "
                f"got array with shape {semantic_vectors.shape}."
            )

        n_facts = len(facts)
        n_sem, d_sem = semantic_vectors.shape

        if d_sem <= 0:
            raise ValueError(
                "semantic_vectors must have a positive semantic dimension, "
                f"got shape {semantic_vectors.shape}."
            )
        if self._semantic_dims is not None and d_sem != self._semantic_dims:
            raise ValueError(
                f"semantic_vectors dimension must equal semantic_dims="
                f"{self._semantic_dims}, got {d_sem}."
            )

        if n_facts != n_sem:
            raise ValueError(
                f"facts and semantic_vectors must have equal length, "
                f"got {n_facts} and {n_sem}."
            )

        if n_facts == 0:
            return np.empty((0, self._temporal_dims + d_sem), dtype=np.float64)

        # Encode temporal subspace
        temporal = self._encoder.encode_facts(facts, reference_time=reference_time)
        expected_temporal_shape = (n_facts, self._temporal_dims)
        if not isinstance(temporal, np.ndarray) or temporal.shape != expected_temporal_shape:
            actual_shape = getattr(temporal, "shape", None)
            raise ValueError(
                "encoder output must have shape "
                f"{expected_temporal_shape}, got {actual_shape}."
            )

        # Concatenate: [e_temp ; e_sem]
        return np.concatenate([temporal, semantic_vectors], axis=1)

    def cosine_similarity(
        self,
        query_embedding: NDArray[Any],
        fact_embeddings: NDArray[Any],
        temporal_weight: float = 1.0,
    ) -> NDArray[Any]:
        """Compute cosine similarity in the combined space.

        Args:
            query_embedding: Exact shape ``(total_dims,)``.
            fact_embeddings: Exact shape ``(n, total_dims)``.
            temporal_weight: Scaling factor for the temporal subspace
                dimensions before computing cosine similarity.
                - 1.0 (default): equal weight per dimension.
                - 0.0: ignore temporal subspace (pure semantic similarity).
                - >1.0: amplify temporal proximity influence.

        Returns:
            np.ndarray of shape (n,) with cosine similarities in [-1, 1].

        Raises:
            ValueError: If either input has the wrong rank or dimensionality.
        """
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a numpy.ndarray.")
        if not isinstance(fact_embeddings, np.ndarray):
            raise TypeError("fact_embeddings must be a numpy.ndarray.")
        if query_embedding.ndim != 1:
            raise ValueError(
                "query_embedding must have shape (total_dims,), "
                f"got {query_embedding.shape}."
            )
        if fact_embeddings.ndim != 2:
            raise ValueError(
                "fact_embeddings must have shape (n, total_dims), "
                f"got {fact_embeddings.shape}."
            )

        total_dims = query_embedding.shape[0]
        if total_dims <= self._temporal_dims:
            raise ValueError(
                "combined embeddings must include a positive semantic dimension; "
                f"got total_dims={total_dims} and temporal_dims={self._temporal_dims}."
            )
        if fact_embeddings.shape[1] != total_dims:
            raise ValueError(
                "query_embedding and fact_embeddings dimensions must match, "
                f"got {total_dims} and {fact_embeddings.shape[1]}."
            )
        if self.total_dims is not None and total_dims != self.total_dims:
            raise ValueError(
                f"embedding dimension must equal configured total_dims={self.total_dims}, "
                f"got {total_dims}."
            )

        t = self._temporal_dims

        # Copy to avoid mutating caller's arrays
        q = query_embedding.astype(np.float64, copy=True)
        f = fact_embeddings.astype(np.float64, copy=True)

        # Apply temporal weighting
        if temporal_weight != 1.0:
            q[:t] *= temporal_weight
            f[:, :t] *= temporal_weight

        # Cosine similarity: dot(q, f_i) / (||q|| * ||f_i||)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return np.zeros(f.shape[0], dtype=np.float64)

        f_norms = np.linalg.norm(f, axis=1)
        # Avoid division by zero
        f_norms = np.where(f_norms == 0.0, 1.0, f_norms)

        dots = f @ q
        similarities: NDArray[Any] = dots / (q_norm * f_norms)
        return similarities

    def __repr__(self) -> str:
        sem_str = (
            f", semantic_dims={self._semantic_dims}"
            if self._semantic_dims is not None
            else ""
        )
        return f"TemporalEmbedder(encoder={self._encoder!r}{sem_str})"
