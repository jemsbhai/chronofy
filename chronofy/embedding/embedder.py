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

import numpy as np

from chronofy.embedding.base import TemporalEncoder


class TemporalEmbedder:
    """Orchestrates temporal + semantic embedding concatenation.

    Given a TemporalEncoder and pre-computed semantic vectors, produces
    the combined embedding e = [e_temp ; e_sem] where e_temp ∈ R^t and
    e_sem ∈ R^{d-t}.

    Args:
        encoder: A TemporalEncoder instance (e.g. SinusoidalEncoder).
        semantic_dims: Expected semantic vector dimensionality. Used for
            total_dims reporting; not enforced at construction (validated
            at embed() time).

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
        self._encoder = encoder
        self._semantic_dims = semantic_dims

    @property
    def encoder(self) -> TemporalEncoder:
        """The active temporal encoder."""
        return self._encoder

    @property
    def temporal_dims(self) -> int:
        """Dimensionality of the temporal subspace."""
        return self._encoder.temporal_dims

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
        return self._encoder.temporal_dims + self._semantic_dims

    def embed(
        self,
        facts: list,
        semantic_vectors: np.ndarray,
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Produce combined embeddings: [e_temp ; e_sem].

        Args:
            facts: List of TemporalFact instances.
            semantic_vectors: Pre-computed semantic vectors, shape (n, d_sem).
            reference_time: Override reference time for the temporal encoder.

        Returns:
            np.ndarray of shape (n, temporal_dims + d_sem).

        Raises:
            ValueError: If facts and semantic_vectors have mismatched lengths.
        """
        n_facts = len(facts)
        n_sem = semantic_vectors.shape[0] if semantic_vectors.ndim > 1 else (
            0 if semantic_vectors.size == 0 else 1
        )

        if n_facts != n_sem:
            raise ValueError(
                f"facts and semantic_vectors must have equal length, "
                f"got {n_facts} and {n_sem}."
            )

        if n_facts == 0:
            d_sem = semantic_vectors.shape[1] if semantic_vectors.ndim == 2 else 0
            return np.empty(
                (0, self._encoder.temporal_dims + d_sem), dtype=np.float64
            )

        # Encode temporal subspace
        temporal = self._encoder.encode_facts(facts, reference_time=reference_time)

        # Ensure semantic vectors are 2D
        if semantic_vectors.ndim == 1:
            semantic_vectors = semantic_vectors.reshape(1, -1)

        # Concatenate: [e_temp ; e_sem]
        return np.concatenate([temporal, semantic_vectors], axis=1)

    def cosine_similarity(
        self,
        query_embedding: np.ndarray,
        fact_embeddings: np.ndarray,
        temporal_weight: float = 1.0,
    ) -> np.ndarray:
        """Compute cosine similarity in the combined space.

        Args:
            query_embedding: Shape (total_dims,) — the query vector.
            fact_embeddings: Shape (n, total_dims) — the fact vectors.
            temporal_weight: Scaling factor for the temporal subspace
                dimensions before computing cosine similarity.
                - 1.0 (default): equal weight per dimension.
                - 0.0: ignore temporal subspace (pure semantic similarity).
                - >1.0: amplify temporal proximity influence.

        Returns:
            np.ndarray of shape (n,) with cosine similarities in [-1, 1].
        """
        t = self._encoder.temporal_dims

        # Copy to avoid mutating caller's arrays
        q = query_embedding.astype(np.float64, copy=True)
        f = fact_embeddings.astype(np.float64, copy=True)

        # Ensure fact_embeddings is 2D
        if f.ndim == 1:
            f = f.reshape(1, -1)

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
        return dots / (q_norm * f_norms)

    def __repr__(self) -> str:
        sem_str = f", semantic_dims={self._semantic_dims}" if self._semantic_dims else ""
        return f"TemporalEmbedder(encoder={self._encoder!r}{sem_str})"
