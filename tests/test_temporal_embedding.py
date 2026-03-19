"""Tests for chronofy.embedding — Layer 1: Temporal Subspace Embeddings.

Validates the temporal encoding and embedding modules that implement
the paper's Layer 1: e = [e_temp ; e_sem] where e_temp encodes fact age
into a dedicated temporal subspace.

Design goals tested:
  - TemporalEncoder ABC enforces the protocol
  - SinusoidalEncoder produces correct-dimension numpy arrays
  - TemporalEmbedder orchestrates temporal + semantic concatenation
  - Modularity: encoders are swappable, dimensions configurable
  - Similarity computation respects both temporal and semantic components
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from chronofy.models import TemporalFact

# ─── Imports under test ───────────────────────────────────────────────
from chronofy.embedding.base import TemporalEncoder
from chronofy.embedding.sinusoidal import SinusoidalEncoder
from chronofy.embedding.embedder import TemporalEmbedder


# ─── Helpers ──────────────────────────────────────────────────────────

QUERY_TIME = datetime(2026, 3, 15)
REF_DATE = datetime(2020, 1, 1)


def _fact(
    days_ago: float,
    fact_type: str = "general",
    quality: float = 1.0,
    content: str = "test",
) -> TemporalFact:
    """Helper: create a fact that is `days_ago` old relative to QUERY_TIME."""
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


def _random_semantic(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit-norm semantic vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# =====================================================================
# TemporalEncoder ABC
# =====================================================================

class TestTemporalEncoderABC:
    """The ABC should enforce the protocol and be non-instantiable."""

    def test_cannot_instantiate_abc(self):
        """TemporalEncoder is abstract — direct instantiation must fail."""
        with pytest.raises(TypeError):
            TemporalEncoder()  # type: ignore[abstract]

    def test_subclass_must_implement_encode(self):
        """A subclass that doesn't implement encode() cannot be instantiated."""

        class BadEncoder(TemporalEncoder):
            @property
            def temporal_dims(self) -> int:
                return 8

        with pytest.raises(TypeError):
            BadEncoder()  # type: ignore[abstract]

    def test_subclass_must_implement_temporal_dims(self):
        """A subclass that doesn't implement temporal_dims cannot be instantiated."""

        class BadEncoder(TemporalEncoder):
            def encode(self, timestamps, reference_time=None):
                return np.zeros((len(timestamps), 8))

        with pytest.raises(TypeError):
            BadEncoder()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self):
        """A complete subclass should instantiate fine."""

        class GoodEncoder(TemporalEncoder):
            @property
            def temporal_dims(self) -> int:
                return 4

            def encode(
                self,
                timestamps: list[datetime],
                reference_time: datetime | None = None,
            ) -> np.ndarray:
                return np.zeros((len(timestamps), 4))

        enc = GoodEncoder()
        assert enc.temporal_dims == 4


# =====================================================================
# SinusoidalEncoder
# =====================================================================

class TestSinusoidalEncoder:
    """SinusoidalEncoder: timestamp → R^t via sinusoidal positional encoding."""

    def test_output_shape_single(self):
        """Single timestamp → shape (1, temporal_dims)."""
        enc = SinusoidalEncoder(temporal_dims=16)
        result = enc.encode([QUERY_TIME])
        assert result.shape == (1, 16)

    def test_output_shape_batch(self):
        """Batch of timestamps → shape (n, temporal_dims)."""
        enc = SinusoidalEncoder(temporal_dims=8)
        timestamps = [QUERY_TIME - timedelta(days=i) for i in range(5)]
        result = enc.encode(timestamps)
        assert result.shape == (5, 8)

    def test_temporal_dims_property(self):
        """temporal_dims property should match construction."""
        for d in [4, 8, 16, 32, 64, 128]:
            enc = SinusoidalEncoder(temporal_dims=d)
            assert enc.temporal_dims == d

    def test_temporal_dims_must_be_even(self):
        """Sinusoidal PE uses sin/cos pairs, so dims must be even."""
        with pytest.raises(ValueError, match="even"):
            SinusoidalEncoder(temporal_dims=7)

    def test_temporal_dims_must_be_positive(self):
        """Zero or negative dims should fail."""
        with pytest.raises(ValueError):
            SinusoidalEncoder(temporal_dims=0)
        with pytest.raises(ValueError):
            SinusoidalEncoder(temporal_dims=-4)

    def test_values_in_minus_one_to_one(self):
        """Sinusoidal encoding values should be in [-1, 1]."""
        enc = SinusoidalEncoder(temporal_dims=32)
        timestamps = [QUERY_TIME - timedelta(days=d) for d in range(0, 365, 10)]
        result = enc.encode(timestamps)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_different_timestamps_produce_different_vectors(self):
        """Two different timestamps must produce different encodings."""
        enc = SinusoidalEncoder(temporal_dims=16)
        t1 = QUERY_TIME
        t2 = QUERY_TIME - timedelta(days=30)
        e1 = enc.encode([t1])
        e2 = enc.encode([t2])
        assert not np.allclose(e1, e2)

    def test_same_timestamp_produces_identical_vectors(self):
        """Same timestamp → identical encoding (deterministic)."""
        enc = SinusoidalEncoder(temporal_dims=16)
        e1 = enc.encode([QUERY_TIME])
        e2 = enc.encode([QUERY_TIME])
        np.testing.assert_array_equal(e1, e2)

    def test_reference_time_shifts_encoding(self):
        """Changing the reference time should change the encoding."""
        enc = SinusoidalEncoder(temporal_dims=16)
        t = QUERY_TIME - timedelta(days=100)
        e1 = enc.encode([t], reference_time=datetime(2020, 1, 1))
        e2 = enc.encode([t], reference_time=datetime(2025, 1, 1))
        assert not np.allclose(e1, e2)

    def test_default_reference_time(self):
        """Constructor reference_time should be used when none passed to encode."""
        ref = datetime(2020, 1, 1)
        enc = SinusoidalEncoder(temporal_dims=16, reference_time=ref)
        e1 = enc.encode([QUERY_TIME])
        e2 = enc.encode([QUERY_TIME], reference_time=ref)
        np.testing.assert_array_equal(e1, e2)

    def test_nearby_timestamps_more_similar(self):
        """Timestamps 1 day apart should be more similar than 365 days apart."""
        enc = SinusoidalEncoder(temporal_dims=32)
        t_now = QUERY_TIME
        t_yesterday = QUERY_TIME - timedelta(days=1)
        t_year_ago = QUERY_TIME - timedelta(days=365)

        e_now = enc.encode([t_now])[0]
        e_yesterday = enc.encode([t_yesterday])[0]
        e_year = enc.encode([t_year_ago])[0]

        cos_near = np.dot(e_now, e_yesterday) / (
            np.linalg.norm(e_now) * np.linalg.norm(e_yesterday)
        )
        cos_far = np.dot(e_now, e_year) / (
            np.linalg.norm(e_now) * np.linalg.norm(e_year)
        )
        assert cos_near > cos_far

    def test_empty_list_returns_empty(self):
        """Empty input → shape (0, temporal_dims)."""
        enc = SinusoidalEncoder(temporal_dims=16)
        result = enc.encode([])
        assert result.shape == (0, 16)

    def test_max_period_days_configurable(self):
        """max_period_days controls the longest wavelength."""
        enc1 = SinusoidalEncoder(temporal_dims=8, max_period_days=365)
        enc2 = SinusoidalEncoder(temporal_dims=8, max_period_days=3650)
        t = QUERY_TIME - timedelta(days=100)
        e1 = enc1.encode([t])
        e2 = enc2.encode([t])
        # Different max_period → different encodings
        assert not np.allclose(e1, e2)

    def test_encode_from_facts(self):
        """Convenience: encode directly from TemporalFact list."""
        enc = SinusoidalEncoder(temporal_dims=16)
        facts = [_fact(0), _fact(10), _fact(100)]
        result = enc.encode_facts(facts)
        assert result.shape == (3, 16)


# =====================================================================
# TemporalEmbedder — orchestrator
# =====================================================================

class TestTemporalEmbedder:
    """TemporalEmbedder combines temporal encoder output with semantic vectors."""

    def test_embed_output_shape(self):
        """Output shape = (n, temporal_dims + semantic_dims)."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        facts = [_fact(0), _fact(10), _fact(100)]
        semantic = _random_semantic(3, 384)
        result = embedder.embed(facts, semantic)
        assert result.shape == (3, 8 + 384)

    def test_embed_temporal_prefix(self):
        """First temporal_dims columns should match encoder output."""
        enc = SinusoidalEncoder(temporal_dims=16)
        embedder = TemporalEmbedder(encoder=enc)
        facts = [_fact(5)]
        semantic = _random_semantic(1, 128)
        combined = embedder.embed(facts, semantic)
        temporal_only = enc.encode_facts(facts)
        np.testing.assert_array_almost_equal(combined[0, :16], temporal_only[0])

    def test_embed_semantic_suffix(self):
        """Columns after temporal_dims should be the semantic vectors."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        facts = [_fact(5)]
        semantic = _random_semantic(1, 64)
        combined = embedder.embed(facts, semantic)
        np.testing.assert_array_almost_equal(combined[0, 8:], semantic[0])

    def test_mismatched_lengths_raises(self):
        """facts and semantic_vectors must have same length."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        facts = [_fact(0), _fact(1)]
        semantic = _random_semantic(3, 64)  # length mismatch
        with pytest.raises(ValueError, match="length"):
            embedder.embed(facts, semantic)

    def test_empty_input(self):
        """Empty input → shape (0, temporal_dims + semantic_dim)."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        semantic = np.empty((0, 64))
        result = embedder.embed([], semantic)
        assert result.shape == (0, 8 + 64)

    def test_total_dims_property(self):
        """total_dims = temporal_dims + semantic_dims."""
        enc = SinusoidalEncoder(temporal_dims=16)
        embedder = TemporalEmbedder(encoder=enc, semantic_dims=384)
        assert embedder.total_dims == 16 + 384

    def test_temporal_dims_property(self):
        """temporal_dims should be accessible via embedder."""
        enc = SinusoidalEncoder(temporal_dims=32)
        embedder = TemporalEmbedder(encoder=enc)
        assert embedder.temporal_dims == 32

    def test_encoder_is_swappable(self):
        """Different encoders should produce different embeddings."""
        enc8 = SinusoidalEncoder(temporal_dims=8)
        enc32 = SinusoidalEncoder(temporal_dims=32)
        emb8 = TemporalEmbedder(encoder=enc8)
        emb32 = TemporalEmbedder(encoder=enc32)
        facts = [_fact(10)]
        sem64 = _random_semantic(1, 64)
        r8 = emb8.embed(facts, sem64)
        r32 = emb32.embed(facts, sem64)
        assert r8.shape == (1, 8 + 64)
        assert r32.shape == (1, 32 + 64)

    def test_reference_time_forwarded(self):
        """reference_time should be forwarded to the encoder."""
        ref = datetime(2020, 1, 1)
        enc = SinusoidalEncoder(temporal_dims=16, reference_time=ref)
        embedder = TemporalEmbedder(encoder=enc)
        facts = [_fact(10)]
        semantic = _random_semantic(1, 64)
        result = embedder.embed(facts, semantic)
        # Should use the encoder's default reference_time
        expected_temporal = enc.encode_facts(facts)
        np.testing.assert_array_almost_equal(result[0, :16], expected_temporal[0])


# =====================================================================
# Similarity computation
# =====================================================================

class TestSimilarity:
    """Cosine similarity in the combined temporal-semantic space."""

    def test_cosine_similarity_shape(self):
        """Similarity of query vs n facts → shape (n,)."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        query_emb = np.random.randn(8 + 64)
        fact_embs = np.random.randn(5, 8 + 64)
        sims = embedder.cosine_similarity(query_emb, fact_embs)
        assert sims.shape == (5,)

    def test_identical_vectors_similarity_one(self):
        """A vector compared to itself should have similarity ~1.0."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        v = np.random.randn(8 + 64)
        sims = embedder.cosine_similarity(v, v.reshape(1, -1))
        assert sims[0] == pytest.approx(1.0, abs=1e-6)

    def test_similarity_in_range(self):
        """Cosine similarity should be in [-1, 1]."""
        enc = SinusoidalEncoder(temporal_dims=16)
        embedder = TemporalEmbedder(encoder=enc)
        query = np.random.randn(16 + 64)
        facts = np.random.randn(20, 16 + 64)
        sims = embedder.cosine_similarity(query, facts)
        assert np.all(sims >= -1.0 - 1e-6)
        assert np.all(sims <= 1.0 + 1e-6)

    def test_temporal_proximity_boosts_similarity(self):
        """Facts with closer timestamps should have higher similarity
        in the combined space, all else equal."""
        enc = SinusoidalEncoder(temporal_dims=32)
        embedder = TemporalEmbedder(encoder=enc)

        # Same semantic content, different ages
        sem = _random_semantic(1, 64)[0]  # one semantic vector
        f_recent = _fact(1, content="same")
        f_old = _fact(365, content="same")

        # Build query embedding at QUERY_TIME
        q_temporal = enc.encode([QUERY_TIME])[0]
        q_emb = np.concatenate([q_temporal, sem])

        # Build fact embeddings
        facts = [f_recent, f_old]
        sem_batch = np.stack([sem, sem])
        fact_embs = embedder.embed(facts, sem_batch)

        sims = embedder.cosine_similarity(q_emb, fact_embs)
        assert sims[0] > sims[1], "Recent fact should be more similar in combined space"


# =====================================================================
# Weighted similarity
# =====================================================================

class TestWeightedSimilarity:
    """Optional temporal_weight parameter for controlling the influence
    of the temporal subspace vs semantic subspace in similarity."""

    def test_weight_zero_ignores_temporal(self):
        """temporal_weight=0 should give pure semantic similarity."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)

        # Same semantic vector for both facts — only temporal age differs
        sem_single = _random_semantic(1, 64)[0]
        sem = np.stack([sem_single, sem_single])  # duplicate
        facts = [_fact(1), _fact(365)]  # very different ages
        combined = embedder.embed(facts, sem)

        # Pure semantic query using the same semantic vector
        q_sem = sem_single
        q_temporal = enc.encode([QUERY_TIME])[0]
        q_full = np.concatenate([q_temporal, q_sem])

        sims_weighted = embedder.cosine_similarity(
            q_full, combined, temporal_weight=0.0
        )
        # With weight=0, temporal dims are zeroed out.
        # Both facts share the same semantic vector, so sims must be equal.
        assert sims_weighted[0] == pytest.approx(sims_weighted[1], abs=1e-6)

    def test_weight_one_is_default(self):
        """temporal_weight=1.0 should equal unweighted similarity."""
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        q = np.random.randn(8 + 64)
        facts = np.random.randn(3, 8 + 64)
        sims_default = embedder.cosine_similarity(q, facts)
        sims_w1 = embedder.cosine_similarity(q, facts, temporal_weight=1.0)
        np.testing.assert_array_almost_equal(sims_default, sims_w1)

    def test_higher_weight_increases_temporal_influence(self):
        """Increasing temporal_weight should increase the gap between
        temporally close and temporally far facts."""
        enc = SinusoidalEncoder(temporal_dims=32)
        embedder = TemporalEmbedder(encoder=enc)

        sem = _random_semantic(1, 64)[0]
        f_near = _fact(1, content="same")
        f_far = _fact(365, content="same")
        facts = [f_near, f_far]
        sem_batch = np.stack([sem, sem])
        fact_embs = embedder.embed(facts, sem_batch)

        q_temporal = enc.encode([QUERY_TIME])[0]
        q = np.concatenate([q_temporal, sem])

        gap_low = (
            embedder.cosine_similarity(q, fact_embs, temporal_weight=0.5)[0]
            - embedder.cosine_similarity(q, fact_embs, temporal_weight=0.5)[1]
        )
        gap_high = (
            embedder.cosine_similarity(q, fact_embs, temporal_weight=2.0)[0]
            - embedder.cosine_similarity(q, fact_embs, temporal_weight=2.0)[1]
        )
        assert gap_high > gap_low


# =====================================================================
# Integration with TemporalScorer
# =====================================================================

class TestIntegrationWithScorer:
    """Embedding module should work seamlessly with TemporalScorer."""

    def test_embedder_similarities_feed_into_scorer(self):
        """Cosine similarities from embedder can be passed to TemporalScorer."""
        from chronofy.scoring.temporal_scorer import TemporalScorer
        from chronofy.decay.exponential import ExponentialDecay

        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)
        decay = ExponentialDecay(beta={"general": 0.5})
        scorer = TemporalScorer(decay_fn=decay)

        facts = [_fact(1), _fact(30), _fact(180)]
        semantic = _random_semantic(3, 64)
        fact_embs = embedder.embed(facts, semantic)

        # Build query embedding
        q_temporal = enc.encode([QUERY_TIME])[0]
        q_emb = np.concatenate([q_temporal, semantic[0]])

        # Get similarities
        sims = embedder.cosine_similarity(q_emb, fact_embs)

        # Feed into scorer
        ranked = scorer.rank(
            facts, sims.tolist(), query_time=QUERY_TIME, top_k=2
        )
        assert len(ranked) == 2
        assert ranked[0].combined_score >= ranked[1].combined_score


# =====================================================================
# Repr and configuration
# =====================================================================

class TestRepr:
    """Repr strings should be informative."""

    def test_sinusoidal_repr(self):
        enc = SinusoidalEncoder(temporal_dims=16, max_period_days=365)
        r = repr(enc)
        assert "SinusoidalEncoder" in r
        assert "16" in r

    def test_embedder_repr(self):
        enc = SinusoidalEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc, semantic_dims=384)
        r = repr(embedder)
        assert "TemporalEmbedder" in r
