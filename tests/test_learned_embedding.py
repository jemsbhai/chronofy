"""Tests for chronofy.embedding learned modules — LearnedEncoder, losses, fine-tuner.

Requires torch (part of the [ml] extra). Tests skip gracefully if torch
is not installed.

Covers:
  - LearnedEncoder: MLP temporal projector, ABC compliance, save/load
  - CKALoss: Linear CKA computation correctness
  - TemporalContrastiveLoss: InfoNCE on temporal subspace
  - TMRLLoss: Combined multi-scale objective
  - TemporalFineTuner: LoRA integration, fit(), checkpoint save/load
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from chronofy.models import TemporalFact

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

# ─── Helpers ──────────────────────────────────────────────────────────

QUERY_TIME = datetime(2026, 3, 15)


def _fact(
    days_ago: float,
    fact_type: str = "general",
    quality: float = 1.0,
    content: str = "test",
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


# =====================================================================
# LearnedEncoder
# =====================================================================


class TestLearnedEncoderConstruction:
    """LearnedEncoder is an nn.Module that subclasses TemporalEncoder."""

    def test_is_nn_module(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        assert isinstance(enc, nn.Module)

    def test_is_temporal_encoder(self):
        from chronofy.embedding.base import TemporalEncoder
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        assert isinstance(enc, TemporalEncoder)

    def test_temporal_dims_property(self):
        from chronofy.embedding.learned import LearnedEncoder

        for d in [4, 8, 16, 32, 64]:
            enc = LearnedEncoder(temporal_dims=d)
            assert enc.temporal_dims == d

    def test_default_hidden_dims(self):
        """Default hidden layer size should be reasonable."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        assert enc.hidden_dims > 0

    def test_custom_hidden_dims(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16, hidden_dims=64)
        assert enc.hidden_dims == 64

    def test_invalid_temporal_dims_raises(self):
        from chronofy.embedding.learned import LearnedEncoder

        with pytest.raises(ValueError):
            LearnedEncoder(temporal_dims=0)
        with pytest.raises(ValueError):
            LearnedEncoder(temporal_dims=-8)

    def test_custom_reference_time(self):
        from chronofy.embedding.learned import LearnedEncoder

        ref = datetime(2020, 1, 1)
        enc = LearnedEncoder(temporal_dims=8, reference_time=ref)
        assert enc.reference_time == ref

    def test_num_input_features(self):
        """Encoder should extract multiple features from timestamps."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        assert enc.num_input_features >= 5  # at least: offset + cyclical pairs


class TestLearnedEncoderForward:
    """Forward pass and encode() method tests."""

    def test_encode_output_shape_single(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        result = enc.encode([QUERY_TIME])
        assert result.shape == (1, 16)

    def test_encode_output_shape_batch(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        timestamps = [QUERY_TIME - timedelta(days=i) for i in range(10)]
        result = enc.encode(timestamps)
        assert result.shape == (10, 8)

    def test_encode_returns_numpy(self):
        """encode() returns numpy for ABC compatibility, not torch tensor."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        result = enc.encode([QUERY_TIME])
        assert isinstance(result, np.ndarray)

    def test_forward_returns_tensor(self):
        """forward() (torch convention) should return a tensor."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        features = enc.extract_features([QUERY_TIME])
        out = enc.forward(features)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 8)

    def test_encode_empty_list(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        result = enc.encode([])
        assert result.shape == (0, 16)

    def test_encode_facts_convenience(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        facts = [_fact(0), _fact(10), _fact(100)]
        result = enc.encode_facts(facts)
        assert result.shape == (3, 8)

    def test_different_timestamps_different_outputs(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        e1 = enc.encode([QUERY_TIME])
        e2 = enc.encode([QUERY_TIME - timedelta(days=365)])
        assert not np.allclose(e1, e2)

    def test_deterministic_in_eval_mode(self):
        """Same timestamp → same output when in eval mode."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16)
        enc.eval()
        e1 = enc.encode([QUERY_TIME])
        e2 = enc.encode([QUERY_TIME])
        np.testing.assert_array_equal(e1, e2)


class TestLearnedEncoderFeatures:
    """Timestamp feature extraction."""

    def test_extract_features_shape(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        features = enc.extract_features([QUERY_TIME, QUERY_TIME - timedelta(days=30)])
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 2
        assert features.shape[1] == enc.num_input_features

    def test_features_include_cyclical(self):
        """Cyclical features (sin/cos) should be in [-1, 1]."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        features = enc.extract_features([QUERY_TIME])
        # All cyclical features (after the first offset feature) should be in [-1, 1]
        cyclical = features[0, 1:]  # skip day offset
        assert torch.all(cyclical >= -1.0)
        assert torch.all(cyclical <= 1.0)


class TestLearnedEncoderGradients:
    """Parameters should be trainable."""

    def test_has_parameters(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        params = list(enc.parameters())
        assert len(params) > 0

    def test_gradients_flow(self):
        """Loss backward should produce gradients on encoder parameters."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        features = enc.extract_features([QUERY_TIME, QUERY_TIME - timedelta(days=10)])
        out = enc.forward(features)
        loss = out.sum()
        loss.backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestLearnedEncoderSaveLoad:
    """Save and load via state_dict."""

    def test_save_and_load_state_dict(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16, hidden_dims=32)
        timestamps = [QUERY_TIME - timedelta(days=i) for i in range(5)]
        original_output = enc.encode(timestamps)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            enc.save(path)
            assert path.exists()

            enc2 = LearnedEncoder(temporal_dims=16, hidden_dims=32)
            enc2.load(path)
            loaded_output = enc2.encode(timestamps)

        np.testing.assert_array_almost_equal(original_output, loaded_output)

    def test_save_includes_config(self):
        """Saved checkpoint should include config for reconstruction."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16, hidden_dims=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            enc.save(path)
            checkpoint = torch.load(path, weights_only=False)

        assert "config" in checkpoint
        assert checkpoint["config"]["temporal_dims"] == 16
        assert checkpoint["config"]["hidden_dims"] == 64

    def test_load_from_checkpoint_classmethod(self):
        """Class method to reconstruct encoder from checkpoint."""
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8, hidden_dims=32)
        timestamps = [QUERY_TIME]
        original_output = enc.encode(timestamps)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            enc.save(path)

            enc2 = LearnedEncoder.from_checkpoint(path)

        assert enc2.temporal_dims == 8
        assert enc2.hidden_dims == 32
        np.testing.assert_array_almost_equal(original_output, enc2.encode(timestamps))


class TestLearnedEncoderIntegration:
    """Integration with TemporalEmbedder — should be swappable with SinusoidalEncoder."""

    def test_works_with_temporal_embedder(self):
        from chronofy.embedding.embedder import TemporalEmbedder
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=8)
        embedder = TemporalEmbedder(encoder=enc)

        facts = [_fact(1), _fact(30), _fact(180)]
        semantic = np.random.randn(3, 64)
        result = embedder.embed(facts, semantic)
        assert result.shape == (3, 8 + 64)

    def test_swappable_with_sinusoidal(self):
        """Same TemporalEmbedder API with different encoder backends."""
        from chronofy.embedding.embedder import TemporalEmbedder
        from chronofy.embedding.learned import LearnedEncoder
        from chronofy.embedding.sinusoidal import SinusoidalEncoder

        facts = [_fact(10)]
        semantic = np.random.randn(1, 64)

        emb_sin = TemporalEmbedder(encoder=SinusoidalEncoder(temporal_dims=8))
        emb_learned = TemporalEmbedder(encoder=LearnedEncoder(temporal_dims=8))

        r_sin = emb_sin.embed(facts, semantic)
        r_learned = emb_learned.embed(facts, semantic)

        assert r_sin.shape == r_learned.shape
        # Semantic suffix should be identical
        np.testing.assert_array_almost_equal(r_sin[0, 8:], r_learned[0, 8:])


# =====================================================================
# CKA Loss
# =====================================================================


class TestCKALoss:
    """Linear CKA loss for structural self-distillation."""

    def test_identical_matrices_give_similarity_one(self):
        """CKA(X, X) should be 1.0."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        similarity = cka.cka_similarity(X, X)
        assert similarity.item() == pytest.approx(1.0, abs=1e-4)

    def test_loss_of_identical_is_zero(self):
        """Loss = 1 - CKA, so identical matrices → loss ≈ 0."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        loss = cka(X, X)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_cka_in_zero_one(self):
        """CKA similarity should be in [0, 1]."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        Y = torch.randn(32, 32)
        sim = cka.cka_similarity(X, Y)
        assert 0.0 <= sim.item() <= 1.0 + 1e-6

    def test_cka_is_symmetric(self):
        """CKA(X, Y) == CKA(Y, X) when dimensions allow."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        Y = torch.randn(32, 64)
        assert cka.cka_similarity(X, Y).item() == pytest.approx(
            cka.cka_similarity(Y, X).item(), abs=1e-5
        )

    def test_gradient_flows(self):
        """CKA loss should be differentiable."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(16, 32, requires_grad=True)
        Y = torch.randn(16, 16)
        loss = cka(X, Y)
        loss.backward()
        assert X.grad is not None

    def test_different_column_dims_allowed(self):
        """CKA works across different dimensionalities (same batch size)."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 128)
        Y = torch.randn(32, 16)
        loss = cka(X, Y)
        assert loss.item() >= 0.0

    def test_mismatched_batch_size_raises(self):
        """Different batch sizes should raise."""
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        Y = torch.randn(16, 64)
        with pytest.raises(ValueError, match="batch"):
            cka(X, Y)


# =====================================================================
# Temporal Contrastive Loss
# =====================================================================


class TestTemporalContrastiveLoss:
    """InfoNCE on temporal subspace — pushes temporally similar pairs
    together and dissimilar pairs apart."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        embeddings = torch.randn(16, 8)  # batch of temporal embeddings
        # Timestamps as day offsets
        timestamps = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings, timestamps)
        assert loss.dim() == 0  # scalar

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        embeddings = torch.randn(16, 8)
        timestamps = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings, timestamps)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        embeddings = torch.randn(16, 8, requires_grad=True)
        timestamps = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings, timestamps)
        loss.backward()
        assert embeddings.grad is not None

    def test_temperature_configurable(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_low_t = TemporalContrastiveLoss(temperature=0.05)
        loss_high_t = TemporalContrastiveLoss(temperature=1.0)
        embeddings = torch.randn(16, 8)
        timestamps = torch.arange(16, dtype=torch.float32)
        l1 = loss_low_t(embeddings, timestamps)
        l2 = loss_high_t(embeddings, timestamps)
        # Different temperatures should give different losses
        assert l1.item() != pytest.approx(l2.item(), abs=0.01)

    def test_proximity_threshold_configurable(self):
        """proximity_threshold controls what counts as a positive pair."""
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss(proximity_threshold=1.0)
        assert loss_fn.proximity_threshold == 1.0


# =====================================================================
# TMRL Loss (combined objective)
# =====================================================================


class TestTMRLLoss:
    """Combined TMRL objective: L = L_semantic + α·L_temp + β·L_CKA."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        # Simulated inputs
        full_emb = torch.randn(16, 64)   # full embeddings
        trunc_emb = torch.randn(16, 16)  # truncated (temporal subspace)
        timestamps = torch.arange(16, dtype=torch.float32)
        # Positive/negative pairs for semantic contrastive
        anchors = torch.randn(16, 64)
        positives = torch.randn(16, 64)
        loss = loss_fn(
            full_embeddings=full_emb,
            truncated_embeddings=trunc_emb,
            temporal_embeddings=trunc_emb,
            timestamps=timestamps,
            anchors=anchors,
            positives=positives,
        )
        assert loss.dim() == 0

    def test_weights_configurable(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss(alpha_temporal=0.5, beta_cka=0.2)
        assert loss_fn.alpha_temporal == 0.5
        assert loss_fn.beta_cka == 0.2

    def test_gradient_flows_through_all_components(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        full_emb = torch.randn(16, 64, requires_grad=True)
        trunc_emb = full_emb[:, :16]  # truncation shares parameters
        timestamps = torch.arange(16, dtype=torch.float32)
        anchors = torch.randn(16, 64)
        positives = torch.randn(16, 64)
        loss = loss_fn(
            full_embeddings=full_emb,
            truncated_embeddings=trunc_emb,
            temporal_embeddings=trunc_emb,
            timestamps=timestamps,
            anchors=anchors,
            positives=positives,
        )
        loss.backward()
        assert full_emb.grad is not None

    def test_individual_loss_components_accessible(self):
        """Should be able to inspect individual loss terms."""
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        full_emb = torch.randn(16, 64)
        trunc_emb = torch.randn(16, 16)
        timestamps = torch.arange(16, dtype=torch.float32)
        anchors = torch.randn(16, 64)
        positives = torch.randn(16, 64)
        loss, components = loss_fn.forward_with_components(
            full_embeddings=full_emb,
            truncated_embeddings=trunc_emb,
            temporal_embeddings=trunc_emb,
            timestamps=timestamps,
            anchors=anchors,
            positives=positives,
        )
        assert "semantic" in components
        assert "temporal" in components
        assert "cka" in components


# =====================================================================
# TemporalFineTuner
# =====================================================================


class TestTemporalFineTuner:
    """Training orchestrator with LoRA integration."""

    def test_construction(self):
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=8)
        tuner = TemporalFineTuner(temporal_encoder=encoder)
        assert tuner.temporal_encoder is encoder

    def test_lora_rank_configurable(self):
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=8)
        tuner = TemporalFineTuner(temporal_encoder=encoder, lora_rank=4)
        assert tuner.lora_rank == 4

    def test_fit_runs_without_error(self):
        """Smoke test: fit() on tiny synthetic data should not crash."""
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=8)
        tuner = TemporalFineTuner(temporal_encoder=encoder, lora_rank=2)

        # Tiny synthetic training data
        train_data = {
            "texts": ["fact A from 2020", "fact B from 2021", "fact C from 2022"] * 4,
            "timestamps": [
                datetime(2020, 1, 1),
                datetime(2021, 6, 15),
                datetime(2022, 12, 1),
            ]
            * 4,
            "positive_pairs": [(0, 3), (1, 4), (2, 5)],  # same facts are positives
        }

        # Should complete 1 epoch without error
        tuner.fit(train_data, epochs=1, batch_size=4)

    def test_fit_reduces_loss(self):
        """Loss should decrease over training on structured data."""
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        torch.manual_seed(42)
        encoder = LearnedEncoder(temporal_dims=8, hidden_dims=32)
        tuner = TemporalFineTuner(
            temporal_encoder=encoder,
            lora_rank=4,
            learning_rate=5e-3,
        )

        # Structured data: 6 temporal clusters of 8 samples each.
        # Positive pairs connect samples within the same cluster,
        # giving a clear temporal contrastive signal.
        n_clusters = 6
        per_cluster = 8
        n = n_clusters * per_cluster
        timestamps = []
        for c in range(n_clusters):
            base = datetime(2020, 1, 1) + timedelta(days=c * 180)
            for j in range(per_cluster):
                timestamps.append(base + timedelta(days=j))  # tight cluster

        positive_pairs = []
        for c in range(n_clusters):
            base_idx = c * per_cluster
            for j in range(0, per_cluster, 2):
                positive_pairs.append((base_idx + j, base_idx + j + 1))

        train_data = {
            "texts": [f"fact {i}" for i in range(n)],
            "timestamps": timestamps,
            "positive_pairs": positive_pairs,
        }

        history = tuner.fit(train_data, epochs=15, batch_size=16)
        # Compare average of first 3 vs last 3 epochs for robustness
        # (individual epochs can be noisy)
        avg_early = sum(history[:3]) / 3
        avg_late = sum(history[-3:]) / 3
        assert avg_late < avg_early, (
            f"Loss should decrease: early avg={avg_early:.4f}, "
            f"late avg={avg_late:.4f}"
        )

    def test_save_and_load_checkpoint(self):
        """Full checkpoint save/load including encoder + LoRA state."""
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=8)
        tuner = TemporalFineTuner(temporal_encoder=encoder, lora_rank=2)

        timestamps = [QUERY_TIME - timedelta(days=i) for i in range(5)]
        original_output = encoder.encode(timestamps)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            tuner.save_checkpoint(path)

            encoder2 = LearnedEncoder(temporal_dims=8)
            tuner2 = TemporalFineTuner(temporal_encoder=encoder2, lora_rank=2)
            tuner2.load_checkpoint(path)

            loaded_output = encoder2.encode(timestamps)

        np.testing.assert_array_almost_equal(original_output, loaded_output)

    def test_learning_rate_configurable(self):
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=8)
        tuner = TemporalFineTuner(
            temporal_encoder=encoder, learning_rate=1e-4
        )
        assert tuner.learning_rate == 1e-4

    def test_matryoshka_scales_configurable(self):
        """Matryoshka truncation scales for multi-scale loss."""
        from chronofy.embedding.fine_tuner import TemporalFineTuner
        from chronofy.embedding.learned import LearnedEncoder

        encoder = LearnedEncoder(temporal_dims=16)
        tuner = TemporalFineTuner(
            temporal_encoder=encoder,
            matryoshka_scales=[4, 8, 16],
        )
        assert tuner.matryoshka_scales == [4, 8, 16]


# =====================================================================
# Repr
# =====================================================================


class TestLearnedRepr:
    def test_learned_encoder_repr(self):
        from chronofy.embedding.learned import LearnedEncoder

        enc = LearnedEncoder(temporal_dims=16, hidden_dims=32)
        r = repr(enc)
        assert "LearnedEncoder" in r
        assert "16" in r
