"""Tests for chronofy.embedding.losses — comprehensive loss function suite.

Covers:
  - TemporalLoss ABC and protocol enforcement
  - LossRegistry: registration, discovery, custom loss plugins
  - CompositeLoss: weighted ensembling of multiple losses
  - All built-in loss functions: mathematical correctness, gradients, edge cases
  - Integration: user-defined custom losses plug into the system

Requires torch (part of the [ml] extra).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# =====================================================================
# TemporalLoss ABC
# =====================================================================


class TestTemporalLossABC:
    """Base class protocol enforcement."""

    def test_cannot_instantiate_bare_base(self):
        """TemporalLoss base has no real name/forward, so calling them raises."""
        from chronofy.embedding.losses import TemporalLoss

        # TemporalLoss itself can be instantiated (nn.Module), but
        # name and forward raise NotImplementedError
        base = TemporalLoss()
        with pytest.raises(NotImplementedError):
            _ = base.name
        with pytest.raises(NotImplementedError):
            base()

    def test_subclass_must_implement_forward(self):
        """Defining a subclass without forward() raises TypeError at class definition."""
        from chronofy.embedding.losses import TemporalLoss

        with pytest.raises(TypeError, match="forward"):
            class BadLoss(TemporalLoss):
                @property
                def name(self) -> str:
                    return "bad"

    def test_subclass_must_implement_name(self):
        """Defining a subclass without name property raises TypeError at class definition."""
        from chronofy.embedding.losses import TemporalLoss

        with pytest.raises(TypeError, match="name"):
            class BadLoss(TemporalLoss):
                def forward(self, **kwargs) -> torch.Tensor:
                    return torch.tensor(0.0)

    def test_valid_subclass_instantiates(self):
        from chronofy.embedding.losses import TemporalLoss

        class GoodLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "good"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(0.0)

        loss = GoodLoss()
        assert loss.name == "good"

    def test_abc_is_nn_module(self):
        """TemporalLoss should inherit from nn.Module."""
        from chronofy.embedding.losses import TemporalLoss

        assert issubclass(TemporalLoss, nn.Module)


# =====================================================================
# LossRegistry
# =====================================================================


class TestLossRegistry:
    """Registration, discovery, and plugin system."""

    def test_register_and_retrieve(self):
        from chronofy.embedding.losses import LossRegistry, TemporalLoss

        registry = LossRegistry()

        class MyLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "my_loss"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(0.0)

        registry.register(MyLoss)
        assert "my_loss" in registry.available()
        instance = registry.create("my_loss")
        assert isinstance(instance, MyLoss)

    def test_register_with_custom_name(self):
        from chronofy.embedding.losses import LossRegistry, TemporalLoss

        registry = LossRegistry()

        class MyLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "default_name"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(0.0)

        registry.register(MyLoss, name="custom_name")
        assert "custom_name" in registry.available()
        assert "default_name" not in registry.available()

    def test_register_duplicate_raises(self):
        from chronofy.embedding.losses import LossRegistry, TemporalLoss

        registry = LossRegistry()

        class MyLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "dup"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(0.0)

        registry.register(MyLoss)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MyLoss)

    def test_create_unknown_raises(self):
        from chronofy.embedding.losses import LossRegistry

        registry = LossRegistry()
        with pytest.raises(KeyError):
            registry.create("nonexistent")

    def test_create_with_kwargs(self):
        from chronofy.embedding.losses import LossRegistry, TemporalLoss

        registry = LossRegistry()

        class ParamLoss(TemporalLoss):
            def __init__(self, temperature: float = 0.1):
                super().__init__()
                self.temperature = temperature

            @property
            def name(self) -> str:
                return "param_loss"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(0.0)

        registry.register(ParamLoss)
        instance = registry.create("param_loss", temperature=0.5)
        assert instance.temperature == 0.5

    def test_available_returns_sorted_list(self):
        from chronofy.embedding.losses import LossRegistry, TemporalLoss

        registry = LossRegistry()

        for n in ["zebra", "alpha", "middle"]:

            class TempLoss(TemporalLoss):
                _name = n

                @property
                def name(self) -> str:
                    return self._name

                def forward(self, **kwargs) -> torch.Tensor:
                    return torch.tensor(0.0)

            registry.register(TempLoss, name=n)

        assert registry.available() == ["alpha", "middle", "zebra"]

    def test_default_registry_has_builtins(self):
        """The module-level default registry should include all built-in losses."""
        from chronofy.embedding.losses import default_registry

        available = default_registry.available()
        expected = [
            "alignment_uniformity",
            "cka",
            "matryoshka_truncation",
            "ntxent",
            "semantic_contrastive",
            "temporal_contrastive",
            "temporal_ordering",
            "temporal_smoothness",
            "triplet",
            "temporal_triplet",
        ]
        for name in expected:
            assert name in available, f"Built-in loss '{name}' missing from default registry"

    def test_register_non_temporal_loss_raises(self):
        from chronofy.embedding.losses import LossRegistry

        registry = LossRegistry()
        with pytest.raises(TypeError, match="TemporalLoss"):
            registry.register(nn.MSELoss)  # type: ignore[arg-type]


# =====================================================================
# CompositeLoss
# =====================================================================


class TestCompositeLoss:
    """Weighted ensembling of multiple losses."""

    def test_single_loss_passthrough(self):
        from chronofy.embedding.losses import CompositeLoss, CKALoss

        cka = CKALoss()
        composite = CompositeLoss(losses={"cka": (cka, 1.0)})
        X = torch.randn(16, 32)
        Y = torch.randn(16, 16)
        result = composite(cka={"X": X, "Y": Y})
        direct = cka(X=X, Y=Y)
        assert result.item() == pytest.approx(direct.item(), abs=1e-5)

    def test_weighted_combination(self):
        """Two losses with different weights should combine correctly."""
        from chronofy.embedding.losses import CompositeLoss, TemporalLoss

        class ConstLoss(TemporalLoss):
            def __init__(self, value: float):
                super().__init__()
                self.value = value

            @property
            def name(self) -> str:
                return f"const_{self.value}"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(self.value)

        l1 = ConstLoss(2.0)
        l2 = ConstLoss(3.0)
        composite = CompositeLoss(losses={"l1": (l1, 0.5), "l2": (l2, 0.3)})
        result = composite(l1={}, l2={})
        expected = 0.5 * 2.0 + 0.3 * 3.0  # 1.0 + 0.9 = 1.9
        assert result.item() == pytest.approx(expected, abs=1e-5)

    def test_gradient_flows_through_all(self):
        from chronofy.embedding.losses import CompositeLoss, CKALoss, TemporalContrastiveLoss

        cka = CKALoss()
        tc = TemporalContrastiveLoss()
        composite = CompositeLoss(losses={"cka": (cka, 0.5), "tc": (tc, 0.3)})

        X = torch.randn(16, 32, requires_grad=True)
        Y = torch.randn(16, 16)
        emb = torch.randn(16, 8, requires_grad=True)
        ts = torch.arange(16, dtype=torch.float32)

        total = composite(
            cka={"X": X, "Y": Y},
            tc={"embeddings": emb, "timestamps": ts},
        )
        total.backward()
        assert X.grad is not None
        assert emb.grad is not None

    def test_forward_with_components(self):
        from chronofy.embedding.losses import CompositeLoss, TemporalLoss

        class ConstLoss(TemporalLoss):
            def __init__(self, value: float, n: str):
                super().__init__()
                self.value = value
                self._name = n

            @property
            def name(self) -> str:
                return self._name

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(self.value)

        composite = CompositeLoss(
            losses={
                "a": (ConstLoss(1.0, "a"), 0.5),
                "b": (ConstLoss(2.0, "b"), 0.3),
            }
        )
        total, components = composite.forward_with_components(a={}, b={})
        assert "a" in components
        assert "b" in components
        assert components["a"] == pytest.approx(1.0, abs=1e-5)
        assert components["b"] == pytest.approx(2.0, abs=1e-5)
        assert total.item() == pytest.approx(0.5 * 1.0 + 0.3 * 2.0, abs=1e-5)

    def test_add_loss_dynamically(self):
        from chronofy.embedding.losses import CompositeLoss, TemporalLoss

        class ConstLoss(TemporalLoss):
            def __init__(self, value: float):
                super().__init__()
                self.value = value

            @property
            def name(self) -> str:
                return "const"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(self.value)

        composite = CompositeLoss()
        composite.add("first", ConstLoss(1.0), weight=0.5)
        assert "first" in composite.loss_names

    def test_remove_loss(self):
        from chronofy.embedding.losses import CompositeLoss, TemporalLoss

        class ConstLoss(TemporalLoss):
            def __init__(self):
                super().__init__()

            @property
            def name(self) -> str:
                return "c"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(1.0)

        composite = CompositeLoss()
        composite.add("first", ConstLoss(), weight=0.5)
        composite.add("second", ConstLoss(), weight=0.3)
        composite.remove("first")
        assert "first" not in composite.loss_names
        assert "second" in composite.loss_names

    def test_update_weight(self):
        from chronofy.embedding.losses import CompositeLoss, TemporalLoss

        class ConstLoss(TemporalLoss):
            def __init__(self):
                super().__init__()

            @property
            def name(self) -> str:
                return "c"

            def forward(self, **kwargs) -> torch.Tensor:
                return torch.tensor(2.0)

        composite = CompositeLoss()
        composite.add("a", ConstLoss(), weight=0.5)
        result1 = composite(a={}).item()
        composite.update_weight("a", 1.0)
        result2 = composite(a={}).item()
        assert result2 == pytest.approx(result1 * 2.0, abs=1e-5)

    def test_empty_composite_returns_zero(self):
        from chronofy.embedding.losses import CompositeLoss

        composite = CompositeLoss()
        result = composite()
        assert result.item() == 0.0


# =====================================================================
# CKALoss
# =====================================================================


class TestCKALoss:
    """Linear CKA — structural self-distillation."""

    def test_identical_matrices_similarity_one(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        assert cka.cka_similarity(X, X).item() == pytest.approx(1.0, abs=1e-4)

    def test_loss_of_identical_is_zero(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        assert cka(X=X, Y=X).item() == pytest.approx(0.0, abs=1e-4)

    def test_cka_in_zero_one(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        Y = torch.randn(32, 32)
        sim = cka.cka_similarity(X, Y)
        assert 0.0 <= sim.item() <= 1.0 + 1e-6

    def test_cka_is_symmetric(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 64)
        Y = torch.randn(32, 64)
        assert cka.cka_similarity(X, Y).item() == pytest.approx(
            cka.cka_similarity(Y, X).item(), abs=1e-5
        )

    def test_gradient_flows(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(16, 32, requires_grad=True)
        Y = torch.randn(16, 16)
        loss = cka(X=X, Y=Y)
        loss.backward()
        assert X.grad is not None

    def test_different_column_dims_allowed(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        X = torch.randn(32, 128)
        Y = torch.randn(32, 16)
        loss = cka(X=X, Y=Y)
        assert loss.item() >= 0.0

    def test_mismatched_batch_raises(self):
        from chronofy.embedding.losses import CKALoss

        cka = CKALoss()
        with pytest.raises(ValueError, match="batch"):
            cka(X=torch.randn(32, 64), Y=torch.randn(16, 64))

    def test_has_name(self):
        from chronofy.embedding.losses import CKALoss

        assert CKALoss().name == "cka"


# =====================================================================
# TemporalContrastiveLoss
# =====================================================================


class TestTemporalContrastiveLoss:
    """InfoNCE on temporal subspace."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings=emb, timestamps=ts)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings=emb, timestamps=ts)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        loss_fn = TemporalContrastiveLoss()
        emb = torch.randn(16, 8, requires_grad=True)
        ts = torch.arange(16, dtype=torch.float32)
        loss = loss_fn(embeddings=emb, timestamps=ts)
        loss.backward()
        assert emb.grad is not None

    def test_temperature_configurable(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        l_low = TemporalContrastiveLoss(temperature=0.05)
        l_high = TemporalContrastiveLoss(temperature=1.0)
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert l_low(embeddings=emb, timestamps=ts).item() != pytest.approx(
            l_high(embeddings=emb, timestamps=ts).item(), abs=0.01
        )

    def test_has_name(self):
        from chronofy.embedding.losses import TemporalContrastiveLoss

        assert TemporalContrastiveLoss().name == "temporal_contrastive"


# =====================================================================
# SemanticContrastiveLoss
# =====================================================================


class TestSemanticContrastiveLoss:
    def test_output_is_scalar(self):
        from chronofy.embedding.losses import SemanticContrastiveLoss

        loss_fn = SemanticContrastiveLoss()
        a = torch.randn(16, 64)
        p = torch.randn(16, 64)
        assert loss_fn(anchors=a, positives=p).dim() == 0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import SemanticContrastiveLoss

        loss_fn = SemanticContrastiveLoss()
        a = torch.randn(16, 64, requires_grad=True)
        p = torch.randn(16, 64)
        loss = loss_fn(anchors=a, positives=p)
        loss.backward()
        assert a.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import SemanticContrastiveLoss

        assert SemanticContrastiveLoss().name == "semantic_contrastive"


# =====================================================================
# NTXentLoss (SimCLR-style)
# =====================================================================


class TestNTXentLoss:
    """Normalized Temperature-scaled Cross Entropy."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import NTXentLoss

        loss_fn = NTXentLoss()
        z_i = torch.randn(16, 64)
        z_j = torch.randn(16, 64)
        assert loss_fn(z_i=z_i, z_j=z_j).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import NTXentLoss

        loss_fn = NTXentLoss()
        z_i = torch.randn(16, 64)
        z_j = torch.randn(16, 64)
        assert loss_fn(z_i=z_i, z_j=z_j).item() >= 0.0

    def test_identical_pairs_low_loss(self):
        """Identical positive pairs should have lower loss than random pairs."""
        from chronofy.embedding.losses import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.randn(16, 64)
        loss_same = loss_fn(z_i=z, z_j=z)
        loss_random = loss_fn(z_i=z, z_j=torch.randn(16, 64))
        assert loss_same.item() < loss_random.item()

    def test_gradient_flows(self):
        from chronofy.embedding.losses import NTXentLoss

        loss_fn = NTXentLoss()
        z_i = torch.randn(16, 64, requires_grad=True)
        z_j = torch.randn(16, 64)
        loss_fn(z_i=z_i, z_j=z_j).backward()
        assert z_i.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import NTXentLoss

        assert NTXentLoss().name == "ntxent"


# =====================================================================
# TripletLoss
# =====================================================================


class TestTripletLoss:
    """Margin-based triplet loss."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TripletLoss

        loss_fn = TripletLoss()
        a = torch.randn(16, 64)
        p = torch.randn(16, 64)
        n = torch.randn(16, 64)
        assert loss_fn(anchor=a, positive=p, negative=n).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TripletLoss

        loss_fn = TripletLoss()
        a = torch.randn(16, 64)
        p = torch.randn(16, 64)
        n = torch.randn(16, 64)
        assert loss_fn(anchor=a, positive=p, negative=n).item() >= 0.0

    def test_well_separated_gives_zero_loss(self):
        """When positives are much closer than negatives, loss → 0."""
        from chronofy.embedding.losses import TripletLoss

        loss_fn = TripletLoss(margin=1.0)
        a = torch.zeros(4, 8)
        p = torch.ones(4, 8) * 0.1  # close to anchor
        n = torch.ones(4, 8) * 10.0  # far from anchor
        loss = loss_fn(anchor=a, positive=p, negative=n)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_margin_configurable(self):
        from chronofy.embedding.losses import TripletLoss

        loss_fn = TripletLoss(margin=2.0)
        assert loss_fn.margin == 2.0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TripletLoss

        loss_fn = TripletLoss()
        a = torch.randn(8, 32, requires_grad=True)
        p = torch.randn(8, 32)
        n = torch.randn(8, 32)
        loss_fn(anchor=a, positive=p, negative=n).backward()
        assert a.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import TripletLoss

        assert TripletLoss().name == "triplet"


# =====================================================================
# TemporalTripletLoss
# =====================================================================


class TestTemporalTripletLoss:
    """Triplet loss with temporal hard negative mining."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TemporalTripletLoss

        loss_fn = TemporalTripletLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TemporalTripletLoss

        loss_fn = TemporalTripletLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).item() >= 0.0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TemporalTripletLoss

        loss_fn = TemporalTripletLoss()
        emb = torch.randn(16, 8, requires_grad=True)
        ts = torch.arange(16, dtype=torch.float32)
        loss_fn(embeddings=emb, timestamps=ts).backward()
        assert emb.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import TemporalTripletLoss

        assert TemporalTripletLoss().name == "temporal_triplet"


# =====================================================================
# TemporalOrderingLoss
# =====================================================================


class TestTemporalOrderingLoss:
    """Learns temporal ordering: before/after relationships."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TemporalOrderingLoss

        loss_fn = TemporalOrderingLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TemporalOrderingLoss

        loss_fn = TemporalOrderingLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).item() >= 0.0

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TemporalOrderingLoss

        loss_fn = TemporalOrderingLoss()
        emb = torch.randn(16, 8, requires_grad=True)
        ts = torch.arange(16, dtype=torch.float32)
        loss_fn(embeddings=emb, timestamps=ts).backward()
        assert emb.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import TemporalOrderingLoss

        assert TemporalOrderingLoss().name == "temporal_ordering"


# =====================================================================
# TemporalSmoothnessLoss
# =====================================================================


class TestTemporalSmoothnessLoss:
    """Nearby timestamps should produce similar embeddings."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        loss_fn = TemporalSmoothnessLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        loss_fn = TemporalSmoothnessLoss()
        emb = torch.randn(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).item() >= 0.0

    def test_identical_embeddings_zero_loss(self):
        """All identical embeddings → smoothness loss = 0."""
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        loss_fn = TemporalSmoothnessLoss()
        emb = torch.ones(16, 8)
        ts = torch.arange(16, dtype=torch.float32)
        assert loss_fn(embeddings=emb, timestamps=ts).item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        loss_fn = TemporalSmoothnessLoss()
        emb = torch.randn(16, 8, requires_grad=True)
        ts = torch.arange(16, dtype=torch.float32)
        loss_fn(embeddings=emb, timestamps=ts).backward()
        assert emb.grad is not None

    def test_bandwidth_configurable(self):
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        loss_fn = TemporalSmoothnessLoss(bandwidth=30.0)
        assert loss_fn.bandwidth == 30.0

    def test_has_name(self):
        from chronofy.embedding.losses import TemporalSmoothnessLoss

        assert TemporalSmoothnessLoss().name == "temporal_smoothness"


# =====================================================================
# AlignmentUniformityLoss
# =====================================================================


class TestAlignmentUniformityLoss:
    """Wang & Isola (ICML 2020) — alignment + uniformity on hypersphere."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import AlignmentUniformityLoss

        loss_fn = AlignmentUniformityLoss()
        x = torch.randn(16, 64)
        y = torch.randn(16, 64)
        assert loss_fn(x_positive=x, y_positive=y).dim() == 0

    def test_loss_components_accessible(self):
        from chronofy.embedding.losses import AlignmentUniformityLoss

        loss_fn = AlignmentUniformityLoss()
        x = torch.randn(16, 64)
        y = torch.randn(16, 64)
        total, components = loss_fn.forward_with_components(x_positive=x, y_positive=y)
        assert "alignment" in components
        assert "uniformity" in components

    def test_alpha_beta_configurable(self):
        from chronofy.embedding.losses import AlignmentUniformityLoss

        loss_fn = AlignmentUniformityLoss(alpha=0.5, beta=0.3)
        assert loss_fn.alpha == 0.5
        assert loss_fn.beta == 0.3

    def test_gradient_flows(self):
        from chronofy.embedding.losses import AlignmentUniformityLoss

        loss_fn = AlignmentUniformityLoss()
        x = torch.randn(16, 64, requires_grad=True)
        y = torch.randn(16, 64)
        loss_fn(x_positive=x, y_positive=y).backward()
        assert x.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import AlignmentUniformityLoss

        assert AlignmentUniformityLoss().name == "alignment_uniformity"


# =====================================================================
# MatryoshkaTruncationLoss
# =====================================================================


class TestMatryoshkaTruncationLoss:
    """Multi-scale self-distillation across truncation levels."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import MatryoshkaTruncationLoss

        loss_fn = MatryoshkaTruncationLoss(scales=[4, 8, 16])
        emb = torch.randn(16, 16)
        assert loss_fn(embeddings=emb).dim() == 0

    def test_loss_is_non_negative(self):
        from chronofy.embedding.losses import MatryoshkaTruncationLoss

        loss_fn = MatryoshkaTruncationLoss(scales=[4, 8, 16])
        emb = torch.randn(16, 16)
        assert loss_fn(embeddings=emb).item() >= 0.0

    def test_scales_configurable(self):
        from chronofy.embedding.losses import MatryoshkaTruncationLoss

        loss_fn = MatryoshkaTruncationLoss(scales=[2, 4, 8, 16, 32])
        assert loss_fn.scales == [2, 4, 8, 16, 32]

    def test_gradient_flows(self):
        from chronofy.embedding.losses import MatryoshkaTruncationLoss

        loss_fn = MatryoshkaTruncationLoss(scales=[4, 8, 16])
        emb = torch.randn(16, 16, requires_grad=True)
        loss_fn(embeddings=emb).backward()
        assert emb.grad is not None

    def test_has_name(self):
        from chronofy.embedding.losses import MatryoshkaTruncationLoss

        assert MatryoshkaTruncationLoss().name == "matryoshka_truncation"


# =====================================================================
# TMRLLoss (combined objective using CompositeLoss)
# =====================================================================


class TestTMRLLoss:
    """Combined TMRL objective: L = L_sem + α·L_temp + β·L_CKA."""

    def test_output_is_scalar(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        loss = loss_fn(
            full_embeddings=torch.randn(16, 64),
            truncated_embeddings=torch.randn(16, 16),
            temporal_embeddings=torch.randn(16, 16),
            timestamps=torch.arange(16, dtype=torch.float32),
            anchors=torch.randn(16, 64),
            positives=torch.randn(16, 64),
        )
        assert loss.dim() == 0

    def test_weights_configurable(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss(alpha_temporal=0.5, beta_cka=0.2)
        assert loss_fn.alpha_temporal == 0.5
        assert loss_fn.beta_cka == 0.2

    def test_gradient_flows(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        full = torch.randn(16, 64, requires_grad=True)
        trunc = full[:, :16]
        loss = loss_fn(
            full_embeddings=full,
            truncated_embeddings=trunc,
            temporal_embeddings=trunc,
            timestamps=torch.arange(16, dtype=torch.float32),
            anchors=torch.randn(16, 64),
            positives=torch.randn(16, 64),
        )
        loss.backward()
        assert full.grad is not None

    def test_forward_with_components(self):
        from chronofy.embedding.losses import TMRLLoss

        loss_fn = TMRLLoss()
        loss, components = loss_fn.forward_with_components(
            full_embeddings=torch.randn(16, 64),
            truncated_embeddings=torch.randn(16, 16),
            temporal_embeddings=torch.randn(16, 16),
            timestamps=torch.arange(16, dtype=torch.float32),
            anchors=torch.randn(16, 64),
            positives=torch.randn(16, 64),
        )
        assert "semantic" in components
        assert "temporal" in components
        assert "cka" in components


# =====================================================================
# Custom User-Defined Loss Plugin
# =====================================================================


class TestCustomLossPlugin:
    """Users should be able to define and register their own losses."""

    def test_custom_loss_registers_and_works(self):
        from chronofy.embedding.losses import TemporalLoss, LossRegistry

        class FocalTemporalLoss(TemporalLoss):
            """User-defined focal loss for temporal embeddings."""

            def __init__(self, gamma: float = 2.0):
                super().__init__()
                self.gamma = gamma

            @property
            def name(self) -> str:
                return "focal_temporal"

            def forward(self, *, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
                # Simplified focal-style: penalize uncertain embeddings
                norms = torch.norm(embeddings, dim=1)
                confidence = torch.sigmoid(norms)
                focal_weight = (1 - confidence) ** self.gamma
                return focal_weight.mean()

        registry = LossRegistry()
        registry.register(FocalTemporalLoss)

        instance = registry.create("focal_temporal", gamma=3.0)
        assert instance.gamma == 3.0

        emb = torch.randn(16, 8, requires_grad=True)
        loss = instance(embeddings=emb)
        assert loss.dim() == 0
        loss.backward()
        assert emb.grad is not None

    def test_custom_loss_in_composite(self):
        """Custom losses should work inside CompositeLoss."""
        from chronofy.embedding.losses import (
            TemporalLoss,
            CompositeLoss,
            CKALoss,
        )

        class MyRegLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "my_reg"

            def forward(self, *, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
                return embeddings.norm(dim=1).mean()

        composite = CompositeLoss()
        composite.add("cka", CKALoss(), weight=0.5)
        composite.add("reg", MyRegLoss(), weight=0.1)

        X = torch.randn(16, 32)
        Y = torch.randn(16, 16)
        emb = torch.randn(16, 8, requires_grad=True)

        total = composite(cka={"X": X, "Y": Y}, reg={"embeddings": emb})
        total.backward()
        assert emb.grad is not None
