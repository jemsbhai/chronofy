"""Chronofy loss function suite.

Comprehensive, extensible collection of loss functions for temporal
embedding training. All losses follow the TemporalLoss protocol and
can be composed via CompositeLoss or discovered via LossRegistry.

Built-in losses:
    cka                     — Linear CKA structural alignment
    temporal_contrastive    — InfoNCE with temporal proximity windowing
    semantic_contrastive    — Standard InfoNCE for anchor-positive pairs
    ntxent                  — Normalized Temperature-scaled Cross Entropy
    triplet                 — Margin-based triplet loss
    temporal_triplet        — Triplet with temporal hard negative mining
    temporal_ordering       — Pairwise ranking for before/after
    temporal_smoothness     — Gaussian-weighted embedding smoothness
    alignment_uniformity    — Wang & Isola alignment + uniformity
    matryoshka_truncation   — Multi-scale CKA self-distillation

Combined objectives:
    TMRLLoss                — L_sem + α·L_temp + β·L_CKA

Infrastructure:
    TemporalLoss            — ABC for all losses
    LossRegistry            — Plugin registration and discovery
    CompositeLoss           — Weighted ensemble of losses

Example::

    from chronofy.embedding.losses import (
        CompositeLoss, CKALoss, TemporalContrastiveLoss,
        TemporalSmoothnessLoss, default_registry,
    )

    # Build a custom ensemble
    composite = CompositeLoss()
    composite.add("cka", CKALoss(), weight=0.5)
    composite.add("tc", TemporalContrastiveLoss(temperature=0.05), weight=0.3)
    composite.add("smooth", TemporalSmoothnessLoss(bandwidth=14.0), weight=0.1)

    # Or use the registry
    cka = default_registry.create("cka")
"""

from chronofy.embedding.losses.base import (
    TemporalLoss,
    LossRegistry,
    CompositeLoss,
)
from chronofy.embedding.losses.cka import CKALoss
from chronofy.embedding.losses.contrastive import (
    TemporalContrastiveLoss,
    SemanticContrastiveLoss,
    NTXentLoss,
)
from chronofy.embedding.losses.triplet import TripletLoss, TemporalTripletLoss
from chronofy.embedding.losses.ordering import TemporalOrderingLoss
from chronofy.embedding.losses.smoothness import TemporalSmoothnessLoss
from chronofy.embedding.losses.alignment import AlignmentUniformityLoss
from chronofy.embedding.losses.matryoshka import MatryoshkaTruncationLoss
from chronofy.embedding.losses.tmrl import TMRLLoss

# ─── Default registry with all built-in losses ───────────────────────

default_registry = LossRegistry()

_BUILTINS = [
    CKALoss,
    TemporalContrastiveLoss,
    SemanticContrastiveLoss,
    NTXentLoss,
    TripletLoss,
    TemporalTripletLoss,
    TemporalOrderingLoss,
    TemporalSmoothnessLoss,
    AlignmentUniformityLoss,
    MatryoshkaTruncationLoss,
]

for _cls in _BUILTINS:
    default_registry.register(_cls)

__all__ = [
    # Infrastructure
    "TemporalLoss",
    "LossRegistry",
    "CompositeLoss",
    "default_registry",
    # Individual losses
    "CKALoss",
    "TemporalContrastiveLoss",
    "SemanticContrastiveLoss",
    "NTXentLoss",
    "TripletLoss",
    "TemporalTripletLoss",
    "TemporalOrderingLoss",
    "TemporalSmoothnessLoss",
    "AlignmentUniformityLoss",
    "MatryoshkaTruncationLoss",
    # Combined objectives
    "TMRLLoss",
]
