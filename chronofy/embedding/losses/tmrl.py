"""TMRL combined training objective.

L_total = L_semantic + α·L_temporal + β·L_CKA

Convenience class that assembles the three standard TMRL loss components
with configurable weights. Uses the individual loss modules internally.

Reference:
    Huynh et al., "TMRL: Temporal-aware Matryoshka Representation Learning", 2026.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chronofy.embedding.losses.cka import CKALoss
from chronofy.embedding.losses.contrastive import (
    SemanticContrastiveLoss,
    TemporalContrastiveLoss,
)


class TMRLLoss(nn.Module):
    """Combined TMRL objective: L = L_sem + α·L_temp + β·L_CKA.

    Args:
        alpha_temporal: Weight for temporal contrastive loss.
        beta_cka: Weight for CKA structural loss.
        temperature_semantic: Temperature for semantic InfoNCE.
        temperature_temporal: Temperature for temporal InfoNCE.
        proximity_threshold: Max temporal distance for positive pairs.
    """

    def __init__(
        self,
        alpha_temporal: float = 0.1,
        beta_cka: float = 0.1,
        temperature_semantic: float = 0.07,
        temperature_temporal: float = 0.1,
        proximity_threshold: float = 7.0,
    ) -> None:
        super().__init__()
        self.alpha_temporal = alpha_temporal
        self.beta_cka = beta_cka

        self.semantic_loss = SemanticContrastiveLoss(
            temperature=temperature_semantic
        )
        self.temporal_loss = TemporalContrastiveLoss(
            temperature=temperature_temporal,
            proximity_threshold=proximity_threshold,
        )
        self.cka_loss = CKALoss()

    def forward(
        self,
        *,
        full_embeddings: torch.Tensor,
        truncated_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        l_sem = self.semantic_loss(anchors=anchors, positives=positives)
        l_temp = self.temporal_loss(
            embeddings=temporal_embeddings, timestamps=timestamps
        )
        l_cka = self.cka_loss(X=full_embeddings, Y=truncated_embeddings)

        return l_sem + self.alpha_temporal * l_temp + self.beta_cka * l_cka

    def forward_with_components(
        self,
        *,
        full_embeddings: torch.Tensor,
        truncated_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_sem = self.semantic_loss(anchors=anchors, positives=positives)
        l_temp = self.temporal_loss(
            embeddings=temporal_embeddings, timestamps=timestamps
        )
        l_cka = self.cka_loss(X=full_embeddings, Y=truncated_embeddings)

        total = l_sem + self.alpha_temporal * l_temp + self.beta_cka * l_cka
        components = {
            "semantic": l_sem.item(),
            "temporal": l_temp.item(),
            "cka": l_cka.item(),
        }
        return total, components
