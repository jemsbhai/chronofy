"""Contrastive loss functions: temporal, semantic, and NTXent.

TemporalContrastiveLoss: InfoNCE on temporal subspace with proximity windowing.
SemanticContrastiveLoss: Standard InfoNCE for anchor-positive pairs.
NTXentLoss: Normalized Temperature-scaled Cross Entropy (SimCLR-style).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chronofy.embedding.losses.base import TemporalLoss


class TemporalContrastiveLoss(TemporalLoss):
    """InfoNCE contrastive loss on temporal embeddings.

    Pairs within a temporal proximity window are positives;
    all others in the batch are negatives.

    Args:
        temperature: Softmax temperature. Lower = sharper.
        proximity_threshold: Max distance (days) for positive pairs.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        proximity_threshold: float = 7.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.proximity_threshold = proximity_threshold

    @property
    def name(self) -> str:
        return "temporal_contrastive"

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n = embeddings.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb_norm = F.normalize(embeddings, dim=1)
        sim_matrix = emb_norm @ emb_norm.T / self.temperature

        time_diffs = torch.abs(timestamps.unsqueeze(1) - timestamps.unsqueeze(0))
        positive_mask = (time_diffs <= self.proximity_threshold).float()
        positive_mask.fill_diagonal_(0.0)

        if positive_mask.sum() < 1.0:
            return torch.tensor(0.0, device=embeddings.device)

        self_mask = torch.eye(n, device=embeddings.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))
        log_probs = F.log_softmax(sim_matrix, dim=1)

        # Use torch.where to avoid 0 * -inf = NaN
        zero = torch.zeros_like(log_probs)
        masked_log_probs = torch.where(positive_mask > 0, log_probs, zero)
        loss = -masked_log_probs.sum() / positive_mask.sum()

        return loss


class SemanticContrastiveLoss(TemporalLoss):
    """Standard InfoNCE for semantic embeddings with anchor-positive pairs.

    Args:
        temperature: Softmax temperature.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "semantic_contrastive"

    def forward(
        self,
        *,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n = anchors.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=anchors.device)

        a_norm = F.normalize(anchors, dim=1)
        p_norm = F.normalize(positives, dim=1)

        all_sim = a_norm @ p_norm.T / self.temperature
        log_probs = F.log_softmax(all_sim, dim=1)
        loss = -log_probs.diag().mean()

        return loss


class NTXentLoss(TemporalLoss):
    """Normalized Temperature-scaled Cross Entropy (SimCLR-style).

    Given two views (z_i, z_j) of the same batch, treats (z_i[k], z_j[k])
    as positive pairs and all other combinations as negatives.

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning
        of Visual Representations", ICML 2020.

    Args:
        temperature: Softmax temperature.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "ntxent"

    def forward(
        self,
        *,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n = z_i.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=z_i.device)

        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)

        # Concatenate both views: [z_i; z_j] → shape (2n, d)
        z = torch.cat([z_i_norm, z_j_norm], dim=0)

        # Full similarity matrix: (2n, 2n)
        sim = z @ z.T / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+n) and (i+n, i)
        labels = torch.cat([
            torch.arange(n, 2 * n, device=z.device),
            torch.arange(0, n, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss
