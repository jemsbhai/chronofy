"""Triplet loss functions: standard margin-based and temporal hard mining.

TripletLoss: Standard triplet loss with configurable margin.
TemporalTripletLoss: Mines hard negatives based on temporal distance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chronofy.embedding.losses.base import TemporalLoss


class TripletLoss(TemporalLoss):
    """Standard margin-based triplet loss.

    L = max(0, d(a,p) - d(a,n) + margin)

    where d is Euclidean distance.

    Args:
        margin: Minimum desired gap between positive and negative distances.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    @property
    def name(self) -> str:
        return "triplet"

    def forward(
        self,
        *,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        losses = F.relu(d_pos - d_neg + self.margin)
        return losses.mean()


class TemporalTripletLoss(TemporalLoss):
    """Triplet loss with temporal hard negative mining.

    For each sample, the positive is the temporally closest other sample,
    and the hard negative is the sample that is temporally distant but
    closest in embedding space (hardest to distinguish).

    Args:
        margin: Triplet margin.
        proximity_threshold: Max temporal distance for positive selection.
    """

    def __init__(
        self,
        margin: float = 1.0,
        proximity_threshold: float = 7.0,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.proximity_threshold = proximity_threshold

    @property
    def name(self) -> str:
        return "temporal_triplet"

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        n = embeddings.shape[0]
        if n < 3:
            return torch.tensor(0.0, device=embeddings.device)

        # Temporal distance matrix
        time_diffs = torch.abs(timestamps.unsqueeze(1) - timestamps.unsqueeze(0))

        # Positive mask: temporally close (excluding self)
        pos_mask = (time_diffs <= self.proximity_threshold).float()
        pos_mask.fill_diagonal_(0.0)

        # Negative mask: temporally far
        neg_mask = (time_diffs > self.proximity_threshold).float()

        # Pairwise embedding distances
        dists = torch.cdist(embeddings, embeddings, p=2)

        # For each anchor:
        #   - positive: closest temporally (smallest time diff among positives)
        #   - hard negative: closest in embedding space among negatives
        total_loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(n):
            # Find positive: temporally closest
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue

            # Select temporally closest positive
            pos_time_diffs = time_diffs[i, pos_indices]
            best_pos_idx = pos_indices[pos_time_diffs.argmin()]

            # Find hard negative: closest in embedding space among negatives
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]
            if len(neg_indices) == 0:
                continue

            neg_dists = dists[i, neg_indices]
            hard_neg_idx = neg_indices[neg_dists.argmin()]

            d_pos = dists[i, best_pos_idx]
            d_neg = dists[i, hard_neg_idx]
            loss_i = F.relu(d_pos - d_neg + self.margin)
            total_loss = total_loss + loss_i
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return total_loss / count
