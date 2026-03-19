"""Temporal ordering loss — learns before/after relationships.

Encourages the temporal subspace to encode directional temporal
information: given two embeddings, a learned function should predict
which one is earlier. Uses a pairwise margin ranking loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chronofy.embedding.losses.base import TemporalLoss


class TemporalOrderingLoss(TemporalLoss):
    """Pairwise ranking loss for temporal ordering.

    For sampled pairs (i, j) where t_i < t_j, encourages:
        score(e_i) < score(e_j)

    where score is the L2 norm projected onto a direction vector.
    Uses MarginRankingLoss under the hood.

    Args:
        margin: Minimum desired gap in scores between earlier and later.
        num_pairs: Number of random pairs to sample per batch.
    """

    def __init__(self, margin: float = 0.1, num_pairs: int | None = None) -> None:
        super().__init__()
        self.margin = margin
        self.num_pairs = num_pairs

    @property
    def name(self) -> str:
        return "temporal_ordering"

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

        # Use the mean of embedding dimensions as a scalar "temporal score"
        scores = embeddings.mean(dim=1)  # (n,)

        # Sample pairs where t_i < t_j
        num_pairs = self.num_pairs or min(n * (n - 1) // 2, n * 4)

        # Generate random pairs
        idx_i = torch.randint(0, n, (num_pairs,), device=embeddings.device)
        idx_j = torch.randint(0, n, (num_pairs,), device=embeddings.device)

        # Ensure i != j
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Determine ordering: target = +1 if t_i < t_j, -1 otherwise
        t_i = timestamps[idx_i]
        t_j = timestamps[idx_j]
        target = torch.sign(t_j - t_i)

        # Remove ties
        non_tie = target != 0
        if non_tie.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        s_i = scores[idx_i][non_tie]
        s_j = scores[idx_j][non_tie]
        target = target[non_tie]

        loss = F.margin_ranking_loss(s_i, s_j, target, margin=self.margin)
        return loss
