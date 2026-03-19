"""Temporal smoothness loss — nearby timestamps should produce similar embeddings.

Uses a Gaussian kernel over temporal distances as soft weights:
the closer two facts are in time, the more their embeddings should agree.
Penalizes large embedding differences between temporally proximate facts.
"""

from __future__ import annotations

import torch

from chronofy.embedding.losses.base import TemporalLoss


class TemporalSmoothnessLoss(TemporalLoss):
    """Weighted MSE loss where weights are Gaussian over temporal distance.

    L = Σ_{i,j} w(t_i, t_j) · ||e_i - e_j||^2 / Σ w

    where w(t_i, t_j) = exp(-|t_i - t_j|^2 / (2·bandwidth^2))

    This encourages embeddings to vary smoothly with time — small temporal
    steps should produce small embedding changes.

    Args:
        bandwidth: Gaussian kernel bandwidth in days. Controls how far
            the smoothness constraint reaches.
    """

    def __init__(self, bandwidth: float = 7.0) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    @property
    def name(self) -> str:
        return "temporal_smoothness"

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

        # Temporal distance matrix: (n, n)
        time_diffs = torch.abs(timestamps.unsqueeze(1) - timestamps.unsqueeze(0))

        # Gaussian kernel weights
        weights = torch.exp(-time_diffs ** 2 / (2.0 * self.bandwidth ** 2))

        # Zero out diagonal (self-comparison)
        weights.fill_diagonal_(0.0)

        weight_sum = weights.sum()
        if weight_sum < 1e-12:
            return torch.tensor(0.0, device=embeddings.device)

        # Pairwise squared L2 distances: ||e_i - e_j||^2
        # Using the expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·b
        sq_norms = (embeddings ** 2).sum(dim=1)  # (n,)
        pairwise_sq = (
            sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * embeddings @ embeddings.T
        )
        # Clamp to avoid negative values from numerical error
        pairwise_sq = pairwise_sq.clamp(min=0.0)

        # Weighted average
        loss = (weights * pairwise_sq).sum() / weight_sum

        return loss
