"""Matryoshka truncation loss — multi-scale self-distillation.

Encourages truncated (smaller-dimensional) prefixes of embeddings to
remain faithful representations of the full-dimensional embedding.
Uses CKA at each truncation scale as the distillation signal.

Reference:
    Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022.
    Huynh et al., "TMRL: Temporal-aware Matryoshka Representation Learning", 2026.
"""

from __future__ import annotations

import torch

from chronofy.embedding.losses.base import TemporalLoss
from chronofy.embedding.losses.cka import CKALoss


class MatryoshkaTruncationLoss(TemporalLoss):
    """Multi-scale CKA distillation across truncation levels.

    For each scale s in scales, computes CKA(full_emb, emb[:, :s])
    and averages across scales. This ensures that information is
    distributed hierarchically: the first s dimensions form a good
    representation at every scale.

    Args:
        scales: List of truncation dimensions, ascending.
            Defaults to [4, 8, 16].
    """

    def __init__(self, scales: list[int] | None = None) -> None:
        super().__init__()
        self.scales = scales or [4, 8, 16]
        self._cka = CKALoss()

    @property
    def name(self) -> str:
        return "matryoshka_truncation"

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute average CKA loss across truncation scales.

        Args:
            embeddings: Full-dimensional embeddings (n, d).
                Each scale s truncates to embeddings[:, :s].

        Returns:
            Scalar average CKA loss.
        """
        d = embeddings.shape[1]
        total = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for s in self.scales:
            if s >= d:
                continue  # skip scales >= full dim (CKA would be trivially 0)
            truncated = embeddings[:, :s]
            total = total + self._cka(X=embeddings, Y=truncated)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return total / count
