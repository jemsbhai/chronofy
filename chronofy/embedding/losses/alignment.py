"""Alignment and Uniformity loss on the hypersphere.

Directly optimizes the two properties that make contrastive representations
effective: alignment (positive pairs should be close) and uniformity
(embeddings should be spread on the hypersphere).

Reference:
    Wang & Isola, "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere", ICML 2020.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chronofy.embedding.losses.base import TemporalLoss


class AlignmentUniformityLoss(TemporalLoss):
    """L = α · L_align + β · L_uniform.

    L_align = E[||f(x) - f(x+)||^p]   (positive pairs should be close)
    L_uniform = log E[exp(-t·||f(x) - f(y)||^2)]  (push apart on sphere)

    Args:
        alpha: Weight for alignment loss.
        beta: Weight for uniformity loss.
        p: Exponent for alignment (default 2 = squared L2).
        t: Temperature for uniformity (default 2).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        p: float = 2.0,
        t: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.t = t

    @property
    def name(self) -> str:
        return "alignment_uniformity"

    def _alignment(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Alignment loss: mean ||x_i - y_i||^p over positive pairs."""
        return (x - y).norm(dim=1).pow(self.p).mean()

    def _uniformity(self, x: torch.Tensor) -> torch.Tensor:
        """Uniformity loss: log-mean-exp of pairwise distances."""
        sq_dists = torch.cdist(x, x, p=2).pow(2)
        n = x.shape[0]
        # Mask diagonal
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        sq_dists = sq_dists[mask].reshape(n, n - 1)
        return torch.logsumexp(-self.t * sq_dists, dim=1).mean()

    def forward(
        self,
        *,
        x_positive: torch.Tensor,
        y_positive: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x_norm = F.normalize(x_positive, dim=1)
        y_norm = F.normalize(y_positive, dim=1)

        l_align = self._alignment(x_norm, y_norm)
        l_uniform = (self._uniformity(x_norm) + self._uniformity(y_norm)) / 2.0

        return self.alpha * l_align + self.beta * l_uniform

    def forward_with_components(
        self,
        *,
        x_positive: torch.Tensor,
        y_positive: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x_norm = F.normalize(x_positive, dim=1)
        y_norm = F.normalize(y_positive, dim=1)

        l_align = self._alignment(x_norm, y_norm)
        l_uniform = (self._uniformity(x_norm) + self._uniformity(y_norm)) / 2.0

        total = self.alpha * l_align + self.beta * l_uniform
        components = {
            "alignment": l_align.item(),
            "uniformity": l_uniform.item(),
        }
        return total, components
