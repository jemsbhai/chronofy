"""Linear Centered Kernel Alignment loss.

CKA measures the similarity between two representations at the batch level.
Used as a structural regularizer in TMRL to ensure that truncated (temporal
subspace) embeddings remain geometrically aligned with full-dimensional
embeddings.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from chronofy.embedding.losses.base import TemporalLoss


class CKALoss(TemporalLoss):
    """Loss = 1 - CKA(X, Y). Minimizing drives X and Y toward alignment."""

    @property
    def name(self) -> str:
        return "cka"

    def cka_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear CKA similarity.

        Args:
            X: (n, d1), Y: (n, d2). Batch sizes must match.

        Returns:
            Scalar in [0, 1].
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"batch sizes must match, got X={X.shape[0]} and Y={Y.shape[0]}"
            )

        x_centered = X - X.mean(dim=0, keepdim=True)
        y_centered = Y - Y.mean(dim=0, keepdim=True)

        cross_covariance = x_centered.T @ y_centered
        cross = (cross_covariance * cross_covariance).sum()

        x_covariance = x_centered.T @ x_centered
        y_covariance = y_centered.T @ y_centered
        norm_x = (x_covariance * x_covariance).sum().sqrt()
        norm_y = (y_covariance * y_covariance).sum().sqrt()

        denom = norm_x * norm_y
        if denom < 1e-12:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        return cross / denom

    def forward(  # noqa: N803 - X/Y are retained as public keyword names.
        self,
        *,
        X: torch.Tensor,
        Y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return cast(torch.Tensor, 1.0 - self.cka_similarity(X, Y))
