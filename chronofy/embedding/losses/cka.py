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

        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)

        XtY = X_c.T @ Y_c
        cross = (XtY * XtY).sum()

        XtX = X_c.T @ X_c
        YtY = Y_c.T @ Y_c
        norm_x = (XtX * XtX).sum().sqrt()
        norm_y = (YtY * YtY).sum().sqrt()

        denom = norm_x * norm_y
        if denom < 1e-12:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        return cross / denom

    def forward(self, *, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> torch.Tensor:
        return 1.0 - self.cka_similarity(X, Y)
