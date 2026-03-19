"""Loss functions for TMRL-style temporal embedding training.

Implements three loss components that together form the TMRL training
objective (Huynh et al., 2026):

    L_total = L_semantic + α·L_temporal + β·L_CKA

Components:
    CKALoss: Linear Centered Kernel Alignment — structural self-distillation
        that keeps truncated embeddings geometrically aligned with full-dim
        embeddings at the batch level.

    TemporalContrastiveLoss: InfoNCE on the temporal subspace — pushes
        embeddings of temporally proximate facts together and temporally
        distant facts apart.

    TMRLLoss: Combined multi-scale objective orchestrating all three
        components with configurable weights.

Requires torch (installed via the [ml] extra).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CKALoss(nn.Module):
    """Linear Centered Kernel Alignment loss.

    CKA measures the similarity between two representations at the batch
    level. Used as a structural regularizer in TMRL to ensure that
    truncated (temporal subspace) embeddings remain geometrically aligned
    with full-dimensional embeddings.

    Loss = 1 - CKA(X, Y), so minimizing drives X and Y toward alignment.

    Reference:
        Kornblith et al., "Similarity of Neural Network Representations
        Revisited", ICML 2019.
    """

    def cka_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear CKA similarity between two representation matrices.

        Args:
            X: Tensor of shape (n, d1) — first representation.
            Y: Tensor of shape (n, d2) — second representation.

        Returns:
            Scalar tensor in [0, 1]. 1.0 means identical geometry.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"batch sizes must match, got X={X.shape[0]} and Y={Y.shape[0]}"
            )

        # Center the columns
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)

        # Cross-covariance: ||X^T Y||_F^2
        # Equivalent to trace((X^T X)(Y^T Y)) but computed via X^T Y
        XtY = X_c.T @ Y_c
        cross = (XtY * XtY).sum()

        # Self-covariances: ||X^T X||_F and ||Y^T Y||_F
        XtX = X_c.T @ X_c
        YtY = Y_c.T @ Y_c
        norm_x = (XtX * XtX).sum().sqrt()
        norm_y = (YtY * YtY).sum().sqrt()

        denom = norm_x * norm_y
        if denom < 1e-12:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)

        return cross / denom

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute CKA loss = 1 - CKA(X, Y).

        Args:
            X: Tensor of shape (n, d1).
            Y: Tensor of shape (n, d2).

        Returns:
            Scalar loss in [0, 1]. 0.0 when X and Y are aligned.
        """
        return 1.0 - self.cka_similarity(X, Y)


class TemporalContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss on temporal embeddings.

    Pairs of facts within a temporal proximity window are treated as
    positives; all other pairs in the batch are negatives. This
    encourages the temporal subspace to cluster temporally close facts.

    Args:
        temperature: Softmax temperature for InfoNCE. Lower = sharper.
        proximity_threshold: Maximum temporal distance (in days) for
            two facts to be considered a positive pair.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        proximity_threshold: float = 7.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.proximity_threshold = proximity_threshold

    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal contrastive loss.

        Args:
            embeddings: Tensor of shape (n, t) — temporal subspace vectors.
            timestamps: Tensor of shape (n,) — day offsets for each sample.

        Returns:
            Scalar InfoNCE loss.
        """
        n = embeddings.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        emb_norm = F.normalize(embeddings, dim=1)

        # Cosine similarity matrix: (n, n)
        sim_matrix = emb_norm @ emb_norm.T / self.temperature

        # Temporal distance matrix: |t_i - t_j|
        time_diffs = torch.abs(
            timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
        )

        # Positive mask: temporally close pairs (excluding self)
        positive_mask = (time_diffs <= self.proximity_threshold).float()
        positive_mask.fill_diagonal_(0.0)

        # If no positive pairs exist, return zero loss
        if positive_mask.sum() < 1.0:
            return torch.tensor(0.0, device=embeddings.device)

        # Self-mask: exclude diagonal from denominator
        self_mask = torch.eye(n, device=embeddings.device, dtype=torch.bool)

        # InfoNCE: for each anchor, sum over positives / sum over all (excl self)
        # log_softmax over non-self entries
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))
        log_probs = F.log_softmax(sim_matrix, dim=1)

        # Average log prob of positive pairs.
        # IMPORTANT: use torch.where to avoid 0 * -inf = NaN on the diagonal
        # (positive_mask is 0 on diagonal, log_probs is -inf on diagonal).
        zero = torch.zeros_like(log_probs)
        masked_log_probs = torch.where(positive_mask > 0, log_probs, zero)
        loss = -masked_log_probs.sum() / positive_mask.sum()

        return loss


class SemanticContrastiveLoss(nn.Module):
    """Standard InfoNCE for semantic embeddings with positive pairs.

    Used as the base semantic MRL loss component. Takes pre-defined
    anchor-positive pairs and treats all other batch elements as negatives.

    Args:
        temperature: Softmax temperature.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute semantic contrastive loss.

        Args:
            anchors: Tensor of shape (n, d) — anchor embeddings.
            positives: Tensor of shape (n, d) — positive match for each anchor.

        Returns:
            Scalar InfoNCE loss.
        """
        n = anchors.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=anchors.device)

        # Normalize
        a_norm = F.normalize(anchors, dim=1)
        p_norm = F.normalize(positives, dim=1)

        # Positive similarities: diagonal of (anchors @ positives.T)
        pos_sim = (a_norm * p_norm).sum(dim=1) / self.temperature

        # All-pairs similarity: anchors vs all positives as candidates
        all_sim = a_norm @ p_norm.T / self.temperature

        # InfoNCE: positive is the diagonal entry
        # log_softmax across columns, take diagonal
        log_probs = F.log_softmax(all_sim, dim=1)
        loss = -log_probs.diag().mean()

        return loss


class TMRLLoss(nn.Module):
    """Combined TMRL training objective.

    L_total = L_semantic + α·L_temporal + β·L_CKA

    Where:
        L_semantic: InfoNCE on full embeddings (semantic contrastive)
        L_temporal: InfoNCE on temporal subspace (temporal contrastive)
        L_CKA: CKA alignment between truncated and full embeddings

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
        full_embeddings: torch.Tensor,
        truncated_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined TMRL loss.

        Args:
            full_embeddings: Full-dim embeddings (n, d) for CKA.
            truncated_embeddings: Truncated embeddings (n, t) for CKA.
            temporal_embeddings: Temporal subspace vectors (n, t) for
                temporal contrastive loss.
            timestamps: Day offsets (n,) for temporal pairs.
            anchors: Anchor embeddings (n, d) for semantic loss.
            positives: Positive embeddings (n, d) for semantic loss.

        Returns:
            Scalar combined loss.
        """
        l_sem = self.semantic_loss(anchors, positives)
        l_temp = self.temporal_loss(temporal_embeddings, timestamps)
        l_cka = self.cka_loss(full_embeddings, truncated_embeddings)

        return l_sem + self.alpha_temporal * l_temp + self.beta_cka * l_cka

    def forward_with_components(
        self,
        full_embeddings: torch.Tensor,
        truncated_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss and return individual components.

        Returns:
            Tuple of (total_loss, {"semantic": ..., "temporal": ..., "cka": ...})
        """
        l_sem = self.semantic_loss(anchors, positives)
        l_temp = self.temporal_loss(temporal_embeddings, timestamps)
        l_cka = self.cka_loss(full_embeddings, truncated_embeddings)

        total = l_sem + self.alpha_temporal * l_temp + self.beta_cka * l_cka

        components = {
            "semantic": l_sem.item(),
            "temporal": l_temp.item(),
            "cka": l_cka.item(),
        }
        return total, components
