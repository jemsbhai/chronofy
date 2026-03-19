"""Temporal fine-tuner — LoRA-based training orchestrator.

Implements the TMRL training pipeline for the LearnedEncoder:
  1. Wraps encoder MLP layers with LoRA adapters
  2. Runs the combined TMRL loss (semantic + temporal + CKA)
  3. Supports multi-scale Matryoshka training
  4. Checkpoint save/load for full training state

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to the
encoder's linear layers, keeping the original weights frozen. This
enables parameter-efficient fine-tuning with minimal overhead.

Requires torch (installed via the [ml] extra).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chronofy.embedding.learned import LearnedEncoder
from chronofy.embedding.losses import TMRLLoss


# =====================================================================
# LoRA layer
# =====================================================================


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds a trainable low-rank decomposition BA to the frozen base weight:
        output = (W_frozen + B @ A) @ x + bias

    where A ∈ R^{rank × in}, B ∈ R^{out × rank}.

    Args:
        base_layer: The original nn.Linear to adapt.
        rank: Rank of the low-rank decomposition.
        alpha: LoRA scaling factor. Effective scaling = alpha / rank.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        base_out = self.base_layer(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


def apply_lora(module: nn.Module, rank: int, alpha: float = 1.0) -> nn.Module:
    """Replace all nn.Linear layers in a module with LoRA-wrapped versions.

    Args:
        module: The module to adapt.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        The module with Linear layers replaced by LoRALinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            apply_lora(child, rank=rank, alpha=alpha)
    return module


def get_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Collect only LoRA parameters (lora_A, lora_B) for optimizer."""
    params = []
    for name, param in module.named_parameters():
        if "lora_" in name:
            params.append(param)
    return params


# =====================================================================
# TemporalFineTuner
# =====================================================================


class TemporalFineTuner:
    """Training orchestrator for TMRL-style temporal encoder fine-tuning.

    Wraps a LearnedEncoder with LoRA adapters and trains it using the
    combined TMRL objective (semantic + temporal contrastive + CKA).

    Args:
        temporal_encoder: The LearnedEncoder to fine-tune.
        lora_rank: Rank for LoRA decomposition. Higher = more capacity.
        lora_alpha: LoRA scaling factor.
        learning_rate: Optimizer learning rate.
        alpha_temporal: Weight for temporal contrastive loss.
        beta_cka: Weight for CKA structural loss.
        matryoshka_scales: Truncation dimensions for multi-scale training.
            If None, uses [temporal_dims] (single scale).

    Example:
        >>> enc = LearnedEncoder(temporal_dims=16)
        >>> tuner = TemporalFineTuner(temporal_encoder=enc, lora_rank=4)
        >>> history = tuner.fit(train_data, epochs=10, batch_size=16)
    """

    def __init__(
        self,
        temporal_encoder: LearnedEncoder,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        learning_rate: float = 1e-3,
        alpha_temporal: float = 0.1,
        beta_cka: float = 0.1,
        matryoshka_scales: list[int] | None = None,
    ) -> None:
        self._encoder = temporal_encoder
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._learning_rate = learning_rate
        self._matryoshka_scales = matryoshka_scales or [
            temporal_encoder.temporal_dims
        ]

        # Apply LoRA to encoder MLP
        apply_lora(self._encoder.mlp, rank=lora_rank, alpha=lora_alpha)

        # Loss function
        self._loss_fn = TMRLLoss(
            alpha_temporal=alpha_temporal,
            beta_cka=beta_cka,
        )

        # Optimizer over LoRA params only
        lora_params = get_lora_parameters(self._encoder)
        if lora_params:
            self._optimizer = torch.optim.AdamW(
                lora_params, lr=learning_rate
            )
        else:
            # Fallback: train all encoder params if no LoRA was applied
            self._optimizer = torch.optim.AdamW(
                self._encoder.parameters(), lr=learning_rate
            )

    @property
    def temporal_encoder(self) -> LearnedEncoder:
        """The encoder being fine-tuned."""
        return self._encoder

    @property
    def lora_rank(self) -> int:
        """LoRA decomposition rank."""
        return self._lora_rank

    @property
    def learning_rate(self) -> float:
        """Optimizer learning rate."""
        return self._learning_rate

    @property
    def matryoshka_scales(self) -> list[int]:
        """Matryoshka truncation scales."""
        return self._matryoshka_scales

    def fit(
        self,
        train_data: dict[str, Any],
        epochs: int = 10,
        batch_size: int = 16,
    ) -> list[float]:
        """Train the temporal encoder on the provided data.

        Args:
            train_data: Dictionary with:
                - "texts": list[str] — text content (used for semantic pairs)
                - "timestamps": list[datetime] — observation timestamps
                - "positive_pairs": list[tuple[int, int]] — indices of
                    semantically similar pairs
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of per-epoch average losses (for monitoring convergence).
        """
        texts = train_data["texts"]
        timestamps = train_data["timestamps"]
        positive_pairs = train_data["positive_pairs"]
        n = len(texts)

        ref = self._encoder.reference_time

        # Precompute day offsets for temporal loss
        day_offsets = torch.tensor(
            [(t - ref).total_seconds() / 86400.0 for t in timestamps],
            dtype=torch.float32,
        )

        # Build positive pair index: for each anchor, its positive
        pair_map: dict[int, int] = {}
        for a, p in positive_pairs:
            pair_map[a] = p
            pair_map[p] = a  # symmetric

        self._encoder.train()
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(n).tolist()
            batch_losses: list[float] = []

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                if len(batch_idx) < 2:
                    continue

                # Extract features for this batch
                batch_timestamps = [timestamps[i] for i in batch_idx]
                features = self._encoder.extract_features(batch_timestamps)

                # Forward: full temporal embeddings
                full_emb = self._encoder.forward(features)
                t_dim = self._encoder.temporal_dims

                # Truncated embeddings (for CKA)
                trunc_emb = full_emb[:, : min(self._matryoshka_scales)]

                # Temporal subspace = full output (it IS the temporal embedding)
                temporal_emb = full_emb

                # Timestamps for this batch
                batch_days = day_offsets[batch_idx]

                # Build anchor/positive pairs for semantic loss
                # Use the embedding as both anchor and positive (self-supervised)
                # For pairs that have defined positives, swap them
                anchor_features = features.clone()
                positive_features = features.clone()
                for local_i, global_i in enumerate(batch_idx):
                    if global_i in pair_map:
                        partner = pair_map[global_i]
                        if partner < n:
                            positive_features[local_i] = (
                                self._encoder.extract_features(
                                    [timestamps[partner]]
                                )[0]
                            )

                anchors = self._encoder.forward(anchor_features)
                positives = self._encoder.forward(positive_features)

                # Combined loss
                loss = self._loss_fn(
                    full_embeddings=full_emb,
                    truncated_embeddings=trunc_emb,
                    temporal_embeddings=temporal_emb,
                    timestamps=batch_days,
                    anchors=anchors,
                    positives=positives,
                )

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                batch_losses.append(loss.item())

            avg_loss = sum(batch_losses) / max(len(batch_losses), 1)
            epoch_losses.append(avg_loss)

        self._encoder.eval()
        return epoch_losses

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training checkpoint.

        Includes encoder state_dict, optimizer state, and config.
        """
        checkpoint = {
            "encoder_config": self._encoder.get_config(),
            "encoder_state_dict": self._encoder.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "learning_rate": self._learning_rate,
            "matryoshka_scales": self._matryoshka_scales,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Restores encoder weights and optimizer state.
        """
        checkpoint = torch.load(path, weights_only=False)
        self._encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
