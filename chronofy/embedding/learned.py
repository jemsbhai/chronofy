"""Learned temporal encoder — MLP-based temporal projector.

A trainable nn.Module that maps timestamp features to dense vectors in R^t,
forming the temporal subspace of the combined embedding. This is the
learnable counterpart to SinusoidalEncoder.

The encoder extracts rich features from timestamps (day offset, cyclical
encodings of day-of-week, month, hour, year fraction) and passes them
through a small MLP to produce the temporal embedding.

Requires torch (installed via the [ml] extra).

Architecture:
    timestamp → extract_features() → [offset, sin/cos cycles] ∈ R^f
              → MLP(f → hidden → temporal_dims) ∈ R^t

Save/load is supported via state_dict checkpoints that include both
model weights and construction config for reconstruction.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from chronofy.embedding.base import TemporalEncoder

_DEFAULT_REFERENCE = datetime(2020, 1, 1)


class LearnedEncoder(TemporalEncoder, nn.Module):
    """Trainable MLP temporal encoder.

    Maps timestamps to dense vectors via a two-layer MLP over hand-crafted
    temporal features. Subclasses both TemporalEncoder (for Chronofy API
    compatibility) and nn.Module (for torch training).

    Args:
        temporal_dims: Output dimensionality of the temporal subspace.
        hidden_dims: Width of the hidden layer. Defaults to 4× temporal_dims.
        reference_time: Epoch for computing relative day offsets.
        dropout: Dropout rate between hidden layers.

    Example:
        >>> enc = LearnedEncoder(temporal_dims=16)
        >>> vecs = enc.encode([datetime(2025, 6, 15)])  # numpy
        >>> vecs.shape
        (1, 16)
    """

    # Number of raw features extracted per timestamp:
    # 1 (day offset, normalized) + 2 (day-of-week sin/cos) + 2 (month sin/cos)
    # + 2 (hour sin/cos) + 2 (year-fraction sin/cos) = 9
    _NUM_INPUT_FEATURES = 9

    def __init__(
        self,
        temporal_dims: int = 16,
        hidden_dims: int | None = None,
        reference_time: datetime | None = None,
        dropout: float = 0.1,
    ) -> None:
        if temporal_dims <= 0:
            raise ValueError(
                f"temporal_dims must be positive, got {temporal_dims}"
            )

        # nn.Module init must come first for proper parameter registration
        nn.Module.__init__(self)

        self._temporal_dims = temporal_dims
        self._hidden_dims = hidden_dims or temporal_dims * 4
        self._reference_time = reference_time or _DEFAULT_REFERENCE
        self._dropout = dropout

        # MLP: input_features → hidden → temporal_dims
        self.mlp = nn.Sequential(
            nn.Linear(self._NUM_INPUT_FEATURES, self._hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self._hidden_dims, self._hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self._hidden_dims, temporal_dims),
            nn.Tanh(),  # bound output to [-1, 1] like sinusoidal
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for stable training."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def temporal_dims(self) -> int:
        """Output dimensionality of the temporal subspace."""
        return self._temporal_dims

    @property
    def hidden_dims(self) -> int:
        """Width of the hidden layer."""
        return self._hidden_dims

    @property
    def reference_time(self) -> datetime:
        """Reference datetime for computing day offsets."""
        return self._reference_time

    @property
    def num_input_features(self) -> int:
        """Number of raw features extracted per timestamp."""
        return self._NUM_INPUT_FEATURES

    def extract_features(self, timestamps: list[datetime]) -> torch.Tensor:
        """Extract rich temporal features from timestamps.

        Features:
            0: normalized day offset from reference (days / 3650)
            1-2: sin/cos of day-of-week (period = 7 days)
            3-4: sin/cos of month (period = 12 months)
            5-6: sin/cos of hour (period = 24 hours)
            7-8: sin/cos of year fraction (period = 365.25 days)

        Args:
            timestamps: List of datetime objects.

        Returns:
            Tensor of shape (n, num_input_features).
        """
        if len(timestamps) == 0:
            return torch.empty(
                (0, self._NUM_INPUT_FEATURES), dtype=torch.float32
            )

        features = []
        for t in timestamps:
            delta_days = (t - self._reference_time).total_seconds() / 86400.0

            # Normalized offset (scaled so ~10 years → 1.0)
            offset = delta_days / 3650.0

            # Day-of-week cyclical (0=Monday, 6=Sunday)
            dow = t.weekday()
            dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
            dow_cos = math.cos(2.0 * math.pi * dow / 7.0)

            # Month cyclical
            month = t.month - 1  # 0-indexed
            month_sin = math.sin(2.0 * math.pi * month / 12.0)
            month_cos = math.cos(2.0 * math.pi * month / 12.0)

            # Hour cyclical
            hour = t.hour + t.minute / 60.0
            hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
            hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

            # Year fraction cyclical
            day_of_year = t.timetuple().tm_yday
            year_frac_sin = math.sin(2.0 * math.pi * day_of_year / 365.25)
            year_frac_cos = math.cos(2.0 * math.pi * day_of_year / 365.25)

            features.append([
                offset,
                dow_sin, dow_cos,
                month_sin, month_cos,
                hour_sin, hour_cos,
                year_frac_sin, year_frac_cos,
            ])

        return torch.tensor(features, dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass: features → temporal embedding.

        Args:
            features: Tensor of shape (n, num_input_features) from
                extract_features().

        Returns:
            Tensor of shape (n, temporal_dims).
        """
        return self.mlp(features)

    def encode(
        self,
        timestamps: list[datetime],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode timestamps into temporal vectors (ABC compliance).

        This is the TemporalEncoder interface method. It extracts features,
        runs the MLP forward pass, and returns numpy.

        Args:
            timestamps: List of datetime objects.
            reference_time: Ignored (uses instance reference_time).
                Accepted for ABC signature compatibility.

        Returns:
            np.ndarray of shape (len(timestamps), temporal_dims).
        """
        if len(timestamps) == 0:
            return np.empty((0, self._temporal_dims), dtype=np.float64)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            features = self.extract_features(timestamps)
            output = self.forward(features)
        if was_training:
            self.train()

        return output.cpu().numpy()

    def encode_facts(
        self,
        facts: list,
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Convenience: encode timestamps extracted from TemporalFacts."""
        timestamps = [f.timestamp for f in facts]
        return self.encode(timestamps, reference_time=reference_time)

    def get_config(self) -> dict[str, Any]:
        """Return construction config for serialization."""
        return {
            "temporal_dims": self._temporal_dims,
            "hidden_dims": self._hidden_dims,
            "reference_time": self._reference_time.isoformat(),
            "dropout": self._dropout,
        }

    def save(self, path: str | Path) -> None:
        """Save encoder weights and config to a checkpoint file.

        Args:
            path: Destination file path (e.g., "encoder.pt").
        """
        checkpoint = {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        """Load encoder weights from a checkpoint file.

        Args:
            path: Source file path. Must match this encoder's config.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LearnedEncoder":
        """Reconstruct a LearnedEncoder from a saved checkpoint.

        Args:
            path: Source file path containing config and state_dict.

        Returns:
            A new LearnedEncoder with restored weights.
        """
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]
        encoder = cls(
            temporal_dims=config["temporal_dims"],
            hidden_dims=config["hidden_dims"],
            reference_time=datetime.fromisoformat(config["reference_time"]),
            dropout=config["dropout"],
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        return encoder

    def __repr__(self) -> str:
        return (
            f"LearnedEncoder(temporal_dims={self._temporal_dims}, "
            f"hidden_dims={self._hidden_dims}, "
            f"dropout={self._dropout})"
        )
