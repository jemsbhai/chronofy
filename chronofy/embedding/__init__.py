"""Layer 1: Temporal Subspace Embeddings.

Implements the paper's embedding layer: e = [e_temp ; e_sem] where the
first t dimensions encode temporal information and the remaining d-t
dimensions encode semantic content. This structural separation ensures
that fact age is irremovable from the representation regardless of
truncation level.

Architecture:
    TemporalEncoder (ABC)          ← timestamp → R^t
      └── SinusoidalEncoder        ← sinusoidal positional encoding (no training)

    TemporalEmbedder               ← orchestrator: [e_temp ; e_sem]
"""

from chronofy.embedding.base import TemporalEncoder
from chronofy.embedding.sinusoidal import SinusoidalEncoder
from chronofy.embedding.embedder import TemporalEmbedder

# Torch-dependent imports (available with [ml] extra)
try:
    from chronofy.embedding.learned import LearnedEncoder
    from chronofy.embedding.losses import (
        CKALoss,
        TemporalContrastiveLoss,
        SemanticContrastiveLoss,
        TMRLLoss,
    )
    from chronofy.embedding.fine_tuner import TemporalFineTuner

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

__all__ = ["TemporalEncoder", "SinusoidalEncoder", "TemporalEmbedder"]

if _ML_AVAILABLE:
    __all__ += [
        "LearnedEncoder",
        "CKALoss",
        "TemporalContrastiveLoss",
        "SemanticContrastiveLoss",
        "TMRLLoss",
        "TemporalFineTuner",
    ]
