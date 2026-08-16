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
from chronofy.embedding.embedder import TemporalEmbedder
from chronofy.embedding.sinusoidal import SinusoidalEncoder

# Torch-dependent imports (available with [ml] extra). Probe only the optional
# package here so defects in Chronofy's own ML modules are never hidden.
try:
    import torch as _torch  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _ML_AVAILABLE = False
else:
    from chronofy.embedding.fine_tuner import TemporalFineTuner
    from chronofy.embedding.learned import LearnedEncoder
    from chronofy.embedding.losses import (
        AlignmentUniformityLoss,
        CKALoss,
        CompositeLoss,
        LossRegistry,
        MatryoshkaTruncationLoss,
        NTXentLoss,
        SemanticContrastiveLoss,
        TemporalContrastiveLoss,
        TemporalLoss,
        TemporalOrderingLoss,
        TemporalSmoothnessLoss,
        TemporalTripletLoss,
        TMRLLoss,
        TripletLoss,
        default_registry,
    )

    _ML_AVAILABLE = True

__all__ = ["TemporalEncoder", "SinusoidalEncoder", "TemporalEmbedder"]

if _ML_AVAILABLE:
    __all__ += [
        "LearnedEncoder",
        "TemporalLoss",
        "LossRegistry",
        "CompositeLoss",
        "default_registry",
        "CKALoss",
        "TemporalContrastiveLoss",
        "SemanticContrastiveLoss",
        "NTXentLoss",
        "TripletLoss",
        "TemporalTripletLoss",
        "TemporalOrderingLoss",
        "TemporalSmoothnessLoss",
        "AlignmentUniformityLoss",
        "MatryoshkaTruncationLoss",
        "TMRLLoss",
        "TemporalFineTuner",
    ]
