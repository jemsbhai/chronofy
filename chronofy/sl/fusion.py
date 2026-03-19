"""Temporal evidence fusion for multi-source evidence.

Solves ASEV Axiom 6 (redundancy / diminishing returns) by combining
Jøsang's opinion aging with Subjective Logic fusion operators.

Pipeline:
    1. Each TemporalFact is decayed to an Opinion via OpinionDecayFunction
    2. (Optional) Byzantine filtering removes adversarial outliers
    3. Decayed opinions are fused via cumulative or averaging fusion

Cumulative fusion (⊕) — for independent sources:
    Combines evidence additively, reducing uncertainty. Adding a second
    independent lab reading tightens the estimate. Exhibits diminishing
    returns: the Nth observation reduces uncertainty less than the (N-1)th,
    satisfying ASEV Axiom 6 (concavity of mutual information).

Averaging fusion (⊘) — for correlated/dependent sources:
    Prevents double-counting when sources share underlying data.
    Idempotent: A ⊘ A = A — fusing the same evidence twice doesn't
    reduce uncertainty. Use when sources are not provably independent.

Byzantine fusion — decay → filter → fuse:
    Removes outlier opinions before fusion. Catches adversarial or
    erroneous evidence (e.g., a miscalibrated sensor, a fabricated
    reading) via pairwise conflict detection.

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §12.3 (Cumulative Fusion), §12.5 (Averaging Fusion),
    §10.4 (Opinion Aging), §12.3.4 (Conflict Detection).
    ASEV Evaluation, Axiom 6: Redundancy / Diminishing Returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from jsonld_ex.confidence_algebra import (
    Opinion,
    averaging_fuse,
    cumulative_fuse,
)
from jsonld_ex.confidence_byzantine import (
    ByzantineConfig,
    byzantine_fuse,
)

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


_VALID_METHODS = {"cumulative", "averaging"}


# ═══════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FusionReport:
    """Result of temporal evidence fusion.

    Carries the fused opinion, per-source decayed opinions, and
    diagnostics for transparency and downstream processing.

    Attributes:
        fused_opinion: The final fused Opinion after decay + fusion.
        decayed_opinions: Per-source opinions after temporal decay
            (same order as input facts), before fusion.
        source_count: Number of input evidence items.
        removed_count: Number of items removed by Byzantine filtering
            (0 if Byzantine mode is disabled).
        removed_indices: Indices of removed items in the original list.
        fusion_method: "cumulative" or "averaging".
        query_time: The reference time used for decay.
    """

    fused_opinion: Opinion
    decayed_opinions: list[Opinion]
    source_count: int
    removed_count: int = 0
    removed_indices: list[int] = field(default_factory=list)
    fusion_method: str = "cumulative"
    query_time: datetime = field(default_factory=datetime.now)

    @property
    def projected_probability(self) -> float:
        """Scalar projected probability P(ω) of the fused opinion.

        Convenience for consumers that need a single float.
        """
        return self.fused_opinion.projected_probability()


# ═══════════════════════════════════════════════════════════════════
# Fusion Engine
# ═══════════════════════════════════════════════════════════════════


class TemporalEvidenceFusion:
    """Decay → (optional Byzantine filter) → fuse pipeline.

    Given multiple TemporalFacts about the same proposition (e.g.,
    repeated potassium readings), decays each by age to an Opinion,
    optionally filters adversarial outliers, and fuses the remainder.

    Args:
        decay_fn: OpinionDecayFunction for converting facts to
            temporally-decayed Opinions.
        fusion_method: "cumulative" (independent sources, default) or
            "averaging" (correlated sources).
        byzantine: If True, apply Byzantine filtering before fusion.
        byzantine_threshold: Discord score threshold for Byzantine
            removal (default 0.15, per Jøsang §12.3.4).

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        fuser = TemporalEvidenceFusion(decay_fn=odf)
        report = fuser.fuse(potassium_readings, query_time=now)
        print(report.fused_opinion)       # Full (b, d, u, a)
        print(report.projected_probability)  # Scalar fallback
    """

    def __init__(
        self,
        decay_fn: OpinionDecayFunction,
        fusion_method: Literal["cumulative", "averaging"] = "cumulative",
        byzantine: bool = False,
        byzantine_threshold: float = 0.15,
    ) -> None:
        if fusion_method not in _VALID_METHODS:
            raise ValueError(
                f"fusion_method must be one of {_VALID_METHODS}, "
                f"got {fusion_method!r}"
            )
        self._decay_fn = decay_fn
        self._fusion_method = fusion_method
        self._byzantine = byzantine
        self._byzantine_threshold = byzantine_threshold

    @property
    def fusion_method(self) -> str:
        return self._fusion_method

    def fuse(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
    ) -> FusionReport:
        """Execute the decay → filter → fuse pipeline.

        Args:
            facts: Evidence items to fuse. All should be about the
                same proposition (e.g., multiple potassium readings).
            query_time: Reference time for temporal decay.

        Returns:
            FusionReport with the fused opinion and diagnostics.

        Raises:
            ValueError: If facts is empty.
        """
        if not facts:
            raise ValueError("fuse requires at least one fact")

        # Step 1: Decay each fact to an Opinion
        decayed = self._decay_fn.compute_opinion_batch(facts, query_time)

        # Step 2: Optional Byzantine filtering
        removed_indices: list[int] = []
        opinions_to_fuse = list(decayed)

        if self._byzantine and len(opinions_to_fuse) >= 3:
            byz_config = ByzantineConfig(
                strategy="most_conflicting",
                threshold=self._byzantine_threshold,
            )
            byz_report = byzantine_fuse(opinions_to_fuse, config=byz_config)
            removed_indices = list(byz_report.removed)
            # The byzantine_fuse returns the fused result directly,
            # but we want to control fusion method ourselves.
            # Reconstruct the filtered list.
            removed_set = set(removed_indices)
            opinions_to_fuse = [
                op for i, op in enumerate(opinions_to_fuse)
                if i not in removed_set
            ]

        # Step 3: Fuse
        if len(opinions_to_fuse) == 1:
            fused = opinions_to_fuse[0]
        elif self._fusion_method == "cumulative":
            fused = cumulative_fuse(*opinions_to_fuse)
        else:
            fused = averaging_fuse(*opinions_to_fuse)

        return FusionReport(
            fused_opinion=fused,
            decayed_opinions=decayed,
            source_count=len(facts),
            removed_count=len(removed_indices),
            removed_indices=removed_indices,
            fusion_method=self._fusion_method,
            query_time=query_time,
        )

    def __repr__(self) -> str:
        return (
            f"TemporalEvidenceFusion("
            f"method={self._fusion_method!r}, "
            f"byzantine={self._byzantine})"
        )
