"""Pairwise conflict detection in retrieved evidence sets.

Detects contradictory evidence BEFORE fusion or LLM consumption,
enabling informed decisions about whether to fuse, discard, or
flag evidence for human review.

Pipeline:
    1. Each TemporalFact is decayed to an Opinion via OpinionDecayFunction
    2. Pairwise conflict matrix computed (Jøsang §12.3.4)
    3. Discord scores computed (mean conflict per opinion)
    4. Conflict pairs flagged above threshold
    5. Cohesion score summarizes group agreement

Mathematical basis (Jøsang §12.3.4):
    Pairwise conflict between opinions A and B:
        con(A, B) = b_A · d_B + d_A · b_B

    Properties:
        - Range: [0, 1]
        - Symmetry: con(A, B) = con(B, A)
        - Zero when opinions agree (same direction)
        - Maximum (1.0) when dogmatically opposed
        - Zero when either opinion is vacuous (no evidence to disagree)

    Internal conflict (balance) for a single opinion:
        conflict_metric(ω) = 1 − |b − d| − u
        Distinguishes contradiction (high b AND d) from ignorance (high u).

Design:
    Standalone module (SRP) — not integrated into TemporalEvidenceFusion.
    Users call ConflictDetector before fusion if desired. The detect()
    method handles the full decay → detect pipeline; detect_from_opinions()
    accepts pre-decayed opinions for users coming from other pipelines
    (e.g., TrustWeightedDecay).

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §12.3.4 (Pairwise Conflict, Discord).
    ASEV Evaluation, Axiom 6: conflict detection is a prerequisite
        for safe fusion — contradictory evidence must be identified
        before combining sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jsonld_ex.confidence_algebra import (
    Opinion,
    conflict_metric,
    pairwise_conflict,
)

from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.models import TemporalFact


# ═══════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConflictReport:
    """Result of pairwise conflict analysis over an evidence set.

    Carries the full conflict matrix, flagged pairs, per-opinion
    discord and internal conflict scores, and an aggregate cohesion
    measure for downstream decision-making.

    Attributes:
        conflict_matrix: n×n symmetric matrix of pairwise conflict
            scores. Diagonal is 0 by convention (pairwise conflict
            is between distinct opinions).
        conflict_pairs: List of (i, j, score) tuples for pairs
            exceeding the threshold, sorted by score descending.
            Canonical ordering: i < j.
        discord_scores: Per-opinion mean pairwise conflict against
            all other opinions. High discord = outlier.
        internal_conflicts: Per-opinion internal conflict score
            via Jøsang's conflict_metric. Measures self-contradiction
            (high b AND high d), distinct from pairwise disagreement.
        cohesion_score: Aggregate group agreement in [0, 1].
            Defined as 1.0 − max(discord_scores). A perfectly
            agreeing group scores 1.0; maximal disagreement scores 0.0.
        decayed_opinions: The opinions analyzed (after temporal decay
            if detect() was used; as-provided if detect_from_opinions()).
        source_count: Number of evidence items analyzed.
        threshold: The conflict threshold used to flag pairs.
    """

    conflict_matrix: list[list[float]]
    conflict_pairs: list[tuple[int, int, float]]
    discord_scores: list[float]
    internal_conflicts: list[float]
    cohesion_score: float
    decayed_opinions: list[Opinion]
    source_count: int
    threshold: float


# ═══════════════════════════════════════════════════════════════════
# Detector
# ═══════════════════════════════════════════════════════════════════


class ConflictDetector:
    """Pairwise conflict detection for retrieved evidence sets.

    Given a set of TemporalFacts (or pre-decayed Opinions), computes
    the full pairwise conflict matrix, identifies outlier opinions
    via discord scores, and flags contradictory pairs above a
    configurable threshold.

    Use before fusion to diagnose whether evidence is safe to combine,
    or before LLM consumption to flag contradictory retrieval results.

    Args:
        decay_fn: OpinionDecayFunction for converting facts to
            temporally-decayed Opinions. Required for detect();
            not used by detect_from_opinions().
        default_threshold: Default conflict threshold for flagging
            pairs. Must be in [0, 1]. Default 0.15.

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        det = ConflictDetector(decay_fn=odf)
        report = det.detect(potassium_readings, query_time=now)
        if report.cohesion_score < 0.5:
            print("WARNING: contradictory evidence detected")
            for i, j, score in report.conflict_pairs:
                print(f"  Facts {i} vs {j}: conflict = {score:.3f}")
    """

    def __init__(
        self,
        decay_fn: OpinionDecayFunction,
        default_threshold: float = 0.15,
    ) -> None:
        if not 0.0 <= default_threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {default_threshold}"
            )
        self._decay_fn = decay_fn
        self._default_threshold = default_threshold

    def detect(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
        threshold: float | None = None,
    ) -> ConflictReport:
        """Decay facts to opinions, then compute conflict analysis.

        Full pipeline: TemporalFact → OpinionDecayFunction → conflict matrix.

        Args:
            facts: Evidence items to analyze.
            query_time: Reference time for temporal decay.
            threshold: Conflict threshold for flagging pairs.
                If None, uses the detector's default_threshold.

        Returns:
            ConflictReport with full diagnostics.

        Raises:
            ValueError: If facts is empty.
        """
        if not facts:
            raise ValueError("detect requires at least one fact")

        opinions = self._decay_fn.compute_opinion_batch(facts, query_time)
        effective_threshold = threshold if threshold is not None else self._default_threshold

        return self._analyze(opinions, effective_threshold)

    @staticmethod
    def detect_from_opinions(
        opinions: list[Opinion],
        threshold: float = 0.15,
    ) -> ConflictReport:
        """Compute conflict analysis on pre-decayed opinions.

        Static method for users who already have Opinions from another
        pipeline (e.g., TrustWeightedDecay, manual construction).

        Args:
            opinions: Pre-decayed opinions to analyze.
            threshold: Conflict threshold for flagging pairs.

        Returns:
            ConflictReport with full diagnostics.

        Raises:
            ValueError: If opinions is empty.
        """
        if not opinions:
            raise ValueError("detect_from_opinions requires at least one opinion")

        return ConflictDetector._analyze(opinions, threshold)

    @staticmethod
    def _analyze(
        opinions: list[Opinion],
        threshold: float,
    ) -> ConflictReport:
        """Core conflict analysis on a list of opinions.

        Computes:
            1. n×n pairwise conflict matrix (symmetric, diagonal = 0)
            2. Discord scores (mean conflict per opinion)
            3. Internal conflict scores (self-contradiction per opinion)
            4. Conflict pairs above threshold (sorted descending)
            5. Cohesion score (1 − max discord)
        """
        n = len(opinions)

        # --- Pairwise conflict matrix ---
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                score = pairwise_conflict(opinions[i], opinions[j])
                matrix[i][j] = score
                matrix[j][i] = score

        # --- Discord scores: mean conflict per opinion ---
        discord_scores: list[float] = []
        for i in range(n):
            if n <= 1:
                discord_scores.append(0.0)
            else:
                total = sum(matrix[i][j] for j in range(n) if j != i)
                discord_scores.append(total / (n - 1))

        # --- Internal conflict scores ---
        internal_conflicts = [conflict_metric(op) for op in opinions]

        # --- Conflict pairs above threshold (canonical: i < j, sorted desc) ---
        pairs: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] > threshold:
                    pairs.append((i, j, matrix[i][j]))
        pairs.sort(key=lambda t: t[2], reverse=True)

        # --- Cohesion score ---
        if n <= 1:
            cohesion = 1.0
        else:
            cohesion = 1.0 - max(discord_scores)
            cohesion = max(0.0, min(1.0, cohesion))  # clamp to [0, 1]

        return ConflictReport(
            conflict_matrix=matrix,
            conflict_pairs=pairs,
            discord_scores=discord_scores,
            internal_conflicts=internal_conflicts,
            cohesion_score=cohesion,
            decayed_opinions=list(opinions),
            source_count=n,
            threshold=threshold,
        )

    def __repr__(self) -> str:
        return (
            f"ConflictDetector("
            f"threshold={self._default_threshold}, "
            f"decay_fn={self._decay_fn!r})"
        )
