"""SL-aware end-to-end pipeline orchestrator for Chronofy.

Standalone parallel to ChronofyPipeline that uses Subjective Logic
components throughout, producing full Opinion-level diagnostics.

Coordinates:
    Layer 2: EpistemicFilter (scalar threshold — OpinionDecayFunction IS DecayFunction)
    Layer 3: OpinionSTLVerifier (full Opinion path)
    Optional: ConflictDetector (pre-LLM conflict analysis)
    Optional: TemporalEvidenceFusion (same-proposition evidence fusion)

Design decision: SLPipeline is a standalone class, NOT a subclass of
ChronofyPipeline. Rationale:
    - Different return types (OpinionSTLResult vs STLResult) violate LSP
    - Different internal composition (optional conflict/fusion components)
    - Consistent with SL module pattern (OpinionSTLVerifier ∥ STLVerifier,
      OpinionScorer ∥ TemporalScorer)

Backward compatibility:
    Accepts both OpinionDecayFunction and plain DecayFunction. With a plain
    DecayFunction, the verifier wraps scalars into dogmatic Opinions (u=0).
    Conflict detection and fusion are NOT available with plain DecayFunction
    (they require OpinionDecayFunction); requesting them raises ValueError.

Default filter_threshold:
    The SL pipeline defaults to filter_threshold=0.6, NOT 0.1 (the scalar
    pipeline default). This is because SL opinion decay approaches base_rate
    (default 0.5) rather than 0, so a threshold ≤ 0.5 would never filter
    any evidence. The threshold must exceed base_rate to be functional.

Two-level aggregation:
    SL fusion operators (⊕ cumulative, ⊘ averaging) are defined for
    opinions about the SAME proposition (Jøsang §12.3). Fusing opinions
    about different propositions is mathematically undefined.

    TLDA therefore uses two-level aggregation:
        Level 1 (within-proposition): SL fusion (⊕ or ⊘)
            → GroupedFusionResult with per-group FusionReports
        Level 2 (across-proposition): possibilistic min (Theorem 1)
            → STL weakest-link bound over all facts

    The grouping key defaults to fact_type but can be overridden by a
    user-supplied function (e.g., grouping by analyte name for clinical
    data where multiple vital signs share the same fact_type).

Usage:
    >>> pipe = SLPipeline.default(half_lives={"vital_sign": 0.5})
    >>> valid, result = pipe.process(
    ...     candidate_facts=facts,
    ...     query_time=now,
    ...     build_trace=lambda facts: my_llm_reason(facts),
    ... )
    >>> if result and not result.satisfied:
    ...     wl = result.weakest_link_opinion
    ...     if wl.uncertainty > 0.8:
    ...         print("Stale — re-acquire same source type")

Requires: pip install chronofy[sl]

References:
    Paper §III (Chronofy Framework), §III-D (STL Verification).
    Jøsang (2016) §10.4 (Opinion Aging), §12.3 (Fusion).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from chronofy.decay.base import DecayFunction
from chronofy.models import ReasoningTrace, TemporalFact
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.sl.conflict import ConflictDetector, ConflictReport
from chronofy.sl.fusion import FusionReport, TemporalEvidenceFusion
from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.sl.stl_opinion import OpinionSTLResult, OpinionSTLVerifier


# Default grouping: by fact_type (decay-rate category).
# This is a coarse proxy — it groups all vital signs together,
# including potassium and sodium which are different propositions.
# Users with domain knowledge should supply a finer-grained function.
_DEFAULT_GROUP_BY: Callable[[TemporalFact], str] = lambda f: f.fact_type


# ═══════════════════════════════════════════════════════════════════
# GroupedFusionResult — two-level aggregation wrapper
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GroupedFusionResult:
    """Per-proposition-group fusion results.

    SL fusion operators (⊕, ⊘) are defined only for opinions about the
    SAME proposition. This wrapper groups facts by proposition before
    fusing within each group, producing one FusionReport per group.

    The cross-proposition weakest-link bound is handled separately by
    STL verification (Theorem 1), which operates on individual facts.

    Attributes:
        group_reports: Per-group FusionReport, keyed by group identifier.
            Ordered by first-seen insertion order.
        group_keys: Group identifiers in first-seen order.
    """

    group_reports: dict[str, FusionReport]
    group_keys: list[str]

    def weakest_group(self) -> tuple[str, FusionReport]:
        """Return the group with the lowest fused projected probability.

        This is the within-proposition analogue of the weakest-link bound:
        the proposition with the weakest combined evidence.

        Returns:
            (group_key, FusionReport) for the weakest group.
        """
        return min(
            self.group_reports.items(),
            key=lambda kv: kv[1].projected_probability,
        )

    def strongest_group(self) -> tuple[str, FusionReport]:
        """Return the group with the highest fused projected probability.

        Returns:
            (group_key, FusionReport) for the strongest group.
        """
        return max(
            self.group_reports.items(),
            key=lambda kv: kv[1].projected_probability,
        )

    def __repr__(self) -> str:
        return (
            f"GroupedFusionResult({len(self.group_reports)} groups: "
            f"{self.group_keys})"
        )


# ═══════════════════════════════════════════════════════════════════
# SLPipelineResult
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SLPipelineResult:
    """Rich result from process_full() with all optional diagnostics.

    Attributes:
        valid_facts: Facts surviving the epistemic filter (Layer 2).
        stl_result: OpinionSTLResult from verification (Layer 3).
            None if no facts survived filtering.
        conflict_report: ConflictReport from optional conflict detection.
            None if conflict detection is disabled or no facts survived.
        fusion_report: GroupedFusionResult from optional per-proposition
            fusion. None if fusion is disabled or no facts survived.
        needs_reacquisition: True if no facts survived filtering.
    """

    valid_facts: list[TemporalFact]
    stl_result: OpinionSTLResult | None = None
    conflict_report: ConflictReport | None = None
    fusion_report: GroupedFusionResult | None = None
    needs_reacquisition: bool = False


# ═══════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════


class SLPipeline:
    """SL-aware end-to-end orchestrator for the Temporal-Logical Decay Architecture.

    Coordinates epistemic filtering (Layer 2) and Opinion-aware STL
    verification (Layer 3), with optional conflict detection and
    per-proposition evidence fusion.

    Args:
        filter: The epistemic filter for Layer 2.
        verifier: The OpinionSTLVerifier for Layer 3.
        conflict_detector: Optional ConflictDetector for pre-LLM analysis.
        fusion: Optional TemporalEvidenceFusion for same-proposition evidence.
        fusion_group_by: Function mapping TemporalFact → group key string.
            Used by process_full() to partition facts into proposition groups
            before fusion. Default: fact_type.
    """

    def __init__(
        self,
        filter: EpistemicFilter,
        verifier: OpinionSTLVerifier,
        conflict_detector: ConflictDetector | None = None,
        fusion: TemporalEvidenceFusion | None = None,
        fusion_group_by: Callable[[TemporalFact], str] = _DEFAULT_GROUP_BY,
    ) -> None:
        self._filter = filter
        self._verifier = verifier
        self._conflict_detector = conflict_detector
        self._fusion = fusion
        self._fusion_group_by = fusion_group_by

    @property
    def filter(self) -> EpistemicFilter:
        return self._filter

    @property
    def verifier(self) -> OpinionSTLVerifier:
        return self._verifier

    @property
    def conflict_detector(self) -> ConflictDetector | None:
        return self._conflict_detector

    @property
    def fusion(self) -> TemporalEvidenceFusion | None:
        return self._fusion

    @classmethod
    def default(
        cls,
        *,
        decay_fn: DecayFunction | None = None,
        half_lives: dict[str, float] | None = None,
        default_half_life: float = 1.39,
        filter_threshold: float = 0.6,
        verifier_threshold: float = 0.5,
        enable_conflict: bool = False,
        conflict_threshold: float = 0.15,
        enable_fusion: bool = False,
        fusion_method: str = "cumulative",
        fusion_group_by: Callable[[TemporalFact], str] = _DEFAULT_GROUP_BY,
    ) -> SLPipeline:
        """Construct an SLPipeline with default settings.

        A single decay function is shared across all components to
        ensure consistent scoring throughout the pipeline.

        Note: filter_threshold defaults to 0.6 (not 0.1 as in the scalar
        ChronofyPipeline). In SL, fully decayed opinions approach base_rate
        (default 0.5), so a threshold ≤ base_rate would never filter
        anything. The threshold must exceed base_rate to be functional.

        Args:
            decay_fn: Pre-built DecayFunction. If provided, half_lives
                and default_half_life are ignored. Supports both
                OpinionDecayFunction and plain DecayFunction.
            half_lives: Per-fact-type half-lives in days (convenience).
                Ignored if decay_fn is provided.
            default_half_life: Fallback half-life for unknown fact types.
            filter_threshold: τ for epistemic filtering (Layer 2).
                Default 0.6 (above SL base_rate of 0.5).
            verifier_threshold: γ for STL verification (Layer 3).
            enable_conflict: Whether to create a ConflictDetector.
                Requires OpinionDecayFunction; raises ValueError if
                a plain DecayFunction is provided.
            conflict_threshold: Threshold for flagging conflict pairs.
            enable_fusion: Whether to create a TemporalEvidenceFusion.
                Requires OpinionDecayFunction; raises ValueError if
                a plain DecayFunction is provided.
            fusion_method: "cumulative" or "averaging".
            fusion_group_by: Function mapping TemporalFact → group key.
                Used by process_full() to partition facts into proposition
                groups before fusion. Default: fact_type.

        Raises:
            ValueError: If enable_conflict or enable_fusion is True but
                decay_fn is not an OpinionDecayFunction.
        """
        if decay_fn is None:
            decay_fn = OpinionDecayFunction(
                half_lives=half_lives,
                default_half_life=default_half_life,
            )

        is_opinion_aware = isinstance(decay_fn, OpinionDecayFunction)

        # Fail loudly if SL-only features requested with plain DecayFunction
        if enable_conflict and not is_opinion_aware:
            raise ValueError(
                "enable_conflict=True requires an OpinionDecayFunction, "
                f"but got {type(decay_fn).__name__}. ConflictDetector needs "
                "full SL Opinions, which plain DecayFunction cannot produce."
            )
        if enable_fusion and not is_opinion_aware:
            raise ValueError(
                "enable_fusion=True requires an OpinionDecayFunction, "
                f"but got {type(decay_fn).__name__}. TemporalEvidenceFusion "
                "needs full SL Opinions, which plain DecayFunction cannot produce."
            )

        ef = EpistemicFilter(decay_fn=decay_fn, threshold=filter_threshold)
        verifier = OpinionSTLVerifier(decay_fn=decay_fn, threshold=verifier_threshold)

        conflict_detector: ConflictDetector | None = None
        if enable_conflict:
            conflict_detector = ConflictDetector(
                decay_fn=decay_fn,  # type: ignore[arg-type]
                default_threshold=conflict_threshold,
            )

        fusion_engine: TemporalEvidenceFusion | None = None
        if enable_fusion:
            fusion_engine = TemporalEvidenceFusion(
                decay_fn=decay_fn,  # type: ignore[arg-type]
                fusion_method=fusion_method,  # type: ignore[arg-type]
            )

        return cls(
            filter=ef,
            verifier=verifier,
            conflict_detector=conflict_detector,
            fusion=fusion_engine,
            fusion_group_by=fusion_group_by,
        )

    # ───────────────────────────────────────────────────────────
    # Layer 2: Filtering
    # ───────────────────────────────────────────────────────────

    def filter_facts(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[TemporalFact]:
        """Layer 2: filter candidate facts by temporal validity."""
        return self._filter.filter(facts, query_time)

    # ───────────────────────────────────────────────────────────
    # Layer 3: Verification
    # ───────────────────────────────────────────────────────────

    def verify_trace(self, trace: ReasoningTrace) -> OpinionSTLResult:
        """Layer 3: verify a reasoning trace via Opinion-aware STL robustness."""
        return self._verifier.verify(trace)

    # ───────────────────────────────────────────────────────────
    # Optional: Conflict detection
    # ───────────────────────────────────────────────────────────

    def detect_conflict(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
    ) -> ConflictReport | None:
        """Run conflict detection on a set of facts.

        Returns None if no ConflictDetector is configured.
        """
        if self._conflict_detector is None:
            return None
        return self._conflict_detector.detect(facts, query_time)

    # ───────────────────────────────────────────────────────────
    # Optional: Fusion (standalone, single-group)
    # ───────────────────────────────────────────────────────────

    def fuse_facts(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
    ) -> FusionReport | None:
        """Fuse same-proposition evidence into a single opinion.

        This method fuses ALL provided facts as a single group.
        Caller is responsible for ensuring all facts are about the
        same proposition. For automatic proposition grouping, use
        process_full() instead.

        Returns None if no TemporalEvidenceFusion is configured.
        """
        if self._fusion is None:
            return None
        return self._fusion.fuse(facts, query_time)

    # ───────────────────────────────────────────────────────────
    # Optional: Grouped fusion (multi-proposition)
    # ───────────────────────────────────────────────────────────

    def _grouped_fuse(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
    ) -> GroupedFusionResult:
        """Partition facts by proposition group, then fuse within each.

        Level 1 of the two-level aggregation:
            Within-proposition: SL fusion (⊕ or ⊘)
        Level 2 is handled by STL verification (possibilistic min).

        Groups are determined by self._fusion_group_by. Order preserves
        first-seen insertion order.
        """
        assert self._fusion is not None  # caller checks

        # Partition into ordered groups
        groups: OrderedDict[str, list[TemporalFact]] = OrderedDict()
        for fact in facts:
            key = self._fusion_group_by(fact)
            if key not in groups:
                groups[key] = []
            groups[key].append(fact)

        # Fuse within each group
        group_reports: dict[str, FusionReport] = {}
        for key, group_facts in groups.items():
            group_reports[key] = self._fusion.fuse(group_facts, query_time)

        return GroupedFusionResult(
            group_reports=group_reports,
            group_keys=list(groups.keys()),
        )

    # ───────────────────────────────────────────────────────────
    # Re-acquisition
    # ───────────────────────────────────────────────────────────

    def needs_reacquisition(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
        min_required: int = 1,
    ) -> bool:
        """Check whether re-acquisition is needed."""
        return self._filter.needs_reacquisition(facts, query_time, min_required)

    # ───────────────────────────────────────────────────────────
    # End-to-end: simple
    # ───────────────────────────────────────────────────────────

    def process(
        self,
        candidate_facts: list[TemporalFact],
        query_time: datetime,
        build_trace: Callable[[list[TemporalFact]], ReasoningTrace],
    ) -> tuple[list[TemporalFact], OpinionSTLResult | None]:
        """End-to-end: filter candidates, build trace, verify.

        Args:
            candidate_facts: Raw candidate facts before filtering.
            query_time: The query timestamp.
            build_trace: A callable that takes the filtered facts and
                returns a ReasoningTrace.

        Returns:
            (valid_facts, stl_result) — the filtered facts and the
            verification result. stl_result is None if no facts survive
            filtering (re-acquisition needed).
        """
        valid = self.filter_facts(candidate_facts, query_time)
        if not valid:
            return valid, None
        trace = build_trace(valid)
        result = self.verify_trace(trace)
        return valid, result

    # ───────────────────────────────────────────────────────────
    # End-to-end: full diagnostics
    # ───────────────────────────────────────────────────────────

    def process_full(
        self,
        candidate_facts: list[TemporalFact],
        query_time: datetime,
        build_trace: Callable[[list[TemporalFact]], ReasoningTrace],
    ) -> SLPipelineResult:
        """End-to-end with all optional diagnostics.

        Pipeline: filter → (conflict detection) → (grouped fusion) → build_trace → verify.

        When fusion is enabled, facts are partitioned by proposition group
        (via fusion_group_by) before fusing within each group. This ensures
        SL fusion operators are only applied to opinions about the same
        proposition, as required by Jøsang §12.3.

        The cross-proposition weakest-link bound (Theorem 1) is computed
        separately by STL verification over individual facts.

        Args:
            candidate_facts: Raw candidate facts before filtering.
            query_time: The query timestamp.
            build_trace: A callable that takes the filtered facts and
                returns a ReasoningTrace.

        Returns:
            SLPipelineResult with valid_facts, stl_result, and optional
            conflict_report and fusion_report (GroupedFusionResult).
        """
        valid = self.filter_facts(candidate_facts, query_time)

        if not valid:
            return SLPipelineResult(
                valid_facts=valid,
                stl_result=None,
                needs_reacquisition=True,
            )

        # Optional: conflict detection on filtered facts
        conflict_report: ConflictReport | None = None
        if self._conflict_detector is not None:
            conflict_report = self._conflict_detector.detect(valid, query_time)

        # Optional: per-proposition-group fusion on filtered facts
        fusion_report: GroupedFusionResult | None = None
        if self._fusion is not None:
            fusion_report = self._grouped_fuse(valid, query_time)

        # Build trace and verify
        trace = build_trace(valid)
        stl_result = self.verify_trace(trace)

        return SLPipelineResult(
            valid_facts=valid,
            stl_result=stl_result,
            conflict_report=conflict_report,
            fusion_report=fusion_report,
            needs_reacquisition=False,
        )

    def __repr__(self) -> str:
        parts = [
            f"filter={self._filter!r}",
            f"verifier={self._verifier!r}",
        ]
        if self._conflict_detector is not None:
            parts.append(f"conflict={self._conflict_detector!r}")
        if self._fusion is not None:
            parts.append(f"fusion={self._fusion!r}")
        return f"SLPipeline({', '.join(parts)})"
