"""Tests for chronofy.sl.pipeline — SLPipeline.

SL-aware end-to-end orchestrator paralleling ChronofyPipeline.
Coordinates:
    Layer 2: EpistemicFilter (scalar — OpinionDecayFunction IS DecayFunction)
    Layer 3: OpinionSTLVerifier (full Opinion path)
    Optional: ConflictDetector, TemporalEvidenceFusion, OpinionScorer

TDD: Tests written before implementation.
"""

from datetime import datetime, timedelta

import pytest

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.sl.conflict import ConflictDetector, ConflictReport
from chronofy.sl.fusion import FusionReport, TemporalEvidenceFusion
from chronofy.sl.opinion_decay import OpinionConfig, OpinionDecayFunction
from chronofy.sl.opinion_scorer import OpinionScorer, OpinionScoredFact
from chronofy.sl.stl_opinion import OpinionSTLResult, OpinionSTLVerifier
from chronofy.sl.pipeline import GroupedFusionResult, SLPipeline, SLPipelineResult


QUERY_TIME = datetime(2026, 3, 15, 12, 0)


def _fact(
    content: str,
    days_ago: float,
    fact_type: str = "general",
    quality: float = 1.0,
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
    )


def _step(index: int, facts: list[TemporalFact]) -> ReasoningStep:
    return ReasoningStep(step_index=index, content=f"Step {index}", facts_used=facts)


def _build_trace(facts: list[TemporalFact]) -> ReasoningTrace:
    """Default trace builder: one step using all facts."""
    return ReasoningTrace(
        steps=[_step(0, facts)],
        query_time=QUERY_TIME,
    )


# ═══════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineDefaultThreshold:
    """Issue 1: SL default filter_threshold must exceed base_rate.

    In SL, fully decayed opinions have P(ω') → base_rate (default 0.5).
    A threshold of 0.1 (the scalar pipeline default) would never filter
    anything because 0.5 > 0.1 always. The SL pipeline default must
    exceed base_rate to be functional.
    """

    def test_default_threshold_exceeds_base_rate(self):
        """SLPipeline.default() uses threshold=0.6, not scalar's 0.1."""
        pipe = SLPipeline.default()
        assert pipe.filter.threshold == 0.6

    def test_default_threshold_filters_stale_sl_opinions(self):
        """With default threshold, fully decayed SL opinions are filtered."""
        pipe = SLPipeline.default(half_lives={"vital_sign": 0.5})
        stale = [_fact("ancient", 100.0, "vital_sign")]
        valid = pipe.filter_facts(stale, QUERY_TIME)
        assert len(valid) == 0


class TestSLPipelineValueErrors:
    """Issue 2: Fail loudly when SL-only features requested with plain decay.

    ConflictDetector and TemporalEvidenceFusion require OpinionDecayFunction.
    If the user passes a plain DecayFunction and requests these features,
    we raise ValueError rather than silently creating a pipeline without them.
    """

    def test_plain_decay_with_conflict_raises(self):
        """enable_conflict=True + plain DecayFunction → ValueError."""
        exp_decay = ExponentialDecay(beta={"general": 0.5})
        with pytest.raises(ValueError, match="OpinionDecayFunction"):
            SLPipeline.default(decay_fn=exp_decay, enable_conflict=True)

    def test_plain_decay_with_fusion_raises(self):
        """enable_fusion=True + plain DecayFunction → ValueError."""
        exp_decay = ExponentialDecay(beta={"general": 0.5})
        with pytest.raises(ValueError, match="OpinionDecayFunction"):
            SLPipeline.default(decay_fn=exp_decay, enable_fusion=True)

    def test_plain_decay_without_sl_features_ok(self):
        """Plain DecayFunction without conflict/fusion is fine."""
        exp_decay = ExponentialDecay(beta={"general": 0.5})
        pipe = SLPipeline.default(decay_fn=exp_decay)
        assert pipe.conflict_detector is None
        assert pipe.fusion is None


class TestSLPipelineConstruction:
    """Test construction from explicit components and default factory."""

    def test_from_components(self):
        """Build SLPipeline from explicit components."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = EpistemicFilter(decay_fn=odf, threshold=0.1)
        verifier = OpinionSTLVerifier(decay_fn=odf, threshold=0.5)
        pipe = SLPipeline(filter=ef, verifier=verifier)
        assert pipe.filter is ef
        assert pipe.verifier is verifier

    def test_from_components_with_optional(self):
        """Build SLPipeline with optional conflict detector and fusion."""
        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        ef = EpistemicFilter(decay_fn=odf, threshold=0.1)
        verifier = OpinionSTLVerifier(decay_fn=odf, threshold=0.5)
        conflict = ConflictDetector(decay_fn=odf)
        fusion = TemporalEvidenceFusion(decay_fn=odf)
        pipe = SLPipeline(
            filter=ef,
            verifier=verifier,
            conflict_detector=conflict,
            fusion=fusion,
        )
        assert pipe.conflict_detector is conflict
        assert pipe.fusion is fusion

    def test_default_factory(self):
        """Build SLPipeline with default settings via classmethod."""
        pipe = SLPipeline.default()
        assert pipe.filter is not None
        assert pipe.verifier is not None
        # Optional components default to None
        assert pipe.conflict_detector is None
        assert pipe.fusion is None

    def test_default_factory_custom_half_lives(self):
        """Default factory accepts half_lives for domain configuration."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.5, "chronic": 100.0},
            filter_threshold=0.2,
            verifier_threshold=0.6,
        )
        assert pipe.filter.threshold == 0.2
        assert pipe.verifier.threshold == 0.6

    def test_default_factory_with_conflict(self):
        """Default factory can enable conflict detection."""
        pipe = SLPipeline.default(enable_conflict=True, conflict_threshold=0.2)
        assert pipe.conflict_detector is not None

    def test_default_factory_with_fusion(self):
        """Default factory can enable fusion."""
        pipe = SLPipeline.default(enable_fusion=True, fusion_method="cumulative")
        assert pipe.fusion is not None

    def test_default_factory_full(self):
        """Default factory with all optional components enabled."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.5},
            filter_threshold=0.15,
            verifier_threshold=0.5,
            enable_conflict=True,
            conflict_threshold=0.2,
            enable_fusion=True,
            fusion_method="averaging",
        )
        assert pipe.filter is not None
        assert pipe.verifier is not None
        assert pipe.conflict_detector is not None
        assert pipe.fusion is not None

    def test_default_factory_with_plain_decay_fn(self):
        """Default factory accepts a plain DecayFunction (scalar fallback)."""
        exp_decay = ExponentialDecay(beta={"vital_sign": 5.0})
        pipe = SLPipeline.default(decay_fn=exp_decay)
        assert pipe.filter is not None
        assert pipe.verifier is not None


# ═══════════════════════════════════════════════════════════════════
# Filtering (Layer 2 pass-through)
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineFiltering:
    """Filter pass-through — same as scalar but via SL decay.

    SL math note: fully decayed opinions approach base_rate (default 0.5),
    NOT zero. So filter_threshold must exceed base_rate to exclude stale
    facts. This is the key behavioral difference from scalar pipelines.
    """

    def test_filter_facts_keeps_fresh(self):
        """Fresh facts survive; stale facts filtered.

        With base_rate=0.5 and threshold=0.6, a fully decayed opinion
        (P ≈ 0.5) is filtered while a fresh opinion (P ≈ 1.0) passes.
        """
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 1.0},
            filter_threshold=0.6,
        )
        facts = [
            _fact("fresh_K", 0.0, "vital_sign"),
            _fact("stale_K", 30.0, "vital_sign"),
        ]
        valid = pipe.filter_facts(facts, QUERY_TIME)
        assert len(valid) == 1
        assert valid[0].content == "fresh_K"

    def test_filter_facts_preserves_stable(self):
        """Low-decay fact types survive even at high age."""
        pipe = SLPipeline.default(
            half_lives={"demographic": 36500.0},
            filter_threshold=0.1,
        )
        facts = [_fact("blood_type_A", 3650.0, "demographic")]
        valid = pipe.filter_facts(facts, QUERY_TIME)
        assert len(valid) == 1


# ═══════════════════════════════════════════════════════════════════
# Verification (Layer 3 pass-through)
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineVerification:
    """STL verification returns OpinionSTLResult (not scalar STLResult)."""

    def test_verify_trace_returns_opinion_result(self):
        """verify_trace returns OpinionSTLResult, not scalar STLResult."""
        pipe = SLPipeline.default(verifier_threshold=0.5)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("fresh", 0.0)])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert isinstance(result, OpinionSTLResult)

    def test_verify_fresh_trace_satisfied(self):
        pipe = SLPipeline.default(verifier_threshold=0.5)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("fresh", 0.0)])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert result.satisfied is True
        assert result.robustness > 0

    def test_verify_stale_trace_violation(self):
        """Stale fact violates STL when threshold > base_rate.

        Fully decayed opinion: P(ω') → base_rate=0.5.
        Threshold=0.6 → robustness = 0.5 − 0.6 < 0 → violation.
        """
        pipe = SLPipeline.default(
            half_lives={"general": 0.5},
            verifier_threshold=0.6,
        )
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("stale", 10.0, "general")])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert result.satisfied is False
        assert result.robustness < 0

    def test_verify_provides_weakest_link_opinion(self):
        """Result carries a full Opinion for the weakest link."""
        pipe = SLPipeline.default(verifier_threshold=0.5)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("fresh", 0.0)])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert result.weakest_link_opinion is not None
        assert hasattr(result.weakest_link_opinion, "belief")
        assert hasattr(result.weakest_link_opinion, "uncertainty")


# ═══════════════════════════════════════════════════════════════════
# End-to-end: process()
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineProcess:
    """End-to-end: filter → build_trace → verify."""

    def test_process_filters_then_verifies(self):
        """process() returns (filtered_facts, OpinionSTLResult).

        threshold=0.6 > base_rate=0.5 ensures stale vital (30 days,
        half_life=1.0) is filtered while chronic (half_life=100) passes.
        """
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 1.0, "chronic_condition": 100.0},
            filter_threshold=0.6,
            verifier_threshold=0.3,
        )
        candidates = [
            _fact("fresh_vital", 0.0, "vital_sign"),
            _fact("stale_vital", 30.0, "vital_sign"),
            _fact("old_chronic", 10.0, "chronic_condition"),
        ]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        # Fresh vital and old chronic survive; stale vital filtered
        assert len(valid) >= 2
        contents = [f.content for f in valid]
        assert "fresh_vital" in contents
        assert "old_chronic" in contents
        assert "stale_vital" not in contents
        # Verification result is an OpinionSTLResult
        assert result is not None
        assert isinstance(result, OpinionSTLResult)

    def test_process_reacquisition_when_all_filtered(self):
        """When all candidates are filtered out, result is None."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.5},
            filter_threshold=0.6,
        )
        candidates = [_fact("ancient", 100.0, "vital_sign")]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert len(valid) == 0
        assert result is None

    def test_process_return_type_is_opinion_stl_result(self):
        """Verify the return type is OpinionSTLResult, not scalar STLResult."""
        pipe = SLPipeline.default()
        candidates = [_fact("fact", 0.0)]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert isinstance(result, OpinionSTLResult)
        # Should NOT be the scalar STLResult
        from chronofy.verification.stl import STLResult
        # OpinionSTLResult is its own type, not STLResult
        assert type(result).__name__ == "OpinionSTLResult"


# ═══════════════════════════════════════════════════════════════════
# Conflict detection integration
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineConflict:
    """Optional conflict detection on filtered facts."""

    def test_detect_conflict_returns_report(self):
        """detect_conflict() returns a ConflictReport."""
        pipe = SLPipeline.default(enable_conflict=True)
        facts = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        report = pipe.detect_conflict(facts, QUERY_TIME)
        assert isinstance(report, ConflictReport)
        assert report.source_count == 2

    def test_detect_conflict_none_when_disabled(self):
        """detect_conflict() returns None when no detector configured."""
        pipe = SLPipeline.default()  # no enable_conflict
        facts = [_fact("K+=4.1", 0.0, "vital_sign")]
        report = pipe.detect_conflict(facts, QUERY_TIME)
        assert report is None

    def test_process_with_conflict_includes_report(self):
        """When conflict detection is enabled, process_full returns conflict report."""
        pipe = SLPipeline.default(
            enable_conflict=True,
            conflict_threshold=0.1,
        )
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.conflict_report is not None
        assert isinstance(result.conflict_report, ConflictReport)


# ═══════════════════════════════════════════════════════════════════
# Fusion integration
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineFusion:
    """Optional fusion of same-proposition evidence."""

    def test_fuse_facts_returns_report(self):
        """fuse_facts() returns a FusionReport."""
        pipe = SLPipeline.default(enable_fusion=True)
        facts = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        report = pipe.fuse_facts(facts, QUERY_TIME)
        assert isinstance(report, FusionReport)
        assert report.source_count == 2

    def test_fuse_facts_none_when_disabled(self):
        """fuse_facts() returns None when no fusion configured."""
        pipe = SLPipeline.default()  # no enable_fusion
        facts = [_fact("K+=4.1", 0.0, "vital_sign")]
        report = pipe.fuse_facts(facts, QUERY_TIME)
        assert report is None


# ═══════════════════════════════════════════════════════════════════
# process_full() — rich result with all diagnostics
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineProcessFull:
    """process_full() returns a SLPipelineResult with all diagnostics."""

    def test_process_full_basic(self):
        """process_full returns a result with valid_facts and stl_result."""
        pipe = SLPipeline.default()
        candidates = [_fact("fact_A", 0.0), _fact("fact_B", 0.5)]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.valid_facts is not None
        assert len(result.valid_facts) > 0
        assert result.stl_result is not None
        assert isinstance(result.stl_result, OpinionSTLResult)

    def test_process_full_with_all_components(self):
        """process_full with conflict + fusion returns all reports."""
        pipe = SLPipeline.default(
            enable_conflict=True,
            enable_fusion=True,
        )
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.valid_facts is not None
        assert result.stl_result is not None
        assert result.conflict_report is not None
        # Fusion result is a GroupedFusionResult, not a raw FusionReport
        assert result.fusion_report is not None
        assert isinstance(result.fusion_report, GroupedFusionResult)

    def test_process_full_reacquisition(self):
        """process_full signals re-acquisition when all facts filtered."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.5},
            filter_threshold=0.6,
        )
        candidates = [_fact("ancient", 100.0, "vital_sign")]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert len(result.valid_facts) == 0
        assert result.stl_result is None
        assert result.needs_reacquisition is True

    def test_process_full_no_reacquisition_when_fresh(self):
        """process_full does not signal re-acquisition when facts survive."""
        pipe = SLPipeline.default()
        candidates = [_fact("fresh", 0.0)]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.needs_reacquisition is False


# ═══════════════════════════════════════════════════════════════════
# Needs re-acquisition
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineReacquisition:
    """needs_reacquisition pass-through."""

    def test_needs_reacquisition_stale(self):
        """Stale fact with P(ω') ≈ 0.5 is below threshold=0.6."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.5},
            filter_threshold=0.6,
        )
        stale = [_fact("old", 100.0, "vital_sign")]
        assert pipe.needs_reacquisition(stale, QUERY_TIME) is True

    def test_needs_reacquisition_fresh(self):
        pipe = SLPipeline.default()
        fresh = [_fact("new", 0.0)]
        assert pipe.needs_reacquisition(fresh, QUERY_TIME) is False

    def test_needs_reacquisition_min_required(self):
        """Needs re-acquisition if fewer than min_required survive."""
        pipe = SLPipeline.default(
            half_lives={"vital_sign": 1.0},
            filter_threshold=0.1,
        )
        # Only 1 fresh fact, but we require 3
        facts = [_fact("one_fresh", 0.0, "vital_sign")]
        assert pipe.needs_reacquisition(facts, QUERY_TIME, min_required=3) is True


# ═══════════════════════════════════════════════════════════════════
# Plain DecayFunction fallback
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelinePlainDecayFallback:
    """SLPipeline should work with a plain DecayFunction (scalar path)."""

    def test_process_with_plain_decay(self):
        """process() works with a plain ExponentialDecay (scalar → dogmatic)."""
        exp_decay = ExponentialDecay(beta={"general": 0.5})
        pipe = SLPipeline.default(decay_fn=exp_decay)
        candidates = [_fact("fact", 0.0)]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert len(valid) == 1
        assert result is not None
        assert isinstance(result, OpinionSTLResult)

    def test_verify_with_plain_decay(self):
        """verify_trace works with plain DecayFunction."""
        exp_decay = ExponentialDecay(beta={"general": 0.5})
        pipe = SLPipeline.default(decay_fn=exp_decay)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("fresh", 0.0)])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert isinstance(result, OpinionSTLResult)
        assert result.satisfied is True


# ═══════════════════════════════════════════════════════════════════
# Repr
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Grouped Fusion — two-level aggregation
# ═══════════════════════════════════════════════════════════════════


class TestSLPipelineGroupedFusion:
    """Issue 3: Fusion must respect proposition boundaries.

    SL fusion operators (⊕ cumulative, ⊘ averaging) are defined for
    opinions about the SAME proposition (Jøsang §12.3). Fusing opinions
    about different propositions (potassium ⊕ blood_pressure) is
    mathematically undefined.

    TLDA has a natural two-level aggregation:
        Level 1 (within-proposition): SL fusion (⊕ or ⊘)
        Level 2 (across-proposition): possibilistic min (Theorem 1)

    The pipeline groups facts by proposition before fusing within groups.
    Default grouping key is fact_type; users can supply a custom function.
    """

    def test_groups_by_fact_type_default(self):
        """process_full groups facts by fact_type for fusion."""
        pipe = SLPipeline.default(enable_fusion=True)
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
            _fact("metformin", 0.0, "medication"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.fusion_report is not None
        gr = result.fusion_report
        assert isinstance(gr, GroupedFusionResult)
        # Two groups: vital_sign and medication
        assert "vital_sign" in gr.group_reports
        assert "medication" in gr.group_reports
        assert len(gr.group_reports) == 2
        # vital_sign group has 2 facts fused
        assert gr.group_reports["vital_sign"].source_count == 2
        # medication group has 1 fact (trivial fusion)
        assert gr.group_reports["medication"].source_count == 1

    def test_custom_group_by(self):
        """Custom grouping function overrides default fact_type grouping."""
        # Group by first word of content (simulating proposition grouping)
        pipe = SLPipeline.default(
            enable_fusion=True,
            fusion_group_by=lambda f: f.content.split("=")[0],
        )
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
            _fact("Na+=138", 0.0, "vital_sign"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        # Custom grouping splits by analyte: K+ and Na+
        assert "K+" in gr.group_reports
        assert "Na+" in gr.group_reports
        assert len(gr.group_reports) == 2
        assert gr.group_reports["K+"].source_count == 2
        assert gr.group_reports["Na+"].source_count == 1

    def test_single_group(self):
        """All facts in one group produces a single-entry GroupedFusionResult."""
        pipe = SLPipeline.default(enable_fusion=True)
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        assert len(gr.group_reports) == 1
        assert "vital_sign" in gr.group_reports

    def test_weakest_group(self):
        """weakest_group() returns the group with lowest fused P(ω')."""
        pipe = SLPipeline.default(
            enable_fusion=True,
            half_lives={"vital_sign": 0.5, "medication": 100.0},
        )
        candidates = [
            # Vital sign: moderately aged → lower P(ω')
            _fact("K+=4.1", 0.5, "vital_sign"),
            # Medication: fresh → higher P(ω')
            _fact("metformin", 0.0, "medication"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        key, report = gr.weakest_group()
        # Vital sign decays fast → it should be the weakest group
        assert key == "vital_sign"

    def test_strongest_group(self):
        """strongest_group() returns the group with highest fused P(ω')."""
        pipe = SLPipeline.default(
            enable_fusion=True,
            half_lives={"vital_sign": 0.5, "medication": 100.0},
        )
        candidates = [
            _fact("K+=4.1", 0.5, "vital_sign"),
            _fact("metformin", 0.0, "medication"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        key, report = gr.strongest_group()
        assert key == "medication"

    def test_group_keys_ordered(self):
        """group_keys preserves insertion order of first-seen groups."""
        pipe = SLPipeline.default(enable_fusion=True)
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("metformin", 0.0, "medication"),
            _fact("K+=4.2", 0.1, "vital_sign"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        # First-seen order: vital_sign (from K+=4.1), medication
        assert gr.group_keys == ["vital_sign", "medication"]

    def test_no_fusion_when_disabled(self):
        """fusion_report is None when fusion is not enabled."""
        pipe = SLPipeline.default()  # no enable_fusion
        candidates = [_fact("K+=4.1", 0.0, "vital_sign")]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        assert result.fusion_report is None

    def test_grouped_fusion_result_repr(self):
        """GroupedFusionResult has a useful repr."""
        pipe = SLPipeline.default(enable_fusion=True)
        candidates = [
            _fact("K+=4.1", 0.0, "vital_sign"),
            _fact("metformin", 0.0, "medication"),
        ]
        result = pipe.process_full(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=_build_trace,
        )
        gr = result.fusion_report
        assert gr is not None
        r = repr(gr)
        assert "GroupedFusionResult" in r
        assert "2 groups" in r


class TestSLPipelineRepr:
    def test_repr(self):
        pipe = SLPipeline.default()
        r = repr(pipe)
        assert "SLPipeline" in r

    def test_repr_with_optional_components(self):
        pipe = SLPipeline.default(enable_conflict=True, enable_fusion=True)
        r = repr(pipe)
        assert "SLPipeline" in r
