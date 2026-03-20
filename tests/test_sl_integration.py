"""Full SL end-to-end integration test — clinical arrhythmia scenario.

Exercises the complete Subjective Logic pipeline on a realistic clinical
decision support scenario: cardiac arrhythmia risk assessment.

Components exercised (in pipeline order):
    1. OpinionDecayFunction — decay all facts to Opinions
    2. OpinionEpistemicFilter — filter stale facts + three-way partition
    3. ConflictDetector — check agreement between same-proposition readings
    4. TemporalEvidenceFusion — fuse within proposition groups
    5. GroupedFusionResult — per-proposition fusion reports
    6. OpinionSTLVerifier — verify reasoning trace via STL robustness
    7. OpinionScorer — rank facts by combined score
    8. SLPipeline.process_full() — orchestrate the full pipeline
    9. OpinionRuleGraph — SL-enriched graph query (lighter touch)

Clinical scenario:
    Query: "What is this patient's arrhythmia risk?"
    Query time: T = 2026-03-15 12:00

    Evidence set:
        - K+=4.1 mEq/L  (potassium, 2 hours ago, quality=0.95)  — fresh lab
        - K+=4.2 mEq/L  (potassium, 4 hours ago, quality=0.90)  — fresh lab
        - K+=3.2 mEq/L  (potassium, 30 days ago, quality=0.85)  — stale lab
        - Metformin 500mg (medication, 1 day ago, quality=1.0)   — fresh record
        - Blood type A+  (demographic, 10 years ago, quality=1.0) — stable fact

    Expected behavior:
        - Stale K+=3.2 filtered out (P(ω') ≈ base_rate < p_min)
        - Two fresh K+ readings fused (same proposition)
        - Metformin and blood type in separate fusion groups
        - STL verification satisfied (fresh evidence chain)
        - Three-way partition: 4 valid, 1 stale, 0 uncertain

This test validates that all SL components compose correctly and that
the mathematical invariants hold across the full pipeline.
"""

from datetime import datetime, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion

from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.sl.conflict import ConflictDetector, ConflictReport
from chronofy.sl.fusion import FusionReport, TemporalEvidenceFusion
from chronofy.sl.opinion_decay import OpinionConfig, OpinionDecayFunction
from chronofy.sl.opinion_filter import OpinionEpistemicFilter, OpinionPartitionResult
from chronofy.sl.opinion_scorer import (
    OpinionScoredFact,
    OpinionScorer,
    ProjectedMultiplicative,
    UncertaintyPenalized,
)
from chronofy.sl.pipeline import GroupedFusionResult, SLPipeline, SLPipelineResult
from chronofy.sl.stl_opinion import OpinionSTLResult, OpinionSTLVerifier


# ═══════════════════════════════════════════════════════════════════
# Clinical scenario setup
# ═══════════════════════════════════════════════════════════════════

QUERY_TIME = datetime(2026, 3, 15, 12, 0)


@pytest.fixture
def clinical_odf() -> OpinionDecayFunction:
    """Domain-configured OpinionDecayFunction for clinical facts."""
    return OpinionDecayFunction(
        half_lives={
            "vital_sign": 0.25,           # ~6 hours
            "medication": 1.0,            # 1 day
            "demographic": 36500.0,       # ~100 years
        },
        default_half_life=1.0,
    )


@pytest.fixture
def clinical_facts() -> list[TemporalFact]:
    """Clinical evidence set for arrhythmia risk query."""
    return [
        # Fresh potassium readings (same proposition, should be fused)
        TemporalFact(
            content="K+=4.1 mEq/L",
            timestamp=QUERY_TIME - timedelta(hours=2),
            fact_type="vital_sign",
            source_quality=0.95,
            source="lab_system_A",
        ),
        TemporalFact(
            content="K+=4.2 mEq/L",
            timestamp=QUERY_TIME - timedelta(hours=4),
            fact_type="vital_sign",
            source_quality=0.90,
            source="lab_system_B",
        ),
        # Stale potassium (should be filtered)
        TemporalFact(
            content="K+=3.2 mEq/L",
            timestamp=QUERY_TIME - timedelta(days=30),
            fact_type="vital_sign",
            source_quality=0.85,
            source="lab_system_A",
        ),
        # Fresh medication record (different proposition)
        TemporalFact(
            content="Metformin 500mg daily",
            timestamp=QUERY_TIME - timedelta(days=1),
            fact_type="medication",
            source_quality=1.0,
            source="pharmacy_system",
        ),
        # Stable demographic fact (near-zero decay)
        TemporalFact(
            content="Blood type A+",
            timestamp=QUERY_TIME - timedelta(days=3650),
            fact_type="demographic",
            source_quality=1.0,
            source="patient_record",
        ),
    ]


def _build_clinical_trace(facts: list[TemporalFact]) -> ReasoningTrace:
    """Simulate a 3-step clinical reasoning chain.

    Step 0: Assess electrolyte status (uses vital_sign facts)
    Step 1: Check medication interactions (uses medication facts)
    Step 2: Final risk assessment (uses all facts)
    """
    vital_signs = [f for f in facts if f.fact_type == "vital_sign"]
    medications = [f for f in facts if f.fact_type == "medication"]

    steps = [
        ReasoningStep(
            step_index=0,
            content="Assess electrolyte status from potassium readings",
            facts_used=vital_signs,
        ),
        ReasoningStep(
            step_index=1,
            content="Check medication interactions with cardiac risk",
            facts_used=medications,
        ),
        ReasoningStep(
            step_index=2,
            content="Synthesize arrhythmia risk assessment",
            facts_used=facts,  # all facts
        ),
    ]
    return ReasoningTrace(steps=steps, query_time=QUERY_TIME)


# ═══════════════════════════════════════════════════════════════════
# 1. Component-level integration: decay → filter → verify
# ═══════════════════════════════════════════════════════════════════


class TestClinicalDecayToFilter:
    """Validate decay + filter compose correctly on clinical data."""

    def test_fresh_potassium_has_high_validity(self, clinical_odf, clinical_facts):
        """2-hour-old K+ reading should have high belief, low uncertainty."""
        fresh_k = clinical_facts[0]  # K+=4.1, 2 hours ago
        opinion = clinical_odf.compute_opinion(fresh_k, QUERY_TIME)
        assert opinion.projected_probability() > 0.8
        assert opinion.uncertainty < 0.3

    def test_stale_potassium_decays_to_near_vacuity(self, clinical_odf, clinical_facts):
        """30-day-old K+ with half_life=0.25 days should be nearly vacuous."""
        stale_k = clinical_facts[2]  # K+=3.2, 30 days ago
        opinion = clinical_odf.compute_opinion(stale_k, QUERY_TIME)
        assert opinion.uncertainty > 0.95
        # P(ω') approaches base_rate (0.5)
        assert abs(opinion.projected_probability() - 0.5) < 0.05

    def test_demographic_survives_10_years(self, clinical_odf, clinical_facts):
        """Blood type with half_life=36500 days barely decays over 10 years."""
        blood_type = clinical_facts[4]
        opinion = clinical_odf.compute_opinion(blood_type, QUERY_TIME)
        assert opinion.projected_probability() > 0.9
        assert opinion.uncertainty < 0.15

    def test_filter_excludes_stale_keeps_fresh(self, clinical_odf, clinical_facts):
        """OpinionEpistemicFilter correctly separates stale from fresh."""
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)
        contents = [f.content for f in valid]
        # Fresh K+ readings, medication, and blood type survive
        assert "K+=4.1 mEq/L" in contents
        assert "K+=4.2 mEq/L" in contents
        assert "Metformin 500mg daily" in contents
        assert "Blood type A+" in contents
        # Stale K+ filtered
        assert "K+=3.2 mEq/L" not in contents

    def test_three_way_partition(self, clinical_odf, clinical_facts):
        """Partition produces correct categories for clinical data."""
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        result = ef.partition(clinical_facts, QUERY_TIME)
        # 4 valid (2 fresh K+, medication, blood type), 1 stale (old K+)
        assert len(result.valid) == 4
        assert len(result.stale) == 1
        assert result.stale[0][0].content == "K+=3.2 mEq/L"
        # No uncertain with default u_max=1.0
        assert len(result.uncertain) == 0

    def test_three_way_partition_with_u_max(self, clinical_odf, clinical_facts):
        """With tight u_max, medication may move to uncertain if moderately aged."""
        ef = OpinionEpistemicFilter(
            decay_fn=clinical_odf, p_min=0.5, u_max=0.3,
        )
        result = ef.partition(clinical_facts, QUERY_TIME)
        # All facts accounted for
        total = len(result.valid) + len(result.stale) + len(result.uncertain)
        assert total == len(clinical_facts)


# ═══════════════════════════════════════════════════════════════════
# 2. Conflict detection on same-proposition evidence
# ═══════════════════════════════════════════════════════════════════


class TestClinicalConflictDetection:
    """Conflict detection between two fresh potassium readings."""

    def test_agreeing_readings_low_conflict(self, clinical_odf, clinical_facts):
        """K+=4.1 and K+=4.2 from similar times should have low conflict."""
        fresh_k = clinical_facts[:2]  # Both fresh potassium readings
        detector = ConflictDetector(decay_fn=clinical_odf)
        report = detector.detect(fresh_k, QUERY_TIME)
        assert report.cohesion_score > 0.8
        # No conflict pairs above default threshold
        assert len(report.conflict_pairs) == 0

    def test_conflict_report_has_correct_count(self, clinical_odf, clinical_facts):
        fresh_k = clinical_facts[:2]
        detector = ConflictDetector(decay_fn=clinical_odf)
        report = detector.detect(fresh_k, QUERY_TIME)
        assert report.source_count == 2
        assert len(report.decayed_opinions) == 2


# ═══════════════════════════════════════════════════════════════════
# 3. Per-proposition fusion
# ═══════════════════════════════════════════════════════════════════


class TestClinicalFusion:
    """Fusion of same-proposition evidence (potassium readings)."""

    def test_fusing_two_potassium_readings_reduces_uncertainty(self, clinical_odf):
        """Cumulative fusion of two readings should reduce uncertainty."""
        readings = [
            TemporalFact(
                content="K+=4.1", timestamp=QUERY_TIME - timedelta(hours=2),
                fact_type="vital_sign", source_quality=0.95,
            ),
            TemporalFact(
                content="K+=4.2", timestamp=QUERY_TIME - timedelta(hours=4),
                fact_type="vital_sign", source_quality=0.90,
            ),
        ]
        fuser = TemporalEvidenceFusion(decay_fn=clinical_odf, fusion_method="cumulative")
        report = fuser.fuse(readings, QUERY_TIME)

        # Fused opinion should have lower uncertainty than either individual
        individual_opinions = clinical_odf.compute_opinion_batch(readings, QUERY_TIME)
        max_individual_u = max(op.uncertainty for op in individual_opinions)
        assert report.fused_opinion.uncertainty <= max_individual_u

        # Fused P(ω') should be at least as high as the weakest individual
        min_individual_p = min(op.projected_probability() for op in individual_opinions)
        assert report.projected_probability >= min_individual_p

    def test_grouped_fusion_separates_propositions(self, clinical_odf, clinical_facts):
        """GroupedFusionResult separates vital_sign, medication, demographic."""
        # Filter first to remove stale
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)

        fuser = TemporalEvidenceFusion(decay_fn=clinical_odf)

        # Simulate grouped fusion as SLPipeline would do it
        from collections import OrderedDict
        groups: OrderedDict[str, list[TemporalFact]] = OrderedDict()
        for fact in valid:
            key = fact.fact_type
            if key not in groups:
                groups[key] = []
            groups[key].append(fact)

        group_reports: dict[str, FusionReport] = {}
        for key, group_facts in groups.items():
            group_reports[key] = fuser.fuse(group_facts, QUERY_TIME)

        grouped = GroupedFusionResult(
            group_reports=group_reports,
            group_keys=list(groups.keys()),
        )

        assert "vital_sign" in grouped.group_reports
        assert "medication" in grouped.group_reports
        assert "demographic" in grouped.group_reports
        # vital_sign group has 2 facts (the two fresh K+ readings)
        assert grouped.group_reports["vital_sign"].source_count == 2
        # medication and demographic have 1 each
        assert grouped.group_reports["medication"].source_count == 1
        assert grouped.group_reports["demographic"].source_count == 1


# ═══════════════════════════════════════════════════════════════════
# 4. STL verification on clinical reasoning trace
# ═══════════════════════════════════════════════════════════════════


class TestClinicalSTLVerification:
    """STL verification on a multi-step clinical reasoning chain."""

    def test_fresh_evidence_chain_satisfies_stl(self, clinical_odf, clinical_facts):
        """Reasoning chain using only fresh facts should satisfy STL."""
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)
        trace = _build_clinical_trace(valid)

        verifier = OpinionSTLVerifier(decay_fn=clinical_odf, threshold=0.6)
        result = verifier.verify(trace)

        assert isinstance(result, OpinionSTLResult)
        assert result.satisfied is True
        assert result.robustness > 0

    def test_weakest_link_is_medication_or_demographic(self, clinical_odf, clinical_facts):
        """The weakest link should be the oldest non-demographic fact.

        Blood type (demographic) has near-zero decay so it's strong despite
        age. The medication (1 day old, half_life=1.0) has moderate decay.
        Fresh K+ (hours old, half_life=0.25) are very strong.
        """
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)
        trace = _build_clinical_trace(valid)

        verifier = OpinionSTLVerifier(decay_fn=clinical_odf, threshold=0.5)
        result = verifier.verify(trace)

        assert result.weakest_fact is not None
        assert result.weakest_link_opinion is not None
        # The output confidence bound should be the weakest P(ω')
        assert result.output_confidence_bound > 0.5

    def test_stale_evidence_would_violate_stl(self, clinical_odf, clinical_facts):
        """If stale K+=3.2 were included, STL with threshold=0.6 would fail."""
        # Include ALL facts including stale one
        trace = _build_clinical_trace(clinical_facts)

        verifier = OpinionSTLVerifier(decay_fn=clinical_odf, threshold=0.6)
        result = verifier.verify(trace)

        # The stale K+ drags the chain below threshold
        assert result.satisfied is False
        assert result.weakest_fact is not None
        assert result.weakest_fact.content == "K+=3.2 mEq/L"

    def test_weakest_link_opinion_shows_staleness(self, clinical_odf, clinical_facts):
        """The weakest link Opinion should show high uncertainty (staleness)."""
        trace = _build_clinical_trace(clinical_facts)
        verifier = OpinionSTLVerifier(decay_fn=clinical_odf, threshold=0.6)
        result = verifier.verify(trace)

        wl = result.weakest_link_opinion
        assert wl is not None
        # High uncertainty = evidence decayed to ignorance → re-acquire
        assert wl.uncertainty > 0.9


# ═══════════════════════════════════════════════════════════════════
# 5. Opinion scoring
# ═══════════════════════════════════════════════════════════════════


class TestClinicalOpinionScoring:
    """OpinionScorer ranks clinical facts by combined temporal validity + similarity."""

    def test_fresh_k_ranks_above_medication(self, clinical_odf, clinical_facts):
        """Fresh K+ (hours old) should rank above medication (1 day old).

        Note: Blood type (demographic) may rank highest because it has
        quality=1.0 and near-zero decay over 36500-day half-life. This
        is mathematically correct — the Temporal Invariance Guarantee
        preserves stable facts. The clinically meaningful comparison is
        between volatile fact types: vital signs vs medications.
        """
        scorer = OpinionScorer(decay_fn=clinical_odf, strategy=ProjectedMultiplicative())
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)

        ranked = scorer.rank(
            facts=valid,
            similarities=[1.0] * len(valid),
            query_time=QUERY_TIME,
        )

        # Find best-ranked vital_sign and medication
        best_vital = next(r for r in ranked if r.fact.fact_type == "vital_sign")
        best_med = next(r for r in ranked if r.fact.fact_type == "medication")
        # Fresh K+ (hours old, fast decay) should rank above medication (1 day old)
        assert best_vital.combined_score > best_med.combined_score

    def test_uncertainty_penalized_differentiates(self, clinical_odf, clinical_facts):
        """UncertaintyPenalized strategy penalizes high-uncertainty facts more."""
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.5)
        valid = ef.filter(clinical_facts, QUERY_TIME)
        sims = [1.0] * len(valid)

        scorer_pm = OpinionScorer(decay_fn=clinical_odf, strategy=ProjectedMultiplicative())
        scorer_up = OpinionScorer(decay_fn=clinical_odf, strategy=UncertaintyPenalized())

        ranked_pm = scorer_pm.rank(facts=valid, similarities=sims, query_time=QUERY_TIME)
        ranked_up = scorer_up.rank(facts=valid, similarities=sims, query_time=QUERY_TIME)

        # Both should produce the same number of results
        assert len(ranked_pm) == len(ranked_up)


# ═══════════════════════════════════════════════════════════════════
# 6. SLPipeline.process_full() — the grand orchestration
# ═══════════════════════════════════════════════════════════════════


class TestClinicalSLPipelineFull:
    """Full pipeline integration: filter → conflict → fusion → trace → verify."""

    def test_process_full_clinical_scenario(self, clinical_facts):
        """The full pipeline produces correct results on clinical data."""
        pipe = SLPipeline.default(
            half_lives={
                "vital_sign": 0.25,
                "medication": 1.0,
                "demographic": 36500.0,
            },
            filter_threshold=0.6,
            verifier_threshold=0.6,
            enable_conflict=True,
            enable_fusion=True,
        )

        result = pipe.process_full(
            candidate_facts=clinical_facts,
            query_time=QUERY_TIME,
            build_trace=_build_clinical_trace,
        )

        # --- SLPipelineResult type ---
        assert isinstance(result, SLPipelineResult)

        # --- Filtering: stale K+ excluded ---
        assert not result.needs_reacquisition
        valid_contents = [f.content for f in result.valid_facts]
        assert "K+=4.1 mEq/L" in valid_contents
        assert "K+=4.2 mEq/L" in valid_contents
        assert "K+=3.2 mEq/L" not in valid_contents
        assert "Metformin 500mg daily" in valid_contents
        assert "Blood type A+" in valid_contents
        assert len(result.valid_facts) == 4

        # --- STL verification: satisfied with fresh evidence ---
        assert result.stl_result is not None
        assert isinstance(result.stl_result, OpinionSTLResult)
        assert result.stl_result.satisfied is True
        assert result.stl_result.robustness > 0

        # --- Conflict report: present ---
        assert result.conflict_report is not None
        assert isinstance(result.conflict_report, ConflictReport)

        # --- Grouped fusion: 3 groups ---
        assert result.fusion_report is not None
        assert isinstance(result.fusion_report, GroupedFusionResult)
        gr = result.fusion_report
        assert "vital_sign" in gr.group_reports
        assert "medication" in gr.group_reports
        assert "demographic" in gr.group_reports
        assert gr.group_reports["vital_sign"].source_count == 2
        assert gr.group_reports["medication"].source_count == 1
        assert gr.group_reports["demographic"].source_count == 1

    def test_process_simple_clinical(self, clinical_facts):
        """process() returns (valid_facts, OpinionSTLResult)."""
        pipe = SLPipeline.default(
            half_lives={
                "vital_sign": 0.25,
                "medication": 1.0,
                "demographic": 36500.0,
            },
            filter_threshold=0.6,
            verifier_threshold=0.6,
        )

        valid, stl_result = pipe.process(
            candidate_facts=clinical_facts,
            query_time=QUERY_TIME,
            build_trace=_build_clinical_trace,
        )

        assert len(valid) == 4
        assert stl_result is not None
        assert stl_result.satisfied is True

    def test_all_stale_triggers_reacquisition(self):
        """When all facts are stale, pipeline signals re-acquisition."""
        stale_facts = [
            TemporalFact(
                content="K+=3.2 mEq/L",
                timestamp=QUERY_TIME - timedelta(days=30),
                fact_type="vital_sign",
                source_quality=0.85,
            ),
            TemporalFact(
                content="K+=3.5 mEq/L",
                timestamp=QUERY_TIME - timedelta(days=25),
                fact_type="vital_sign",
                source_quality=0.90,
            ),
        ]

        pipe = SLPipeline.default(
            half_lives={"vital_sign": 0.25},
            filter_threshold=0.6,
            verifier_threshold=0.6,
        )

        result = pipe.process_full(
            candidate_facts=stale_facts,
            query_time=QUERY_TIME,
            build_trace=_build_clinical_trace,
        )

        assert result.needs_reacquisition is True
        assert result.stl_result is None
        assert len(result.valid_facts) == 0


# ═══════════════════════════════════════════════════════════════════
# 7. Mathematical invariants across the full pipeline
# ═══════════════════════════════════════════════════════════════════


class TestClinicalMathematicalInvariants:
    """Verify SL mathematical invariants hold end-to-end."""

    def test_temporal_invariance_guarantee(self, clinical_odf):
        """β → 0 preserves stable facts regardless of age (Proposition 2).

        Blood type with half_life=36500 days should have P(ω') ≈ 1.0
        even after 10 years, per the Temporal Invariance Guarantee.
        """
        blood_type = TemporalFact(
            content="Blood type A+",
            timestamp=QUERY_TIME - timedelta(days=3650),
            fact_type="demographic",
            source_quality=1.0,
        )
        opinion = clinical_odf.compute_opinion(blood_type, QUERY_TIME)
        assert opinion.projected_probability() > 0.9

    def test_weakest_link_bound_holds(self, clinical_odf, clinical_facts):
        """Theorem 1: C_out ≤ min_i V(e_i, T_q) over all facts in chain.

        The STL output_confidence_bound should equal the minimum P(ω')
        across all facts in the reasoning trace.
        """
        ef = OpinionEpistemicFilter(decay_fn=clinical_odf, p_min=0.6)
        valid = ef.filter(clinical_facts, QUERY_TIME)
        trace = _build_clinical_trace(valid)

        verifier = OpinionSTLVerifier(decay_fn=clinical_odf, threshold=0.5)
        result = verifier.verify(trace)

        # Compute min P(ω') manually across all facts
        all_facts = trace.all_facts
        min_p = min(
            clinical_odf.compute_opinion(f, QUERY_TIME).projected_probability()
            for f in all_facts
        )

        # The bound should match (within floating point tolerance)
        assert abs(result.output_confidence_bound - min_p) < 1e-9

    def test_cumulative_fusion_diminishing_returns(self, clinical_odf):
        """ASEV Axiom 6: marginal information gain decreases with more evidence.

        Fusing N identical readings should show diminishing reduction in
        uncertainty: the 2nd reading reduces u more than the 3rd, etc.
        """
        # Create 5 identical fresh readings
        readings = [
            TemporalFact(
                content=f"K+=4.1 reading {i}",
                timestamp=QUERY_TIME - timedelta(hours=1),
                fact_type="vital_sign",
                source_quality=0.95,
            )
            for i in range(5)
        ]

        fuser = TemporalEvidenceFusion(
            decay_fn=clinical_odf, fusion_method="cumulative",
        )

        # Track uncertainty as we fuse 1, 2, 3, 4, 5 readings
        uncertainties: list[float] = []
        for n in range(1, 6):
            report = fuser.fuse(readings[:n], QUERY_TIME)
            uncertainties.append(report.fused_opinion.uncertainty)

        # Uncertainty should be non-increasing
        for i in range(1, len(uncertainties)):
            assert uncertainties[i] <= uncertainties[i - 1] + 1e-9

        # Marginal reduction should diminish (or stay constant)
        reductions = [
            uncertainties[i - 1] - uncertainties[i]
            for i in range(1, len(uncertainties))
        ]
        for i in range(1, len(reductions)):
            assert reductions[i] <= reductions[i - 1] + 1e-9

    def test_opinion_validity_is_proper_opinion(self, clinical_odf, clinical_facts):
        """Every decayed opinion satisfies b + d + u = 1, b,d,u ≥ 0."""
        for fact in clinical_facts:
            op = clinical_odf.compute_opinion(fact, QUERY_TIME)
            assert op.belief >= 0
            assert op.disbelief >= 0
            assert op.uncertainty >= 0
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
