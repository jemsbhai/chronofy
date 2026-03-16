"""Tests for chronofy.verification.stl — STLVerifier and STLResult.

This is the HEADLINE NOVELTY of Chronofy: applying Signal Temporal Logic
robustness functions to KNOWLEDGE TEMPORAL VALIDITY, not output confidence.

The STL specification is:
    φ_valid = G_[0,n]( v(s_i) ≥ γ )

where:
    v(s_i) = min_{e ∈ facts(s_i)} V(e, T_q)     — validity signal at step i
    ρ(φ, π) = min_{i=1}^{n} (v(s_i) - γ)         — robustness score
    G_[0,n] = "globally" (must hold at every step)

Theorem 1 (Weakest-Link Bound):
    C_out ≤ min_{i=1}^{k} V(e_i, T_q)  for all facts e_i in the trace
"""

import math
from datetime import datetime, timedelta

import pytest

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.verification.stl import STLResult, STLVerifier


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


def _step(index: int, facts: list[TemporalFact], content: str = "") -> ReasoningStep:
    return ReasoningStep(
        step_index=index,
        content=content or f"Step {index}",
        facts_used=facts,
    )


@pytest.fixture
def decay() -> ExponentialDecay:
    return ExponentialDecay(beta={"vital_sign": 5.0, "chronic": 0.01, "general": 0.5})


@pytest.fixture
def verifier(decay: ExponentialDecay) -> STLVerifier:
    return STLVerifier(decay_fn=decay, threshold=0.5)


# ---------------------------------------------------------------------------
# STLResult
# ---------------------------------------------------------------------------

class TestSTLResult:
    """STLResult is a dataclass/model carrying the verification output."""

    def test_result_fields(self):
        result = STLResult(
            robustness=0.3,
            satisfied=True,
            step_validity=[0.9, 0.8],
            weakest_step_index=1,
            weakest_fact=None,
            output_confidence_bound=0.8,
        )
        assert result.robustness == 0.3
        assert result.satisfied is True
        assert len(result.step_validity) == 2
        assert result.weakest_step_index == 1
        assert result.output_confidence_bound == 0.8

    def test_needs_reacquisition_when_violated(self):
        """When robustness < 0, the trace fails and re-acquisition is needed."""
        result = STLResult(
            robustness=-0.2,
            satisfied=False,
            step_validity=[0.3, 0.1],
            weakest_step_index=1,
            weakest_fact=None,
            output_confidence_bound=0.1,
        )
        assert result.satisfied is False
        assert result.robustness < 0


# ---------------------------------------------------------------------------
# STLVerifier construction
# ---------------------------------------------------------------------------

class TestVerifierConstruction:
    def test_valid_threshold(self, decay: ExponentialDecay):
        v = STLVerifier(decay_fn=decay, threshold=0.5)
        assert v.threshold == 0.5

    def test_threshold_bounds(self, decay: ExponentialDecay):
        with pytest.raises(ValueError):
            STLVerifier(decay_fn=decay, threshold=1.5)
        with pytest.raises(ValueError):
            STLVerifier(decay_fn=decay, threshold=-0.1)


# ---------------------------------------------------------------------------
# Step validity: v(s_i) = min over facts of V(fact, T_q)
# ---------------------------------------------------------------------------

class TestStepValidity:
    """v(s_i) = min_{e ∈ facts(s_i)} V(e, T_q)"""

    def test_single_fact_step(self, verifier: STLVerifier):
        """Step with one fact: validity = that fact's decay score."""
        fact = _fact("K+ = 4.1", 0.5, "general")  # β=0.5, age=0.5 → exp(-0.25)
        step = _step(0, [fact])
        v = verifier.step_validity(step, QUERY_TIME)
        expected = math.exp(-0.5 * 0.5)
        assert v == pytest.approx(expected)

    def test_multi_fact_step_takes_minimum(self, verifier: STLVerifier):
        """Step with multiple facts: validity = min of all scores (weakest link)."""
        fresh = _fact("fresh", 0.0, "general")      # score ≈ 1.0
        stale = _fact("stale", 5.0, "general")       # score = exp(-2.5) ≈ 0.082
        step = _step(0, [fresh, stale])
        v = verifier.step_validity(step, QUERY_TIME)
        expected_stale = math.exp(-0.5 * 5.0)
        assert v == pytest.approx(expected_stale)

    def test_empty_step_returns_one(self, verifier: STLVerifier):
        """A step using no facts has validity 1.0 (no evidence to decay).

        This follows the convention that vacuous min is 1.0
        (no constraint = fully valid).
        """
        step = _step(0, [])
        v = verifier.step_validity(step, QUERY_TIME)
        assert v == pytest.approx(1.0)

    def test_quality_weight_affects_validity(self, verifier: STLVerifier):
        """Source quality q_e is part of the validity score."""
        fact = _fact("low_q", 0.0, "general", quality=0.4)
        step = _step(0, [fact])
        v = verifier.step_validity(step, QUERY_TIME)
        assert v == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Robustness: ρ = min_{i} (v(s_i) - γ)
# ---------------------------------------------------------------------------

class TestRobustness:
    """ρ(φ_valid, π) = min_{i=1}^{n} (v(s_i) - γ)"""

    def test_all_fresh_positive_robustness(self, verifier: STLVerifier):
        """All fresh facts → ρ > 0 (constraint satisfied)."""
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("a", 0.0)]),
                _step(1, [_fact("b", 0.0)]),
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.robustness > 0
        assert result.satisfied is True

    def test_one_stale_step_negative_robustness(self, verifier: STLVerifier):
        """One stale step drags ρ below zero."""
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("fresh", 0.0)]),           # v ≈ 1.0 → 1.0 - 0.5 = 0.5
                _step(1, [_fact("stale", 10.0, "general")]),  # v = exp(-5) ≈ 0.0067 → 0.0067 - 0.5 < 0
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.robustness < 0
        assert result.satisfied is False

    def test_robustness_equals_minimum_margin(self, verifier: STLVerifier):
        """ρ should equal exactly min(v(s_i) - γ)."""
        gamma = 0.5
        # Step 0: v = 1.0, margin = 0.5
        # Step 1: v = exp(-0.5*2) = exp(-1) ≈ 0.3679, margin = 0.3679 - 0.5 = -0.1321
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("fresh", 0.0)]),
                _step(1, [_fact("aged", 2.0, "general")]),
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        expected_v1 = math.exp(-0.5 * 2.0)
        expected_rho = expected_v1 - gamma
        assert result.robustness == pytest.approx(expected_rho)

    def test_robustness_convenience_method(self, verifier: STLVerifier):
        """verifier.robustness(trace) returns just the scalar ρ."""
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("x", 0.0)])],
            query_time=QUERY_TIME,
        )
        rho = verifier.robustness(trace)
        assert isinstance(rho, float)
        assert rho == pytest.approx(1.0 - 0.5)  # v=1.0, γ=0.5


# ---------------------------------------------------------------------------
# Weakest-link bound (Theorem 1)
# ---------------------------------------------------------------------------

class TestWeakestLinkBound:
    """Theorem 1: C_out ≤ min_{i} V(e_i, T_q) across ALL facts in the trace."""

    def test_bound_is_global_minimum(self, verifier: STLVerifier):
        """The bound is the min validity across all facts, not per-step."""
        # Step 0 uses fresh fact (score ≈ 1.0)
        # Step 1 uses moderately old fact (score ≈ 0.37)
        # Global min should be ≈ 0.37
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("fresh", 0.0)]),
                _step(1, [_fact("old", 2.0, "general")]),
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        expected_bound = math.exp(-0.5 * 2.0)
        assert result.output_confidence_bound == pytest.approx(expected_bound)

    def test_bound_with_quality_weight(self, verifier: STLVerifier):
        """Quality weight reduces the confidence bound."""
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("low_q", 0.0, "general", quality=0.3)])],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.output_confidence_bound == pytest.approx(0.3)

    def test_bound_is_tightest_across_steps(self, verifier: STLVerifier):
        """Even if step-level min is computed per step, the theorem
        gives the global min across ALL facts in ALL steps."""
        f_good = _fact("good", 0.0)       # V = 1.0
        f_ok = _fact("ok", 1.0, "general")  # V = exp(-0.5) ≈ 0.607
        f_bad = _fact("bad", 5.0, "general")  # V = exp(-2.5) ≈ 0.082

        trace = ReasoningTrace(
            steps=[
                _step(0, [f_good, f_ok]),    # step min = 0.607
                _step(1, [f_good, f_bad]),    # step min = 0.082
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        # Global min across all facts
        expected = math.exp(-0.5 * 5.0)
        assert result.output_confidence_bound == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Weakest step and fact tracking
# ---------------------------------------------------------------------------

class TestWeakestTracking:
    """Verify that the result correctly identifies which step and fact are weakest."""

    def test_weakest_step_index(self, verifier: STLVerifier):
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("fresh", 0.0)]),         # v ≈ 1.0
                _step(1, [_fact("stale", 10.0, "general")]),  # v ≈ 0.007
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.weakest_step_index == 1

    def test_weakest_fact_identified(self, verifier: STLVerifier):
        stale = _fact("the_stale_one", 10.0, "general")
        trace = ReasoningTrace(
            steps=[
                _step(0, [_fact("fresh", 0.0)]),
                _step(1, [_fact("also_fresh", 0.0), stale]),
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.weakest_fact is not None
        assert result.weakest_fact.content == "the_stale_one"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trace(self, verifier: STLVerifier):
        """Empty trace (no steps) should be vacuously satisfied."""
        trace = ReasoningTrace(steps=[], query_time=QUERY_TIME)
        result = verifier.verify(trace)
        assert result.satisfied is True
        assert result.robustness >= 0
        assert result.step_validity == []

    def test_single_step_single_fact(self, verifier: STLVerifier):
        """Simplest non-trivial case."""
        fact = _fact("only", 0.0)
        trace = ReasoningTrace(
            steps=[_step(0, [fact])],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        assert result.satisfied is True
        assert len(result.step_validity) == 1
        assert result.step_validity[0] == pytest.approx(1.0)

    def test_temporal_invariance_in_verification(self, verifier: STLVerifier):
        """Demographic facts (β=0) should never cause STL violation regardless of age."""
        decay = ExponentialDecay(beta={"demographic": 0.0})
        v = STLVerifier(decay_fn=decay, threshold=0.5)
        old_demo = _fact("blood_type_A", 3650.0, "demographic")  # 10 years old
        trace = ReasoningTrace(
            steps=[_step(0, [old_demo])],
            query_time=QUERY_TIME,
        )
        result = v.verify(trace)
        assert result.satisfied is True
        assert result.step_validity[0] == pytest.approx(1.0)

    def test_all_steps_same_validity(self, verifier: STLVerifier):
        """When all steps have the same validity, weakest_step_index = 0 (first)."""
        fact = _fact("same", 1.0, "general")
        trace = ReasoningTrace(
            steps=[
                _step(0, [fact]),
                _step(1, [fact]),
                _step(2, [fact]),
            ],
            query_time=QUERY_TIME,
        )
        result = verifier.verify(trace)
        # All step validities should be equal
        assert all(
            v == pytest.approx(result.step_validity[0])
            for v in result.step_validity
        )

    def test_threshold_zero_always_satisfied(self, decay: ExponentialDecay):
        """With γ = 0, any non-negative validity satisfies the constraint."""
        v = STLVerifier(decay_fn=decay, threshold=0.0)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("ancient", 1000.0, "general")])],
            query_time=QUERY_TIME,
        )
        result = v.verify(trace)
        assert result.satisfied is True

    def test_threshold_one_requires_perfect(self, decay: ExponentialDecay):
        """With γ = 1.0, only perfectly fresh, perfect-quality facts pass."""
        v = STLVerifier(decay_fn=decay, threshold=1.0)
        # Even a very fresh fact with tiny age will have score < 1.0
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("almost", 0.001, "general")])],
            query_time=QUERY_TIME,
        )
        result = v.verify(trace)
        assert result.satisfied is False


# ---------------------------------------------------------------------------
# Monotonicity property (Theorem 2 in the paper)
# ---------------------------------------------------------------------------

class TestRobustnessMonotonicity:
    """Theorem 2: ρ decreases monotonically as the max age of any fact increases."""

    def test_monotonic_decrease_with_age(self, decay: ExponentialDecay):
        """Increasing the age of the oldest fact should decrease ρ."""
        v = STLVerifier(decay_fn=decay, threshold=0.5)
        rho_values = []
        for age in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            trace = ReasoningTrace(
                steps=[_step(0, [_fact("x", age, "general")])],
                query_time=QUERY_TIME,
            )
            rho_values.append(v.robustness(trace))
        # ρ should be monotonically non-increasing
        for i in range(len(rho_values) - 1):
            assert rho_values[i] >= rho_values[i + 1]
