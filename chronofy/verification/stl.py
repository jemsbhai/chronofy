"""Signal Temporal Logic verification for knowledge temporal validity.

THIS IS THE HEADLINE NOVELTY OF CHRONOFY.

Prior work (Mao et al., 2025; "Confidence over Time", 2026) applies STL
robustness to LLM *output confidence trajectories*. Chronofy applies STL
robustness to the *temporal validity of retrieved knowledge* feeding the
reasoning chain — a fundamentally different signal that catches cases where
the LLM is confidently wrong due to temporally stale retrieval.

STL specification:
    φ_valid = G_[0,n]( v(s_i) ≥ γ )

    where:
        v(s_i) = min_{e ∈ facts(s_i)} V(e, T_q)
        V(e, T_q) = q_e · exp(-β_j · (T_q - t_e))
        ρ(φ, π) = min_{i=1}^{n} (v(s_i) - γ)

Theorem 1 (Weakest-Link Bound / Decay Propagation Bound):
    C_out ≤ min_{i=1}^{k} V(e_i, T_q)

    This follows from the min-aggregation principle of possibilistic logic:
    the certainty of a derived conclusion equals the minimum certainty
    among its premises.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from chronofy.decay.base import DecayFunction
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact


class STLResult(BaseModel):
    """Result of STL verification over a reasoning trace.

    Carries the robustness score ρ, per-step validity signals,
    and the weakest-link confidence bound (Theorem 1).
    """

    robustness: float = Field(
        ...,
        description="STL robustness score ρ = min_i (v(s_i) - γ). "
        "Positive means the trace satisfies the temporal validity constraint.",
    )
    satisfied: bool = Field(
        ...,
        description="Whether ρ ≥ 0 (the STL formula is satisfied).",
    )
    step_validity: list[float] = Field(
        ...,
        description="Per-step validity signal v(s_i) = min over facts of V(e, T_q).",
    )
    weakest_step_index: int | None = Field(
        default=None,
        description="Index of the reasoning step with the lowest validity signal.",
    )
    weakest_fact: TemporalFact | None = Field(
        default=None,
        description="The single fact with the lowest temporal validity across the entire trace.",
    )
    output_confidence_bound: float = Field(
        ...,
        description="Theorem 1 bound: C_out ≤ min_i V(e_i, T_q) across ALL facts in the trace.",
    )


class STLVerifier:
    """Verifies temporal validity of reasoning traces via STL robustness.

    Applies the STL formula φ_valid = G_[0,n]( v(s_i) ≥ γ ) to a reasoning
    trace, computing the continuous robustness score ρ that quantifies how
    robustly the temporal validity constraint is satisfied or violated.

    Args:
        decay_fn: The decay function for computing temporal validity scores.
        threshold: The minimum acceptable validity γ ∈ [0, 1].

    Example:
        >>> from chronofy import ExponentialDecay, STLVerifier
        >>> decay = ExponentialDecay(beta={"vital_sign": 5.0})
        >>> verifier = STLVerifier(decay_fn=decay, threshold=0.5)
        >>> result = verifier.verify(trace)
        >>> if not result.satisfied:
        ...     print(f"Re-acquisition needed for: {result.weakest_fact}")
    """

    def __init__(self, decay_fn: DecayFunction, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold γ must be in [0, 1], got {threshold}")
        self._decay_fn = decay_fn
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The minimum acceptable validity threshold γ."""
        return self._threshold

    def step_validity(self, step: ReasoningStep, query_time: datetime) -> float:
        """Compute the temporal validity signal for a single reasoning step.

        v(s_i) = min_{e ∈ facts(s_i)} V(e, T_q)

        If the step uses no facts, returns 1.0 (vacuous truth — no evidence
        to decay means no constraint is violated).
        """
        if not step.facts_used:
            return 1.0
        scores = self._decay_fn.compute_batch(step.facts_used, query_time)
        return min(scores)

    def verify(self, trace: ReasoningTrace) -> STLResult:
        """Verify a complete reasoning trace against the STL validity formula.

        Computes:
            - Per-step validity signals v(s_i)
            - Robustness ρ = min_i (v(s_i) - γ)
            - Weakest-link confidence bound (Theorem 1)
            - The weakest step and fact for diagnostics

        Returns:
            STLResult with all verification outputs.
        """
        if not trace.steps:
            return STLResult(
                robustness=0.0,
                satisfied=True,
                step_validity=[],
                weakest_step_index=None,
                weakest_fact=None,
                output_confidence_bound=1.0,
            )

        # Compute per-step validity signals
        step_vals = [
            self.step_validity(step, trace.query_time) for step in trace.steps
        ]

        # Robustness: ρ = min_i (v(s_i) - γ)
        margins = [v - self._threshold for v in step_vals]
        rho = min(margins)

        # Weakest step
        weakest_idx = margins.index(min(margins))

        # Weakest-link bound (Theorem 1): min over ALL facts in ALL steps
        # Also track which single fact is the global weakest
        global_min_score = float("inf")
        global_weakest_fact: TemporalFact | None = None

        for step in trace.steps:
            for fact in step.facts_used:
                score = self._decay_fn.compute(fact, trace.query_time)
                if score < global_min_score:
                    global_min_score = score
                    global_weakest_fact = fact

        # If no facts at all, bound is 1.0
        if global_min_score == float("inf"):
            global_min_score = 1.0

        return STLResult(
            robustness=rho,
            satisfied=rho >= 0.0,
            step_validity=step_vals,
            weakest_step_index=weakest_idx,
            weakest_fact=global_weakest_fact,
            output_confidence_bound=global_min_score,
        )

    def robustness(self, trace: ReasoningTrace) -> float:
        """Convenience method: return just the scalar robustness score ρ."""
        return self.verify(trace).robustness

    def __repr__(self) -> str:
        return f"STLVerifier(decay_fn={self._decay_fn!r}, threshold={self._threshold})"
