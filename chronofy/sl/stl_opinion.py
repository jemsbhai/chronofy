"""Opinion-aware STL verification for knowledge temporal validity.

Extends the scalar STLVerifier to track full Subjective Logic Opinions
per reasoning step, enabling richer weakest-link bounds and nuanced
re-acquisition diagnostics.

Key advantage over scalar STLVerifier:
    The scalar verifier produces ρ < 0 → "violation." But it cannot
    distinguish WHY:
        - Stale evidence (high u): "we once had data but it decayed"
          → re-acquire the same source type
        - Negative evidence (high d, low u): "we have fresh evidence
          that the proposition is false" → real finding, not staleness
        - No evidence (u ≈ 1): "we never observed this"
          → need a completely different source

    The Opinion-aware verifier tracks (b, d, u, a) through the chain,
    making these distinctions inspectable.

Backward compatibility:
    OpinionSTLResult carries scalar step_validity, robustness, and
    output_confidence_bound that are numerically identical to what
    the scalar STLVerifier would produce (when using the same decay fn).

    OpinionSTLVerifier accepts BOTH OpinionDecayFunction (full SL path)
    and plain DecayFunction (scalar → dogmatic Opinion wrapper).

STL specification (same as scalar verifier):
    φ_valid = G_[0,n]( v(s_i) ≥ γ )
    ρ(φ, π) = min_{i=1}^{n} (v(s_i) - γ)

    v(s_i) = P(ω_i)  where ω_i is the step opinion (min over facts)

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §10.4 (Opinion Aging), §3.2 (Projected Probability).
    Paper Theorem 1 (Decay Propagation Bound).
    Paper Eq. 3-5 (validity signal, STL formula, robustness).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.base import DecayFunction
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


# Dogmatic full-belief opinion: used for no-fact steps (vacuous truth).
_FULL_VALIDITY = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)


# ═══════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════


class OpinionSTLResult:
    """Result of Opinion-aware STL verification over a reasoning trace.

    Carries the full per-step Opinions alongside scalar backward-compatible
    outputs (robustness ρ, step_validity, output_confidence_bound).

    Attributes:
        robustness: STL robustness score ρ = min_i(P(ω_i) - γ).
            Positive → trace satisfies temporal validity constraint.
        satisfied: Whether ρ ≥ 0.
        step_opinions: Per-step Opinion (the weakest fact's Opinion at
            each step). None-fact steps get dogmatic full-belief.
        step_validity: Per-step scalar validity = P(step_opinions[i]).
            Backward compatible with scalar STLVerifier.
        weakest_step_index: Index of the step with lowest validity.
            None if trace is empty.
        weakest_step_opinion: Opinion at the weakest step.
            None if trace is empty.
        weakest_fact: The single fact with lowest temporal validity
            across the entire trace. None if no facts used.
        weakest_link_opinion: Opinion of the weakest fact.
            None if no facts used.
        output_confidence_bound: Scalar P(weakest_link_opinion).
            Theorem 1 bound. 1.0 if no facts used.
    """

    __slots__ = (
        "robustness",
        "satisfied",
        "step_opinions",
        "step_validity",
        "weakest_step_index",
        "weakest_step_opinion",
        "weakest_fact",
        "weakest_link_opinion",
        "output_confidence_bound",
    )

    def __init__(
        self,
        *,
        robustness: float,
        satisfied: bool,
        step_opinions: list[Opinion],
        step_validity: list[float],
        weakest_step_index: int | None,
        weakest_step_opinion: Opinion | None,
        weakest_fact: TemporalFact | None,
        weakest_link_opinion: Opinion | None,
        output_confidence_bound: float,
    ) -> None:
        self.robustness = robustness
        self.satisfied = satisfied
        self.step_opinions = step_opinions
        self.step_validity = step_validity
        self.weakest_step_index = weakest_step_index
        self.weakest_step_opinion = weakest_step_opinion
        self.weakest_fact = weakest_fact
        self.weakest_link_opinion = weakest_link_opinion
        self.output_confidence_bound = output_confidence_bound

    def __repr__(self) -> str:
        return (
            f"OpinionSTLResult(ρ={self.robustness:.4f}, "
            f"satisfied={self.satisfied}, "
            f"steps={len(self.step_opinions)}, "
            f"bound={self.output_confidence_bound:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════
# Verifier
# ═══════════════════════════════════════════════════════════════════


class OpinionSTLVerifier:
    """Opinion-aware STL verifier for temporal knowledge validity.

    Extends STLVerifier by tracking full (b, d, u, a) Opinions through
    the reasoning chain, enabling richer diagnostics while preserving
    scalar backward compatibility.

    Accepts both OpinionDecayFunction (full SL path) and plain
    DecayFunction (scalar outputs wrapped into dogmatic Opinions).

    Args:
        decay_fn: OpinionDecayFunction or plain DecayFunction.
        threshold: Minimum acceptable validity γ ∈ [0, 1]. Default 0.5.

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        verifier = OpinionSTLVerifier(decay_fn=odf, threshold=0.5)
        result = verifier.verify(trace)

        if not result.satisfied:
            wl = result.weakest_link_opinion
            if wl.uncertainty > 0.8:
                print("Stale evidence — re-acquire same source type")
            elif wl.disbelief > wl.belief:
                print("Negative evidence — real finding, not staleness")
    """

    def __init__(
        self,
        decay_fn: DecayFunction,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        self._decay_fn = decay_fn
        self._threshold = threshold
        self._is_opinion_aware = isinstance(decay_fn, OpinionDecayFunction)

    @property
    def threshold(self) -> float:
        """The minimum acceptable validity threshold γ."""
        return self._threshold

    def _compute_fact_opinion(
        self, fact: TemporalFact, query_time: datetime
    ) -> Opinion:
        """Compute the temporal validity Opinion for a single fact.

        If the decay function is an OpinionDecayFunction, calls
        compute_opinion() for the full SL path. Otherwise, calls
        compute() and wraps the scalar into a dogmatic Opinion.
        """
        if self._is_opinion_aware:
            return self._decay_fn.compute_opinion(fact, query_time)  # type: ignore[union-attr]
        else:
            scalar = self._decay_fn.compute(fact, query_time)
            return Opinion.from_confidence(scalar, uncertainty=0.0)

    def _step_opinion(
        self, step: ReasoningStep, query_time: datetime
    ) -> tuple[Opinion, TemporalFact | None]:
        """Compute the Opinion for a reasoning step.

        Returns the Opinion of the fact with the lowest projected
        probability (weakest link within the step), plus that fact.

        No-fact steps return (dogmatic full-belief, None).
        """
        if not step.facts_used:
            return _FULL_VALIDITY, None

        min_prob = float("inf")
        min_opinion: Opinion | None = None
        min_fact: TemporalFact | None = None

        for fact in step.facts_used:
            opinion = self._compute_fact_opinion(fact, query_time)
            prob = opinion.projected_probability()
            if prob < min_prob:
                min_prob = prob
                min_opinion = opinion
                min_fact = fact

        # min_opinion is guaranteed non-None since facts_used is non-empty
        return min_opinion, min_fact  # type: ignore[return-value]

    def verify(self, trace: ReasoningTrace) -> OpinionSTLResult:
        """Verify a complete reasoning trace against the STL validity formula.

        Computes:
            - Per-step Opinions and scalar validity signals
            - Robustness ρ = min_i(P(ω_i) - γ)
            - Weakest-link Opinion and scalar bound (Theorem 1)
            - Weakest step and fact for diagnostics

        Returns:
            OpinionSTLResult with full Opinion-level and scalar outputs.
        """
        if not trace.steps:
            return OpinionSTLResult(
                robustness=0.0,
                satisfied=True,
                step_opinions=[],
                step_validity=[],
                weakest_step_index=None,
                weakest_step_opinion=None,
                weakest_fact=None,
                weakest_link_opinion=None,
                output_confidence_bound=1.0,
            )

        # --- Per-step opinions and validity ---
        step_opinions: list[Opinion] = []
        step_validity: list[float] = []
        step_weakest_facts: list[TemporalFact | None] = []

        for step in trace.steps:
            opinion, fact = self._step_opinion(step, trace.query_time)
            step_opinions.append(opinion)
            step_validity.append(opinion.projected_probability())
            step_weakest_facts.append(fact)

        # --- Robustness: ρ = min_i(v(s_i) - γ) ---
        margins = [v - self._threshold for v in step_validity]
        rho = min(margins)
        weakest_step_idx = margins.index(rho)

        # --- Global weakest fact across ALL steps ---
        global_min_prob = float("inf")
        global_weakest_opinion: Opinion | None = None
        global_weakest_fact: TemporalFact | None = None

        for step in trace.steps:
            for fact in step.facts_used:
                opinion = self._compute_fact_opinion(fact, trace.query_time)
                prob = opinion.projected_probability()
                if prob < global_min_prob:
                    global_min_prob = prob
                    global_weakest_opinion = opinion
                    global_weakest_fact = fact

        # If no facts at all across the trace, bound is 1.0
        if global_min_prob == float("inf"):
            global_min_prob = 1.0

        return OpinionSTLResult(
            robustness=rho,
            satisfied=rho >= 0.0,
            step_opinions=step_opinions,
            step_validity=step_validity,
            weakest_step_index=weakest_step_idx,
            weakest_step_opinion=step_opinions[weakest_step_idx],
            weakest_fact=global_weakest_fact,
            weakest_link_opinion=global_weakest_opinion,
            output_confidence_bound=global_min_prob,
        )

    def robustness(self, trace: ReasoningTrace) -> float:
        """Convenience: return just the scalar robustness score ρ."""
        return self.verify(trace).robustness

    def __repr__(self) -> str:
        mode = "opinion" if self._is_opinion_aware else "scalar-wrapped"
        return (
            f"OpinionSTLVerifier("
            f"mode={mode!r}, "
            f"threshold={self._threshold})"
        )
