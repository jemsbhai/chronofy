"""End-to-end Chronofy pipeline orchestrator.

Ties together the three TLDA layers:
    Layer 1: Temporal embedding (future — pluggable)
    Layer 2: Decay-weighted epistemic filtering
    Layer 3: STL robustness verification

Usage:
    >>> pipe = ChronofyPipeline.default(beta={"vital_sign": 5.0})
    >>> valid_facts, result = pipe.process(
    ...     candidate_facts=facts,
    ...     query_time=now,
    ...     build_trace=lambda facts: my_llm_reason(facts),
    ... )
    >>> if result and not result.satisfied:
    ...     print("Re-acquisition needed:", result.weakest_fact)
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningTrace, TemporalFact
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.verification.stl import STLResult, STLVerifier


class ChronofyPipeline:
    """End-to-end orchestrator for the Temporal-Logical Decay Architecture.

    Coordinates epistemic filtering (Layer 2) and STL verification (Layer 3).
    Layer 1 (temporal embeddings) is not yet implemented and will be
    integrated as a pre-processing step in a future release.

    Args:
        filter: The epistemic filter for Layer 2.
        verifier: The STL verifier for Layer 3.
    """

    def __init__(self, filter: EpistemicFilter, verifier: STLVerifier) -> None:
        self._filter = filter
        self._verifier = verifier

    @property
    def filter(self) -> EpistemicFilter:
        return self._filter

    @property
    def verifier(self) -> STLVerifier:
        return self._verifier

    @classmethod
    def default(
        cls,
        *,
        beta: dict[str, float] | None = None,
        default_beta: float = 0.5,
        time_unit: str = "days",
        filter_threshold: float = 0.1,
        verifier_threshold: float = 0.5,
    ) -> ChronofyPipeline:
        """Construct a pipeline with default settings.

        A single decay function is shared between the filter and verifier
        to ensure consistent scoring throughout the pipeline.

        Args:
            beta: Per-fact-type decay coefficients.
            default_beta: Fallback β for unknown fact types.
            time_unit: Time unit for age computation.
            filter_threshold: τ for epistemic filtering (Layer 2).
            verifier_threshold: γ for STL verification (Layer 3).
        """
        decay = ExponentialDecay(
            beta=beta,
            default_beta=default_beta,
            time_unit=time_unit,
        )
        return cls(
            filter=EpistemicFilter(decay_fn=decay, threshold=filter_threshold),
            verifier=STLVerifier(decay_fn=decay, threshold=verifier_threshold),
        )

    def filter_facts(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[TemporalFact]:
        """Layer 2: filter candidate facts by temporal validity."""
        return self._filter.filter(facts, query_time)

    def verify_trace(self, trace: ReasoningTrace) -> STLResult:
        """Layer 3: verify a reasoning trace via STL robustness."""
        return self._verifier.verify(trace)

    def needs_reacquisition(
        self,
        facts: list[TemporalFact],
        query_time: datetime,
        min_required: int = 1,
    ) -> bool:
        """Check whether re-acquisition is needed."""
        return self._filter.needs_reacquisition(facts, query_time, min_required)

    def process(
        self,
        candidate_facts: list[TemporalFact],
        query_time: datetime,
        build_trace: Callable[[list[TemporalFact]], ReasoningTrace],
    ) -> tuple[list[TemporalFact], STLResult | None]:
        """End-to-end: filter candidates, build trace, verify.

        Args:
            candidate_facts: Raw candidate facts before filtering.
            query_time: The query timestamp.
            build_trace: A callable that takes the filtered facts and
                returns a ReasoningTrace. In practice this wraps the
                LLM reasoning step.

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

    def __repr__(self) -> str:
        return f"ChronofyPipeline(filter={self._filter!r}, verifier={self._verifier!r})"
