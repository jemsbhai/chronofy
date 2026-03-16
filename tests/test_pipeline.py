"""Tests for chronofy.pipeline — ChronofyPipeline.

The pipeline ties all three TLDA layers together:
    Layer 1: Temporal embedding (future — not tested here)
    Layer 2: Decay-weighted epistemic filtering
    Layer 3: STL robustness verification

The pipeline orchestrates: filter → (LLM generates trace) → verify → result.
Since Layer 1 (embedding) is not yet implemented, the pipeline operates
on pre-retrieved TemporalFact lists.
"""

import math
from datetime import datetime, timedelta

import pytest

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.pipeline import ChronofyPipeline
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.verification.stl import STLVerifier


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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_from_components(self):
        """Build pipeline from explicit components."""
        decay = ExponentialDecay()
        ef = EpistemicFilter(decay_fn=decay, threshold=0.1)
        sv = STLVerifier(decay_fn=decay, threshold=0.5)
        pipe = ChronofyPipeline(filter=ef, verifier=sv)
        assert pipe.filter is ef
        assert pipe.verifier is sv

    def test_default_factory(self):
        """Build pipeline with default settings via class method."""
        pipe = ChronofyPipeline.default()
        assert pipe.filter is not None
        assert pipe.verifier is not None

    def test_custom_factory(self):
        """Build pipeline with custom β and thresholds via class method."""
        pipe = ChronofyPipeline.default(
            beta={"vital_sign": 10.0},
            filter_threshold=0.2,
            verifier_threshold=0.3,
        )
        assert pipe.filter.threshold == 0.2
        assert pipe.verifier.threshold == 0.3


# ---------------------------------------------------------------------------
# Filtering (Layer 2 pass-through)
# ---------------------------------------------------------------------------

class TestPipelineFiltering:
    def test_filter_facts(self):
        """Pipeline exposes filtering as a convenience method."""
        pipe = ChronofyPipeline.default(
            beta={"vital_sign": 5.0},
            filter_threshold=0.1,
        )
        facts = [
            _fact("fresh", 0.0, "vital_sign"),
            _fact("stale", 10.0, "vital_sign"),
        ]
        valid = pipe.filter_facts(facts, QUERY_TIME)
        assert len(valid) == 1
        assert valid[0].content == "fresh"


# ---------------------------------------------------------------------------
# Verification (Layer 3 pass-through)
# ---------------------------------------------------------------------------

class TestPipelineVerification:
    def test_verify_trace(self):
        """Pipeline exposes verification as a convenience method."""
        pipe = ChronofyPipeline.default(verifier_threshold=0.5)
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("fresh", 0.0)])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert result.satisfied is True

    def test_verify_catches_stale(self):
        pipe = ChronofyPipeline.default(
            beta={"general": 0.5},
            verifier_threshold=0.5,
        )
        trace = ReasoningTrace(
            steps=[_step(0, [_fact("stale", 10.0, "general")])],
            query_time=QUERY_TIME,
        )
        result = pipe.verify_trace(trace)
        assert result.satisfied is False


# ---------------------------------------------------------------------------
# End-to-end: filter + verify
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_process_filters_then_verifies(self):
        """process() filters candidate facts, then verifies a trace built from survivors."""
        pipe = ChronofyPipeline.default(
            beta={"vital_sign": 5.0, "chronic": 0.01},
            filter_threshold=0.1,
            verifier_threshold=0.5,
        )
        candidates = [
            _fact("fresh_vital", 0.0, "vital_sign"),   # survives filter
            _fact("stale_vital", 5.0, "vital_sign"),    # filtered out
            _fact("old_chronic", 30.0, "chronic"),       # survives (low β)
        ]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=lambda facts: ReasoningTrace(
                steps=[_step(0, facts)],
                query_time=QUERY_TIME,
            ),
        )
        # Filter should keep 2 facts
        assert len(valid) == 2
        contents = [f.content for f in valid]
        assert "fresh_vital" in contents
        assert "old_chronic" in contents
        assert "stale_vital" not in contents
        # Verification result should exist
        assert result is not None
        assert isinstance(result.robustness, float)

    def test_process_reacquisition_signal(self):
        """When all candidates are filtered out, process signals re-acquisition."""
        pipe = ChronofyPipeline.default(
            beta={"vital_sign": 5.0},
            filter_threshold=0.1,
        )
        candidates = [_fact("ancient", 100.0, "vital_sign")]
        valid, result = pipe.process(
            candidate_facts=candidates,
            query_time=QUERY_TIME,
            build_trace=lambda facts: ReasoningTrace(
                steps=[_step(0, facts)],
                query_time=QUERY_TIME,
            ),
        )
        assert len(valid) == 0
        assert result is None  # No trace to verify when nothing survives

    def test_needs_reacquisition(self):
        """Pipeline exposes the re-acquisition check."""
        pipe = ChronofyPipeline.default(
            beta={"vital_sign": 5.0},
            filter_threshold=0.1,
        )
        stale = [_fact("old", 100.0, "vital_sign")]
        assert pipe.needs_reacquisition(stale, QUERY_TIME) is True

        fresh = [_fact("new", 0.0, "vital_sign")]
        assert pipe.needs_reacquisition(fresh, QUERY_TIME) is False


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestPipelineRepr:
    def test_repr(self):
        pipe = ChronofyPipeline.default()
        r = repr(pipe)
        assert "ChronofyPipeline" in r
