"""Tests for chronofy.models — TemporalFact, ReasoningStep, ReasoningTrace.

Validates the evidence tuple (c, t_e, q, m) representation and the
reasoning trace structure that enables STL verification.
"""

from datetime import datetime, timedelta

import pytest

from chronofy.models import TemporalFact, ReasoningStep, ReasoningTrace


# ---------------------------------------------------------------------------
# TemporalFact
# ---------------------------------------------------------------------------

class TestTemporalFact:
    """Tests for the TemporalFact evidence tuple."""

    def test_minimal_construction(self):
        """A fact needs only content and timestamp."""
        fact = TemporalFact(content="K+ = 4.1 mEq/L", timestamp=datetime(2026, 3, 15))
        assert fact.content == "K+ = 4.1 mEq/L"
        assert fact.timestamp == datetime(2026, 3, 15)

    def test_defaults(self):
        """Verify sensible defaults for optional fields."""
        fact = TemporalFact(content="test", timestamp=datetime(2026, 1, 1))
        assert fact.fact_type == "general"
        assert fact.source_quality == 1.0
        assert fact.publication_timestamp is None
        assert fact.source is None
        assert fact.metadata == {}

    def test_full_construction(self):
        """All fields populated — the full (c, t_e, q, m) tuple."""
        obs_time = datetime(2026, 3, 14, 8, 0)
        pub_time = datetime(2026, 3, 14, 10, 0)
        fact = TemporalFact(
            content="K+ = 4.1 mEq/L",
            timestamp=obs_time,
            fact_type="lab_result",
            source_quality=0.95,
            publication_timestamp=pub_time,
            source="hospital_lis",
            metadata={"unit": "mEq/L", "method": "ISE"},
        )
        assert fact.fact_type == "lab_result"
        assert fact.source_quality == 0.95
        assert fact.publication_timestamp == pub_time
        assert fact.source == "hospital_lis"
        assert fact.metadata["unit"] == "mEq/L"

    def test_source_quality_bounds(self):
        """q_e must be in [0, 1]."""
        with pytest.raises(ValueError):
            TemporalFact(content="x", timestamp=datetime.now(), source_quality=1.5)
        with pytest.raises(ValueError):
            TemporalFact(content="x", timestamp=datetime.now(), source_quality=-0.1)

    def test_source_quality_boundary_values(self):
        """q_e = 0.0 and q_e = 1.0 are both valid."""
        f0 = TemporalFact(content="x", timestamp=datetime.now(), source_quality=0.0)
        f1 = TemporalFact(content="x", timestamp=datetime.now(), source_quality=1.0)
        assert f0.source_quality == 0.0
        assert f1.source_quality == 1.0

    def test_age_at_positive(self):
        """age_at returns correct age in days for a past fact."""
        fact = TemporalFact(content="test", timestamp=datetime(2026, 3, 10))
        query_time = datetime(2026, 3, 15)
        age = fact.age_at(query_time)
        assert age == pytest.approx(5.0)

    def test_age_at_zero(self):
        """age_at returns 0 when query_time == timestamp."""
        now = datetime(2026, 3, 15, 12, 0)
        fact = TemporalFact(content="test", timestamp=now)
        assert fact.age_at(now) == pytest.approx(0.0)

    def test_age_at_future_clamped(self):
        """age_at returns 0 (clamped) when fact is in the future relative to query."""
        fact = TemporalFact(content="test", timestamp=datetime(2026, 3, 20))
        query_time = datetime(2026, 3, 15)
        assert fact.age_at(query_time) == pytest.approx(0.0)

    def test_age_at_fractional_days(self):
        """age_at handles sub-day precision correctly."""
        fact = TemporalFact(content="test", timestamp=datetime(2026, 3, 15, 0, 0))
        query_time = datetime(2026, 3, 15, 12, 0)  # 12 hours later
        assert fact.age_at(query_time) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ReasoningStep
# ---------------------------------------------------------------------------

class TestReasoningStep:
    """Tests for a single reasoning step in a chain-of-thought trace."""

    def test_minimal_construction(self):
        step = ReasoningStep(step_index=0, content="Patient presents with chest pain.")
        assert step.step_index == 0
        assert step.content == "Patient presents with chest pain."
        assert step.facts_used == []

    def test_with_facts(self):
        """A step references specific temporal facts."""
        f1 = TemporalFact(content="K+ = 4.1", timestamp=datetime(2026, 3, 14))
        f2 = TemporalFact(content="HR = 72", timestamp=datetime(2026, 3, 14))
        step = ReasoningStep(step_index=1, content="Vitals are stable.", facts_used=[f1, f2])
        assert len(step.facts_used) == 2
        assert step.facts_used[0].content == "K+ = 4.1"


# ---------------------------------------------------------------------------
# ReasoningTrace
# ---------------------------------------------------------------------------

class TestReasoningTrace:
    """Tests for the full chain-of-thought reasoning trace π = (s_1, ..., s_n)."""

    def _make_trace(self) -> ReasoningTrace:
        """Helper: build a 3-step trace with overlapping facts."""
        f1 = TemporalFact(content="K+ = 4.1", timestamp=datetime(2026, 3, 14))
        f2 = TemporalFact(content="HR = 72", timestamp=datetime(2026, 3, 14))
        f3 = TemporalFact(content="Dx: HTN", timestamp=datetime(2025, 1, 1))

        s1 = ReasoningStep(step_index=0, content="Review labs.", facts_used=[f1])
        s2 = ReasoningStep(step_index=1, content="Check vitals.", facts_used=[f1, f2])
        s3 = ReasoningStep(step_index=2, content="Review history.", facts_used=[f3])

        return ReasoningTrace(
            steps=[s1, s2, s3],
            query_time=datetime(2026, 3, 15),
        )

    def test_empty_trace(self):
        trace = ReasoningTrace(query_time=datetime(2026, 3, 15))
        assert trace.steps == []
        assert trace.all_facts == []

    def test_step_count(self):
        trace = self._make_trace()
        assert len(trace.steps) == 3

    def test_all_facts_deduplication(self):
        """all_facts should return each unique fact only once."""
        trace = self._make_trace()
        all_facts = trace.all_facts
        # f1 appears in steps 0 and 1, but should only appear once in all_facts
        assert len(all_facts) == 3
        contents = [f.content for f in all_facts]
        assert "K+ = 4.1" in contents
        assert "HR = 72" in contents
        assert "Dx: HTN" in contents

    def test_all_facts_preserves_order(self):
        """all_facts should return facts in the order first encountered."""
        trace = self._make_trace()
        contents = [f.content for f in trace.all_facts]
        assert contents == ["K+ = 4.1", "HR = 72", "Dx: HTN"]

    def test_query_time(self):
        trace = self._make_trace()
        assert trace.query_time == datetime(2026, 3, 15)
