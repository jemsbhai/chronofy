"""Core data models for Chronofy.

Implements the evidence tuple representation (c, t_e, q, m) from the
Temporal-Logical Decay Architecture. Each evidence item carries its content,
observation timestamp, source reliability weight, and metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TemporalFact(BaseModel):
    """A timestamped evidence item with provenance metadata.

    Represents the tuple e = (c, t_e, q, m) where:
        c: content — the factual claim
        t_e: observation timestamp — when the fact was observed (not published)
        q: source reliability weight in (0, 1]
        m: metadata — source, modality, fact type, etc.
    """

    content: str = Field(..., description="The factual content of the evidence item")
    timestamp: datetime = Field(
        ..., description="Observation timestamp (when the fact was observed)"
    )
    fact_type: str = Field(
        default="general",
        description="Category determining which decay rate β_j applies",
    )
    source_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Source reliability weight q ∈ (0, 1]. "
        "A peer-reviewed lab result should score higher than a patient self-report.",
    )
    publication_timestamp: datetime | None = Field(
        default=None,
        description="When the fact was published/reported (may differ from observation time)",
    )
    source: str | None = Field(default=None, description="Provenance identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (modality, confidence, etc.)"
    )

    def age_at(self, query_time: datetime) -> float:
        """Compute wall-clock age in days at the given query time.

        Uses observation timestamp (not publication timestamp) by default,
        following the principle that age should reflect when the underlying
        state was actually measured.
        """
        delta = query_time - self.timestamp
        return max(delta.total_seconds() / 86400.0, 0.0)


class ReasoningStep(BaseModel):
    """A single step in a chain-of-thought reasoning trace.

    Tracks which facts were utilized at this step, enabling
    the STL verification layer to compute per-step validity signals.
    """

    step_index: int = Field(..., description="Position in the reasoning chain (0-indexed)")
    content: str = Field(..., description="The reasoning content at this step")
    facts_used: list[TemporalFact] = Field(
        default_factory=list,
        description="Facts utilized in this reasoning step",
    )


class ReasoningTrace(BaseModel):
    """A complete chain-of-thought reasoning trace.

    Represents π = (s_1, s_2, ..., s_n) where each s_i is a reasoning step
    that may reference one or more temporal facts.
    """

    steps: list[ReasoningStep] = Field(
        default_factory=list, description="Ordered sequence of reasoning steps"
    )
    query_time: datetime = Field(
        default_factory=datetime.now,
        description="The time at which the query was issued",
    )

    @property
    def all_facts(self) -> list[TemporalFact]:
        """Return all unique facts used across the entire trace."""
        seen_contents: set[str] = set()
        unique: list[TemporalFact] = []
        for step in self.steps:
            for fact in step.facts_used:
                if fact.content not in seen_contents:
                    seen_contents.add(fact.content)
                    unique.append(fact)
        return unique
