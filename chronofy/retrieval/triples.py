"""Temporal triple store for timestamped knowledge graph facts.

Implements the data layer for Layer 2 of the Chronofy framework.
A TemporalTriple extends the standard (subject, relation, object) KG triple
with an observation timestamp, source quality, and fact type — bridging
the relational KG world with Chronofy's TemporalFact model.

The TemporalKnowledgeGraph provides indexed storage over triples,
supporting both directed (default, matches ICEWS/GDELT) and undirected
graph semantics, with filtering by field, time range, and neighbor queries.

Design notes (SOLID):
- Single Responsibility: this module handles ONLY triple storage and retrieval.
  Rule mining, graph construction, and PageRank live in separate modules.
- Open/Closed: TemporalKnowledgeGraph is open for extension (subclass for
  custom indexing) but closed for modification of the core query interface.
- Dependency Inversion: no dependency on networkx, decay functions, or any
  higher-level module. Pure data layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from chronofy.models import TemporalFact


class TemporalTriple(BaseModel):
    """A timestamped knowledge graph triple: (subject, relation, object, t, q).

    Bridges the KG representation with Chronofy's TemporalFact model via
    the ``to_temporal_fact()`` method.

    Args:
        subject: The source entity.
        relation: The relation type.
        object: The target entity.
        timestamp: Observation timestamp.
        source_quality: Reliability weight q ∈ [0, 1].
        fact_type: Category determining which decay rate β_j applies.
    """

    subject: str
    relation: str
    object: str
    timestamp: datetime
    source_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    fact_type: str = Field(default="general")

    def to_temporal_fact(self) -> TemporalFact:
        """Convert to a TemporalFact for use with decay functions and filters.

        The content field encodes the full triple in a human-readable format.
        """
        return TemporalFact(
            content=f"{self.subject} {self.relation} {self.object}",
            timestamp=self.timestamp,
            fact_type=self.fact_type,
            source_quality=self.source_quality,
            source=f"KG:({self.subject}, {self.relation}, {self.object})",
        )


class TemporalKnowledgeGraph:
    """Indexed store for timestamped knowledge graph triples.

    Supports both directed (default, matching ICEWS/GDELT convention)
    and undirected graph semantics. Provides efficient filtering by
    subject, relation, object, and time range, plus neighbor queries
    needed by the graph traversal layer.

    This is a pure data structure — it does NOT depend on networkx,
    decay functions, or any ML components. It stores raw triples and
    provides query access. Graph algorithms live in ``graph.py``.

    Args:
        directed: If True (default), edges have direction (subject → object).
            If False, edges are bidirectional.
    """

    def __init__(self, directed: bool = True) -> None:
        self._directed = directed
        self._triples: list[TemporalTriple] = []

        # Indexes for O(1)-ish lookups by field
        self._by_subject: dict[str, list[int]] = {}
        self._by_relation: dict[str, list[int]] = {}
        self._by_object: dict[str, list[int]] = {}

    @property
    def is_directed(self) -> bool:
        """Whether the graph treats edges as directed."""
        return self._directed

    def __len__(self) -> int:
        """Number of triples stored."""
        return len(self._triples)

    @property
    def entities(self) -> set[str]:
        """All unique entities (subjects and objects)."""
        return set(self._by_subject.keys()) | set(self._by_object.keys())

    @property
    def relations(self) -> set[str]:
        """All unique relation types."""
        return set(self._by_relation.keys())

    def add_triple(self, triple: TemporalTriple) -> None:
        """Add a single triple to the store, updating indexes."""
        idx = len(self._triples)
        self._triples.append(triple)
        self._by_subject.setdefault(triple.subject, []).append(idx)
        self._by_relation.setdefault(triple.relation, []).append(idx)
        self._by_object.setdefault(triple.object, []).append(idx)

    def add_triples(self, triples: list[TemporalTriple]) -> None:
        """Add a batch of triples."""
        for t in triples:
            self.add_triple(t)

    def get_triples(
        self,
        *,
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[TemporalTriple]:
        """Query triples by field and/or time range.

        All filters are conjunctive (AND). Omitted filters match everything.

        Args:
            subject: Filter by source entity.
            relation: Filter by relation type.
            object: Filter by target entity.
            after: Include only triples with timestamp >= after.
            before: Include only triples with timestamp <= before.

        Returns:
            List of matching triples, in insertion order.
        """
        # Start with the most selective index to minimize scanning
        candidate_idxs = self._resolve_candidate_indexes(subject, relation, object)

        results: list[TemporalTriple] = []
        for idx in candidate_idxs:
            t = self._triples[idx]
            if subject is not None and t.subject != subject:
                continue
            if relation is not None and t.relation != relation:
                continue
            if object is not None and t.object != object:
                continue
            if after is not None and t.timestamp < after:
                continue
            if before is not None and t.timestamp > before:
                continue
            results.append(t)
        return results

    def get_neighbors(
        self,
        entity: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[tuple[str, str, TemporalTriple]]:
        """Get neighboring entities connected by triples.

        For directed graphs, ``direction`` controls whether to follow
        outgoing edges (entity is subject), incoming edges (entity is object),
        or both. For undirected graphs, direction is ignored — all connected
        triples are returned.

        Returns:
            List of (neighbor_entity, relation, triple) tuples.
        """
        if entity not in self._by_subject and entity not in self._by_object:
            return []

        results: list[tuple[str, str, TemporalTriple]] = []

        # For undirected, always get both directions
        effective_direction = "both" if not self._directed else direction

        if effective_direction in ("outgoing", "both"):
            for idx in self._by_subject.get(entity, []):
                t = self._triples[idx]
                results.append((t.object, t.relation, t))

        if effective_direction in ("incoming", "both"):
            for idx in self._by_object.get(entity, []):
                t = self._triples[idx]
                results.append((t.subject, t.relation, t))

        return results

    def _resolve_candidate_indexes(
        self,
        subject: str | None,
        relation: str | None,
        object: str | None,
    ) -> list[int] | range:
        """Pick the smallest index set to scan, or fall back to full scan."""
        candidates: list[list[int]] = []
        if subject is not None:
            candidates.append(self._by_subject.get(subject, []))
        if relation is not None:
            candidates.append(self._by_relation.get(relation, []))
        if object is not None:
            candidates.append(self._by_object.get(object, []))

        if not candidates:
            return range(len(self._triples))

        # Intersect starting from the smallest set
        candidates.sort(key=len)
        result_set = set(candidates[0])
        for c in candidates[1:]:
            result_set &= set(c)

        # Return in insertion order
        return sorted(result_set)

    def __repr__(self) -> str:
        return (
            f"TemporalKnowledgeGraph(triples={len(self)}, "
            f"entities={len(self.entities)}, "
            f"relations={len(self.relations)}, "
            f"directed={self._directed})"
        )
