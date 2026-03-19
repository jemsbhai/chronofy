"""Temporal rule mining via Apriori-style level-wise extraction.

Implements the rule abstraction layer for Layer 2 of Chronofy. Individual
timestamped triples are abstracted into categorical rule nodes that capture
repeating temporal patterns in the knowledge graph.

A TemporalRule represents a pattern like:
    body_relation_1(X,Y) ∧ body_relation_2(Y,Z) → head_relation(X,Z)
with associated confidence, support count, and temporal gap statistics.

The RuleMiner implements level-wise Apriori extraction:
    Level 1: Single-hop rules  r_body(X,Y) → r_head(X,Z) or r_head(Y,Z)
    Level 2: Two-hop chains     r1(X,Y) ∧ r2(Y,Z) → r_head(X,Z)
    ...up to max_length.

Design notes (SOLID):
- Single Responsibility: this module handles ONLY rule extraction.
  It reads from TemporalKnowledgeGraph and produces TemporalRule objects.
  No graph construction, no decay weighting, no PageRank.
- Open/Closed: RuleMiner can be subclassed for alternative mining strategies.
- Dependency Inversion: depends on TemporalKnowledgeGraph abstraction,
  not on any concrete storage implementation.
- Interface Segregation: TemporalRule is a pure data object; RuleMiner
  is a pure computation object. They don't cross concerns.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime
from itertools import product as cartesian_product

from pydantic import BaseModel, Field, field_validator

from chronofy.retrieval.triples import TemporalKnowledgeGraph, TemporalTriple


class TemporalRule(BaseModel):
    """An abstracted temporal rule with confidence, support, and gap statistics.

    Represents:
        body_relations[0](X,Y) [∧ body_relations[1](Y,Z) ...] → head_relation(X,Z)

    Args:
        head_relation: The predicted relation.
        body_relations: Ordered list of premise relations forming a chain.
        confidence: Rule confidence = support / (support + counterexamples).
        support: Number of distinct grounding instances in the KG.
        temporal_gaps: Per-instance total time gap (days) from last body event
            to head event. Length should equal support.
    """

    head_relation: str
    body_relations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    support: int = Field(gt=0)
    temporal_gaps: list[float] = Field(default_factory=list)

    @field_validator("body_relations")
    @classmethod
    def _body_nonempty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("body_relations must be non-empty")
        return v

    @property
    def length(self) -> int:
        """Rule length = number of body relations (hops)."""
        return len(self.body_relations)

    @property
    def temporal_gap_mean(self) -> float:
        """Mean temporal gap across all grounding instances."""
        if not self.temporal_gaps:
            return 0.0
        return statistics.mean(self.temporal_gaps)

    @property
    def temporal_gap_std(self) -> float:
        """Standard deviation of temporal gaps.

        Returns 0.0 for single-instance rules (avoids statistics error).
        """
        if len(self.temporal_gaps) < 2:
            return 0.0
        return statistics.stdev(self.temporal_gaps)

    def __repr__(self) -> str:
        body = " ∧ ".join(self.body_relations)
        return (
            f"TemporalRule({body} → {self.head_relation}, "
            f"conf={self.confidence:.2f}, sup={self.support})"
        )


class RuleMiner:
    """Apriori-style level-wise temporal rule extraction.

    Mines repeating temporal patterns from a TemporalKnowledgeGraph by:
    1. Enumerating candidate rule bodies of increasing length (1, 2, ..., max_length)
    2. For each candidate, counting grounding instances (support)
    3. Computing confidence = support / total head occurrences with matching entities
    4. Recording temporal gaps for each grounding instance

    A "grounding instance" of a length-k rule is a sequence of k+1 triples
    where:
    - The k body triples form a chain (object_i == subject_{i+1})
    - The head triple connects the chain's first subject to its last object
    - All body triples precede the head triple temporally

    Args:
        tkg: The temporal knowledge graph to mine.
        min_support: Minimum support count for a rule to be retained.
        min_confidence: Minimum confidence for a rule to be retained.
        max_length: Maximum rule body length (number of hops).
    """

    def __init__(
        self,
        tkg: TemporalKnowledgeGraph,
        min_support: int = 2,
        min_confidence: float = 0.1,
        max_length: int = 2,
    ) -> None:
        self._tkg = tkg
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._max_length = max_length

    @property
    def min_support(self) -> int:
        return self._min_support

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def max_length(self) -> int:
        return self._max_length

    def mine(self) -> list[TemporalRule]:
        """Execute the mining process and return discovered rules.

        Returns:
            List of TemporalRule objects satisfying min_support and min_confidence.
        """
        if len(self._tkg) == 0:
            return []

        rules: list[TemporalRule] = []

        # Index all triples by relation for efficient enumeration
        triples_by_relation = self._index_by_relation()

        # All unique relations
        all_relations = list(self._tkg.relations)

        for length in range(1, self._max_length + 1):
            if length == 1:
                rules.extend(self._mine_length_1(triples_by_relation, all_relations))
            else:
                rules.extend(
                    self._mine_length_k(triples_by_relation, all_relations, length)
                )

        return rules

    def _index_by_relation(self) -> dict[str, list[TemporalTriple]]:
        """Group all triples by their relation type."""
        index: dict[str, list[TemporalTriple]] = defaultdict(list)
        for rel in self._tkg.relations:
            index[rel] = self._tkg.get_triples(relation=rel)
        return dict(index)

    def _mine_length_1(
        self,
        triples_by_rel: dict[str, list[TemporalTriple]],
        all_relations: list[str],
    ) -> list[TemporalRule]:
        """Mine length-1 rules: r_body(X,Y) → r_head(?,?).

        We consider four linking patterns for directed graphs:
            (a) r_body(X,Y) → r_head(X,Y)  same direction
            (b) r_body(X,Y) → r_head(Y,X)  inverse
            (c) r_body(X,Y) → r_head(X,Z)  shared subject
            (d) r_body(X,Y) → r_head(Z,Y)  shared object

        For tractability, we focus on patterns (a) and (b) which are the
        most common in temporal KG literature (TLogic, TempValid).
        """
        rules: list[TemporalRule] = []

        for body_rel, head_rel in cartesian_product(all_relations, repeat=2):
            if body_rel == head_rel:
                continue

            body_triples = triples_by_rel.get(body_rel, [])
            head_triples = triples_by_rel.get(head_rel, [])
            if not body_triples or not head_triples:
                continue

            # Index head triples by entity pairs for fast lookup
            head_by_pair: dict[tuple[str, str], list[TemporalTriple]] = defaultdict(list)
            for ht in head_triples:
                head_by_pair[(ht.subject, ht.object)].append(ht)

            # Pattern (a): same direction — body(X,Y) → head(X,Y)
            gaps_same = self._find_grounding_gaps(
                body_triples, head_by_pair, same_direction=True
            )
            if len(gaps_same) >= self._min_support:
                conf = self._compute_confidence(
                    len(gaps_same), body_triples, head_by_pair, same_direction=True
                )
                if conf >= self._min_confidence:
                    rules.append(TemporalRule(
                        head_relation=head_rel,
                        body_relations=[body_rel],
                        confidence=conf,
                        support=len(gaps_same),
                        temporal_gaps=gaps_same,
                    ))

            # Pattern (b): inverse — body(X,Y) → head(Y,X)
            gaps_inv = self._find_grounding_gaps(
                body_triples, head_by_pair, same_direction=False
            )
            if len(gaps_inv) >= self._min_support:
                conf = self._compute_confidence(
                    len(gaps_inv), body_triples, head_by_pair, same_direction=False
                )
                if conf >= self._min_confidence:
                    rules.append(TemporalRule(
                        head_relation=head_rel,
                        body_relations=[body_rel],
                        confidence=conf,
                        support=len(gaps_inv),
                        temporal_gaps=gaps_inv,
                    ))

        return rules

    def _mine_length_k(
        self,
        triples_by_rel: dict[str, list[TemporalTriple]],
        all_relations: list[str],
        k: int,
    ) -> list[TemporalRule]:
        """Mine length-k rules: r1(X,Y) ∧ r2(Y,Z) ∧ ... → r_head(X, last_entity).

        A length-k chain grounds as:
            t1: r1(e0, e1) at time t1
            t2: r2(e1, e2) at time t2 > t1
            ...
            tk: rk(e_{k-1}, e_k) at time tk > t_{k-1}
            head: r_head(e0, e_k) at time t_head > tk
        """
        rules: list[TemporalRule] = []

        # Generate all k-tuples of body relations
        for body_rels in cartesian_product(all_relations, repeat=k):
            for head_rel in all_relations:
                if head_rel in body_rels:
                    continue

                head_triples = triples_by_rel.get(head_rel, [])
                if not head_triples:
                    continue

                # Index head triples by (subject, object) pair
                head_by_pair: dict[tuple[str, str], list[TemporalTriple]] = defaultdict(
                    list
                )
                for ht in head_triples:
                    head_by_pair[(ht.subject, ht.object)].append(ht)

                # Find all valid chain groundings
                gaps = self._find_chain_groundings(
                    list(body_rels), triples_by_rel, head_by_pair
                )

                if len(gaps) >= self._min_support:
                    total_possible = self._count_chain_starts(
                        list(body_rels), triples_by_rel
                    )
                    conf = min(len(gaps) / max(total_possible, 1), 1.0)
                    if conf >= self._min_confidence:
                        rules.append(TemporalRule(
                            head_relation=head_rel,
                            body_relations=list(body_rels),
                            confidence=conf,
                            support=len(gaps),
                            temporal_gaps=gaps,
                        ))

        return rules

    def _find_grounding_gaps(
        self,
        body_triples: list[TemporalTriple],
        head_by_pair: dict[tuple[str, str], list[TemporalTriple]],
        same_direction: bool,
    ) -> list[float]:
        """Find temporal gaps for length-1 rule groundings.

        For each body triple, checks if a matching head triple exists
        at a later timestamp and records the gap.
        """
        gaps: list[float] = []
        used_body_timestamps: set[datetime] = set()

        for bt in body_triples:
            # Each body triple grounds at most once
            if bt.timestamp in used_body_timestamps:
                continue

            if same_direction:
                key = (bt.subject, bt.object)
            else:
                key = (bt.object, bt.subject)

            matching_heads = head_by_pair.get(key, [])
            for ht in matching_heads:
                if ht.timestamp <= bt.timestamp:
                    continue

                used_body_timestamps.add(bt.timestamp)
                gap_days = (ht.timestamp - bt.timestamp).total_seconds() / 86400.0
                gaps.append(gap_days)
                break  # Only count closest matching head per body instance

        return gaps

    def _compute_confidence(
        self,
        support: int,
        body_triples: list[TemporalTriple],
        head_by_pair: dict[tuple[str, str], list[TemporalTriple]],
        same_direction: bool,
    ) -> float:
        """Confidence = support / number of body triples that COULD match a head."""
        total = 0
        for bt in body_triples:
            if same_direction:
                key = (bt.subject, bt.object)
            else:
                key = (bt.object, bt.subject)
            # Count body triples where the entity pair exists in head_by_pair
            # regardless of temporal ordering
            if key in head_by_pair:
                total += 1

        return support / max(total, 1)

    def _find_chain_groundings(
        self,
        body_rels: list[str],
        triples_by_rel: dict[str, list[TemporalTriple]],
        head_by_pair: dict[tuple[str, str], list[TemporalTriple]],
    ) -> list[float]:
        """Find temporal gaps for length-k chain groundings.

        Recursively extends chains: at each step, the next body triple must
        have its subject equal to the current chain's last object, and its
        timestamp must be strictly after the previous body triple.
        """
        first_rel_triples = triples_by_rel.get(body_rels[0], [])
        gaps: list[float] = []

        for start_triple in first_rel_triples:
            # Build chains starting from this triple
            self._extend_chain(
                current_chain=[start_triple],
                remaining_rels=body_rels[1:],
                triples_by_rel=triples_by_rel,
                head_by_pair=head_by_pair,
                gaps=gaps,
            )

        return gaps

    def _extend_chain(
        self,
        current_chain: list[TemporalTriple],
        remaining_rels: list[str],
        triples_by_rel: dict[str, list[TemporalTriple]],
        head_by_pair: dict[tuple[str, str], list[TemporalTriple]],
        gaps: list[float],
    ) -> None:
        """Recursively extend a chain and check for head match."""
        last_triple = current_chain[-1]

        if not remaining_rels:
            # Chain complete — check for head match
            chain_start_entity = current_chain[0].subject
            chain_end_entity = last_triple.object
            key = (chain_start_entity, chain_end_entity)

            for ht in head_by_pair.get(key, []):
                if ht.timestamp > last_triple.timestamp:
                    gap = (ht.timestamp - last_triple.timestamp).total_seconds() / 86400.0
                    gaps.append(gap)
                    break  # One match per chain
            return

        # Extend chain with next relation
        next_rel = remaining_rels[0]
        for candidate in triples_by_rel.get(next_rel, []):
            # Chain linkage: candidate subject == last triple's object
            if candidate.subject != last_triple.object:
                continue
            # Temporal ordering: strictly after
            if candidate.timestamp <= last_triple.timestamp:
                continue

            self._extend_chain(
                current_chain=current_chain + [candidate],
                remaining_rels=remaining_rels[1:],
                triples_by_rel=triples_by_rel,
                head_by_pair=head_by_pair,
                gaps=gaps,
            )

    def _count_chain_starts(
        self,
        body_rels: list[str],
        triples_by_rel: dict[str, list[TemporalTriple]],
    ) -> int:
        """Count valid chain starts for confidence denominator."""
        first_rel_triples = triples_by_rel.get(body_rels[0], [])
        if len(body_rels) == 1:
            return len(first_rel_triples)

        count = 0
        for start in first_rel_triples:
            # Check if chain can be extended at least one step
            next_rel = body_rels[1]
            for candidate in triples_by_rel.get(next_rel, []):
                if (
                    candidate.subject == start.object
                    and candidate.timestamp > start.timestamp
                ):
                    count += 1
                    break
        return max(count, 1)

    def __repr__(self) -> str:
        return (
            f"RuleMiner(min_support={self._min_support}, "
            f"min_confidence={self._min_confidence}, "
            f"max_length={self._max_length})"
        )
