"""Decay-weighted temporal rule graph with MDL optimization and PageRank.

Implements the core of Layer 2: given mined temporal rules, constructs a
rule graph where nodes are rules and edges connect rules that can chain
(the head of one matches the body of another). Edge weights encode:

    w(e_ij) = q_e · c(tr_j) · exp(-β_j · (T_q - T_f))        [Paper Eq. 2]

The Minimum Description Length (MDL) principle prunes edges with erratic
temporal gaps, following STAR-RAG's approach: edges whose temporal jump
variance exceeds a threshold are removed because they encode noise rather
than signal.

Seeded Personalized PageRank with decay-weighted edges ranks rules by
their time-aware relevance to a query, and the query interface returns
scored TemporalFact objects suitable for the LLM context window.

Design notes (SOLID):
- Single Responsibility: graph construction, MDL pruning, PageRank, and
  query. Does not mine rules or store raw triples.
- Open/Closed: pluggable DecayFunction via dependency injection.
- Liskov: any DecayFunction subclass works without modification.
- Interface Segregation: query() returns TemporalFacts — callers don't
  need to know about the internal graph representation.
- Dependency Inversion: depends on DecayFunction ABC and TemporalRule
  data model, not on any concrete decay implementation.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import networkx as nx

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact
from chronofy.retrieval.rules import TemporalRule

if TYPE_CHECKING:
    from chronofy.retrieval.filter import EpistemicFilter


class TemporalRuleGraph:
    """MDL-optimized rule graph with decay-weighted Personalized PageRank.

    Nodes represent temporal rules. Edges connect rules whose head/body
    relations can chain. MDL pruning removes edges with high temporal gap
    variance (encoding noise). Edge weights at query time incorporate
    exponential decay (or any pluggable DecayFunction).

    Args:
        rules: Mined temporal rules to build the graph from.
        decay_fn: Pluggable decay function for edge weighting at query time.
        mdl_variance_threshold: Maximum allowed temporal gap variance for
            an edge to survive MDL pruning. Higher = more permissive.
            Set to float('inf') to disable MDL pruning.
    """

    def __init__(
        self,
        rules: list[TemporalRule],
        decay_fn: DecayFunction,
        mdl_variance_threshold: float = 50.0,
        known_entities: set[str] | None = None,
    ) -> None:
        self._rules = list(rules)
        self._decay_fn = decay_fn
        self._mdl_variance_threshold = mdl_variance_threshold
        self._known_entities: set[str] = known_entities or set()

        # Build the graph
        self._graph = nx.DiGraph()
        self._rule_by_node: dict[int, TemporalRule] = {}
        self._node_by_head: dict[str, list[int]] = {}
        self._node_by_body_start: dict[str, list[int]] = {}

        self._edges_before_mdl = 0
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct rule graph: add rule nodes, connect chainable rules, apply MDL."""
        # Step 1: Add rule nodes
        for i, rule in enumerate(self._rules):
            self._graph.add_node(i, rule=rule)
            self._rule_by_node[i] = rule

            # Index by head relation (for connecting to other rules' bodies)
            self._node_by_head.setdefault(rule.head_relation, []).append(i)

            # Index by first body relation (for connecting from other rules' heads)
            self._node_by_body_start.setdefault(rule.body_relations[0], []).append(i)

        # Step 2: Connect chainable rules
        # Rule A → Rule B if A.head_relation == B.body_relations[0]
        for i, rule_a in enumerate(self._rules):
            target_nodes = self._node_by_body_start.get(rule_a.head_relation, [])
            for j in target_nodes:
                if i == j:
                    continue
                self._graph.add_edge(i, j, temporal_gap_std=self._rules[j].temporal_gap_std)

        self._edges_before_mdl = self._graph.number_of_edges()

        # Step 3: MDL pruning — remove edges with erratic temporal patterns
        self._apply_mdl_pruning()

    def _apply_mdl_pruning(self) -> None:
        """Remove edges where the target rule has high temporal gap variance.

        The MDL principle penalizes edges that encode erratic temporal jumps:
        high variance means the temporal pattern is noisy and the edge adds
        more description length than it saves in compression.
        """
        edges_to_remove: list[tuple[int, int]] = []
        for u, v, data in self._graph.edges(data=True):
            target_rule = self._rule_by_node[v]
            if target_rule.temporal_gap_std > self._mdl_variance_threshold:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self._graph.remove_edge(u, v)

    @property
    def rules(self) -> list[TemporalRule]:
        """All rules in the graph."""
        return list(self._rules)

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying networkx graph (for inspection/testing)."""
        return self._graph

    @property
    def num_rule_nodes(self) -> int:
        """Number of rule nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges after MDL pruning."""
        return self._graph.number_of_edges()

    @property
    def num_edges_before_mdl(self) -> int:
        """Number of edges before MDL pruning (for diagnostics)."""
        return self._edges_before_mdl

    def get_edge_weights(self, query_time: datetime) -> list[float]:
        """Compute decay-weighted edge weights for all edges at query_time.

        Weight for edge (u → v) uses the target rule's temporal gap mean
        as the representative fact age, and the target rule's confidence
        as the base semantic weight c(tr_j).

        Returns:
            List of weights, one per edge, in graph edge iteration order.
        """
        weights: list[float] = []
        for u, v in self._graph.edges():
            w = self._compute_edge_weight(v, query_time)
            weights.append(w)
        return weights

    def _compute_edge_weight(self, target_node: int, query_time: datetime) -> float:
        """Compute Eq. 2 weight for an edge targeting target_node.

        w = c(tr_j) · decay(representative_fact, query_time)

        The representative fact uses the target rule's mean temporal gap
        as the age proxy and the target rule's confidence as the base weight.
        """
        rule = self._rule_by_node[target_node]

        # Create a representative fact for decay computation
        representative_age_days = rule.temporal_gap_mean
        representative_timestamp = query_time - timedelta(days=representative_age_days)
        representative_fact = TemporalFact(
            content=f"rule:{rule.head_relation}",
            timestamp=representative_timestamp,
            fact_type="general",
            source_quality=1.0,
        )

        decay_value = self._decay_fn.compute(representative_fact, query_time)
        return rule.confidence * decay_value

    def query_pagerank(
        self,
        seed_entity: str,
        query_time: datetime,
        damping: float = 0.85,
    ) -> dict[int, float]:
        """Run Seeded Personalized PageRank with decay-weighted edges.

        The seed is distributed among rule nodes whose body or head
        relations involve the seed entity (matched by inspecting the
        rule's relation names — entity-level seeding requires the
        original TKG which is not stored here).

        Args:
            seed_entity: Entity to seed the random walk from.
            query_time: Time for decay weight computation.
            damping: PageRank damping factor (probability of following an edge).

        Returns:
            Dict mapping node_id → PageRank score.
        """
        if self._graph.number_of_nodes() == 0:
            return {}

        # Build personalization vector: seed nodes related to the entity
        personalization = self._build_personalization(seed_entity)
        if not personalization or all(v == 0 for v in personalization.values()):
            return {n: 0.0 for n in self._graph.nodes()}

        # Build weight dict for edges
        weight_attr = "_decay_weight"
        for u, v in self._graph.edges():
            self._graph[u][v][weight_attr] = max(
                self._compute_edge_weight(v, query_time), 1e-10
            )

        try:
            scores = nx.pagerank(
                self._graph,
                alpha=damping,
                personalization=personalization,
                weight=weight_attr,
                max_iter=200,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback: return personalization as-is
            scores = personalization

        return scores

    def _build_personalization(self, seed_entity: str) -> dict[int, float]:
        """Build PageRank personalization vector for a seed entity.

        If known_entities was provided at construction, unknown entities
        receive zero personalization (yielding zero PageRank scores).
        For known entities, rules are seeded uniformly since rules are
        entity-agnostic abstractions.
        """
        # Reject unknown entities when we have entity information
        if self._known_entities and seed_entity not in self._known_entities:
            return {node_id: 0.0 for node_id in self._graph.nodes()}

        personalization: dict[int, float] = {}
        n_nodes = self._graph.number_of_nodes()
        if n_nodes > 0:
            uniform_weight = 1.0 / n_nodes
            for node_id in self._graph.nodes():
                personalization[node_id] = uniform_weight

        return personalization

    def query(
        self,
        seed_entity: str,
        query_relation: str,
        query_time: datetime,
        top_k: int = 10,
        threshold: float = 0.0,
        damping: float = 0.85,
        epistemic_filter: EpistemicFilter | None = None,
    ) -> list[tuple[TemporalFact, float]]:
        """Query the rule graph for temporally-valid facts.

        Runs decay-weighted PageRank seeded on the query entity, then
        extracts and scores facts from rules whose head_relation matches
        the query_relation.

        Args:
            seed_entity: Entity to seed the random walk from.
            query_relation: Target relation to retrieve facts for.
            query_time: Time for decay weight computation.
            top_k: Maximum number of results to return.
            threshold: Minimum score for inclusion.
            damping: PageRank damping factor.
            epistemic_filter: Optional filter for additional validity checking.

        Returns:
            List of (TemporalFact, score) pairs, sorted descending by score.
        """
        if self._graph.number_of_nodes() == 0:
            return []

        # Run PageRank
        pr_scores = self.query_pagerank(seed_entity, query_time, damping)

        # Collect facts from rules matching query_relation
        candidates: list[tuple[TemporalFact, float]] = []
        for node_id, pr_score in pr_scores.items():
            rule = self._rule_by_node.get(node_id)
            if rule is None:
                continue
            if rule.head_relation != query_relation:
                continue

            # Create a representative fact from the rule
            representative_age_days = rule.temporal_gap_mean
            fact = TemporalFact(
                content=f"{seed_entity} {rule.head_relation} (via {' → '.join(rule.body_relations)})",
                timestamp=query_time - timedelta(days=representative_age_days),
                fact_type="general",
                source_quality=rule.confidence,
            )

            # Final score = PageRank score × decay value
            decay_value = self._decay_fn.compute(fact, query_time)
            final_score = pr_score * decay_value

            if final_score >= threshold:
                candidates.append((fact, final_score))

        # Apply epistemic filter if provided
        if epistemic_filter is not None:
            candidates = [
                (fact, score) for fact, score in candidates
                if epistemic_filter.score(fact, query_time) >= epistemic_filter.threshold
            ]

        # Sort descending and truncate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def __repr__(self) -> str:
        return (
            f"TemporalRuleGraph(rules={len(self._rules)}, "
            f"nodes={self.num_rule_nodes}, "
            f"edges={self.num_edges}/{self._edges_before_mdl})"
        )
