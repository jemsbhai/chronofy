"""Grounded, decay-weighted temporal rule graph.

Given mined temporal rules, this module constructs a rule graph where nodes
are rules and edges connect rules that can chain (the head of one matches the
first body relation of another). Minimum Description Length (MDL) pruning
removes edges with erratic temporal gaps.

At query time, edge weights, entity personalization, and returned facts are
grounded in a caller-supplied :class:`TemporalKnowledgeGraph`. The graph never
invents evidence from aggregate rule statistics: query results retain the
timestamp, content, quality, fact type, and provenance of stored triples.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact
from chronofy.retrieval.rules import TemporalRule
from chronofy.retrieval.triples import TemporalKnowledgeGraph, TemporalTriple

if TYPE_CHECKING:
    from chronofy.retrieval.filter import EpistemicFilter

    RuleDiGraph = nx.DiGraph[int, dict[str, Any], dict[str, Any]]
else:
    # NetworkX's stubs are generic, but the runtime class is not subscriptable.
    RuleDiGraph = nx.DiGraph


class TemporalRuleGraph:
    """MDL-optimized rule graph with grounded Personalized PageRank.

    Nodes represent temporal rules. Edges connect rules whose head/body
    relations can chain. MDL pruning removes edges with high temporal gap
    variance. Edge weights and personalization use real stored triples and a
    pluggable :class:`DecayFunction`.

    Args:
        rules: Mined temporal rules to build the graph from.
        decay_fn: Pluggable decay function for edge weighting at query time.
        mdl_variance_threshold: Maximum allowed temporal gap variance for an
            edge to survive MDL pruning. Set to ``float("inf")`` to disable.
        known_entities: Optional legacy allowlist. It can reject unknown seeds
            but does not supply grounding by itself.
        knowledge_graph: Triple store supplying entity grounding and fact
            provenance. The graph snapshots its triples during construction.
            Without it, ranking scores and edge weights are zero and queries
            return no facts rather than synthesizing evidence.
    """

    def __init__(
        self,
        rules: list[TemporalRule],
        decay_fn: DecayFunction,
        mdl_variance_threshold: float = 50.0,
        known_entities: set[str] | None = None,
        knowledge_graph: TemporalKnowledgeGraph | None = None,
    ) -> None:
        self._rules = list(rules)
        self._decay_fn = decay_fn
        self._mdl_variance_threshold = mdl_variance_threshold
        self._known_entities: set[str] = known_entities or set()

        self._grounding_triples = (
            knowledge_graph.get_triples() if knowledge_graph is not None else []
        )
        self._triples_by_relation: dict[str, list[TemporalTriple]] = defaultdict(list)
        self._triples_by_entity: dict[str, list[TemporalTriple]] = defaultdict(list)
        for triple in self._grounding_triples:
            self._triples_by_relation[triple.relation].append(triple)
            self._triples_by_entity[triple.subject].append(triple)
            if triple.object != triple.subject:
                self._triples_by_entity[triple.object].append(triple)

        self._graph: RuleDiGraph = nx.DiGraph()
        self._rule_by_node: dict[int, TemporalRule] = {}
        self._node_by_head: dict[str, list[int]] = {}
        self._node_by_body_start: dict[str, list[int]] = {}

        self._edges_before_mdl = 0
        self._build_graph()

    def _build_graph(self) -> None:
        """Add rule nodes, connect chainable rules, and apply MDL pruning."""
        for node_id, rule in enumerate(self._rules):
            self._graph.add_node(node_id, rule=rule)
            self._rule_by_node[node_id] = rule
            self._node_by_head.setdefault(rule.head_relation, []).append(node_id)
            self._node_by_body_start.setdefault(rule.body_relations[0], []).append(
                node_id
            )

        for source_id, source_rule in enumerate(self._rules):
            target_nodes = self._node_by_body_start.get(
                source_rule.head_relation, []
            )
            for target_id in target_nodes:
                if source_id == target_id:
                    continue
                self._graph.add_edge(
                    source_id,
                    target_id,
                    temporal_gap_std=self._rules[target_id].temporal_gap_std,
                )

        self._edges_before_mdl = self._graph.number_of_edges()
        self._apply_mdl_pruning()

    def _apply_mdl_pruning(self) -> None:
        """Remove edges whose target rule has excessive gap variance."""
        edges_to_remove = [
            (source_id, target_id)
            for source_id, target_id in self._graph.edges()
            if self._rule_by_node[target_id].temporal_gap_std
            > self._mdl_variance_threshold
        ]
        self._graph.remove_edges_from(edges_to_remove)

    @property
    def rules(self) -> list[TemporalRule]:
        """All rules in the graph."""
        return list(self._rules)

    @property
    def graph(self) -> RuleDiGraph:
        """The underlying NetworkX graph for inspection and testing."""
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
        """Number of edges before MDL pruning."""
        return self._edges_before_mdl

    def get_edge_weights(self, query_time: datetime) -> list[float]:
        """Return grounded decay weights in graph edge iteration order.

        The weight of ``source -> target`` is the target rule confidence
        multiplied by the mean temporal validity of real triples having the
        target rule's head relation. An ungrounded target has weight zero.
        """
        return [
            self._compute_edge_weight(target_id, query_time)
            for _, target_id in self._graph.edges()
        ]

    def _compute_edge_weight(self, target_node: int, query_time: datetime) -> float:
        """Compute a target rule's confidence-weighted grounded validity."""
        rule = self._rule_by_node[target_node]
        validities = [
            self._decay_fn.compute(triple.to_temporal_fact(), query_time)
            for triple in self._triples_by_relation.get(rule.head_relation, [])
            if triple.timestamp <= query_time
        ]
        if not validities:
            return 0.0
        return float(rule.confidence * (sum(validities) / len(validities)))

    def query_pagerank(
        self,
        seed_entity: str,
        query_time: datetime,
        damping: float = 0.85,
    ) -> dict[int, float]:
        """Run entity-grounded Personalized PageRank.

        A rule is seeded only by real triples incident to ``seed_entity``
        whose relation appears in that rule's body or head. Incident means
        the entity may be either subject or object, including in directed
        knowledge graphs. Grounded validity determines the initial weight,
        allowing different entity evidence to produce different rankings.

        Unknown entities, future-only evidence, and graphs constructed
        without ``knowledge_graph`` safely return zero scores.
        """
        if self._graph.number_of_nodes() == 0:
            return {}

        personalization = self._build_personalization(seed_entity, query_time)
        if not personalization or all(
            weight == 0.0 for weight in personalization.values()
        ):
            return {node_id: 0.0 for node_id in self._graph.nodes()}

        weight_attr = "_decay_weight"
        for source_id, target_id in self._graph.edges():
            self._graph[source_id][target_id][weight_attr] = (
                self._compute_edge_weight(target_id, query_time)
            )

        try:
            return cast(
                dict[int, float],
                nx.pagerank(
                    self._graph,
                    alpha=damping,
                    personalization=personalization,
                    weight=weight_attr,
                    max_iter=200,
                    tol=1e-6,
                ),
            )
        except nx.PowerIterationFailedConvergence:
            return personalization

    def _build_personalization(
        self,
        seed_entity: str,
        query_time: datetime,
    ) -> dict[int, float]:
        """Build normalized rule weights from the seed's grounded triples."""
        zero_scores = {node_id: 0.0 for node_id in self._graph.nodes()}
        if self._known_entities and seed_entity not in self._known_entities:
            return zero_scores

        entity_triples = [
            triple
            for triple in self._triples_by_entity.get(seed_entity, [])
            if triple.timestamp <= query_time
        ]
        if not entity_triples:
            return zero_scores

        raw_weights = zero_scores.copy()
        for node_id, rule in self._rule_by_node.items():
            rule_relations = set(rule.body_relations)
            rule_relations.add(rule.head_relation)
            raw_weights[node_id] = sum(
                self._decay_fn.compute(triple.to_temporal_fact(), query_time)
                for triple in entity_triples
                if triple.relation in rule_relations
            )

        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            return zero_scores
        return {
            node_id: weight / total_weight
            for node_id, weight in raw_weights.items()
        }

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
        """Return scored, provenance-preserving facts for an entity.

        PageRank is seeded from grounded triples incident to ``seed_entity``.
        Candidates must be real, non-future triples with ``query_relation``
        and the seed as either subject or object. Each result comes directly
        from :meth:`TemporalTriple.to_temporal_fact`; no content, timestamp,
        quality, or fact type is fabricated.
        """
        if self._graph.number_of_nodes() == 0:
            return []

        pr_scores = self.query_pagerank(seed_entity, query_time, damping)
        matching_nodes = self._node_by_head.get(query_relation, [])
        relation_score = sum(
            pr_scores.get(node_id, 0.0) for node_id in matching_nodes
        )
        if relation_score <= 0.0:
            return []

        candidates: list[tuple[TemporalFact, float]] = []
        seen_triples: set[tuple[object, ...]] = set()
        for triple in self._triples_by_relation.get(query_relation, []):
            if seed_entity not in (triple.subject, triple.object):
                continue
            if triple.timestamp > query_time:
                continue

            triple_key = (
                triple.subject,
                triple.relation,
                triple.object,
                triple.timestamp,
                triple.source_quality,
                triple.fact_type,
            )
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            fact = triple.to_temporal_fact()
            final_score = relation_score * self._decay_fn.compute(fact, query_time)
            if final_score > 0.0 and final_score >= threshold:
                candidates.append((fact, final_score))

        if epistemic_filter is not None:
            candidates = [
                (fact, score)
                for fact, score in candidates
                if epistemic_filter.score(fact, query_time)
                >= epistemic_filter.threshold
            ]

        candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].timestamp.isoformat(),
                item[0].content,
                item[0].source or "",
            )
        )
        return candidates[:top_k]

    def __repr__(self) -> str:
        return (
            f"TemporalRuleGraph(rules={len(self._rules)}, "
            f"nodes={self.num_rule_nodes}, "
            f"edges={self.num_edges}/{self._edges_before_mdl}, "
            f"groundings={len(self._grounding_triples)})"
        )
