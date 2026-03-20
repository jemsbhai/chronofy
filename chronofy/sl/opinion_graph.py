"""SL-aware wrapper for TemporalRuleGraph.

Composes TemporalRuleGraph (Layer 2) with OpinionScorer to enrich
query results with full Subjective Logic Opinions, adding dual
filtering (scalar epistemic + SL opinion thresholds).

Design: Option B — composition wrapper. Does NOT modify TemporalRuleGraph.
All existing functionality is delegated unchanged. New SL enrichment is
added via opinion_query().

Consistency guarantee:
    The from_rules() factory builds the inner TemporalRuleGraph with the
    SAME OpinionDecayFunction used for Opinion scoring. Since
    OpinionDecayFunction IS DecayFunction, it works for both PageRank
    edge weights (scalar path via compute()) and Opinion enrichment
    (full SL path via compute_opinion()). This ensures that the ranking
    produced by PageRank and the Opinion validity scores agree — a fact
    ranked highly by PageRank will also have a high P(ω').

    Users wrapping a pre-built TemporalRuleGraph should ensure the inner
    graph's decay function is consistent with the OpinionDecayFunction
    passed to the wrapper. The from_rules() factory handles this
    automatically.

Two-level filtering in opinion_query():
    1. Scalar epistemic_filter: same as TemporalRuleGraph.query()
    2. SL opinion thresholds:
       - p_min: projected probability floor P(ω') ≥ p_min
       - u_max: uncertainty ceiling u ≤ u_max

    These are complementary, not redundant:
    - epistemic_filter operates on compute() (scalar)
    - p_min operates on projected_probability() (SL)
    - u_max catches high-uncertainty opinions that may still have
      acceptable P(ω') (e.g., P=0.6 with u=0.8 is different from
      P=0.6 with u=0.1 — the former is ignorance, the latter is
      genuine evidence)

Requires: pip install chronofy[sl]

References:
    Paper §III-C (Layer 2: Decay-Weighted Graph Traversal).
    Jøsang (2016) §3.2 (Projected Probability), §10.4 (Opinion Aging).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import networkx as nx

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact
from chronofy.retrieval.graph import TemporalRuleGraph
from chronofy.retrieval.rules import TemporalRule
from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.sl.opinion_scorer import (
    OpinionScoredFact,
    OpinionScorer,
    ProjectedMultiplicative,
)

if TYPE_CHECKING:
    from chronofy.retrieval.filter import EpistemicFilter


class OpinionRuleGraph:
    """SL-aware wrapper around TemporalRuleGraph.

    Delegates all TemporalRuleGraph functionality unchanged and adds
    opinion_query() for SL-enriched retrieval with full (b, d, u, a)
    Opinions per result.

    Args:
        graph: Pre-built TemporalRuleGraph to wrap.
        opinion_decay_fn: OpinionDecayFunction for computing Opinions.
        scorer: OpinionScorer for combining PageRank similarity with
            Opinion validity. If None, a default scorer with
            ProjectedMultiplicative strategy is created.
    """

    def __init__(
        self,
        graph: TemporalRuleGraph,
        opinion_decay_fn: OpinionDecayFunction,
        scorer: OpinionScorer | None = None,
    ) -> None:
        self._graph = graph
        self._opinion_decay_fn = opinion_decay_fn
        self._scorer = scorer or OpinionScorer(
            decay_fn=opinion_decay_fn,
            strategy=ProjectedMultiplicative(),
        )

    @property
    def scorer(self) -> OpinionScorer:
        """The OpinionScorer used for opinion_query()."""
        return self._scorer

    # ───────────────────────────────────────────────────────────
    # Factory
    # ───────────────────────────────────────────────────────────

    @classmethod
    def from_rules(
        cls,
        rules: list[TemporalRule],
        opinion_decay_fn: OpinionDecayFunction,
        mdl_variance_threshold: float = 50.0,
        known_entities: set[str] | None = None,
        scorer: OpinionScorer | None = None,
    ) -> OpinionRuleGraph:
        """Build a TemporalRuleGraph and wrap it in one step.

        Guarantees consistency: the inner graph uses the SAME
        OpinionDecayFunction for PageRank edge weights (via compute())
        and Opinion scoring (via compute_opinion()).

        Args:
            rules: Mined temporal rules.
            opinion_decay_fn: Shared decay function for both edge
                weights and Opinion scoring.
            mdl_variance_threshold: MDL pruning threshold for inner graph.
            known_entities: Known entity set for PageRank personalization.
            scorer: Optional custom OpinionScorer.
        """
        inner = TemporalRuleGraph(
            rules=rules,
            decay_fn=opinion_decay_fn,
            mdl_variance_threshold=mdl_variance_threshold,
            known_entities=known_entities,
        )
        return cls(
            graph=inner,
            opinion_decay_fn=opinion_decay_fn,
            scorer=scorer,
        )

    # ───────────────────────────────────────────────────────────
    # Delegation — full TemporalRuleGraph API
    # ───────────────────────────────────────────────────────────

    @property
    def graph(self) -> TemporalRuleGraph:
        """The underlying TemporalRuleGraph."""
        return self._graph

    @property
    def rules(self) -> list[TemporalRule]:
        return self._graph.rules

    @property
    def nx_graph(self) -> nx.DiGraph:
        """The underlying networkx graph."""
        return self._graph.graph

    @property
    def num_rule_nodes(self) -> int:
        return self._graph.num_rule_nodes

    @property
    def num_edges(self) -> int:
        return self._graph.num_edges

    @property
    def num_edges_before_mdl(self) -> int:
        return self._graph.num_edges_before_mdl

    def get_edge_weights(self, query_time: datetime) -> list[float]:
        """Delegate: compute decay-weighted edge weights."""
        return self._graph.get_edge_weights(query_time)

    def query_pagerank(
        self,
        seed_entity: str,
        query_time: datetime,
        damping: float = 0.85,
    ) -> dict[int, float]:
        """Delegate: run Seeded Personalized PageRank."""
        return self._graph.query_pagerank(seed_entity, query_time, damping)

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
        """Delegate: scalar query returning (TemporalFact, score) pairs."""
        return self._graph.query(
            seed_entity=seed_entity,
            query_relation=query_relation,
            query_time=query_time,
            top_k=top_k,
            threshold=threshold,
            damping=damping,
            epistemic_filter=epistemic_filter,
        )

    # ───────────────────────────────────────────────────────────
    # SL enrichment: opinion_query
    # ───────────────────────────────────────────────────────────

    def opinion_query(
        self,
        seed_entity: str,
        query_relation: str,
        query_time: datetime,
        top_k: int = 10,
        threshold: float = 0.0,
        damping: float = 0.85,
        epistemic_filter: EpistemicFilter | None = None,
        p_min: float | None = None,
        u_max: float | None = None,
    ) -> list[OpinionScoredFact]:
        """Query with full SL Opinion enrichment and dual filtering.

        Runs the scalar query() pipeline, then enriches each result
        with an OpinionScoredFact via OpinionScorer. The PageRank-based
        score from query() becomes the ``similarity`` input to the
        scoring strategy.

        Filtering is applied in order:
            1. Scalar epistemic_filter (during inner query)
            2. p_min: exclude results where P(ω') < p_min
            3. u_max: exclude results where uncertainty > u_max

        Args:
            seed_entity: Entity to seed PageRank from.
            query_relation: Target relation to retrieve.
            query_time: Time for decay computation.
            top_k: Maximum results to return (applied after all filtering).
            threshold: Minimum scalar score for inner query.
            damping: PageRank damping factor.
            epistemic_filter: Optional scalar filter (Layer 2).
            p_min: Minimum projected probability P(ω') for inclusion.
            u_max: Maximum uncertainty for inclusion.

        Returns:
            List of OpinionScoredFact, sorted descending by combined_score.
        """
        # Step 1: Run scalar query (includes epistemic_filter if provided)
        # Request more than top_k since SL filtering may remove some
        scalar_limit = top_k * 3 if (p_min is not None or u_max is not None) else top_k
        scalar_results = self._graph.query(
            seed_entity=seed_entity,
            query_relation=query_relation,
            query_time=query_time,
            top_k=scalar_limit,
            threshold=threshold,
            damping=damping,
            epistemic_filter=epistemic_filter,
        )

        if not scalar_results:
            return []

        # Step 2: Enrich each result with OpinionScoredFact
        # The PageRank × decay score from query() becomes "similarity"
        # Normalize scores to [0,1] for use as similarity input
        max_score = max(s for _, s in scalar_results)
        if max_score <= 0:
            return []

        enriched: list[OpinionScoredFact] = []
        for fact, score in scalar_results:
            similarity = score / max_score  # normalize to [0, 1]
            scored = self._scorer.score_fact(
                fact=fact,
                similarity=similarity,
                query_time=query_time,
            )
            enriched.append(scored)

        # Step 3: Apply SL opinion thresholds
        if p_min is not None:
            enriched = [r for r in enriched if r.validity >= p_min]

        if u_max is not None:
            enriched = [r for r in enriched if r.validity_opinion.uncertainty <= u_max]

        # Step 4: Sort descending by combined_score and truncate
        enriched.sort(key=lambda r: r.combined_score, reverse=True)
        return enriched[:top_k]

    def __repr__(self) -> str:
        return (
            f"OpinionRuleGraph("
            f"nodes={self.num_rule_nodes}, "
            f"edges={self.num_edges}, "
            f"scorer={self._scorer!r})"
        )
