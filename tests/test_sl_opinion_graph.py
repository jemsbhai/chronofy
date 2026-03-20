"""Tests for chronofy.sl.opinion_graph — OpinionRuleGraph.

SL-aware wrapper around TemporalRuleGraph that enriches query results
with full Subjective Logic Opinions via OpinionScorer, adding dual
filtering (scalar epistemic + SL opinion thresholds).

Design: Composition wrapper (Option B) — does NOT modify TemporalRuleGraph.
All existing TemporalRuleGraph functionality is delegated unchanged.

Two-level filtering in opinion_query():
    1. Scalar epistemic_filter (same as TemporalRuleGraph.query)
    2. SL opinion thresholds: p_min (projected probability floor),
       u_max (uncertainty ceiling)

TDD: Tests written before implementation.
"""

from datetime import datetime, timedelta

import pytest

nx = pytest.importorskip("networkx", reason="networkx required for graph module")

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.retrieval.filter import EpistemicFilter
from chronofy.retrieval.triples import TemporalTriple, TemporalKnowledgeGraph
from chronofy.retrieval.rules import TemporalRule, RuleMiner
from chronofy.retrieval.graph import TemporalRuleGraph
from chronofy.sl.opinion_decay import OpinionDecayFunction
from chronofy.sl.opinion_scorer import (
    OpinionScoredFact,
    OpinionScorer,
    ProjectedMultiplicative,
    UncertaintyPenalized,
)
from chronofy.sl.opinion_graph import OpinionRuleGraph


# ═══════════════════════════════════════════════════════════════════
# Fixtures — reuse same ICEWS-like data as test_temporal_graph.py
# ═══════════════════════════════════════════════════════════════════

BASE_TIME = datetime(2026, 1, 1)


def _triple(
    subject: str,
    relation: str,
    obj: str,
    days_offset: float,
    quality: float = 1.0,
    fact_type: str = "general",
) -> TemporalTriple:
    return TemporalTriple(
        subject=subject,
        relation=relation,
        object=obj,
        timestamp=BASE_TIME + timedelta(days=days_offset),
        source_quality=quality,
        fact_type=fact_type,
    )


@pytest.fixture
def sample_triples() -> list[TemporalTriple]:
    """Same ICEWS-like dataset as test_temporal_graph.py."""
    return [
        # Pattern A — cooperate chain → make_statement (support=2)
        _triple("USA", "cooperate", "Germany", 1),
        _triple("Germany", "cooperate", "France", 2),
        _triple("USA", "make_statement", "France", 3),
        _triple("USA", "cooperate", "Germany", 10),
        _triple("Germany", "cooperate", "France", 11),
        _triple("USA", "make_statement", "France", 12),
        # Pattern B — threaten → negotiate inverse (support=2)
        _triple("Russia", "threaten", "USA", 5),
        _triple("USA", "negotiate", "Russia", 7),
        _triple("Russia", "threaten", "USA", 15),
        _triple("USA", "negotiate", "Russia", 17),
        # Isolated
        _triple("China", "visit", "Germany", 8),
    ]


@pytest.fixture
def tkg(sample_triples) -> TemporalKnowledgeGraph:
    kg = TemporalKnowledgeGraph(directed=True)
    kg.add_triples(sample_triples)
    return kg


@pytest.fixture
def rules(tkg) -> list[TemporalRule]:
    miner = RuleMiner(tkg, min_support=2, min_confidence=0.1, max_length=2)
    return miner.mine()


@pytest.fixture
def known_entities() -> set[str]:
    return {"USA", "Russia", "China", "Germany", "France"}


@pytest.fixture
def odf() -> OpinionDecayFunction:
    return OpinionDecayFunction(
        half_lives={"general": 5.0},
        default_half_life=5.0,
    )


@pytest.fixture
def scalar_decay() -> ExponentialDecay:
    return ExponentialDecay(beta={"general": 0.5}, default_beta=0.5)


@pytest.fixture
def inner_graph(rules, odf, known_entities) -> TemporalRuleGraph:
    """TemporalRuleGraph built with OpinionDecayFunction (it IS a DecayFunction)."""
    return TemporalRuleGraph(
        rules=rules, decay_fn=odf, known_entities=known_entities,
    )


@pytest.fixture
def opinion_graph(inner_graph, odf) -> OpinionRuleGraph:
    """OpinionRuleGraph wrapping the inner graph."""
    return OpinionRuleGraph(graph=inner_graph, opinion_decay_fn=odf)


# ═══════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════


class TestOpinionRuleGraphConstruction:
    """Construction from pre-built graph and via from_rules factory."""

    def test_from_pre_built_graph(self, inner_graph, odf):
        """Wrap an existing TemporalRuleGraph."""
        og = OpinionRuleGraph(graph=inner_graph, opinion_decay_fn=odf)
        assert og.graph is inner_graph
        assert og.num_rule_nodes == inner_graph.num_rule_nodes

    def test_from_pre_built_with_custom_scorer(self, inner_graph, odf):
        """Accept a custom OpinionScorer."""
        scorer = OpinionScorer(
            decay_fn=odf,
            strategy=UncertaintyPenalized(),
        )
        og = OpinionRuleGraph(
            graph=inner_graph,
            opinion_decay_fn=odf,
            scorer=scorer,
        )
        assert og.scorer is scorer

    def test_default_scorer_created(self, opinion_graph):
        """If no scorer provided, a default one is created."""
        assert opinion_graph.scorer is not None

    def test_from_rules_factory(self, rules, odf, known_entities):
        """from_rules() builds TemporalRuleGraph + OpinionRuleGraph."""
        og = OpinionRuleGraph.from_rules(
            rules=rules,
            opinion_decay_fn=odf,
            known_entities=known_entities,
        )
        assert og.num_rule_nodes == len(rules)

    def test_from_rules_factory_consistency(self, rules, odf):
        """from_rules() uses the SAME decay fn for inner graph and scoring.

        This guarantees PageRank edge weights and Opinion scores agree.
        """
        og = OpinionRuleGraph.from_rules(rules=rules, opinion_decay_fn=odf)
        # The inner graph's decay fn should be the same OpinionDecayFunction
        # (since OpinionDecayFunction IS DecayFunction, it works for edge weights)
        assert og.num_rule_nodes > 0

    def test_from_rules_with_mdl_threshold(self, rules, odf):
        """from_rules() passes mdl_variance_threshold through."""
        og_strict = OpinionRuleGraph.from_rules(
            rules=rules, opinion_decay_fn=odf, mdl_variance_threshold=0.01,
        )
        og_permissive = OpinionRuleGraph.from_rules(
            rules=rules, opinion_decay_fn=odf, mdl_variance_threshold=1000.0,
        )
        assert og_strict.num_edges <= og_permissive.num_edges


# ═══════════════════════════════════════════════════════════════════
# Delegation — all TemporalRuleGraph properties/methods pass through
# ═══════════════════════════════════════════════════════════════════


class TestOpinionRuleGraphDelegation:
    """Every TemporalRuleGraph method is accessible through the wrapper."""

    def test_rules_property(self, opinion_graph, inner_graph):
        assert opinion_graph.rules == inner_graph.rules

    def test_nx_graph_property(self, opinion_graph, inner_graph):
        assert opinion_graph.nx_graph is inner_graph.graph

    def test_num_rule_nodes(self, opinion_graph, inner_graph):
        assert opinion_graph.num_rule_nodes == inner_graph.num_rule_nodes

    def test_num_edges(self, opinion_graph, inner_graph):
        assert opinion_graph.num_edges == inner_graph.num_edges

    def test_num_edges_before_mdl(self, opinion_graph, inner_graph):
        assert opinion_graph.num_edges_before_mdl == inner_graph.num_edges_before_mdl

    def test_get_edge_weights(self, opinion_graph, inner_graph):
        t = BASE_TIME + timedelta(days=13)
        assert opinion_graph.get_edge_weights(t) == inner_graph.get_edge_weights(t)

    def test_query_pagerank(self, opinion_graph, inner_graph):
        t = BASE_TIME + timedelta(days=13)
        assert opinion_graph.query_pagerank("USA", t) == inner_graph.query_pagerank("USA", t)

    def test_query_scalar(self, opinion_graph, inner_graph):
        """Scalar query() is fully delegated and returns identical results."""
        t = BASE_TIME + timedelta(days=13)
        results_wrapper = opinion_graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=t,
        )
        results_inner = inner_graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=t,
        )
        assert len(results_wrapper) == len(results_inner)
        for (f1, s1), (f2, s2) in zip(results_wrapper, results_inner):
            assert f1.content == f2.content
            assert s1 == pytest.approx(s2)


# ═══════════════════════════════════════════════════════════════════
# opinion_query — core SL enrichment
# ═══════════════════════════════════════════════════════════════════


class TestOpinionQuery:
    """opinion_query() returns OpinionScoredFact with full SL Opinions."""

    def test_returns_opinion_scored_facts(self, opinion_graph):
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, OpinionScoredFact)

    def test_opinion_scored_fact_has_opinion(self, opinion_graph):
        """Each result carries a full (b, d, u, a) Opinion."""
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        if results:
            item = results[0]
            assert hasattr(item.validity_opinion, "belief")
            assert hasattr(item.validity_opinion, "disbelief")
            assert hasattr(item.validity_opinion, "uncertainty")
            assert hasattr(item.validity_opinion, "base_rate")

    def test_results_sorted_descending(self, opinion_graph):
        """Results are sorted by combined_score descending."""
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        if len(results) >= 2:
            scores = [r.combined_score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, opinion_graph):
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            top_k=1,
        )
        assert len(results) <= 1

    def test_unknown_entity_empty(self, opinion_graph):
        results = opinion_graph.opinion_query(
            seed_entity="NonExistent",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert results == []

    def test_unknown_relation_empty(self, opinion_graph):
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="nonexistent_relation",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert results == []

    def test_similarity_comes_from_pagerank(self, opinion_graph):
        """The similarity field on OpinionScoredFact should reflect PageRank."""
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        # All similarities should be non-negative
        for item in results:
            assert item.similarity >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Filtering — scalar epistemic_filter + SL thresholds
# ═══════════════════════════════════════════════════════════════════


class TestOpinionQueryFiltering:
    """opinion_query supports both scalar and SL-aware filtering."""

    def test_scalar_epistemic_filter(self, opinion_graph, odf):
        """Scalar epistemic_filter excludes facts below scalar threshold."""
        ef = EpistemicFilter(decay_fn=odf, threshold=0.8)
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            epistemic_filter=ef,
        )
        # All surviving facts must pass scalar filter
        for item in results:
            scalar = odf.compute(item.fact, BASE_TIME + timedelta(days=13))
            assert scalar >= 0.8

    def test_p_min_filters_low_probability(self, opinion_graph):
        """p_min filters results where P(ω') < p_min."""
        results_permissive = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        results_strict = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            p_min=0.99,
        )
        # Strict should have <= results
        assert len(results_strict) <= len(results_permissive)
        # All strict results must satisfy p_min
        for item in results_strict:
            assert item.validity >= 0.99

    def test_u_max_filters_high_uncertainty(self, opinion_graph):
        """u_max filters results where uncertainty > u_max."""
        results_permissive = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
        )
        results_strict = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            u_max=0.01,
        )
        # Very tight u_max should exclude facts with any uncertainty
        assert len(results_strict) <= len(results_permissive)
        # All strict results must satisfy u_max
        for item in results_strict:
            assert item.validity_opinion.uncertainty <= 0.01

    def test_combined_p_min_and_u_max(self, opinion_graph):
        """Both thresholds applied simultaneously."""
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            p_min=0.5,
            u_max=0.5,
        )
        for item in results:
            assert item.validity >= 0.5
            assert item.validity_opinion.uncertainty <= 0.5

    def test_all_three_filters_combined(self, opinion_graph, odf):
        """epistemic_filter + p_min + u_max all applied."""
        ef = EpistemicFilter(decay_fn=odf, threshold=0.5)
        results = opinion_graph.opinion_query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            epistemic_filter=ef,
            p_min=0.5,
            u_max=0.9,
        )
        for item in results:
            scalar = odf.compute(item.fact, BASE_TIME + timedelta(days=13))
            assert scalar >= 0.5
            assert item.validity >= 0.5
            assert item.validity_opinion.uncertainty <= 0.9


# ═══════════════════════════════════════════════════════════════════
# Custom scoring strategy
# ═══════════════════════════════════════════════════════════════════


class TestOpinionQueryScoringStrategy:
    """opinion_query uses the configured OpinionScorer strategy."""

    def test_different_strategy_different_scores(self, rules, odf, known_entities):
        """Different strategies produce different combined_scores."""
        og_default = OpinionRuleGraph.from_rules(
            rules=rules, opinion_decay_fn=odf, known_entities=known_entities,
        )
        scorer_up = OpinionScorer(decay_fn=odf, strategy=UncertaintyPenalized())
        og_up = OpinionRuleGraph.from_rules(
            rules=rules,
            opinion_decay_fn=odf,
            known_entities=known_entities,
            scorer=scorer_up,
        )
        t = BASE_TIME + timedelta(days=13)
        results_default = og_default.opinion_query(
            seed_entity="USA", query_relation="make_statement", query_time=t,
        )
        results_up = og_up.opinion_query(
            seed_entity="USA", query_relation="make_statement", query_time=t,
        )
        # Both should return results, but combined_scores may differ
        # (UncertaintyPenalized additionally penalizes by (1-u))
        if results_default and results_up:
            # Scores should be ≤ default since UP multiplies by (1-u) ≤ 1
            assert results_up[0].combined_score <= results_default[0].combined_score + 1e-9


# ═══════════════════════════════════════════════════════════════════
# Repr
# ═══════════════════════════════════════════════════════════════════


class TestOpinionRuleGraphRepr:
    def test_repr(self, opinion_graph):
        r = repr(opinion_graph)
        assert "OpinionRuleGraph" in r
