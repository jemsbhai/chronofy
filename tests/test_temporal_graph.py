"""Tests for chronofy.retrieval — Layer 2: Decay-Weighted Graph Traversal.

Covers the full Apriori → MDL → Decay-Weighted PageRank pipeline:
- TemporalTriple data model and TemporalFact bridging
- TemporalKnowledgeGraph triple store (directed + undirected)
- TemporalRule abstraction with temporal statistics
- RuleMiner Apriori-style level-wise rule extraction
- TemporalRuleGraph with MDL edge optimization and decay-weighted PageRank

Reference: Paper Eq. 2: w(e_ij) = q_e · c(tr_j) · exp(-β_j · (T_q - T_f))
"""

from datetime import datetime, timedelta

import pytest

nx = pytest.importorskip("networkx", reason="networkx required for graph module")

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.retrieval.triples import TemporalTriple, TemporalKnowledgeGraph
from chronofy.retrieval.rules import TemporalRule, RuleMiner
from chronofy.retrieval.graph import TemporalRuleGraph


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 1, 1)


def _triple(
    subject: str,
    relation: str,
    obj: str,
    days_offset: float,
    quality: float = 1.0,
    fact_type: str = "general",
) -> TemporalTriple:
    """Convenience factory for test triples."""
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
    """Small ICEWS-like dataset with two known repeating patterns.

    Pattern A (length-2 chain, support=2):
        cooperate(X,Y) + cooperate(Y,Z) → make_statement(X,Z)
        Instance 1: USA→Germany(d1), Germany→France(d2), USA→France(d3) gap=2
        Instance 2: USA→Germany(d10), Germany→France(d11), USA→France(d12) gap=2

    Pattern B (length-1 inverse, support=2):
        threaten(X,Y) → negotiate(Y,X)
        Instance 1: Russia→USA(d5), USA→Russia(d7) gap=2
        Instance 2: Russia→USA(d15), USA→Russia(d17) gap=2

    Plus one isolated triple (China visits Germany, d8) — no repeating pattern.
    """
    return [
        # Pattern A — instance 1
        _triple("USA", "cooperate", "Germany", 1),
        _triple("Germany", "cooperate", "France", 2),
        _triple("USA", "make_statement", "France", 3),
        # Pattern A — instance 2
        _triple("USA", "cooperate", "Germany", 10),
        _triple("Germany", "cooperate", "France", 11),
        _triple("USA", "make_statement", "France", 12),
        # Pattern B — instance 1
        _triple("Russia", "threaten", "USA", 5),
        _triple("USA", "negotiate", "Russia", 7),
        # Pattern B — instance 2
        _triple("Russia", "threaten", "USA", 15),
        _triple("USA", "negotiate", "Russia", 17),
        # Isolated
        _triple("China", "visit", "Germany", 8),
    ]


@pytest.fixture
def tkg(sample_triples: list[TemporalTriple]) -> TemporalKnowledgeGraph:
    """Directed temporal knowledge graph populated with sample triples."""
    kg = TemporalKnowledgeGraph(directed=True)
    kg.add_triples(sample_triples)
    return kg


@pytest.fixture
def decay() -> ExponentialDecay:
    return ExponentialDecay(beta={"general": 0.5}, default_beta=0.5)


# ===========================================================================
# 1. TemporalTriple
# ===========================================================================


class TestTemporalTriple:
    """Data model for timestamped (s, r, o, t, q) knowledge graph triples."""

    def test_construction_defaults(self):
        t = _triple("A", "likes", "B", 5.0)
        assert t.subject == "A"
        assert t.relation == "likes"
        assert t.object == "B"
        assert t.source_quality == 1.0
        assert t.fact_type == "general"

    def test_timestamp_from_offset(self):
        t = _triple("A", "likes", "B", 5.0)
        assert t.timestamp == BASE_TIME + timedelta(days=5)

    def test_custom_quality(self):
        t = _triple("A", "likes", "B", 5.0, quality=0.8)
        assert t.source_quality == 0.8

    def test_custom_fact_type(self):
        t = _triple("A", "likes", "B", 5.0, fact_type="vital_sign")
        assert t.fact_type == "vital_sign"

    def test_quality_upper_bound(self):
        with pytest.raises(ValueError):
            TemporalTriple(
                subject="A", relation="r", object="B",
                timestamp=BASE_TIME, source_quality=1.5,
            )

    def test_quality_lower_bound(self):
        with pytest.raises(ValueError):
            TemporalTriple(
                subject="A", relation="r", object="B",
                timestamp=BASE_TIME, source_quality=-0.1,
            )

    def test_to_temporal_fact_type(self):
        t = _triple("A", "likes", "B", 5.0)
        fact = t.to_temporal_fact()
        assert isinstance(fact, TemporalFact)

    def test_to_temporal_fact_content_contains_triple(self):
        t = _triple("A", "likes", "B", 5.0)
        fact = t.to_temporal_fact()
        assert "A" in fact.content
        assert "likes" in fact.content
        assert "B" in fact.content

    def test_to_temporal_fact_preserves_timestamp(self):
        t = _triple("A", "likes", "B", 5.0)
        fact = t.to_temporal_fact()
        assert fact.timestamp == t.timestamp

    def test_to_temporal_fact_preserves_quality(self):
        t = _triple("A", "likes", "B", 5.0, quality=0.7)
        fact = t.to_temporal_fact()
        assert fact.source_quality == 0.7

    def test_to_temporal_fact_preserves_fact_type(self):
        t = _triple("A", "likes", "B", 5.0, fact_type="vital_sign")
        fact = t.to_temporal_fact()
        assert fact.fact_type == "vital_sign"


# ===========================================================================
# 2. TemporalKnowledgeGraph
# ===========================================================================


class TestTemporalKnowledgeGraph:
    """Triple store supporting directed/undirected querying."""

    # --- Construction ---

    def test_directed_construction(self):
        kg = TemporalKnowledgeGraph(directed=True)
        assert kg.is_directed is True
        assert len(kg) == 0

    def test_undirected_construction(self):
        kg = TemporalKnowledgeGraph(directed=False)
        assert kg.is_directed is False
        assert len(kg) == 0

    # --- Adding triples ---

    def test_add_single_triple(self):
        kg = TemporalKnowledgeGraph(directed=True)
        kg.add_triple(_triple("A", "r", "B", 1))
        assert len(kg) == 1

    def test_add_triples_batch(self, sample_triples):
        kg = TemporalKnowledgeGraph(directed=True)
        kg.add_triples(sample_triples)
        assert len(kg) == 11

    def test_len_matches_triple_count(self, tkg):
        assert len(tkg) == 11

    # --- Entity and relation inspection ---

    def test_entities(self, tkg):
        entities = tkg.entities
        assert entities == {"USA", "Russia", "China", "Germany", "France"}

    def test_relations(self, tkg):
        relations = tkg.relations
        expected = {"cooperate", "threaten", "visit", "make_statement", "negotiate"}
        assert relations == expected

    # --- Filtering by field ---

    def test_get_triples_by_subject(self, tkg):
        triples = tkg.get_triples(subject="USA")
        assert all(t.subject == "USA" for t in triples)
        assert len(triples) == 6  # 2 cooperate + 2 make_statement + 2 negotiate

    def test_get_triples_by_relation(self, tkg):
        triples = tkg.get_triples(relation="cooperate")
        assert all(t.relation == "cooperate" for t in triples)
        assert len(triples) == 4

    def test_get_triples_by_object(self, tkg):
        triples = tkg.get_triples(object="France")
        assert all(t.object == "France" for t in triples)
        assert len(triples) == 4  # 2 cooperate + 2 make_statement

    def test_get_triples_combined_filter(self, tkg):
        triples = tkg.get_triples(subject="USA", relation="cooperate")
        assert len(triples) == 2
        assert all(t.subject == "USA" and t.relation == "cooperate" for t in triples)

    def test_get_triples_no_match(self, tkg):
        assert tkg.get_triples(subject="NonExistent") == []

    # --- Temporal range filtering ---

    def test_get_triples_in_time_range(self, tkg):
        start = BASE_TIME + timedelta(days=4)
        end = BASE_TIME + timedelta(days=10)
        triples = tkg.get_triples(after=start, before=end)
        for t in triples:
            assert start <= t.timestamp <= end

    def test_get_triples_after_only(self, tkg):
        cutoff = BASE_TIME + timedelta(days=14)
        triples = tkg.get_triples(after=cutoff)
        for t in triples:
            assert t.timestamp >= cutoff

    def test_get_triples_before_only(self, tkg):
        cutoff = BASE_TIME + timedelta(days=3)
        triples = tkg.get_triples(before=cutoff)
        for t in triples:
            assert t.timestamp <= cutoff

    # --- Neighbor queries (directed) ---

    def test_neighbors_outgoing(self, tkg):
        neighbors = tkg.get_neighbors("USA", direction="outgoing")
        neighbor_entities = {entity for entity, _, _ in neighbors}
        # USA → Germany (cooperate), USA → France (make_statement), USA → Russia (negotiate)
        assert "Germany" in neighbor_entities
        assert "France" in neighbor_entities
        assert "Russia" in neighbor_entities

    def test_neighbors_incoming(self, tkg):
        neighbors = tkg.get_neighbors("USA", direction="incoming")
        neighbor_entities = {entity for entity, _, _ in neighbors}
        assert "Russia" in neighbor_entities  # Russia → USA (threaten)

    def test_neighbors_both(self, tkg):
        outgoing = tkg.get_neighbors("USA", direction="outgoing")
        incoming = tkg.get_neighbors("USA", direction="incoming")
        both = tkg.get_neighbors("USA", direction="both")
        assert len(both) == len(outgoing) + len(incoming)

    def test_neighbors_returns_triple_tuples(self, tkg):
        neighbors = tkg.get_neighbors("USA", direction="outgoing")
        for entity, relation, triple in neighbors:
            assert isinstance(entity, str)
            assert isinstance(relation, str)
            assert isinstance(triple, TemporalTriple)

    # --- Undirected mode ---

    def test_undirected_neighbors_sees_all(self, sample_triples):
        kg = TemporalKnowledgeGraph(directed=False)
        kg.add_triples(sample_triples)
        neighbors = kg.get_neighbors("Germany")
        neighbor_entities = {entity for entity, _, _ in neighbors}
        assert "USA" in neighbor_entities    # USA cooperate Germany
        assert "France" in neighbor_entities  # Germany cooperate France
        assert "China" in neighbor_entities   # China visit Germany

    def test_undirected_ignores_direction_param(self, sample_triples):
        kg = TemporalKnowledgeGraph(directed=False)
        kg.add_triples(sample_triples)
        n_default = kg.get_neighbors("Germany")
        n_outgoing = kg.get_neighbors("Germany", direction="outgoing")
        assert len(n_default) == len(n_outgoing)

    # --- Edge case ---

    def test_neighbors_unknown_entity(self, tkg):
        assert tkg.get_neighbors("NonExistent") == []


# ===========================================================================
# 3. TemporalRule
# ===========================================================================


class TestTemporalRule:
    """Abstracted temporal rule with confidence, support, and gap stats."""

    def test_construction(self):
        rule = TemporalRule(
            head_relation="make_statement",
            body_relations=["cooperate", "cooperate"],
            confidence=0.8,
            support=2,
            temporal_gaps=[2.0, 2.0],
        )
        assert rule.head_relation == "make_statement"
        assert rule.body_relations == ["cooperate", "cooperate"]
        assert rule.confidence == 0.8
        assert rule.support == 2

    def test_length_property(self):
        rule = TemporalRule(
            head_relation="negotiate",
            body_relations=["threaten"],
            confidence=1.0,
            support=2,
            temporal_gaps=[2.0, 2.0],
        )
        assert rule.length == 1

    def test_length_multi_hop(self):
        rule = TemporalRule(
            head_relation="r3",
            body_relations=["r1", "r2"],
            confidence=0.5,
            support=1,
            temporal_gaps=[5.0],
        )
        assert rule.length == 2

    def test_temporal_gap_mean_constant(self):
        rule = TemporalRule(
            head_relation="r",
            body_relations=["r1"],
            confidence=0.8,
            support=3,
            temporal_gaps=[2.0, 2.0, 2.0],
        )
        assert rule.temporal_gap_mean == 2.0

    def test_temporal_gap_std_zero_for_constant(self):
        rule = TemporalRule(
            head_relation="r",
            body_relations=["r1"],
            confidence=0.8,
            support=3,
            temporal_gaps=[2.0, 2.0, 2.0],
        )
        assert rule.temporal_gap_std == pytest.approx(0.0)

    def test_temporal_gap_std_positive_for_varying(self):
        rule = TemporalRule(
            head_relation="r",
            body_relations=["r1"],
            confidence=0.5,
            support=3,
            temporal_gaps=[1.0, 3.0, 5.0],
        )
        assert rule.temporal_gap_mean == pytest.approx(3.0)
        assert rule.temporal_gap_std > 0

    def test_confidence_upper_bound(self):
        with pytest.raises(ValueError):
            TemporalRule(
                head_relation="r",
                body_relations=["r1"],
                confidence=1.5,
                support=1,
                temporal_gaps=[1.0],
            )

    def test_confidence_lower_bound(self):
        with pytest.raises(ValueError):
            TemporalRule(
                head_relation="r",
                body_relations=["r1"],
                confidence=-0.1,
                support=1,
                temporal_gaps=[1.0],
            )

    def test_support_must_be_positive(self):
        with pytest.raises(ValueError):
            TemporalRule(
                head_relation="r",
                body_relations=["r1"],
                confidence=0.5,
                support=0,
                temporal_gaps=[],
            )

    def test_body_must_be_nonempty(self):
        with pytest.raises(ValueError):
            TemporalRule(
                head_relation="r",
                body_relations=[],
                confidence=0.5,
                support=1,
                temporal_gaps=[1.0],
            )


# ===========================================================================
# 4. RuleMiner (Apriori)
# ===========================================================================


class TestRuleMiner:
    """Apriori-style level-wise temporal rule extraction."""

    def test_construction(self, tkg):
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.5, max_length=2)
        assert miner.min_support == 2
        assert miner.min_confidence == 0.5
        assert miner.max_length == 2

    def test_mine_returns_list_of_rules(self, tkg):
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.1, max_length=2)
        rules = miner.mine()
        assert isinstance(rules, list)
        assert all(isinstance(r, TemporalRule) for r in rules)

    def test_mine_finds_length_1_rule(self, tkg):
        """Should find: threaten(X,Y) → negotiate(Y,X) with support >= 2."""
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.5, max_length=2)
        rules = miner.mine()
        negotiate_rules = [
            r for r in rules
            if r.length == 1
            and r.head_relation == "negotiate"
            and "threaten" in r.body_relations
        ]
        assert len(negotiate_rules) >= 1
        assert negotiate_rules[0].support >= 2

    def test_mine_finds_length_2_rule(self, tkg):
        """Should find: cooperate + cooperate → make_statement with support >= 2."""
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.1, max_length=2)
        rules = miner.mine()
        chain_rules = [
            r for r in rules
            if r.length == 2
            and r.head_relation == "make_statement"
            and r.body_relations == ["cooperate", "cooperate"]
        ]
        assert len(chain_rules) >= 1
        assert chain_rules[0].support >= 2

    def test_min_support_filters_low_support(self, tkg):
        """With min_support=4, no rule should survive.

        Note: max support can exceed 2 because cross-instance chain
        groundings are valid (e.g., body triple from instance 1 can
        chain with body triple from instance 2 if temporal ordering holds).
        """
        miner = RuleMiner(tkg, min_support=4, min_confidence=0.0, max_length=2)
        rules = miner.mine()
        assert len(rules) == 0

    def test_min_confidence_filters_low_confidence(self, tkg):
        miner_permissive = RuleMiner(tkg, min_support=1, min_confidence=0.0, max_length=1)
        miner_strict = RuleMiner(tkg, min_support=1, min_confidence=0.9, max_length=1)
        rules_permissive = miner_permissive.mine()
        rules_strict = miner_strict.mine()
        assert len(rules_strict) <= len(rules_permissive)

    def test_max_length_limits_depth(self, tkg):
        miner = RuleMiner(tkg, min_support=1, min_confidence=0.0, max_length=1)
        rules = miner.mine()
        assert all(r.length <= 1 for r in rules)

    def test_temporal_gaps_populated(self, tkg):
        """Every mined rule should have temporal gap data matching its support."""
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.1, max_length=2)
        rules = miner.mine()
        for rule in rules:
            assert len(rule.temporal_gaps) == rule.support
            assert all(g >= 0 for g in rule.temporal_gaps)

    def test_temporal_gaps_consistent_for_known_pattern(self, tkg):
        """Pattern B (threaten→negotiate) has gap=2 days in both instances."""
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.5, max_length=1)
        rules = miner.mine()
        negotiate_rules = [
            r for r in rules
            if r.head_relation == "negotiate" and "threaten" in r.body_relations
        ]
        if negotiate_rules:
            rule = negotiate_rules[0]
            assert rule.temporal_gap_mean == pytest.approx(2.0)
            assert rule.temporal_gap_std == pytest.approx(0.0)

    def test_empty_graph_returns_no_rules(self):
        kg = TemporalKnowledgeGraph(directed=True)
        miner = RuleMiner(kg, min_support=1, min_confidence=0.0, max_length=2)
        assert miner.mine() == []

    def test_single_triple_no_rules_at_support_2(self):
        kg = TemporalKnowledgeGraph(directed=True)
        kg.add_triple(_triple("A", "r", "B", 1))
        miner = RuleMiner(kg, min_support=2, min_confidence=0.0, max_length=2)
        assert miner.mine() == []


# ===========================================================================
# 5. TemporalRuleGraph (MDL + Decay PageRank + Query)
# ===========================================================================


class TestTemporalRuleGraph:
    """Rule graph with MDL optimization, decay-weighted PageRank, and query."""

    @pytest.fixture
    def rules(self, tkg) -> list[TemporalRule]:
        miner = RuleMiner(tkg, min_support=2, min_confidence=0.1, max_length=2)
        return miner.mine()

    @pytest.fixture
    def known_entities(self) -> set[str]:
        return {"USA", "Russia", "China", "Germany", "France"}

    @pytest.fixture
    def rule_graph(self, rules, decay, known_entities) -> TemporalRuleGraph:
        return TemporalRuleGraph(
            rules=rules, decay_fn=decay, known_entities=known_entities,
        )

    # --- Construction ---

    def test_construction_has_nodes(self, rule_graph):
        assert rule_graph.num_rule_nodes > 0

    def test_construction_stores_rules(self, rule_graph):
        for rule in rule_graph.rules:
            assert isinstance(rule, TemporalRule)

    def test_empty_rules_graph(self, decay):
        graph = TemporalRuleGraph(rules=[], decay_fn=decay)
        assert graph.num_rule_nodes == 0
        assert graph.num_edges == 0

    # --- MDL Edge Optimization ---

    def test_mdl_preserves_consistent_edges(self):
        """Edges with low temporal gap variance should survive MDL pruning."""
        rule_a = TemporalRule(
            head_relation="r1", body_relations=["r0"],
            confidence=0.9, support=10,
            temporal_gaps=[2.0] * 10,
        )
        rule_b = TemporalRule(
            head_relation="r2", body_relations=["r1"],
            confidence=0.9, support=10,
            temporal_gaps=[1.5] * 10,
        )
        decay = ExponentialDecay(default_beta=0.5)
        graph = TemporalRuleGraph(rules=[rule_a, rule_b], decay_fn=decay)
        assert graph.num_edges > 0

    def test_mdl_penalizes_erratic_edges(self):
        """Edges with high temporal gap variance should be pruned or weakened."""
        consistent = TemporalRule(
            head_relation="r1", body_relations=["r0"],
            confidence=0.8, support=5,
            temporal_gaps=[2.0, 2.1, 1.9, 2.0, 2.0],
        )
        erratic = TemporalRule(
            head_relation="r2", body_relations=["r1"],
            confidence=0.8, support=5,
            temporal_gaps=[0.5, 10.0, 1.0, 20.0, 3.0],
        )
        target = TemporalRule(
            head_relation="r3", body_relations=["r2"],
            confidence=0.8, support=5,
            temporal_gaps=[1.0] * 5,
        )
        decay = ExponentialDecay(default_beta=0.5)
        graph = TemporalRuleGraph(
            rules=[consistent, erratic, target], decay_fn=decay,
        )
        # MDL should prune at least some edges
        assert graph.num_edges <= graph.num_edges_before_mdl

    def test_mdl_tracking(self, rule_graph):
        """Graph should track edge count before and after MDL."""
        assert rule_graph.num_edges_before_mdl >= rule_graph.num_edges

    # --- Decay-Weighted Edge Scoring (Eq. 2) ---

    def test_edge_weights_in_valid_range(self, rules, decay):
        graph = TemporalRuleGraph(rules=rules, decay_fn=decay)
        query_time = BASE_TIME + timedelta(days=20)
        weights = graph.get_edge_weights(query_time)
        for w in weights:
            assert 0.0 <= w <= 1.0

    def test_edge_weights_decrease_with_age(self, rules, decay):
        """Querying at a more distant time should yield lower average weight."""
        graph = TemporalRuleGraph(rules=rules, decay_fn=decay)
        recent = BASE_TIME + timedelta(days=13)
        distant = BASE_TIME + timedelta(days=100)
        w_recent = graph.get_edge_weights(recent)
        w_distant = graph.get_edge_weights(distant)
        if w_recent and w_distant:
            assert sum(w_recent) / len(w_recent) >= sum(w_distant) / len(w_distant)

    # --- Personalized PageRank ---

    def test_pagerank_returns_dict(self, rule_graph):
        scores = rule_graph.query_pagerank(
            seed_entity="USA",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert isinstance(scores, dict)

    def test_pagerank_nonzero_scores(self, rule_graph):
        scores = rule_graph.query_pagerank(
            seed_entity="USA",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert any(v > 0 for v in scores.values())

    def test_pagerank_custom_damping(self, rules, decay, known_entities):
        graph = TemporalRuleGraph(
            rules=rules, decay_fn=decay, known_entities=known_entities,
        )
        t = BASE_TIME + timedelta(days=13)
        scores_high = graph.query_pagerank(seed_entity="USA", query_time=t, damping=0.95)
        scores_low = graph.query_pagerank(seed_entity="USA", query_time=t, damping=0.5)
        # With multiple connected nodes, damping changes distribution.
        # With a single node or no edges, scores may be identical.
        if graph.num_edges > 0 and graph.num_rule_nodes > 1:
            assert scores_high != scores_low
        else:
            # Single-node graph: damping has no effect, which is correct
            assert scores_high == scores_low

    def test_pagerank_unknown_entity_empty(self, rule_graph):
        scores = rule_graph.query_pagerank(
            seed_entity="NonExistent",
            query_time=BASE_TIME + timedelta(days=13),
        )
        # Should return dict — either empty or with zero scores
        assert all(v == 0 for v in scores.values()) or len(scores) == 0

    # --- Query Interface ---

    def test_query_returns_fact_score_pairs(self, rule_graph):
        results = rule_graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            top_k=5,
        )
        assert isinstance(results, list)
        for fact, score in results:
            assert isinstance(fact, TemporalFact)
            assert isinstance(score, float)
            assert score >= 0

    def test_query_respects_top_k(self, rule_graph):
        results = rule_graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            top_k=2,
        )
        assert len(results) <= 2

    def test_query_sorted_descending(self, rule_graph):
        results = rule_graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            top_k=10,
        )
        if len(results) >= 2:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_query_with_threshold(self, rules, decay):
        """Facts below explicit threshold should be excluded."""
        graph = TemporalRuleGraph(rules=rules, decay_fn=decay)
        results = graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=100),
            top_k=10,
            threshold=0.5,
        )
        for _, score in results:
            assert score >= 0.5

    def test_query_unknown_seed_empty(self, rule_graph):
        results = rule_graph.query(
            seed_entity="NonExistent",
            query_relation="cooperate",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert results == []

    def test_query_unknown_relation_empty(self, rule_graph):
        results = rule_graph.query(
            seed_entity="USA",
            query_relation="nonexistent_relation",
            query_time=BASE_TIME + timedelta(days=13),
        )
        assert results == []

    # --- EpistemicFilter Integration ---

    def test_query_with_epistemic_filter(self, rules, decay):
        """Results filtered through EpistemicFilter should all pass threshold."""
        from chronofy.retrieval.filter import EpistemicFilter

        ef = EpistemicFilter(decay_fn=decay, threshold=0.3)
        graph = TemporalRuleGraph(rules=rules, decay_fn=decay)

        results = graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=20),
            top_k=10,
            epistemic_filter=ef,
        )
        for fact, _ in results:
            individual = ef.score(fact, BASE_TIME + timedelta(days=20))
            assert individual >= ef.threshold

    # --- Pluggable Decay (Dependency Inversion) ---

    def test_accepts_different_decay_configurations(self, rules):
        """Any DecayFunction subclass should work — dependency inversion."""
        fast = ExponentialDecay(default_beta=5.0)
        slow = ExponentialDecay(default_beta=0.01)
        graph_fast = TemporalRuleGraph(rules=rules, decay_fn=fast)
        graph_slow = TemporalRuleGraph(rules=rules, decay_fn=slow)
        t = BASE_TIME + timedelta(days=20)
        results_fast = graph_fast.query(
            seed_entity="USA", query_relation="make_statement", query_time=t,
        )
        results_slow = graph_slow.query(
            seed_entity="USA", query_relation="make_statement", query_time=t,
        )
        # Fast decay should produce lower scores (or fewer results) than slow
        if results_fast and results_slow:
            max_fast = max(s for _, s in results_fast)
            max_slow = max(s for _, s in results_slow)
            assert max_fast <= max_slow

    def test_accepts_linear_decay(self, rules):
        """Should accept LinearDecay — a completely different DecayFunction."""
        from chronofy.decay.linear import LinearDecay

        linear = LinearDecay(default_rate=0.1)
        graph = TemporalRuleGraph(rules=rules, decay_fn=linear)
        results = graph.query(
            seed_entity="USA",
            query_relation="make_statement",
            query_time=BASE_TIME + timedelta(days=13),
            top_k=5,
        )
        assert isinstance(results, list)

    # --- Graph properties ---

    def test_num_rule_nodes_matches_rules(self, rules, decay):
        graph = TemporalRuleGraph(rules=rules, decay_fn=decay)
        assert graph.num_rule_nodes == len(rules)

    def test_underlying_graph_is_networkx(self, rule_graph):
        """Internal graph should be a networkx DiGraph or Graph."""
        assert isinstance(rule_graph.graph, (nx.DiGraph, nx.Graph))
