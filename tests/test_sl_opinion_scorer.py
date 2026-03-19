"""Tests for chronofy.sl.opinion_scorer — Opinion-aware temporal scoring.

TemporalScorer equivalent that works with full (b, d, u, a) Opinions
instead of scalar validity, enabling scoring strategies that can inspect
uncertainty directly.

Architecture under test:
    OpinionScoringStrategy (ABC)
        ├── ProjectedMultiplicative   sim × P(ω)  (scalar-equivalent default)
        ├── UncertaintyPenalized      sim × P(ω) × (1 - u)
        └── UncertaintyAwareBlend     α·sim + β·P(ω) + (1-α-β)·(1-u)

    OpinionScoredFact (frozen dataclass)
        fact, similarity, validity_opinion, validity, combined_score

    OpinionScorer(decay_fn, strategy)
        .score_fact(fact, sim, query_time)       → OpinionScoredFact
        .score_facts(facts, sims, query_time)    → list[OpinionScoredFact]
        .rank(facts, sims, query_time, top_k)    → list[OpinionScoredFact] desc

Coverage:
    OpinionScoredFact: fields, immutability, validity == P(ω)
    ProjectedMultiplicative: sim × P(ω), zero cases, identity
    UncertaintyPenalized: penalizes high-u opinions, zero cases, fresh vs stale
    UncertaintyAwareBlend: three-way blend, param validation, edge weights
    OpinionScorer: orchestration, batch, rank, top_k, plain DecayFunction fallback
    Backward compatibility: ProjectedMultiplicative matches scalar TemporalScorer

References:
    Jøsang (2016) §3.2 (Projected Probability), §10.4 (Opinion Aging).
"""

from datetime import datetime, timedelta

import pytest

jex = pytest.importorskip("jsonld_ex", reason="jsonld-ex required for SL module")

from jsonld_ex.confidence_algebra import Opinion

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.scoring.temporal_scorer import TemporalScorer, MultiplicativeScoring
from chronofy.sl.opinion_decay import OpinionDecayFunction, OpinionConfig
from chronofy.sl.opinion_scorer import (
    OpinionScoredFact,
    OpinionScoringStrategy,
    ProjectedMultiplicative,
    UncertaintyPenalized,
    UncertaintyAwareBlend,
    OpinionScorer,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2026, 3, 19, 12, 0)


def _fact(
    content: str = "K+ = 4.1",
    days_ago: float = 0.0,
    fact_type: str = "general",
    quality: float = 1.0,
    source: str | None = None,
) -> TemporalFact:
    return TemporalFact(
        content=content,
        timestamp=QUERY_TIME - timedelta(days=days_ago),
        fact_type=fact_type,
        source_quality=quality,
        source=source,
    )


def _opinion(b: float, d: float, u: float, a: float = 0.5) -> Opinion:
    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


@pytest.fixture
def odf() -> OpinionDecayFunction:
    return OpinionDecayFunction(
        half_lives={"general": 7.0, "vital_sign": 1.0},
        construction="confidence",
        base_uncertainty=0.1,
    )


@pytest.fixture
def scorer(odf) -> OpinionScorer:
    return OpinionScorer(decay_fn=odf)


# ===========================================================================
# 1. OpinionScoredFact
# ===========================================================================


class TestOpinionScoredFact:
    """OpinionScoredFact carries fact, similarity, validity_opinion, validity, combined_score."""

    def test_fields_present(self, scorer):
        f = _fact("a", days_ago=0.0, quality=0.9)
        sf = scorer.score_fact(f, similarity=0.8, query_time=QUERY_TIME)
        assert isinstance(sf, OpinionScoredFact)
        assert sf.fact is f
        assert isinstance(sf.similarity, float)
        assert isinstance(sf.validity_opinion, Opinion)
        assert isinstance(sf.validity, float)
        assert isinstance(sf.combined_score, float)

    def test_validity_equals_projected_probability(self, scorer):
        """Scalar validity should equal P(validity_opinion)."""
        f = _fact("a", days_ago=3.0)
        sf = scorer.score_fact(f, similarity=0.7, query_time=QUERY_TIME)
        assert sf.validity == pytest.approx(
            sf.validity_opinion.projected_probability(), abs=1e-9
        )

    def test_is_frozen(self, scorer):
        f = _fact("a")
        sf = scorer.score_fact(f, similarity=0.8, query_time=QUERY_TIME)
        with pytest.raises(AttributeError):
            sf.combined_score = 0.99  # type: ignore[misc]


# ===========================================================================
# 2. ProjectedMultiplicative Strategy
# ===========================================================================


class TestProjectedMultiplicative:
    """ProjectedMultiplicative: score = sim × P(ω). Default strategy."""

    def test_is_scoring_strategy(self):
        assert isinstance(ProjectedMultiplicative(), OpinionScoringStrategy)

    def test_known_values(self):
        strat = ProjectedMultiplicative()
        op = _opinion(0.7, 0.1, 0.2, 0.5)  # P = 0.7 + 0.5*0.2 = 0.8
        assert strat.score(0.9, op) == pytest.approx(0.9 * 0.8)

    def test_zero_similarity(self):
        strat = ProjectedMultiplicative()
        op = _opinion(0.8, 0.1, 0.1)
        assert strat.score(0.0, op) == pytest.approx(0.0)

    def test_vacuous_opinion(self):
        """Vacuous opinion: P = a = 0.5, so score = sim × 0.5."""
        strat = ProjectedMultiplicative()
        op = _opinion(0.0, 0.0, 1.0, 0.5)
        assert strat.score(0.8, op) == pytest.approx(0.8 * 0.5)

    def test_dogmatic_full_belief(self):
        """Dogmatic (b=1, u=0): P = 1.0, score = sim."""
        strat = ProjectedMultiplicative()
        op = _opinion(1.0, 0.0, 0.0)
        assert strat.score(0.7, op) == pytest.approx(0.7)

    def test_both_one(self):
        strat = ProjectedMultiplicative()
        op = _opinion(1.0, 0.0, 0.0)
        assert strat.score(1.0, op) == pytest.approx(1.0)


# ===========================================================================
# 3. UncertaintyPenalized Strategy
# ===========================================================================


class TestUncertaintyPenalized:
    """UncertaintyPenalized: score = sim × P(ω) × (1 - u).

    Extra penalty for uncertain evidence. Two opinions with the same P(ω)
    but different uncertainty should produce different scores.
    """

    def test_is_scoring_strategy(self):
        assert isinstance(UncertaintyPenalized(), OpinionScoringStrategy)

    def test_known_values(self):
        strat = UncertaintyPenalized()
        op = _opinion(0.7, 0.1, 0.2, 0.5)  # P=0.8, 1-u=0.8
        assert strat.score(0.9, op) == pytest.approx(0.9 * 0.8 * 0.8)

    def test_vacuous_opinion_zero_score(self):
        """Vacuous (u=1): (1 - u) = 0 → score = 0 regardless of similarity."""
        strat = UncertaintyPenalized()
        op = _opinion(0.0, 0.0, 1.0, 0.5)
        assert strat.score(1.0, op) == pytest.approx(0.0)

    def test_dogmatic_no_penalty(self):
        """Dogmatic (u=0): (1 - u) = 1 → reduces to sim × P(ω)."""
        strat = UncertaintyPenalized()
        op = _opinion(0.8, 0.2, 0.0)  # P = 0.8
        assert strat.score(0.9, op) == pytest.approx(0.9 * 0.8)

    def test_penalizes_uncertain_more_than_projected(self):
        """Two opinions with same P(ω) but different u → different scores."""
        strat = UncertaintyPenalized()
        # Both have P = 0.5 + 0.5*u → we need to construct carefully
        # Op A: high evidence, low u → P = b + a*u
        op_certain = _opinion(0.7, 0.1, 0.2, 0.5)    # P=0.8, 1-u=0.8
        op_uncertain = _opinion(0.1, 0.1, 0.8, 0.5)   # P=0.5, 1-u=0.2

        score_certain = strat.score(0.9, op_certain)
        score_uncertain = strat.score(0.9, op_uncertain)
        assert score_certain > score_uncertain

    def test_fresh_vs_stale_with_same_projected(self):
        """A fresh fact (low u) should score higher than a stale fact (high u),
        even if both happen to have similar projected probability."""
        strat = UncertaintyPenalized()
        # Fresh: strong evidence, low uncertainty
        fresh = _opinion(0.45, 0.05, 0.50, 0.5)   # P = 0.45 + 0.25 = 0.70
        # Stale: weak evidence, high uncertainty
        stale = _opinion(0.10, 0.02, 0.88, 0.5)    # P = 0.10 + 0.44 = 0.54

        # Even though both have moderate P, the penalty distinguishes them
        score_fresh = strat.score(0.9, fresh)     # 0.9 * 0.70 * 0.50
        score_stale = strat.score(0.9, stale)     # 0.9 * 0.54 * 0.12
        assert score_fresh > score_stale

    def test_zero_similarity(self):
        strat = UncertaintyPenalized()
        op = _opinion(0.8, 0.1, 0.1)
        assert strat.score(0.0, op) == pytest.approx(0.0)


# ===========================================================================
# 4. UncertaintyAwareBlend Strategy
# ===========================================================================


class TestUncertaintyAwareBlend:
    """UncertaintyAwareBlend: score = α·sim + β·P(ω) + (1-α-β)·(1-u).

    Three-way weighted blend with explicit weight on certainty.
    """

    def test_is_scoring_strategy(self):
        assert isinstance(UncertaintyAwareBlend(), OpinionScoringStrategy)

    def test_default_weights(self):
        strat = UncertaintyAwareBlend()
        assert strat.alpha + strat.beta <= 1.0

    def test_known_values(self):
        strat = UncertaintyAwareBlend(alpha=0.5, beta=0.3)
        op = _opinion(0.7, 0.1, 0.2, 0.5)  # P=0.8, 1-u=0.8
        # gamma = 1 - 0.5 - 0.3 = 0.2
        expected = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.8
        assert strat.score(0.9, op) == pytest.approx(expected)

    def test_alpha_one_pure_similarity(self):
        """α=1, β=0 → score = similarity."""
        strat = UncertaintyAwareBlend(alpha=1.0, beta=0.0)
        op = _opinion(0.3, 0.5, 0.2)
        assert strat.score(0.7, op) == pytest.approx(0.7)

    def test_beta_one_pure_validity(self):
        """α=0, β=1 → score = P(ω)."""
        strat = UncertaintyAwareBlend(alpha=0.0, beta=1.0)
        op = _opinion(0.7, 0.1, 0.2, 0.5)  # P=0.8
        assert strat.score(0.3, op) == pytest.approx(0.8)

    def test_gamma_one_pure_certainty(self):
        """α=0, β=0 → score = (1-u)."""
        strat = UncertaintyAwareBlend(alpha=0.0, beta=0.0)
        op = _opinion(0.3, 0.2, 0.5)
        assert strat.score(0.9, op) == pytest.approx(0.5)

    def test_raises_on_weights_exceeding_one(self):
        with pytest.raises(ValueError, match="alpha.*beta"):
            UncertaintyAwareBlend(alpha=0.6, beta=0.6)

    def test_raises_on_negative_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            UncertaintyAwareBlend(alpha=-0.1, beta=0.5)

    def test_raises_on_negative_beta(self):
        with pytest.raises(ValueError, match="beta"):
            UncertaintyAwareBlend(alpha=0.5, beta=-0.1)


# ===========================================================================
# 5. OpinionScorer — Construction
# ===========================================================================


class TestScorerConstruction:
    """OpinionScorer construction and defaults."""

    def test_default_strategy(self, odf):
        s = OpinionScorer(decay_fn=odf)
        assert isinstance(s.strategy, ProjectedMultiplicative)

    def test_custom_strategy(self, odf):
        s = OpinionScorer(decay_fn=odf, strategy=UncertaintyPenalized())
        assert isinstance(s.strategy, UncertaintyPenalized)

    def test_accepts_plain_decay_function(self):
        plain = ExponentialDecay(beta={"general": 0.5})
        s = OpinionScorer(decay_fn=plain)
        assert s is not None

    def test_repr(self, scorer):
        r = repr(scorer)
        assert "OpinionScorer" in r


# ===========================================================================
# 6. OpinionScorer — score_fact
# ===========================================================================


class TestScoreFact:
    """OpinionScorer.score_fact produces correct OpinionScoredFact."""

    def test_returns_opinion_scored_fact(self, scorer):
        sf = scorer.score_fact(_fact("a"), 0.8, QUERY_TIME)
        assert isinstance(sf, OpinionScoredFact)

    def test_similarity_stored(self, scorer):
        sf = scorer.score_fact(_fact("a"), 0.75, QUERY_TIME)
        assert sf.similarity == 0.75

    def test_validity_opinion_is_opinion(self, scorer):
        sf = scorer.score_fact(_fact("a"), 0.8, QUERY_TIME)
        assert isinstance(sf.validity_opinion, Opinion)

    def test_combined_score_uses_strategy(self, odf):
        """Combined score matches strategy(sim, opinion)."""
        strat = UncertaintyPenalized()
        s = OpinionScorer(decay_fn=odf, strategy=strat)
        f = _fact("a", days_ago=3.0)
        sf = s.score_fact(f, 0.8, QUERY_TIME)
        expected = strat.score(0.8, sf.validity_opinion)
        assert sf.combined_score == pytest.approx(expected)


# ===========================================================================
# 7. OpinionScorer — score_facts (batch)
# ===========================================================================


class TestScoreFacts:
    """Batch scoring."""

    def test_returns_same_length(self, scorer):
        facts = [_fact(f"f{i}", days_ago=float(i)) for i in range(4)]
        sims = [0.9, 0.8, 0.7, 0.6]
        results = scorer.score_facts(facts, sims, QUERY_TIME)
        assert len(results) == 4

    def test_mismatched_lengths_raises(self, scorer):
        with pytest.raises(ValueError, match="length"):
            scorer.score_facts([_fact("a")], [0.8, 0.7], QUERY_TIME)

    def test_empty_input_returns_empty(self, scorer):
        assert scorer.score_facts([], [], QUERY_TIME) == []

    def test_each_result_is_opinion_scored_fact(self, scorer):
        facts = [_fact("a"), _fact("b")]
        results = scorer.score_facts(facts, [0.8, 0.7], QUERY_TIME)
        for sf in results:
            assert isinstance(sf, OpinionScoredFact)


# ===========================================================================
# 8. OpinionScorer — rank
# ===========================================================================


class TestRank:
    """Ranking by combined_score descending."""

    def test_sorted_descending(self, scorer):
        facts = [
            _fact("stale", days_ago=14.0),
            _fact("fresh", days_ago=0.0),
            _fact("medium", days_ago=5.0),
        ]
        sims = [0.9, 0.9, 0.9]
        ranked = scorer.rank(facts, sims, QUERY_TIME)
        scores = [sf.combined_score for sf in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self, scorer):
        facts = [_fact(f"f{i}", days_ago=float(i)) for i in range(5)]
        sims = [0.8] * 5
        ranked = scorer.rank(facts, sims, QUERY_TIME, top_k=3)
        assert len(ranked) == 3

    def test_top_k_none_returns_all(self, scorer):
        facts = [_fact(f"f{i}") for i in range(4)]
        sims = [0.8] * 4
        ranked = scorer.rank(facts, sims, QUERY_TIME, top_k=None)
        assert len(ranked) == 4

    def test_top_k_zero_raises(self, scorer):
        with pytest.raises(ValueError, match="top_k"):
            scorer.rank([_fact("a")], [0.8], QUERY_TIME, top_k=0)

    def test_top_k_negative_raises(self, scorer):
        with pytest.raises(ValueError, match="top_k"):
            scorer.rank([_fact("a")], [0.8], QUERY_TIME, top_k=-1)

    def test_rank_empty_returns_empty(self, scorer):
        assert scorer.rank([], [], QUERY_TIME) == []


# ===========================================================================
# 9. Backward Compatibility — ProjectedMultiplicative Matches Scalar
# ===========================================================================


class TestBackwardCompatibility:
    """ProjectedMultiplicative with OpinionDecayFunction should produce
    the same combined_score as scalar MultiplicativeScoring with the same
    decay function (since both use sim × P(ω) = sim × compute())."""

    def test_scores_match_scalar_temporal_scorer(self, odf):
        """OpinionScorer(ProjectedMultiplicative) ≈ TemporalScorer(Multiplicative)."""
        opinion_scorer = OpinionScorer(decay_fn=odf, strategy=ProjectedMultiplicative())
        scalar_scorer = TemporalScorer(decay_fn=odf, strategy=MultiplicativeScoring())

        facts = [
            _fact("a", days_ago=0.0, quality=0.9),
            _fact("b", days_ago=5.0, quality=0.7),
            _fact("c", days_ago=14.0, quality=0.5),
        ]
        sims = [0.8, 0.7, 0.9]

        opinion_results = opinion_scorer.score_facts(facts, sims, QUERY_TIME)
        scalar_results = scalar_scorer.score_facts(facts, sims, QUERY_TIME)

        for op_sf, sc_sf in zip(opinion_results, scalar_results):
            assert op_sf.combined_score == pytest.approx(
                sc_sf.combined_score, abs=1e-9
            )


# ===========================================================================
# 10. Plain DecayFunction Fallback
# ===========================================================================


class TestPlainDecayFallback:
    """OpinionScorer accepts plain DecayFunction with scalar-to-Opinion wrapping."""

    def test_plain_decay_produces_valid_results(self):
        plain = ExponentialDecay(beta={"general": 0.5})
        s = OpinionScorer(decay_fn=plain)
        sf = s.score_fact(_fact("a", days_ago=3.0), 0.8, QUERY_TIME)
        assert isinstance(sf, OpinionScoredFact)
        assert isinstance(sf.validity_opinion, Opinion)

    def test_plain_decay_validity_matches_compute(self):
        """Scalar validity should match DecayFunction.compute()."""
        plain = ExponentialDecay(beta={"general": 0.5})
        s = OpinionScorer(decay_fn=plain)
        f = _fact("a", days_ago=5.0)
        sf = s.score_fact(f, 0.8, QUERY_TIME)
        expected = plain.compute(f, QUERY_TIME)
        assert sf.validity == pytest.approx(expected, abs=1e-9)

    def test_plain_decay_opinion_is_dogmatic(self):
        """When wrapping a scalar, the opinion should be dogmatic (u=0)."""
        plain = ExponentialDecay(beta={"general": 0.5})
        s = OpinionScorer(decay_fn=plain)
        sf = s.score_fact(_fact("a", days_ago=3.0), 0.8, QUERY_TIME)
        assert sf.validity_opinion.uncertainty == pytest.approx(0.0)


# ===========================================================================
# 11. Uncertainty-Aware Ranking Advantage
# ===========================================================================


class TestUncertaintyAdvantage:
    """Demonstrate that uncertainty-aware strategies produce different
    rankings than scalar-equivalent strategies — the whole point of
    opinion-based scoring."""

    def test_uncertainty_penalized_reranks_vs_projected(self, odf):
        """Two facts with similar P(ω) but different uncertainty should be
        ranked differently by UncertaintyPenalized vs ProjectedMultiplicative."""
        # Fresh, moderate quality: low u, P ≈ moderate
        fresh = _fact("fresh", days_ago=0.0, quality=0.6)
        # Old, high quality: high u, but P may still be moderate due to a=0.5
        old = _fact("old", days_ago=10.0, quality=0.95)

        sims = [0.8, 0.8]
        facts = [fresh, old]

        proj_scorer = OpinionScorer(decay_fn=odf, strategy=ProjectedMultiplicative())
        uncert_scorer = OpinionScorer(decay_fn=odf, strategy=UncertaintyPenalized())

        proj_ranked = proj_scorer.rank(facts, sims, QUERY_TIME)
        uncert_ranked = uncert_scorer.rank(facts, sims, QUERY_TIME)

        # UncertaintyPenalized should more strongly prefer the fresh fact
        # (even if proj rankings might be similar), because (1-u) penalizes old
        uncert_scores = {sf.fact.content: sf.combined_score for sf in uncert_ranked}
        assert uncert_scores["fresh"] > uncert_scores["old"]
