"""Tests for chronofy.scoring — temporal retrieval scoring.

TDD: written BEFORE the implementation.

Architecture under test:
  ScoringStrategy (ABC)
    ├── MultiplicativeScoring   sim * val
    ├── HarmonicScoring         2*sim*val / (sim+val)
    ├── WeightedBlendScoring    α*sim + (1-α)*val
    └── PowerScoring            sim^α * val^(1-α)

  ScoredFact (frozen dataclass)
    fact, similarity, validity, combined_score

  TemporalScorer(decay_fn, strategy)
    .score_fact(fact, sim, query_time)       → ScoredFact
    .score_facts(facts, sims, query_time)    → list[ScoredFact]
    .rank(facts, sims, query_time, top_k)    → list[ScoredFact] desc

Coverage plan:
  ScoredFact
    - fields present and correct
    - is frozen (immutable)
    - combined_score is the score passed in, not recomputed

  ScoringStrategy protocol
    - All four builtins are instances of ScoringStrategy
    - Custom subclass works
    - score() returns float in [0, 1]

  MultiplicativeScoring
    - score = sim * val (known values)
    - zero sim → zero score
    - zero val → zero score
    - both 1.0 → 1.0
    - identity when val=1 (score equals sim)

  HarmonicScoring
    - known values
    - both equal → equals that value
    - zero sim → zero
    - zero val → zero
    - symmetric: score(a,b) == score(b,a)
    - always <= min(sim, val)

  WeightedBlendScoring
    - alpha=1.0 → pure similarity
    - alpha=0.0 → pure validity
    - alpha=0.5 → arithmetic mean
    - known values
    - raises on alpha out of [0,1]

  PowerScoring
    - alpha=1.0 → pure similarity
    - alpha=0.0 → pure validity
    - alpha=0.5 → geometric mean (sqrt(sim*val))
    - known values
    - raises on alpha out of [0,1]
    - zero base → zero (handles 0^0 gracefully)

  TemporalScorer
    - default strategy is MultiplicativeScoring
    - score_fact returns ScoredFact
    - score_fact validity matches decay_fn output
    - score_fact combined_score matches strategy output
    - score_facts returns list of same length
    - score_facts raises on mismatched lengths
    - rank returns sorted descending by combined_score
    - rank top_k limits output length
    - rank top_k=None returns all
    - rank raises on top_k <= 0
    - rank with ties is stable (preserves input order for equal scores)
    - score_facts with empty input returns empty list
    - strategy is swappable at construction time
    - works with all four built-in strategies

  validate_scoring_strategy (plugins)
    - valid custom strategy passes
    - all built-ins pass
    - score() returns above 1.0 → raises
    - score() returns negative → raises
    - score() returns NaN → raises
    - score() returns wrong type → raises
    - non-ScoringStrategy object → raises
    - returns the object unchanged on success
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from chronofy.scoring.temporal_scorer import (
    HarmonicScoring,
    MultiplicativeScoring,
    PowerScoring,
    ScoredFact,
    ScoringStrategy,
    TemporalScorer,
    WeightedBlendScoring,
)
from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.plugins import PluginValidationError, validate_scoring_strategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2024, 3, 1, 12, 0, 0)


def make_fact(age_days: float, fact_type: str = "general") -> TemporalFact:
    return TemporalFact(
        content=f"fact aged {age_days}d",
        timestamp=QUERY_TIME - timedelta(days=age_days),
        fact_type=fact_type,
        source_quality=1.0,
    )


def fast_decay() -> ExponentialDecay:
    """β=1.0/day for 'general'."""
    return ExponentialDecay(beta={"general": 1.0}, default_beta=1.0)


def zero_decay() -> ExponentialDecay:
    """β=0 — validity always 1.0."""
    return ExponentialDecay(beta={"general": 0.0}, default_beta=0.0)


# ---------------------------------------------------------------------------
# 1. ScoredFact
# ---------------------------------------------------------------------------


class TestScoredFact:
    def test_fields_present(self):
        fact = make_fact(1.0)
        sf = ScoredFact(fact=fact, similarity=0.8, validity=0.6, combined_score=0.48)
        assert sf.fact is fact
        assert math.isclose(sf.similarity, 0.8)
        assert math.isclose(sf.validity, 0.6)
        assert math.isclose(sf.combined_score, 0.48)

    def test_is_frozen(self):
        fact = make_fact(1.0)
        sf = ScoredFact(fact=fact, similarity=0.8, validity=0.6, combined_score=0.48)
        with pytest.raises((AttributeError, TypeError)):
            sf.combined_score = 0.99  # type: ignore[misc]

    def test_combined_score_is_stored_not_recomputed(self):
        # ScoredFact stores whatever combined_score it is given
        fact = make_fact(1.0)
        sf = ScoredFact(fact=fact, similarity=0.9, validity=0.9, combined_score=0.42)
        assert math.isclose(sf.combined_score, 0.42)


# ---------------------------------------------------------------------------
# 2. ScoringStrategy protocol
# ---------------------------------------------------------------------------


class TestScoringStrategyProtocol:
    def test_multiplicative_is_strategy(self):
        assert isinstance(MultiplicativeScoring(), ScoringStrategy)

    def test_harmonic_is_strategy(self):
        assert isinstance(HarmonicScoring(), ScoringStrategy)

    def test_weighted_blend_is_strategy(self):
        assert isinstance(WeightedBlendScoring(alpha=0.5), ScoringStrategy)

    def test_power_is_strategy(self):
        assert isinstance(PowerScoring(alpha=0.5), ScoringStrategy)

    def test_custom_strategy_works(self):
        class MinStrategy(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return min(similarity, validity)

        s = MinStrategy()
        assert math.isclose(s.score(0.8, 0.6), 0.6)

    def test_score_returns_float(self):
        for strategy in [
            MultiplicativeScoring(),
            HarmonicScoring(),
            WeightedBlendScoring(0.5),
            PowerScoring(0.5),
        ]:
            result = strategy.score(0.7, 0.8)
            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 3. MultiplicativeScoring
# ---------------------------------------------------------------------------


class TestMultiplicativeScoring:
    def setup_method(self):
        self.s = MultiplicativeScoring()

    def test_known_value(self):
        assert math.isclose(self.s.score(0.8, 0.5), 0.4, rel_tol=1e-9)

    def test_both_one(self):
        assert math.isclose(self.s.score(1.0, 1.0), 1.0)

    def test_zero_similarity(self):
        assert math.isclose(self.s.score(0.0, 0.9), 0.0)

    def test_zero_validity(self):
        assert math.isclose(self.s.score(0.9, 0.0), 0.0)

    def test_identity_when_validity_one(self):
        assert math.isclose(self.s.score(0.73, 1.0), 0.73)

    def test_identity_when_similarity_one(self):
        assert math.isclose(self.s.score(1.0, 0.63), 0.63)

    def test_result_in_range(self):
        result = self.s.score(0.7, 0.8)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 4. HarmonicScoring
# ---------------------------------------------------------------------------


class TestHarmonicScoring:
    def setup_method(self):
        self.s = HarmonicScoring()

    def test_known_value(self):
        # H(0.8, 0.5) = 2*0.8*0.5 / (0.8+0.5) = 0.8/1.3 ≈ 0.6154
        expected = 2 * 0.8 * 0.5 / (0.8 + 0.5)
        assert math.isclose(self.s.score(0.8, 0.5), expected, rel_tol=1e-9)

    def test_equal_inputs(self):
        # H(x, x) = x
        assert math.isclose(self.s.score(0.7, 0.7), 0.7, rel_tol=1e-9)

    def test_zero_similarity(self):
        assert math.isclose(self.s.score(0.0, 0.9), 0.0)

    def test_zero_validity(self):
        assert math.isclose(self.s.score(0.9, 0.0), 0.0)

    def test_symmetric(self):
        assert math.isclose(
            self.s.score(0.3, 0.7), self.s.score(0.7, 0.3), rel_tol=1e-9
        )

    def test_always_geq_min(self):
        # Harmonic mean satisfies min(a,b) ≤ H(a,b) ≤ max(a,b)
        # It is bounded BELOW by the minimum, not above.
        for sim, val in [(0.8, 0.5), (0.2, 0.9), (0.6, 0.6), (1.0, 0.3)]:
            assert self.s.score(sim, val) >= min(sim, val) - 1e-9

    def test_always_leq_max(self):
        # Harmonic mean is also bounded ABOVE by the maximum.
        for sim, val in [(0.8, 0.5), (0.2, 0.9), (0.6, 0.6), (1.0, 0.3)]:
            assert self.s.score(sim, val) <= max(sim, val) + 1e-9

    def test_both_one(self):
        assert math.isclose(self.s.score(1.0, 1.0), 1.0)

    def test_result_in_range(self):
        result = self.s.score(0.6, 0.9)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 5. WeightedBlendScoring
# ---------------------------------------------------------------------------


class TestWeightedBlendScoring:
    def test_alpha_one_is_pure_similarity(self):
        s = WeightedBlendScoring(alpha=1.0)
        assert math.isclose(s.score(0.73, 0.2), 0.73, rel_tol=1e-9)

    def test_alpha_zero_is_pure_validity(self):
        s = WeightedBlendScoring(alpha=0.0)
        assert math.isclose(s.score(0.73, 0.2), 0.2, rel_tol=1e-9)

    def test_alpha_half_is_arithmetic_mean(self):
        s = WeightedBlendScoring(alpha=0.5)
        assert math.isclose(s.score(0.6, 0.4), 0.5, rel_tol=1e-9)

    def test_known_value(self):
        # α=0.7: 0.7*0.8 + 0.3*0.6 = 0.56 + 0.18 = 0.74
        s = WeightedBlendScoring(alpha=0.7)
        assert math.isclose(s.score(0.8, 0.6), 0.74, rel_tol=1e-9)

    def test_raises_on_alpha_negative(self):
        with pytest.raises(ValueError):
            WeightedBlendScoring(alpha=-0.1)

    def test_raises_on_alpha_above_one(self):
        with pytest.raises(ValueError):
            WeightedBlendScoring(alpha=1.1)

    def test_result_in_range(self):
        s = WeightedBlendScoring(alpha=0.6)
        result = s.score(0.7, 0.4)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 6. PowerScoring
# ---------------------------------------------------------------------------


class TestPowerScoring:
    def test_alpha_one_is_pure_similarity(self):
        s = PowerScoring(alpha=1.0)
        assert math.isclose(s.score(0.73, 0.2), 0.73, rel_tol=1e-9)

    def test_alpha_zero_is_pure_validity(self):
        s = PowerScoring(alpha=0.0)
        assert math.isclose(s.score(0.73, 0.2), 0.2, rel_tol=1e-9)

    def test_alpha_half_is_geometric_mean(self):
        s = PowerScoring(alpha=0.5)
        expected = math.sqrt(0.6 * 0.4)
        assert math.isclose(s.score(0.6, 0.4), expected, rel_tol=1e-9)

    def test_known_value(self):
        # α=0.7: 0.8^0.7 * 0.6^0.3
        s = PowerScoring(alpha=0.7)
        expected = (0.8 ** 0.7) * (0.6 ** 0.3)
        assert math.isclose(s.score(0.8, 0.6), expected, rel_tol=1e-9)

    def test_raises_on_alpha_negative(self):
        with pytest.raises(ValueError):
            PowerScoring(alpha=-0.1)

    def test_raises_on_alpha_above_one(self):
        with pytest.raises(ValueError):
            PowerScoring(alpha=1.1)

    def test_zero_similarity_gives_zero(self):
        s = PowerScoring(alpha=0.5)
        assert math.isclose(s.score(0.0, 0.8), 0.0)

    def test_zero_validity_gives_zero(self):
        s = PowerScoring(alpha=0.5)
        assert math.isclose(s.score(0.8, 0.0), 0.0)

    def test_result_in_range(self):
        s = PowerScoring(alpha=0.4)
        result = s.score(0.7, 0.9)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 7. TemporalScorer — score_fact
# ---------------------------------------------------------------------------


class TestTemporalScorerScoreFact:
    def setup_method(self):
        self.decay = fast_decay()
        self.scorer = TemporalScorer(decay_fn=self.decay)

    def test_returns_scored_fact(self):
        fact = make_fact(1.0)
        result = self.scorer.score_fact(fact, similarity=0.8, query_time=QUERY_TIME)
        assert isinstance(result, ScoredFact)

    def test_fact_field_is_same_object(self):
        fact = make_fact(1.0)
        result = self.scorer.score_fact(fact, similarity=0.8, query_time=QUERY_TIME)
        assert result.fact is fact

    def test_similarity_stored_correctly(self):
        fact = make_fact(1.0)
        result = self.scorer.score_fact(fact, similarity=0.75, query_time=QUERY_TIME)
        assert math.isclose(result.similarity, 0.75)

    def test_validity_matches_decay_fn(self):
        fact = make_fact(1.0)
        expected_validity = self.decay.compute(fact, QUERY_TIME)
        result = self.scorer.score_fact(fact, similarity=0.8, query_time=QUERY_TIME)
        assert math.isclose(result.validity, expected_validity, rel_tol=1e-9)

    def test_combined_score_uses_strategy(self):
        # Default strategy is multiplicative: combined = sim * validity
        fact = make_fact(1.0)
        expected_validity = self.decay.compute(fact, QUERY_TIME)
        result = self.scorer.score_fact(fact, similarity=0.8, query_time=QUERY_TIME)
        assert math.isclose(result.combined_score, 0.8 * expected_validity, rel_tol=1e-9)

    def test_default_strategy_is_multiplicative(self):
        scorer_explicit = TemporalScorer(
            decay_fn=self.decay, strategy=MultiplicativeScoring()
        )
        scorer_default = TemporalScorer(decay_fn=self.decay)
        fact = make_fact(5.0)
        r1 = scorer_explicit.score_fact(fact, 0.7, QUERY_TIME)
        r2 = scorer_default.score_fact(fact, 0.7, QUERY_TIME)
        assert math.isclose(r1.combined_score, r2.combined_score, rel_tol=1e-9)

    def test_zero_decay_validity_is_one(self):
        scorer = TemporalScorer(decay_fn=zero_decay())
        fact = make_fact(1000.0)
        result = scorer.score_fact(fact, similarity=0.9, query_time=QUERY_TIME)
        assert math.isclose(result.validity, 1.0, rel_tol=1e-9)
        assert math.isclose(result.combined_score, 0.9, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 8. TemporalScorer — score_facts
# ---------------------------------------------------------------------------


class TestTemporalScorerScoreFacts:
    def setup_method(self):
        self.decay = fast_decay()
        self.scorer = TemporalScorer(decay_fn=self.decay)
        self.facts = [make_fact(float(d)) for d in [1, 5, 10]]
        self.sims = [0.9, 0.7, 0.5]

    def test_returns_list_of_scored_facts(self):
        results = self.scorer.score_facts(self.facts, self.sims, QUERY_TIME)
        assert isinstance(results, list)
        assert all(isinstance(r, ScoredFact) for r in results)

    def test_output_length_matches_input(self):
        results = self.scorer.score_facts(self.facts, self.sims, QUERY_TIME)
        assert len(results) == len(self.facts)

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            self.scorer.score_facts(self.facts, [0.9, 0.7], QUERY_TIME)

    def test_empty_input_returns_empty(self):
        results = self.scorer.score_facts([], [], QUERY_TIME)
        assert results == []

    def test_order_preserved(self):
        results = self.scorer.score_facts(self.facts, self.sims, QUERY_TIME)
        for i, (fact, sim) in enumerate(zip(self.facts, self.sims)):
            assert results[i].fact is fact
            assert math.isclose(results[i].similarity, sim)

    def test_scores_consistent_with_score_fact(self):
        results = self.scorer.score_facts(self.facts, self.sims, QUERY_TIME)
        for i, (fact, sim) in enumerate(zip(self.facts, self.sims)):
            individual = self.scorer.score_fact(fact, sim, QUERY_TIME)
            assert math.isclose(
                results[i].combined_score, individual.combined_score, rel_tol=1e-9
            )


# ---------------------------------------------------------------------------
# 9. TemporalScorer — rank
# ---------------------------------------------------------------------------


class TestTemporalScorerRank:
    def setup_method(self):
        self.decay = fast_decay()
        self.scorer = TemporalScorer(decay_fn=self.decay)
        # age=1: high validity, sim=0.9 → high score
        # age=30: low validity, sim=0.95 → lower score (decay dominates)
        # age=5: medium validity, sim=0.5 → lowest score
        self.facts = [make_fact(1.0), make_fact(30.0), make_fact(5.0)]
        self.sims = [0.9, 0.95, 0.5]

    def test_returns_list_of_scored_facts(self):
        ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME)
        assert all(isinstance(r, ScoredFact) for r in ranked)

    def test_sorted_descending(self):
        ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME)
        scores = [r.combined_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=2)
        assert len(ranked) == 2

    def test_top_k_none_returns_all(self):
        ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=None)
        assert len(ranked) == len(self.facts)

    def test_top_k_larger_than_corpus_returns_all(self):
        ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=100)
        assert len(ranked) == len(self.facts)

    def test_raises_on_top_k_zero(self):
        with pytest.raises(ValueError):
            self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=0)

    def test_raises_on_top_k_negative(self):
        with pytest.raises(ValueError):
            self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=-1)

    def test_empty_input_returns_empty(self):
        assert self.scorer.rank([], [], QUERY_TIME) == []

    def test_top_k_returns_highest_scores(self):
        all_ranked = self.scorer.rank(self.facts, self.sims, QUERY_TIME)
        top2 = self.scorer.rank(self.facts, self.sims, QUERY_TIME, top_k=2)
        assert top2[0].combined_score == all_ranked[0].combined_score
        assert top2[1].combined_score == all_ranked[1].combined_score


# ---------------------------------------------------------------------------
# 10. Strategy swappability
# ---------------------------------------------------------------------------


class TestStrategySwappability:
    def test_harmonic_strategy_used(self):
        scorer = TemporalScorer(
            decay_fn=zero_decay(), strategy=HarmonicScoring()
        )
        fact = make_fact(0.0)
        result = scorer.score_fact(fact, similarity=0.8, query_time=QUERY_TIME)
        # validity=1.0, sim=0.8 → H(0.8, 1.0) = 2*0.8*1.0/(0.8+1.0) = 1.6/1.8
        expected = 2 * 0.8 * 1.0 / (0.8 + 1.0)
        assert math.isclose(result.combined_score, expected, rel_tol=1e-9)

    def test_weighted_blend_strategy_used(self):
        scorer = TemporalScorer(
            decay_fn=zero_decay(), strategy=WeightedBlendScoring(alpha=0.3)
        )
        fact = make_fact(0.0)
        result = scorer.score_fact(fact, similarity=0.6, query_time=QUERY_TIME)
        # validity=1.0, sim=0.6 → 0.3*0.6 + 0.7*1.0 = 0.88
        assert math.isclose(result.combined_score, 0.88, rel_tol=1e-9)

    def test_power_strategy_used(self):
        scorer = TemporalScorer(
            decay_fn=zero_decay(), strategy=PowerScoring(alpha=0.5)
        )
        fact = make_fact(0.0)
        result = scorer.score_fact(fact, similarity=0.64, query_time=QUERY_TIME)
        # validity=1.0, sim=0.64 → sqrt(0.64 * 1.0) = 0.8
        assert math.isclose(result.combined_score, 0.8, rel_tol=1e-9)

    def test_custom_strategy_used_in_scorer(self):
        class ConstantStrategy(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return 0.5

        scorer = TemporalScorer(
            decay_fn=zero_decay(), strategy=ConstantStrategy()
        )
        fact = make_fact(1.0)
        result = scorer.score_fact(fact, similarity=0.9, query_time=QUERY_TIME)
        assert math.isclose(result.combined_score, 0.5)


# ---------------------------------------------------------------------------
# 11. validate_scoring_strategy (plugins)
# ---------------------------------------------------------------------------


class TestValidateScoringStrategy:
    def test_multiplicative_passes(self):
        s = MultiplicativeScoring()
        assert validate_scoring_strategy(s) is s

    def test_harmonic_passes(self):
        assert validate_scoring_strategy(HarmonicScoring()) is not None

    def test_weighted_blend_passes(self):
        assert validate_scoring_strategy(WeightedBlendScoring(0.5)) is not None

    def test_power_passes(self):
        assert validate_scoring_strategy(PowerScoring(0.5)) is not None

    def test_valid_custom_passes(self):
        class GoodStrategy(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return similarity * validity

        s = validate_scoring_strategy(GoodStrategy())
        assert s is not None

    def test_score_above_one_raises(self):
        class BadHigh(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return 1.5

        with pytest.raises(PluginValidationError, match="[Rr]ange|above|1\\.0"):
            validate_scoring_strategy(BadHigh())

    def test_score_negative_raises(self):
        class BadNeg(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return -0.1

        with pytest.raises(PluginValidationError, match="[Rr]ange|negative|below"):
            validate_scoring_strategy(BadNeg())

    def test_score_nan_raises(self):
        class NaNStrategy(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return float("nan")

        with pytest.raises(PluginValidationError, match="[Nn]aN|finite"):
            validate_scoring_strategy(NaNStrategy())

    def test_score_wrong_type_raises(self):
        class StrStrategy(ScoringStrategy):
            def score(self, similarity: float, validity: float) -> float:
                return "high"  # type: ignore[return-value]

        with pytest.raises(PluginValidationError, match="[Tt]ype|float|numeric"):
            validate_scoring_strategy(StrStrategy())

    def test_non_strategy_raises(self):
        with pytest.raises((PluginValidationError, TypeError, AttributeError)):
            validate_scoring_strategy("not a strategy")

    def test_returns_object_unchanged(self):
        s = MultiplicativeScoring()
        result = validate_scoring_strategy(s)
        assert result is s
