"""Tests for chronofy.comparison.decay_comparison.

TDD: written BEFORE the implementation.

Architecture under test:
  DecayComparisonResult (frozen dataclass)
    name, decay_fn, auc_roc, brier_score, log_loss, ranking_correlation

  DecayComparison(decay_fns, facts, labels, query_time)
    .compare()              → list[DecayComparisonResult] sorted by auc_roc desc
    .best(metric=...)       → single DecayComparisonResult
    .summary()              → dict

Coverage plan:
  DecayComparisonResult
    - has all required fields
    - is frozen (immutable)
    - to_dict() returns all fields

  DecayComparison construction
    - raises on empty decay_fns
    - raises on mismatched facts/labels
    - raises on empty facts
    - accepts a single decay function
    - accepts multiple decay functions

  compare()
    - returns list of same length as decay_fns
    - each result has a name matching the decay function's repr or provided name
    - sorted by auc_roc descending by default
    - a perfect classifier (validity=1 for valid, 0 for invalid) has auc_roc=1.0
    - a random classifier has auc_roc≈0.5
    - a reversed classifier has auc_roc=0.0
    - results are deterministic (same output on repeated calls)

  AUROC
    - all valid: auc = 0.5 (no discrimination possible)
    - all invalid: auc = 0.5
    - perfect separation: auc = 1.0
    - reversed: auc = 0.0
    - known 3-point example

  Brier score
    - all correct (validity=1, label=True): brier=0.0
    - all wrong (validity=0, label=True): brier=1.0
    - known values
    - always in [0.0, 1.0]

  Log loss
    - always non-negative
    - lower for better predictions
    - clipped to avoid log(0)

  Ranking correlation (Spearman)
    - perfect agreement: rho=1.0
    - perfect reversal: rho=-1.0
    - known 4-point example
    - all tied scores: rho=0.0

  best()
    - raises on unknown metric
    - returns single DecayComparisonResult
    - metric="auc_roc" returns highest auc_roc
    - metric="brier_score" returns lowest brier_score (lower is better)
    - metric="log_loss" returns lowest log_loss

  summary()
    - returns dict with required keys
    - best_by_auc in summary
    - best_by_brier in summary

  Integration: multiple decay functions
    - a well-calibrated decay scores better than a random/wrong one
    - exponential with correct beta outperforms one with wrong beta
    - linear decay is comparable when domain matches
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from chronofy.comparison.decay_comparison import (
    DecayComparison,
    DecayComparisonResult,
)
from chronofy.decay.exponential import ExponentialDecay
from chronofy.decay.linear import LinearDecay
from chronofy.models import TemporalFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUERY_TIME = datetime(2024, 6, 1, 12, 0, 0)


def make_fact(age_days: float, fact_type: str = "general") -> TemporalFact:
    return TemporalFact(
        content=f"fact aged {age_days}d",
        timestamp=QUERY_TIME - timedelta(days=age_days),
        fact_type=fact_type,
        source_quality=1.0,
    )


def perfect_decay(facts: list[TemporalFact], labels: list[bool]) -> ExponentialDecay:
    """A decay that returns ~1.0 for valid facts and ~0.0 for invalid ones.

    Built by assigning tiny beta to valid-only facts and huge beta to invalid-only.
    For testing we use a trick: all valid facts are age 0 (validity=1.0),
    all invalid facts are age 1000 days (validity≈0 for any reasonable beta).
    """
    return ExponentialDecay(beta={"general": 0.1}, default_beta=0.1)


def random_decay() -> ExponentialDecay:
    """A near-zero decay that gives ~equal scores to everything."""
    return ExponentialDecay(beta={"general": 0.0}, default_beta=0.0)


# ---------------------------------------------------------------------------
# 1. DecayComparisonResult structure
# ---------------------------------------------------------------------------


class TestDecayComparisonResult:
    def _make_result(self) -> DecayComparisonResult:
        return DecayComparisonResult(
            name="test",
            decay_fn=ExponentialDecay(),
            auc_roc=0.75,
            brier_score=0.15,
            log_loss=0.4,
            ranking_correlation=0.6,
        )

    def test_has_all_fields(self):
        r = self._make_result()
        assert r.name == "test"
        assert math.isclose(r.auc_roc, 0.75)
        assert math.isclose(r.brier_score, 0.15)
        assert math.isclose(r.log_loss, 0.4)
        assert math.isclose(r.ranking_correlation, 0.6)
        assert r.decay_fn is not None

    def test_is_frozen(self):
        r = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.auc_roc = 0.99  # type: ignore[misc]

    def test_to_dict_has_all_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert {"name", "auc_roc", "brier_score",
                "log_loss", "ranking_correlation"}.issubset(d.keys())

    def test_to_dict_values_match(self):
        r = self._make_result()
        d = r.to_dict()
        assert math.isclose(d["auc_roc"], 0.75)
        assert math.isclose(d["brier_score"], 0.15)


# ---------------------------------------------------------------------------
# 2. DecayComparison construction
# ---------------------------------------------------------------------------


class TestDecayComparisonConstruction:
    def _simple(self):
        facts = [make_fact(1.0), make_fact(10.0)]
        labels = [True, False]
        return facts, labels

    def test_raises_on_empty_decay_fns(self):
        facts, labels = self._simple()
        with pytest.raises(ValueError, match="[Dd]ecay"):
            DecayComparison(
                decay_fns={},
                facts=facts,
                labels=labels,
                query_time=QUERY_TIME,
            )

    def test_raises_on_empty_facts(self):
        with pytest.raises(ValueError, match="[Ff]acts|[Ee]mpty"):
            DecayComparison(
                decay_fns={"exp": ExponentialDecay()},
                facts=[],
                labels=[],
                query_time=QUERY_TIME,
            )

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="[Ll]ength|[Mm]ismatch"):
            DecayComparison(
                decay_fns={"exp": ExponentialDecay()},
                facts=[make_fact(1.0), make_fact(2.0)],
                labels=[True],
                query_time=QUERY_TIME,
            )

    def test_accepts_single_decay(self):
        facts, labels = self._simple()
        dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay()},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        assert dc is not None

    def test_accepts_multiple_decays(self):
        facts, labels = self._simple()
        dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay(), "lin": LinearDecay()},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        assert dc is not None


# ---------------------------------------------------------------------------
# 3. compare() — structure
# ---------------------------------------------------------------------------


class TestCompareStructure:
    def setup_method(self):
        self.facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        self.labels = [True, True, False, False]
        self.decay_fns = {
            "fast": ExponentialDecay(beta={"general": 2.0}, default_beta=2.0),
            "slow": ExponentialDecay(beta={"general": 0.01}, default_beta=0.01),
        }
        self.dc = DecayComparison(
            decay_fns=self.decay_fns,
            facts=self.facts,
            labels=self.labels,
            query_time=QUERY_TIME,
        )

    def test_returns_list(self):
        results = self.dc.compare()
        assert isinstance(results, list)

    def test_length_matches_decay_fns(self):
        results = self.dc.compare()
        assert len(results) == len(self.decay_fns)

    def test_all_results_are_decay_comparison_result(self):
        results = self.dc.compare()
        assert all(isinstance(r, DecayComparisonResult) for r in results)

    def test_names_match_keys(self):
        results = self.dc.compare()
        names = {r.name for r in results}
        assert names == set(self.decay_fns.keys())

    def test_sorted_by_auc_roc_descending(self):
        results = self.dc.compare()
        aucs = [r.auc_roc for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_deterministic(self):
        r1 = self.dc.compare()
        r2 = self.dc.compare()
        for a, b in zip(r1, r2):
            assert math.isclose(a.auc_roc, b.auc_roc)
            assert math.isclose(a.brier_score, b.brier_score)

    def test_all_metrics_in_valid_ranges(self):
        results = self.dc.compare()
        for r in results:
            assert 0.0 <= r.auc_roc <= 1.0
            assert 0.0 <= r.brier_score <= 1.0
            assert r.log_loss >= 0.0
            assert -1.0 <= r.ranking_correlation <= 1.0


# ---------------------------------------------------------------------------
# 4. AUROC — analytical cases
# ---------------------------------------------------------------------------


class TestAUROC:
    def test_perfect_separation_auc_one(self):
        # age=0 → validity=1.0 (valid), age=1000 → validity≈0 (invalid)
        # Fast decay perfectly separates them
        facts = [make_fact(0.0), make_fact(1000.0)]
        labels = [True, False]
        dc = DecayComparison(
            decay_fns={"fast": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].auc_roc >= 0.99

    def test_reversed_separation_auc_zero(self):
        # Low validity for valid facts, high for invalid → AUROC = 0
        # Achieve by: valid facts are OLD (low validity), invalid are NEW (high validity)
        facts = [make_fact(1000.0), make_fact(0.0)]
        labels = [True, False]
        dc = DecayComparison(
            decay_fns={"fast": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].auc_roc <= 0.01

    def test_zero_decay_auc_half(self):
        # Zero decay: all validities equal → random classifier → AUC = 0.5
        facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        labels = [True, False, True, False]
        dc = DecayComparison(
            decay_fns={"zero": ExponentialDecay(
                beta={"general": 0.0}, default_beta=0.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert math.isclose(results[0].auc_roc, 0.5, abs_tol=0.01)

    def test_known_three_point_example(self):
        # scores: [0.9 (valid), 0.4 (valid), 0.1 (invalid)]
        # AUC = fraction of (valid, invalid) pairs where valid scores higher
        # Pairs: (0.9, 0.1) valid higher ✓, (0.4, 0.1) valid higher ✓ → AUC = 1.0
        # Approximate with age=0→v=1.0, age=1→v≈0.37, age=10→v≈0.00005
        facts = [make_fact(0.0), make_fact(1.0), make_fact(10.0)]
        labels = [True, True, False]
        dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].auc_roc >= 0.99


# ---------------------------------------------------------------------------
# 5. Brier score — analytical cases
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_predictions_brier_zero(self):
        # Zero decay: validity=1.0 for all; all labels True → Brier=0.0
        facts = [make_fact(0.0)] * 4
        labels = [True] * 4
        dc = DecayComparison(
            decay_fns={"zero": ExponentialDecay(
                beta={"general": 0.0}, default_beta=0.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert math.isclose(results[0].brier_score, 0.0, abs_tol=1e-9)

    def test_worst_predictions_brier_one(self):
        # Validity≈0 (age=1000, fast decay) for all, all labels True → Brier≈1.0
        facts = [make_fact(1000.0)] * 4
        labels = [True] * 4
        dc = DecayComparison(
            decay_fns={"fast": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].brier_score >= 0.99

    def test_brier_always_in_range(self):
        facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        labels = [True, False, True, False]
        dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay()},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert 0.0 <= results[0].brier_score <= 1.0


# ---------------------------------------------------------------------------
# 6. Ranking correlation — analytical cases
# ---------------------------------------------------------------------------


class TestRankingCorrelation:
    def test_perfect_agreement_rho_near_one(self):
        # Valid = high validity, invalid = low validity → rho as high as possible.
        # NOTE: With binary labels [1,1,0,0] the label ranks are always tied at
        # (3.5, 3.5, 1.5, 1.5). Tied ranks cap Spearman ρ at 4/√20 ≈ 0.894
        # regardless of how perfect the score separation is. Threshold is 0.85.
        facts = [make_fact(0.0), make_fact(0.1), make_fact(100.0), make_fact(200.0)]
        labels = [True, True, False, False]
        dc = DecayComparison(
            decay_fns={"fast": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].ranking_correlation >= 0.85

    def test_perfect_reversal_rho_near_minus_one(self):
        # Valid = low validity (old), invalid = high validity (new).
        # Same tied-rank ceiling applies: minimum is −4/√20 ≈ −0.894.
        facts = [make_fact(200.0), make_fact(100.0), make_fact(0.1), make_fact(0.0)]
        labels = [True, True, False, False]
        dc = DecayComparison(
            decay_fns={"fast": ExponentialDecay(
                beta={"general": 1.0}, default_beta=1.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert results[0].ranking_correlation <= -0.85

    def test_all_equal_scores_rho_zero(self):
        # Zero decay: all scores equal → no ranking signal → rho = 0
        facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        labels = [True, False, True, False]
        dc = DecayComparison(
            decay_fns={"zero": ExponentialDecay(
                beta={"general": 0.0}, default_beta=0.0
            )},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert math.isclose(results[0].ranking_correlation, 0.0, abs_tol=1e-9)

    def test_ranking_correlation_in_range(self):
        facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        labels = [True, False, True, False]
        dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay()},
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        assert -1.0 <= results[0].ranking_correlation <= 1.0


# ---------------------------------------------------------------------------
# 7. best()
# ---------------------------------------------------------------------------


class TestBest:
    def setup_method(self):
        self.facts = [make_fact(0.0), make_fact(0.5), make_fact(50.0), make_fact(200.0)]
        self.labels = [True, True, False, False]
        self.dc = DecayComparison(
            decay_fns={
                "fast": ExponentialDecay(beta={"general": 1.0}, default_beta=1.0),
                "slow": ExponentialDecay(beta={"general": 0.001}, default_beta=0.001),
            },
            facts=self.facts,
            labels=self.labels,
            query_time=QUERY_TIME,
        )

    def test_best_returns_single_result(self):
        result = self.dc.best()
        assert isinstance(result, DecayComparisonResult)

    def test_best_auc_roc_returns_highest(self):
        all_results = self.dc.compare()
        best = self.dc.best(metric="auc_roc")
        max_auc = max(r.auc_roc for r in all_results)
        assert math.isclose(best.auc_roc, max_auc)

    def test_best_brier_returns_lowest(self):
        all_results = self.dc.compare()
        best = self.dc.best(metric="brier_score")
        min_brier = min(r.brier_score for r in all_results)
        assert math.isclose(best.brier_score, min_brier)

    def test_best_log_loss_returns_lowest(self):
        all_results = self.dc.compare()
        best = self.dc.best(metric="log_loss")
        min_ll = min(r.log_loss for r in all_results)
        assert math.isclose(best.log_loss, min_ll)

    def test_best_default_metric_is_auc_roc(self):
        best_default = self.dc.best()
        best_auc = self.dc.best(metric="auc_roc")
        assert best_default.name == best_auc.name

    def test_raises_on_unknown_metric(self):
        with pytest.raises(ValueError, match="[Mm]etric|[Uu]nknown"):
            self.dc.best(metric="made_up_metric")


# ---------------------------------------------------------------------------
# 8. summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def setup_method(self):
        self.facts = [make_fact(float(d)) for d in [1, 5, 10, 30]]
        self.labels = [True, True, False, False]
        self.dc = DecayComparison(
            decay_fns={"exp": ExponentialDecay(), "lin": LinearDecay()},
            facts=self.facts,
            labels=self.labels,
            query_time=QUERY_TIME,
        )

    def test_summary_returns_dict(self):
        assert isinstance(self.dc.summary(), dict)

    def test_summary_has_required_keys(self):
        s = self.dc.summary()
        assert {"n_facts", "n_valid", "n_decay_fns",
                "best_by_auc", "best_by_brier", "results"}.issubset(s.keys())

    def test_summary_n_facts_correct(self):
        s = self.dc.summary()
        assert s["n_facts"] == 4

    def test_summary_n_valid_correct(self):
        s = self.dc.summary()
        assert s["n_valid"] == 2

    def test_summary_n_decay_fns_correct(self):
        s = self.dc.summary()
        assert s["n_decay_fns"] == 2

    def test_summary_best_by_auc_is_name_string(self):
        s = self.dc.summary()
        assert isinstance(s["best_by_auc"], str)

    def test_summary_results_is_list_of_dicts(self):
        s = self.dc.summary()
        assert isinstance(s["results"], list)
        assert all(isinstance(r, dict) for r in s["results"])


# ---------------------------------------------------------------------------
# 9. Integration: better-calibrated decay wins
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_correct_beta_beats_wrong_beta(self):
        # Ground truth: facts go invalid around day 5
        # Correct β ≈ 0.14 (half-life ≈ 5 days), wrong β = 0.001 (too slow)
        facts = [make_fact(float(d)) for d in [1, 2, 3, 7, 10, 20, 30]]
        labels = [True, True, True, False, False, False, False]

        dc = DecayComparison(
            decay_fns={
                "correct": ExponentialDecay(
                    beta={"general": 0.14}, default_beta=0.14
                ),
                "too_slow": ExponentialDecay(
                    beta={"general": 0.001}, default_beta=0.001
                ),
            },
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        correct = next(r for r in results if r.name == "correct")
        too_slow = next(r for r in results if r.name == "too_slow")
        assert correct.auc_roc >= too_slow.auc_roc
        assert correct.brier_score <= too_slow.brier_score

    def test_three_way_comparison(self):
        facts = [make_fact(float(d)) for d in [0.5, 1, 2, 10, 20, 30]]
        labels = [True, True, True, False, False, False]

        dc = DecayComparison(
            decay_fns={
                "exponential": ExponentialDecay(
                    beta={"general": 0.5}, default_beta=0.5
                ),
                "linear": LinearDecay(
                    rate={"general": 0.05}, default_rate=0.05
                ),
                "no_decay": ExponentialDecay(
                    beta={"general": 0.0}, default_beta=0.0
                ),
            },
            facts=facts,
            labels=labels,
            query_time=QUERY_TIME,
        )
        results = dc.compare()
        # Should have 3 results
        assert len(results) == 3
        # All sorted by AUC descending
        aucs = [r.auc_roc for r in results]
        assert aucs == sorted(aucs, reverse=True)
        # No decay should be worst (AUC=0.5)
        no_decay = next(r for r in results if r.name == "no_decay")
        assert math.isclose(no_decay.auc_roc, 0.5, abs_tol=0.01)
