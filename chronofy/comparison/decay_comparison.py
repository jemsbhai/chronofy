"""Comparative evaluation of decay function backends on labelled fact corpora.

Given a set of labelled observations (fact, still_valid: bool) and multiple
decay backends, computes standard binary classification metrics treating each
decay function as a scorer. This formalises domain-specific backend selection:
rather than guessing which decay family fits a domain, measure it empirically.

Metrics:
  AUROC              — Area under ROC curve (threshold-free ranking quality)
  Brier score        — Mean squared error between validity score and label
  Log loss           — Cross-entropy between validity score and label
  Ranking correlation — Spearman ρ between validity scores and binary labels

All metrics implemented in pure Python — no NumPy or SciPy dependency.

Architecture:
  DecayComparisonResult  ← frozen dataclass: one result per decay backend
  DecayComparison(...)   ← orchestrator: compare(), best(), summary()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact

# Clip validity scores away from 0 and 1 for log loss stability
_LOG_CLIP = 1e-9

# Valid metric names and whether lower is better
_METRICS: dict[str, bool] = {
    "auc_roc": False,           # higher is better
    "brier_score": True,        # lower is better
    "log_loss": True,           # lower is better
    "ranking_correlation": False,  # higher is better
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayComparisonResult:
    """Evaluation metrics for a single decay function on a labelled corpus.

    Attributes:
        name:                 Identifier matching the key in decay_fns dict.
        decay_fn:             The evaluated DecayFunction instance.
        auc_roc:              Area under ROC curve ∈ [0, 1].
                              0.5 = random; 1.0 = perfect separation.
        brier_score:          Mean squared error ∈ [0, 1].
                              0.0 = perfect; 1.0 = worst possible.
        log_loss:             Cross-entropy ≥ 0.
                              Lower is better.
        ranking_correlation:  Spearman ρ between validity scores and labels
                              ∈ [-1, 1]. 1.0 = perfect rank agreement.
    """

    name: str
    decay_fn: DecayFunction
    auc_roc: float
    brier_score: float
    log_loss: float
    ranking_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "auc_roc": self.auc_roc,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "ranking_correlation": self.ranking_correlation,
        }


# ---------------------------------------------------------------------------
# DecayComparison — orchestrator
# ---------------------------------------------------------------------------


class DecayComparison:
    """Evaluate and rank multiple decay backends on a labelled fact corpus.

    Treats each decay function as a binary classifier: the validity score it
    assigns to a fact is the predicted probability that the fact is still valid.
    Standard calibration and ranking metrics are then computed against the
    ground-truth labels.

    Args:
        decay_fns:  Dict mapping name → DecayFunction. At least one required.
        facts:      The TemporalFact corpus to evaluate on.
        labels:     Parallel list of booleans — True if the fact is still valid.
        query_time: Reference time for validity computation.

    Raises:
        ValueError: If decay_fns is empty, facts is empty, or lengths mismatch.

    Example:
        >>> dc = DecayComparison(
        ...     decay_fns={
        ...         "exponential": ExponentialDecay(beta={"lab": 2.0}),
        ...         "linear":      LinearDecay(rate={"lab": 0.1}),
        ...     },
        ...     facts=my_facts,
        ...     labels=my_labels,
        ...     query_time=datetime.now(),
        ... )
        >>> results = dc.compare()
        >>> for r in results:
        ...     print(f"{r.name:20s}  AUC={r.auc_roc:.3f}  Brier={r.brier_score:.3f}")
        >>> best = dc.best(metric="auc_roc")
        >>> fitted_decay = best.decay_fn
    """

    def __init__(
        self,
        decay_fns: dict[str, DecayFunction],
        facts: list[TemporalFact],
        labels: list[bool],
        query_time: datetime,
    ) -> None:
        if not decay_fns:
            raise ValueError(
                "decay_fns must contain at least one DecayFunction."
            )
        if not facts:
            raise ValueError(
                "facts must not be empty."
            )
        if len(facts) != len(labels):
            raise ValueError(
                f"facts and labels length mismatch: "
                f"{len(facts)} facts vs {len(labels)} labels."
            )

        self._decay_fns = dict(decay_fns)
        self._facts = list(facts)
        self._labels = list(labels)
        self._query_time = query_time
        self._cache: list[DecayComparisonResult] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self) -> list[DecayComparisonResult]:
        """Evaluate all decay backends and return results sorted by AUROC desc.

        Results are cached — repeated calls return the same list.

        Returns:
            List of DecayComparisonResult, one per decay function,
            sorted by auc_roc descending.
        """
        if self._cache is None:
            self._cache = self._run_comparison()
        return list(self._cache)

    def best(self, metric: str = "auc_roc") -> DecayComparisonResult:
        """Return the single best-performing decay backend by a given metric.

        Args:
            metric: One of "auc_roc", "brier_score", "log_loss",
                    "ranking_correlation".

        Returns:
            The DecayComparisonResult with the best value for the metric.

        Raises:
            ValueError: If metric is not a recognised name.
        """
        if metric not in _METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. "
                f"Valid options: {sorted(_METRICS.keys())}."
            )
        results = self.compare()
        lower_is_better = _METRICS[metric]
        return min(results, key=lambda r: getattr(r, metric)) \
            if lower_is_better \
            else max(results, key=lambda r: getattr(r, metric))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the comparison.

        Returns:
            Dict with keys: n_facts, n_valid, n_decay_fns,
            best_by_auc, best_by_brier, results (list of dicts).
        """
        results = self.compare()
        best_auc = self.best(metric="auc_roc")
        best_brier = self.best(metric="brier_score")
        return {
            "n_facts": len(self._facts),
            "n_valid": sum(self._labels),
            "n_decay_fns": len(self._decay_fns),
            "query_time": self._query_time.isoformat(),
            "best_by_auc": best_auc.name,
            "best_by_brier": best_brier.name,
            "results": [r.to_dict() for r in results],
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_comparison(self) -> list[DecayComparisonResult]:
        results = []
        int_labels = [1 if v else 0 for v in self._labels]

        for name, decay_fn in self._decay_fns.items():
            scores = decay_fn.compute_batch(self._facts, self._query_time)

            auc = _auroc(scores, int_labels)
            brier = _brier_score(scores, int_labels)
            ll = _log_loss(scores, int_labels)
            rho = _spearman(scores, int_labels)

            results.append(DecayComparisonResult(
                name=name,
                decay_fn=decay_fn,
                auc_roc=auc,
                brier_score=brier,
                log_loss=ll,
                ranking_correlation=rho,
            ))

        # Sort by AUROC descending; stable sort preserves dict insertion order
        # for ties
        return sorted(results, key=lambda r: r.auc_roc, reverse=True)


# ---------------------------------------------------------------------------
# Pure-Python metric implementations
# ---------------------------------------------------------------------------


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via the Wilcoxon-Mann-Whitney statistic.

    AUC = P(score(positive) > score(negative)), estimated from all
    (positive, negative) pairs. This is equivalent to the trapezoidal
    area under the ROC curve and requires no threshold selection.

    Returns 0.5 when all scores are equal (no discrimination).
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined — return chance level

    # Count concordant pairs (pos > neg) + 0.5 × ties
    concordant = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                concordant += 1.0
            elif p == n:
                concordant += 0.5

    return concordant / (n_pos * n_neg)


def _brier_score(scores: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted validity and binary label.

    Brier = (1/N) Σ (score_i - label_i)²

    0.0 = perfect calibration; 1.0 = worst possible.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    return sum((s - l) ** 2 for s, l in zip(scores, labels)) / n


def _log_loss(scores: list[float], labels: list[int]) -> float:
    """Binary cross-entropy between predicted validity and binary label.

    Clips scores away from 0 and 1 to avoid log(0).
    Lower is better; 0.0 = perfect.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    total = 0.0
    for s, l in zip(scores, labels):
        p = max(_LOG_CLIP, min(1.0 - _LOG_CLIP, s))
        if l == 1:
            total -= math.log(p)
        else:
            total -= math.log(1.0 - p)
    return total / n


def _spearman(scores: list[float], labels: list[int]) -> float:
    """Spearman rank correlation between validity scores and binary labels.

    Measures whether higher-validity facts tend to be labelled valid.
    Returns 0.0 when all scores are tied (no ranking signal).
    """
    n = len(scores)
    if n < 2:
        return 0.0

    score_ranks = _rank(scores)
    label_ranks = _rank([float(l) for l in labels])

    # Pearson correlation on ranks = Spearman correlation
    return _pearson(score_ranks, label_ranks)


def _rank(xs: list[float]) -> list[float]:
    """Assign average ranks to a list of values (handles ties)."""
    n = len(xs)
    indexed = sorted(range(n), key=lambda i: xs[i])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find the end of the tied group
        while j < n - 1 and xs[indexed[j + 1]] == xs[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0

    return cov / denom
