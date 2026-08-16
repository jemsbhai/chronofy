"""Experiment 1: Decay-Weighted Temporal Knowledge Graph Retrieval

Goal: Validate that exponential decay weighting improves fact retrieval
accuracy over static methods on standard TKG benchmarks.

Dataset: ICEWS14 (Integrated Crisis Early Warning System, 2014 events)
  - Timestamped relational triples: (subject, relation, object, timestamp)
  - Standard train/valid/test splits
  - ~7,128 entities, 230 relations, 90,730 triples

Approach:
  For each test query (s, r, ?, t_q):
    1. Retrieve candidate facts from training data matching (s, r, *)
    2. Score candidates using different temporal strategies
    3. Rank candidates by score
    4. Evaluate Hits@1, Hits@3, Hits@10, MRR, Temporal Precision@k

Baselines:
  - Static: uniform weight (no temporal signal)
  - Recency-only: weight = 1/rank_by_recency (always prefer newest)
  - Frequency: weight = count of (s, r, o) in training data
  - Chronofy decay: weight = exp(-β · Δt) with learned/swept β

Usage:
  python experiments/exp1_tkg_decay.py --download   # Download ICEWS14
  python experiments/exp1_tkg_decay.py --run         # Run experiment
  python experiments/exp1_tkg_decay.py --results     # Print results table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path so we can import chronofy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproducibility import EVOEXPLORE_REVISION, ICEWS14_FILE_SHA256

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data" / "icews14"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ICEWS14 — from EvoExplore (Zhang et al., 2022)
# Format: tab-separated integers (subject, relation, object, timestamp)
_EVOEXPLORE_BASE = (
    "https://raw.githubusercontent.com/zjs123/EvoExplore/" f"{EVOEXPLORE_REVISION}/dataset/ICEWS14"
)
ICEWS14_URLS = {
    "train": f"{_EVOEXPLORE_BASE}/train.txt",
    "valid": f"{_EVOEXPLORE_BASE}/valid.txt",
    "test": f"{_EVOEXPLORE_BASE}/test.txt",
    "stat": f"{_EVOEXPLORE_BASE}/stat.txt",
}

# β values to sweep
BETA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


# ============================================================
# DATA LOADING
# ============================================================


@dataclass
class Triple:
    """A timestamped knowledge graph triple."""

    subject: int
    relation: int
    obj: int
    timestamp: int  # Integer timestep (day index for ICEWS14)


@dataclass
class TKGDataset:
    """A temporal knowledge graph dataset with train/valid/test splits."""

    train: list[Triple] = field(default_factory=list)
    valid: list[Triple] = field(default_factory=list)
    test: list[Triple] = field(default_factory=list)
    num_entities: int = 0
    num_relations: int = 0
    num_timestamps: int = 0

    @property
    def summary(self) -> str:
        return (
            f"TKGDataset: {self.num_entities} entities, "
            f"{self.num_relations} relations, "
            f"{self.num_timestamps} timestamps\n"
            f"  Train: {len(self.train)} triples\n"
            f"  Valid: {len(self.valid)} triples\n"
            f"  Test:  {len(self.test)} triples"
        )


def download_icews14() -> None:
    """Download and checksum ICEWS14 files from a pinned EvoExplore revision."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in ICEWS14_URLS.items():
        dest = DATA_DIR / f"{name}.txt"
        if dest.exists():
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            expected = ICEWS14_FILE_SHA256[dest.name]
            if actual != expected:
                raise ValueError(f"Checksum mismatch for existing file: {dest}")
            print(f"  [skip] {dest} already exists (checksum verified)")
            continue
        print(f"  [download] {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            expected = ICEWS14_FILE_SHA256[dest.name]
            if actual != expected:
                raise ValueError(f"Checksum mismatch for downloaded file: {dest}")
            print(f"  [saved] {dest} ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {name}.txt: {e}")
            print(f"  Please manually place ICEWS14 files in: {DATA_DIR}")
            sys.exit(1)


# Reference date for converting ICEWS14 date strings to integer day indices
_ICEWS14_EPOCH = datetime(2014, 1, 1)


def _parse_timestamp(ts_str: str) -> int:
    """Parse a timestamp field — handles both date strings and integer indices."""
    ts_str = ts_str.strip()
    if "-" in ts_str:  # Date string like "2014-05-13"
        dt = datetime.strptime(ts_str, "%Y-%m-%d")
        return (dt - _ICEWS14_EPOCH).days
    return int(ts_str)  # Already an integer index


def load_triples(filepath: Path) -> list[Triple]:
    """Load triples from a tab-separated file: subject relation object timestamp.

    Handles:
      - Integer timestamps (day indices)
      - Date string timestamps ("2014-05-13" → day-of-year index)
      - 4 or 5+ column formats
    """
    triples = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                s, r, o = int(parts[0]), int(parts[1]), int(parts[2])
                t = _parse_timestamp(parts[3])
                triples.append(Triple(s, r, o, t))
    return triples


def load_icews14() -> TKGDataset:
    """Load the full ICEWS14 dataset."""
    train = load_triples(DATA_DIR / "train.txt")
    valid = load_triples(DATA_DIR / "valid.txt")
    test = load_triples(DATA_DIR / "test.txt")

    # Load stats — handles both tab-separated and space-separated formats
    stat_path = DATA_DIR / "stat.txt"
    if stat_path.exists():
        with open(stat_path) as f:
            raw = f.read().strip()
            # EvoExplore format: "6869 230 365 72826 8789 8871" (space-separated)
            # Other format: "7128\t230" (tab-separated, just entities and relations)
            parts = raw.replace("\t", " ").split()
            num_ent, num_rel = int(parts[0]), int(parts[1])
    else:
        all_triples = train + valid + test
        num_ent = max(max(t.subject, t.obj) for t in all_triples) + 1
        num_rel = max(t.relation for t in all_triples) + 1

    all_triples = train + valid + test
    num_ts = max(t.timestamp for t in all_triples) + 1

    return TKGDataset(
        train=train,
        valid=valid,
        test=test,
        num_entities=num_ent,
        num_relations=num_rel,
        num_timestamps=num_ts,
    )


# ============================================================
# RETRIEVAL INDEX
# ============================================================


class TemporalRetrievalIndex:
    """Index training triples for efficient retrieval by (subject, relation).

    For a query (s, r, ?, t_q), returns candidate objects with their
    temporal metadata (timestamps and frequencies).
    """

    def __init__(self, triples: list[Triple]) -> None:
        # (subject, relation) -> list of (object, timestamp)
        self._sr_index: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        # (subject, relation, object) -> list of timestamps
        self._sro_timestamps: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        # All unique objects per (subject, relation) for candidate set
        self._sr_objects: dict[tuple[int, int], set[int]] = defaultdict(set)

        for t in triples:
            self._sr_index[(t.subject, t.relation)].append((t.obj, t.timestamp))
            self._sro_timestamps[(t.subject, t.relation, t.obj)].append(t.timestamp)
            self._sr_objects[(t.subject, t.relation)].add(t.obj)

    def get_candidates(self, subject: int, relation: int) -> set[int]:
        """Return all objects ever seen with (subject, relation) in training."""
        return self._sr_objects.get((subject, relation), set())

    def get_timestamps(self, subject: int, relation: int, obj: int) -> list[int]:
        """Return all timestamps where (subject, relation, object) appeared."""
        return self._sro_timestamps.get((subject, relation, obj), [])

    def get_all_facts(self, subject: int, relation: int) -> list[tuple[int, int]]:
        """Return all (object, timestamp) pairs for (subject, relation)."""
        return self._sr_index.get((subject, relation), [])


# ============================================================
# SCORING STRATEGIES
# ============================================================


def score_static(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
) -> dict[int, float]:
    """Static baseline: score = frequency count (no temporal signal)."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        scores[obj] = float(len(timestamps))
    return scores


def score_recency(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
) -> dict[int, float]:
    """Recency-only: score = 1 / (1 + min_time_gap). Always prefer newest."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            min_gap = min(abs(query_time - t) for t in timestamps)
            scores[obj] = 1.0 / (1.0 + min_gap)
        else:
            scores[obj] = 0.0
    return scores


def score_chronofy_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    beta: float,
) -> dict[int, float]:
    """Chronofy decay: score = sum of exp(-β · |Δt|) across all timestamps.

    Aggregates over all historical occurrences, with each occurrence
    weighted by its temporal proximity to the query time.
    """
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum(np.exp(-beta * abs(query_time - t)) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


def score_chronofy_max_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    beta: float,
) -> dict[int, float]:
    """Chronofy decay (max): score = max exp(-β · |Δt|) across timestamps.

    Uses the single most recent/relevant occurrence rather than aggregating.
    """
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            best = max(np.exp(-beta * abs(query_time - t)) for t in timestamps)
            scores[obj] = float(best)
        else:
            scores[obj] = 0.0
    return scores


# ============================================================
# ALTERNATIVE DECAY SCORING STRATEGIES
# ============================================================


def score_half_life_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    half_life: float,
) -> dict[int, float]:
    """Half-life decay: score = sum of 0.5^(|Δt|/h) across timestamps."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum(0.5 ** (abs(query_time - t) / half_life) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


def score_linear_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    rate: float,
) -> dict[int, float]:
    """Linear decay: score = sum of max(0, 1 - α·|Δt|) across timestamps."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum(max(0.0, 1.0 - rate * abs(query_time - t)) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


def score_power_law_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    exponent: float,
) -> dict[int, float]:
    """Power-law decay: score = sum of (1 + |Δt|)^(-α) across timestamps."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum((1.0 + abs(query_time - t)) ** (-exponent) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


def score_weibull_decay(
    index: TemporalRetrievalIndex,
    subject: int,
    relation: int,
    candidates: set[int],
    query_time: int,
    scale: float,
    shape: float,
) -> dict[int, float]:
    """Weibull decay: score = sum of exp(-(|Δt|/λ)^k) across timestamps."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum(np.exp(-((abs(query_time - t) / scale) ** shape)) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


# Parameter grids for alternative decay sweeps
HALF_LIFE_VALUES = [1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0]
LINEAR_RATE_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
POWER_LAW_EXPONENTS = [0.5, 1.0, 1.5, 2.0, 3.0]
WEIBULL_CONFIGS = [
    (7.0, 0.5),  # Decelerating obsolescence
    (7.0, 1.0),  # Exponential (baseline)
    (7.0, 2.0),  # Accelerating obsolescence
    (1.0, 1.0),  # Fast exponential
    (1.0, 2.0),  # Fast accelerating
    (30.0, 0.5),  # Slow decelerating
    (30.0, 2.0),  # Slow accelerating
]


# ============================================================
# EVALUATION METRICS
# ============================================================


@dataclass
class EvalMetrics:
    """Standard TKG evaluation metrics."""

    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_10: float = 0.0
    mrr: float = 0.0
    temporal_precision_at_10: float = 0.0
    num_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "hits@1": round(self.hits_at_1, 4),
            "hits@3": round(self.hits_at_3, 4),
            "hits@10": round(self.hits_at_10, 4),
            "mrr": round(self.mrr, 4),
            "temp_prec@10": round(self.temporal_precision_at_10, 4),
            "num_queries": self.num_queries,
        }


def compute_rank(scores: dict[int, float], target: int) -> int:
    """Compute the rank of the target entity in the scored candidate list.

    Uses filtered setting: only rank among candidates that appear in training.
    """
    if target not in scores:
        return len(scores) + 1
    target_score = scores[target]
    rank = 1
    for obj, s in scores.items():
        if obj != target and s > target_score:
            rank += 1
    return rank


def evaluate(
    test_triples: list[Triple],
    index: TemporalRetrievalIndex,
    score_fn,
    max_queries: int | None = None,
    temporal_window: int = 30,
) -> EvalMetrics:
    """Evaluate a scoring strategy on the test set.

    Args:
        test_triples: Test triples to evaluate.
        index: Training data index for retrieval.
        score_fn: Callable(index, subject, relation, candidates, query_time) -> scores dict.
        max_queries: Limit number of queries (for faster iteration).
        temporal_window: Window (in timesteps) to consider a fact "temporally valid"
            for Temporal Precision@k.
    """
    hits1, hits3, hits10 = 0, 0, 0
    mrr_sum = 0.0
    temp_prec_sum = 0.0
    evaluated = 0
    skipped = 0

    for i, triple in enumerate(test_triples):
        if max_queries and evaluated >= max_queries:
            break

        candidates = index.get_candidates(triple.subject, triple.relation)
        if not candidates:
            skipped += 1
            continue

        # Ensure the target is in the candidate set (open-world assumption)
        candidates = candidates | {triple.obj}

        scores = score_fn(index, triple.subject, triple.relation, candidates, triple.timestamp)

        rank = compute_rank(scores, triple.obj)

        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        if rank <= 10:
            hits10 += 1
        mrr_sum += 1.0 / rank

        # Temporal Precision@10: of top-10 retrieved, how many are
        # temporally close to the query time?
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        temp_valid = 0
        for obj, _ in sorted_candidates:
            ts_list = index.get_timestamps(triple.subject, triple.relation, obj)
            if ts_list and min(abs(triple.timestamp - t) for t in ts_list) <= temporal_window:
                temp_valid += 1
        temp_prec_sum += temp_valid / min(10, len(sorted_candidates)) if sorted_candidates else 0

        evaluated += 1

    if evaluated == 0:
        return EvalMetrics()

    return EvalMetrics(
        hits_at_1=hits1 / evaluated,
        hits_at_3=hits3 / evaluated,
        hits_at_10=hits10 / evaluated,
        mrr=mrr_sum / evaluated,
        temporal_precision_at_10=temp_prec_sum / evaluated,
        num_queries=evaluated,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def run_experiment(max_queries: int | None = None) -> dict[str, EvalMetrics]:
    """Run the full Experiment 1 evaluation."""
    print("=" * 70)
    print("EXPERIMENT 1: Decay-Weighted TKG Retrieval")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading ICEWS14 dataset...")
    dataset = load_icews14()
    print(f"  {dataset.summary}")

    # Build index
    print("\n[2/4] Building retrieval index...")
    index = TemporalRetrievalIndex(dataset.train)
    print(f"  Indexed {len(dataset.train)} training triples")

    results: dict[str, EvalMetrics] = {}
    test = dataset.test
    query_label = f" (first {max_queries})" if max_queries else ""
    print(f"\n[3/4] Evaluating on {len(test)} test triples{query_label}...")

    # Baseline: Static (frequency)
    print("\n  → Static (frequency)...")
    results["static"] = evaluate(test, index, score_static, max_queries=max_queries)
    print(
        f"    MRR={results['static'].mrr:.4f}  H@1={results['static'].hits_at_1:.4f}  "
        f"H@10={results['static'].hits_at_10:.4f}  "
        f"TP@10={results['static'].temporal_precision_at_10:.4f}"
    )

    # Baseline: Recency-only
    print("\n  → Recency-only...")
    results["recency"] = evaluate(test, index, score_recency, max_queries=max_queries)
    print(
        f"    MRR={results['recency'].mrr:.4f}  H@1={results['recency'].hits_at_1:.4f}  "
        f"H@10={results['recency'].hits_at_10:.4f}  "
        f"TP@10={results['recency'].temporal_precision_at_10:.4f}"
    )

    # Chronofy decay: sweep β values
    print("\n  → Chronofy decay (sweeping β)...")
    best_beta = 0.0
    best_mrr = 0.0

    for beta in BETA_VALUES:

        def score_fn(idx, s, r, c, t, _beta=beta):
            return score_chronofy_decay(idx, s, r, c, t, _beta)

        label = f"chronofy_sum_β={beta}"
        results[label] = evaluate(test, index, score_fn, max_queries=max_queries)
        mrr = results[label].mrr
        print(
            f"    β={beta:<6}  MRR={mrr:.4f}  H@1={results[label].hits_at_1:.4f}  "
            f"H@10={results[label].hits_at_10:.4f}  "
            f"TP@10={results[label].temporal_precision_at_10:.4f}"
        )

        if mrr > best_mrr:
            best_mrr = mrr
            best_beta = beta

    # Chronofy decay (max): sweep β values
    print("\n  → Chronofy decay-max (sweeping β)...")
    best_beta_max = 0.0
    best_mrr_max = 0.0

    for beta in BETA_VALUES:

        def score_fn(idx, s, r, c, t, _beta=beta):
            return score_chronofy_max_decay(idx, s, r, c, t, _beta)

        label = f"chronofy_max_β={beta}"
        results[label] = evaluate(test, index, score_fn, max_queries=max_queries)
        mrr = results[label].mrr
        print(
            f"    β={beta:<6}  MRR={mrr:.4f}  H@1={results[label].hits_at_1:.4f}  "
            f"H@10={results[label].hits_at_10:.4f}  "
            f"TP@10={results[label].temporal_precision_at_10:.4f}"
        )

        if mrr > best_mrr_max:
            best_mrr_max = mrr
            best_beta_max = beta

    # Summary
    print("\n" + "=" * 70)
    print(f"Best Chronofy (sum): β={best_beta}, MRR={best_mrr:.4f}")
    print(f"Best Chronofy (max): β={best_beta_max}, MRR={best_mrr_max:.4f}")
    print(f"Static baseline:     MRR={results['static'].mrr:.4f}")
    print(f"Recency baseline:    MRR={results['recency'].mrr:.4f}")
    print("=" * 70)

    return results


def save_results(results: dict[str, EvalMetrics]) -> None:
    """Save results to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "exp1_tkg_decay.json"
    json_data = {k: v.to_dict() for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[saved] {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "exp1_tkg_decay.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "hits@1", "hits@3", "hits@10", "mrr", "temp_prec@10", "num_queries"]
        )
        for method, metrics in results.items():
            d = metrics.to_dict()
            writer.writerow(
                [
                    method,
                    d["hits@1"],
                    d["hits@3"],
                    d["hits@10"],
                    d["mrr"],
                    d["temp_prec@10"],
                    d["num_queries"],
                ]
            )
    print(f"[saved] {csv_path}")


def print_results_table() -> None:
    """Load and print saved results."""
    json_path = RESULTS_DIR / "exp1_tkg_decay.json"
    if not json_path.exists():
        print("No results found. Run the experiment first: --run")
        return
    with open(json_path) as f:
        data = json.load(f)

    print(f"\n{'Method':<30} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8} {'TP@10':>8}")
    print("-" * 80)
    for method, metrics in data.items():
        print(
            f"{method:<30} {metrics['mrr']:>8.4f} {metrics['hits@1']:>8.4f} "
            f"{metrics['hits@3']:>8.4f} {metrics['hits@10']:>8.4f} {metrics['temp_prec@10']:>8.4f}"
        )


# ============================================================
# ALTERNATIVE DECAY EXPERIMENT
# ============================================================


def run_alternatives(max_queries: int | None = None) -> dict[str, EvalMetrics]:
    """Run alternative decay function comparison."""
    print("=" * 70)
    print("EXPERIMENT 1B: Alternative Decay Function Comparison")
    print("=" * 70)

    print("\n[1/3] Loading ICEWS14 dataset...")
    dataset = load_icews14()
    print(f"  {dataset.summary}")

    print("\n[2/3] Building retrieval index...")
    index = TemporalRetrievalIndex(dataset.train)

    results: dict[str, EvalMetrics] = {}
    test = dataset.test
    query_label = f" (first {max_queries})" if max_queries else ""
    print(
        "\n[3/3] Evaluating alternative decay functions on "
        f"{len(test)} test triples{query_label}..."
    )

    def _eval(label, fn):
        r = evaluate(test, index, fn, max_queries=max_queries)
        results[label] = r
        print(
            f"    {label:<35} MRR={r.mrr:.4f}  H@1={r.hits_at_1:.4f}  " f"H@10={r.hits_at_10:.4f}"
        )
        return r

    # Half-life sweep
    print("\n  --- Half-Life Decay ---")
    for h in HALF_LIFE_VALUES:
        _eval(
            f"half_life_h={h}",
            lambda idx, s, r, c, t, _h=h: score_half_life_decay(idx, s, r, c, t, _h),
        )

    # Linear sweep
    print("\n  --- Linear Decay ---")
    for alpha in LINEAR_RATE_VALUES:
        _eval(
            f"linear_a={alpha}",
            lambda idx, s, r, c, t, _a=alpha: score_linear_decay(idx, s, r, c, t, _a),
        )

    # Power-law sweep
    print("\n  --- Power-Law Decay ---")
    for alpha in POWER_LAW_EXPONENTS:
        _eval(
            f"power_law_a={alpha}",
            lambda idx, s, r, c, t, _a=alpha: score_power_law_decay(idx, s, r, c, t, _a),
        )

    # Weibull sweep
    print("\n  --- Weibull Decay ---")
    for lam, k in WEIBULL_CONFIGS:
        _eval(
            f"weibull_l={lam}_k={k}",
            lambda idx, s, r, c, t, _l=lam, _k=k: score_weibull_decay(idx, s, r, c, t, _l, _k),
        )

    # Summary: find best in each family
    print("\n" + "=" * 70)
    families = {
        "half_life": [k for k in results if k.startswith("half_life")],
        "linear": [k for k in results if k.startswith("linear")],
        "power_law": [k for k in results if k.startswith("power_law")],
        "weibull": [k for k in results if k.startswith("weibull")],
    }
    for family, keys in families.items():
        best_key = max(keys, key=lambda k: results[k].mrr)
        print(f"Best {family}: {best_key}, MRR={results[best_key].mrr:.4f}")
    print("=" * 70)

    return results


def save_alternatives(results: dict[str, EvalMetrics]) -> None:
    """Save alternative decay results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "exp1b_decay_alternatives.json"
    json_data = {k: v.to_dict() for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[saved] {json_path}")

    csv_path = RESULTS_DIR / "exp1b_decay_alternatives.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "hits@1", "hits@3", "hits@10", "mrr", "temp_prec@10", "num_queries"]
        )
        for method, metrics in results.items():
            d = metrics.to_dict()
            writer.writerow(
                [
                    method,
                    d["hits@1"],
                    d["hits@3"],
                    d["hits@10"],
                    d["mrr"],
                    d["temp_prec@10"],
                    d["num_queries"],
                ]
            )
    print(f"[saved] {csv_path}")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Decay-Weighted TKG Retrieval")
    parser.add_argument("--download", action="store_true", help="Download ICEWS14 dataset")
    parser.add_argument("--run", action="store_true", help="Run main experiment")
    parser.add_argument(
        "--alternatives", action="store_true", help="Run alternative decay comparison"
    )
    parser.add_argument("--quick", action="store_true", help="Limit to first 1000 queries")
    parser.add_argument("--results", action="store_true", help="Print saved results")
    args = parser.parse_args()

    max_q = 1000 if args.quick else None

    if args.download:
        print("Downloading ICEWS14 dataset...")
        download_icews14()
        print("Done.")

    if args.run:
        results = run_experiment(max_queries=max_q)
        save_results(results)

    if args.alternatives:
        results = run_alternatives(max_queries=max_q)
        save_alternatives(results)

    if args.results:
        print_results_table()
        # Also print alternatives if available
        alt_path = RESULTS_DIR / "exp1b_decay_alternatives.json"
        if alt_path.exists():
            print("\n--- Alternative Decay Functions ---")
            with open(alt_path) as f:
                data = json.load(f)
            print(f"\n{'Method':<35} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}")
            print("-" * 75)
            for method, metrics in data.items():
                print(
                    f"{method:<35} {metrics['mrr']:>8.4f} {metrics['hits@1']:>8.4f} "
                    f"{metrics['hits@3']:>8.4f} {metrics['hits@10']:>8.4f}"
                )

    if not any([args.download, args.run, args.alternatives, args.quick, args.results]):
        parser.print_help()


if __name__ == "__main__":
    main()
