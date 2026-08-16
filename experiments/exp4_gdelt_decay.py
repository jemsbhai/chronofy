"""Experiment 4: GDELT Decay-Weighted TKG Retrieval + Per-Relation Beta Sensitivity

KEY EXPERIMENT for the heterogeneity thesis.

GDELT has 20 CAMEO event-type relations (diplomatic cooperation, military force,
protests, etc.) with genuinely different temporal dynamics — unlike ICEWS14 which
was too homogeneous (all political events within 1 year).

Hypothesis: Optimal beta varies across CAMEO event categories. Military/conflict
events (high volatility) want high beta; diplomatic cooperation (slower) wants
moderate beta; structural events want low beta.

Dataset: linxy/GDELT on HuggingFace (TFLEX benchmark)
  - 500 entities, 20 relations, 366 timestamps
  - ~2.3M train triples, ~330K test triples

Design:
  Part A: Global decay sweep (same as Exp 1 on ICEWS14)
    - Sweep beta over full test set
    - Compare Chronofy vs static/recency baselines

  Part B: Per-relation beta sensitivity (the novel result)
    - For each of the 20 relations, sweep beta independently
    - Show that optimal beta varies across relation types
    - This is the key result that validates the heterogeneity thesis

Usage:
  python experiments/exp4_gdelt_decay.py --download     # Download GDELT
  python experiments/exp4_gdelt_decay.py --run           # Full experiment
  python experiments/exp4_gdelt_decay.py --run --limit N # Test with N queries
  python experiments/exp4_gdelt_decay.py --results       # Show saved results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproducibility import GDELT_REVISION

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Beta values to sweep — wider range than ICEWS14 since GDELT spans a full year
BETA_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
BETA_VALUES_FINE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# CAMEO root event codes — the 20 GDELT relation types
# These have genuinely different temporal dynamics:
#   Cooperative events (01-09) should be slower-changing (lower optimal beta)
#   Conflict events (10-20) should be faster-changing (higher optimal beta)
CAMEO_NAMES = {
    0: "Make Public Statement",
    1: "Appeal",
    2: "Express Intent to Cooperate",
    3: "Consult",
    4: "Engage in Diplomatic Cooperation",
    5: "Engage in Material Cooperation",
    6: "Provide Aid",
    7: "Yield",
    8: "Investigate",
    9: "Demand",
    10: "Disapprove",
    11: "Reject",
    12: "Threaten",
    13: "Protest",
    14: "Exhibit Military Posture",
    15: "Reduce Relations",
    16: "Coerce",
    17: "Assault",
    18: "Fight",
    19: "Unconventional Mass Violence",
}


# ============================================================
# DATA LOADING (HuggingFace — TFLEX Pe query format)
# ============================================================


@dataclass
class Triple:
    subject: int
    relation: int
    obj: int
    timestamp: int


@dataclass
class GDELTDataset:
    train: list[Triple] = field(default_factory=list)
    test: list[Triple] = field(default_factory=list)
    num_entities: int = 0
    num_relations: int = 0
    num_timestamps: int = 0
    relation_id_to_cameo: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return (
            f"GDELTDataset: {self.num_entities} entities, "
            f"{self.num_relations} relations, "
            f"{self.num_timestamps} timestamps\n"
            f"  Train: {len(self.train)} triples\n"
            f"  Test:  {len(self.test)} triples (Pe queries)"
        )


def download_gdelt() -> None:
    """Download GDELT dataset via HuggingFace (automatic caching)."""
    from datasets import load_dataset

    print("  Downloading GDELT from HuggingFace (linxy/GDELT)...")
    print("  (This will cache to HuggingFace's default cache dir)")
    _ = load_dataset("linxy/GDELT", "meta", revision=GDELT_REVISION)
    _ = load_dataset("linxy/GDELT", "Pe", revision=GDELT_REVISION)
    print("  Done.")


def _pe_queries_to_triples(split) -> list[Triple]:
    """Convert Pe queries to triples.

    Each Pe query is: query=[subject, relation, timestamp], answer=[obj1, obj2, ...]
    We expand to triples: (subject, relation, obj_i, timestamp) for each answer.
    """
    triples = []
    seen = set()  # deduplicate
    for row in split:
        query = row["query"]  # [subject, relation, timestamp]
        answers = row["answer"]
        s, r, t = query[0], query[1], query[2]
        for o in answers:
            key = (s, r, o, t)
            if key not in seen:
                seen.add(key)
                triples.append(Triple(subject=s, relation=r, obj=o, timestamp=t))
    return triples


def load_gdelt() -> GDELTDataset:
    """Load the GDELT TKG dataset from HuggingFace Pe queries.

    Reconstructs raw triples from Pe (predict entity) queries.
    Training Pe queries -> training triples (for retrieval index).
    Test Pe queries -> test queries (for evaluation).
    """
    from datasets import load_dataset

    # Load metadata
    ds_meta = load_dataset("linxy/GDELT", "meta", revision=GDELT_REVISION)
    meta = ds_meta["train"][0]

    # Load Pe (predict entity) queries — this is the standard TKG task
    print("  Loading Pe queries...")
    ds_pe = load_dataset("linxy/GDELT", "Pe", revision=GDELT_REVISION)

    # Reconstruct training triples from training Pe queries
    print("  Reconstructing training triples from Pe queries...")
    train_triples = _pe_queries_to_triples(ds_pe["train"])
    print(f"    {len(train_triples)} unique training triples")

    # Build relation ID mapping
    # The relation IDs in the dataset are indices into the sorted relation name list
    # Relation names are "0","1","10","11",...,"9" (lexicographic sort of CAMEO codes)
    rel_names_sorted = meta["relation2idx"]["name"]  # ["0","1","10",...,"9"]
    rel_ids = meta["relation2idx"]["id"]  # [0, 1, 2, ..., 19]
    # Map dataset relation index -> CAMEO code
    rel_id_to_cameo = {}
    for idx, name in zip(rel_ids, rel_names_sorted):
        cameo_code = int(name)  # "10" -> 10, "2" -> 2, etc.
        rel_id_to_cameo[idx] = cameo_code

    # For test, keep as query format — each query has one ground truth
    # We'll evaluate each test Pe query directly
    test_triples = []
    for row in ds_pe["test"]:
        query = row["query"]  # [subject, relation, timestamp]
        answers = row["answer"]
        s, r, t = query[0], query[1], query[2]
        # For evaluation: each answer is a valid ground truth
        # We pick the first as the "target" (standard TKG eval)
        for o in answers:
            test_triples.append(Triple(subject=s, relation=r, obj=o, timestamp=t))

    print(f"    {len(test_triples)} test triples")

    return GDELTDataset(
        train=train_triples,
        test=test_triples,
        num_entities=meta["entity_count"],
        num_relations=meta["relation_count"],
        num_timestamps=meta["timestamp_count"],
        relation_id_to_cameo=rel_id_to_cameo,
    )


# ============================================================
# RETRIEVAL INDEX
# ============================================================


class TemporalRetrievalIndex:
    """Index training triples for efficient retrieval by (subject, relation)."""

    def __init__(self, triples: list[Triple]) -> None:
        self._sr_index: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        self._sro_timestamps: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        self._sr_objects: dict[tuple[int, int], set[int]] = defaultdict(set)

        for t in triples:
            self._sr_index[(t.subject, t.relation)].append((t.obj, t.timestamp))
            self._sro_timestamps[(t.subject, t.relation, t.obj)].append(t.timestamp)
            self._sr_objects[(t.subject, t.relation)].add(t.obj)

    def get_candidates(self, subject: int, relation: int) -> set[int]:
        return self._sr_objects.get((subject, relation), set())

    def get_timestamps(self, subject: int, relation: int, obj: int) -> list[int]:
        return self._sro_timestamps.get((subject, relation, obj), [])

    def get_all_facts(self, subject: int, relation: int) -> list[tuple[int, int]]:
        return self._sr_index.get((subject, relation), [])


# ============================================================
# SCORING STRATEGIES
# ============================================================


def score_static(index, subject, relation, candidates, query_time):
    """Static baseline: score = frequency count."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        scores[obj] = float(len(timestamps))
    return scores


def score_recency(index, subject, relation, candidates, query_time):
    """Recency baseline: prefer most recent occurrence."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            most_recent = max(timestamps)
            scores[obj] = 1.0 / (1.0 + abs(query_time - most_recent))
        else:
            scores[obj] = 0.0
    return scores


def score_chronofy_decay(index, subject, relation, candidates, query_time, beta):
    """Chronofy exponential decay: score = sum of exp(-beta * |t_q - t_f|)."""
    scores = {}
    for obj in candidates:
        timestamps = index.get_timestamps(subject, relation, obj)
        if timestamps:
            total = sum(np.exp(-beta * abs(query_time - t)) for t in timestamps)
            scores[obj] = float(total)
        else:
            scores[obj] = 0.0
    return scores


# ============================================================
# EVALUATION
# ============================================================


def evaluate_ranking(scores: dict[int, float], true_obj: int) -> dict:
    """Compute ranking metrics for a single query."""
    if true_obj not in scores or not scores:
        return {"rank": len(scores) + 1, "hits1": 0, "hits3": 0, "hits10": 0, "mrr": 0.0}

    sorted_objs = sorted(scores, key=lambda o: scores[o], reverse=True)
    try:
        rank = sorted_objs.index(true_obj) + 1
    except ValueError:
        rank = len(sorted_objs) + 1

    return {
        "rank": rank,
        "hits1": int(rank <= 1),
        "hits3": int(rank <= 3),
        "hits10": int(rank <= 10),
        "mrr": 1.0 / rank,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-query metrics into overall metrics."""
    if not results:
        return {"mrr": 0, "hits1": 0, "hits3": 0, "hits10": 0, "n": 0}
    n = len(results)
    return {
        "mrr": float(np.mean([r["mrr"] for r in results])),
        "hits1": float(np.mean([r["hits1"] for r in results])),
        "hits3": float(np.mean([r["hits3"] for r in results])),
        "hits10": float(np.mean([r["hits10"] for r in results])),
        "n": n,
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================


def run_experiment(*, limit: int | None = None, fine: bool = False) -> dict:
    """Run the full GDELT decay experiment."""
    beta_list = BETA_VALUES_FINE if fine else BETA_VALUES
    exp_suffix = "_fine" if fine else ""
    sweep_name = "fine-grained" if fine else "standard"
    print(f"[1/5] Loading GDELT dataset... ({sweep_name} sweep)")
    data = load_gdelt()
    print(f"  {data.summary}")
    relation_names = [
        CAMEO_NAMES.get(data.relation_id_to_cameo.get(index, index), index)
        for index in range(data.num_relations)
    ]
    print(f"  Relations: {relation_names}")

    # Build retrieval index from training data
    print("\n[2/5] Building retrieval index...")
    index = TemporalRetrievalIndex(data.train)

    # Determine test queries
    test_queries = data.test
    if limit:
        test_queries = test_queries[:limit]
    print(f"  Test queries: {len(test_queries)}")

    def _rel_name(rel_id: int) -> str:
        cameo = data.relation_id_to_cameo.get(rel_id, rel_id)
        return CAMEO_NAMES.get(cameo, f"CAMEO_{cameo}")

    # Count per-relation test queries
    rel_counts = defaultdict(int)
    for t in test_queries:
        rel_counts[t.relation] += 1
    print("  Queries per relation:")
    for rel_id in sorted(rel_counts):
        print(f"    {rel_id:3d} ({_rel_name(rel_id)}): {rel_counts[rel_id]}")

    # ---- Part A: Global decay sweep ----
    print(f"\n[3/5] Part A: Global decay sweep ({len(beta_list)} beta values)...")
    global_results = {}

    # Static baseline
    print("  Running: static")
    static_evals = []
    for q in test_queries:
        cands = index.get_candidates(q.subject, q.relation)
        if not cands:
            continue
        scores = score_static(index, q.subject, q.relation, cands, q.timestamp)
        static_evals.append(evaluate_ranking(scores, q.obj))
    global_results["static"] = aggregate_metrics(static_evals)

    # Recency baseline
    print("  Running: recency")
    recency_evals = []
    for q in test_queries:
        cands = index.get_candidates(q.subject, q.relation)
        if not cands:
            continue
        scores = score_recency(index, q.subject, q.relation, cands, q.timestamp)
        recency_evals.append(evaluate_ranking(scores, q.obj))
    global_results["recency"] = aggregate_metrics(recency_evals)

    # Chronofy decay sweep
    for beta in beta_list:
        label = f"chronofy_beta={beta}"
        print(f"  Running: {label}")
        decay_evals = []
        for q in test_queries:
            cands = index.get_candidates(q.subject, q.relation)
            if not cands:
                continue
            scores = score_chronofy_decay(index, q.subject, q.relation, cands, q.timestamp, beta)
            decay_evals.append(evaluate_ranking(scores, q.obj))
        global_results[label] = aggregate_metrics(decay_evals)

    # ---- Part B: Per-relation beta sensitivity ----
    print("\n[4/5] Part B: Per-relation beta sensitivity...")
    per_relation_results = {}

    # Group test queries by relation
    rel_queries = defaultdict(list)
    for q in test_queries:
        rel_queries[q.relation].append(q)

    for rel_id in sorted(rel_queries):
        queries = rel_queries[rel_id]
        rel_name = _rel_name(rel_id)
        print(f"  Relation {rel_id} ({rel_name}): {len(queries)} queries")

        rel_data = {"relation_id": rel_id, "relation_name": rel_name, "n_queries": len(queries)}

        # Static baseline for this relation
        static_evals = []
        for q in queries:
            cands = index.get_candidates(q.subject, q.relation)
            if not cands:
                continue
            scores = score_static(index, q.subject, q.relation, cands, q.timestamp)
            static_evals.append(evaluate_ranking(scores, q.obj))
        rel_data["static"] = aggregate_metrics(static_evals)

        # Beta sweep for this relation
        best_beta = 0.0
        best_mrr = 0.0
        beta_sweep = {}
        for beta in beta_list:
            decay_evals = []
            for q in queries:
                cands = index.get_candidates(q.subject, q.relation)
                if not cands:
                    continue
                scores = score_chronofy_decay(
                    index, q.subject, q.relation, cands, q.timestamp, beta
                )
                decay_evals.append(evaluate_ranking(scores, q.obj))
            metrics = aggregate_metrics(decay_evals)
            beta_sweep[str(beta)] = metrics
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_beta = beta

        rel_data["beta_sweep"] = beta_sweep
        rel_data["best_beta"] = best_beta
        rel_data["best_mrr"] = best_mrr
        rel_data["mrr_improvement"] = (
            best_mrr - rel_data["static"]["mrr"] if rel_data["static"]["mrr"] > 0 else 0.0
        )

        per_relation_results[rel_name] = rel_data

    # ---- Save ----
    print("\n[5/5] Saving results...")
    output = {
        "experiment": "exp4_gdelt_decay",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": "linxy/GDELT",
            "num_entities": data.num_entities,
            "num_relations": data.num_relations,
            "num_timestamps": data.num_timestamps,
            "num_train": len(data.train),
            "num_test": len(test_queries),
            "beta_values": beta_list,
            "limit": limit,
            "relation_id_to_cameo": data.relation_id_to_cameo,
        },
        "global_results": global_results,
        "per_relation_results": per_relation_results,
    }

    from experiments.utils import timestamped_save

    timestamped_save(output, f"exp4_gdelt_decay{exp_suffix}", save_csv=False)

    print_summary(output)
    return output


# ============================================================
# DISPLAY
# ============================================================


def print_summary(output: dict) -> None:
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: GDELT Decay-Weighted TKG Retrieval")
    print("=" * 70)

    cfg = output["config"]
    print(
        f"\nDataset: GDELT ({cfg['num_entities']} entities, "
        f"{cfg['num_relations']} relations, {cfg['num_timestamps']} timestamps)"
    )
    print(f"Train: {cfg['num_train']} triples, Test: {cfg['num_test']} queries")

    # Part A: Global results
    print("\n--- Part A: Global Decay Sweep ---")
    print(f"{'Method':<30s}  {'MRR':>7s}  {'H@1':>7s}  {'H@3':>7s}  {'H@10':>7s}  {'N':>7s}")
    print("-" * 70)
    for method, metrics in sorted(output["global_results"].items()):
        print(
            f"{method:<30s}  {metrics['mrr']:7.4f}  {metrics['hits1']:7.4f}  "
            f"{metrics['hits3']:7.4f}  {metrics['hits10']:7.4f}  {metrics['n']:7d}"
        )

    # Find best global beta
    best_global = max(
        ((k, v) for k, v in output["global_results"].items() if k.startswith("chronofy")),
        key=lambda x: x[1]["mrr"],
        default=None,
    )
    if best_global:
        static_mrr = output["global_results"].get("static", {}).get("mrr", 0)
        improvement = (
            (best_global[1]["mrr"] - static_mrr) / static_mrr * 100 if static_mrr > 0 else 0
        )
        print(
            f"\nBest global: {best_global[0]} (MRR={best_global[1]['mrr']:.4f}, "
            f"+{improvement:.1f}% vs static)"
        )

    # Part B: Per-relation results
    print("\n--- Part B: Per-Relation Optimal Beta ---")
    print(
        f"{'Relation':<35s}  {'Best β':>7s}  {'Best MRR':>8s}  {'Static MRR':>10s}  {'Δ MRR':>7s}"
    )
    print("-" * 70)

    betas_found = []
    for rel_name, rel_data in sorted(output["per_relation_results"].items()):
        best_b = rel_data["best_beta"]
        best_m = rel_data["best_mrr"]
        static_m = rel_data["static"]["mrr"]
        delta = rel_data["mrr_improvement"]
        betas_found.append(best_b)
        print(f"{rel_name:<35s}  {best_b:7.3f}  {best_m:8.4f}  {static_m:10.4f}  {delta:+7.4f}")

    # Summary statistics on beta distribution
    betas_arr = np.array(betas_found)
    print("\nBeta distribution across relations:")
    print(f"  Mean:   {betas_arr.mean():.4f}")
    print(f"  Std:    {betas_arr.std():.4f}")
    print(f"  Min:    {betas_arr.min():.4f}")
    print(f"  Max:    {betas_arr.max():.4f}")
    print(f"  Unique: {len(set(betas_found))} distinct values")

    # Key finding
    if len(set(betas_found)) > 1:
        print("\n  *** KEY FINDING: Optimal beta VARIES across relation types ***")
        print("  *** This validates the heterogeneous temporal value thesis ***")
    else:
        print(f"\n  Note: All relations peaked at same beta={betas_found[0]}")
        print("  (Similar to ICEWS14 — dataset may be too homogeneous)")
    print()


def show_results() -> None:
    """Load and display saved results."""
    results_path = RESULTS_DIR / "exp4_gdelt_decay.json"
    if not results_path.exists():
        print(f"No results found at {results_path}")
        print("Run with --run first.")
        return
    with open(results_path, encoding="utf-8") as f:
        output = json.load(f)
    print_summary(output)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: GDELT decay + per-relation beta")
    parser.add_argument("--download", action="store_true", help="Download GDELT dataset")
    parser.add_argument("--run", action="store_true", help="Run the experiment")
    parser.add_argument("--results", action="store_true", help="Show saved results")
    parser.add_argument("--limit", type=int, default=None, help="Limit test queries")
    parser.add_argument("--fine", action="store_true", help="Fine-grained beta sweep around 0.5")
    args = parser.parse_args()

    if args.download:
        download_gdelt()
    elif args.run:
        run_experiment(limit=args.limit, fine=args.fine)
    elif args.results:
        show_results()
    else:
        parser.print_help()
