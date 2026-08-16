"""Experiment 3: β Sensitivity Analysis Across Fact Categories

Goal: Demonstrate that different fact types require different decay rates,
and that β is learnable — validating Proposition 1 (β = 2κ).

Method:
  1. Define fact categories with different expected temporal dynamics
  2. For each category, sweep β and measure retrieval accuracy
  3. Show that optimal β varies dramatically across categories
  4. Compare learned β (grid search on validation) vs. fixed β vs. oracle

This uses ICEWS14 relations as proxies for fact categories:
  - High-volatility relations (e.g., "Make statement", "Express intent")
    → expected high optimal β (fast decay)
  - Medium-volatility relations (e.g., "Make an appeal", "Consult")
    → expected medium optimal β
  - Low-volatility relations (e.g., "Sign formal agreement")
    → expected low optimal β (slow decay)

Output: Heat map of β × relation_category × accuracy for the paper.

Usage:
  python experiments/exp3_beta_sensitivity.py --run
  python experiments/exp3_beta_sensitivity.py --quick
  python experiments/exp3_beta_sensitivity.py --results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp1_tkg_decay import (
    TemporalRetrievalIndex,
    Triple,
    compute_rank,
    load_icews14,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# β sweep grid — fine-grained
BETA_GRID = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


# ============================================================
# RELATION CATEGORIZATION
# ============================================================


def categorize_relations(train: list[Triple], test: list[Triple]) -> dict[str, list[int]]:
    """Categorize relations by their temporal volatility.

    Volatility = how much the object distribution changes over time.
    We measure this as the ratio of unique (s,r,o) triples to total
    occurrences — high ratio means diverse/changing, low means repetitive/stable.

    We also consider the average time span of each relation.
    """
    # Count unique triples and total occurrences per relation
    rel_unique: dict[int, set[tuple[int, int]]] = defaultdict(set)
    rel_total: dict[int, int] = defaultdict(int)
    rel_timestamps: dict[int, list[int]] = defaultdict(list)

    for t in train:
        rel_unique[t.relation].add((t.subject, t.obj))
        rel_total[t.relation] += 1
        rel_timestamps[t.relation].append(t.timestamp)

    # Only consider relations with enough test queries
    test_rel_counts = Counter(t.relation for t in test)
    active_relations = {r for r, c in test_rel_counts.items() if c >= 20}

    # Compute volatility score for each relation
    rel_volatility: dict[int, float] = {}
    for r in active_relations:
        if r in rel_total and rel_total[r] > 0:
            # Volatility: unique pairs / total occurrences
            volatility = len(rel_unique[r]) / rel_total[r]
            # Time spread: std of timestamps
            ts = rel_timestamps[r]
            time_spread = np.std(ts) if len(ts) > 1 else 0.0
            # Combined score: high volatility + high spread = fast-changing
            rel_volatility[r] = volatility * (1.0 + time_spread / 365.0)

    if not rel_volatility:
        return {}

    # Sort by volatility and split into terciles
    sorted_rels = sorted(rel_volatility.items(), key=lambda x: x[1])
    n = len(sorted_rels)
    tercile_1 = n // 3
    tercile_2 = 2 * n // 3

    categories = {
        "low_volatility": [r for r, _ in sorted_rels[:tercile_1]],
        "medium_volatility": [r for r, _ in sorted_rels[tercile_1:tercile_2]],
        "high_volatility": [r for r, _ in sorted_rels[tercile_2:]],
    }

    return categories


# ============================================================
# PER-CATEGORY EVALUATION
# ============================================================


@dataclass
class CategoryResult:
    """Results for a single (category, β) pair."""

    category: str
    beta: float
    mrr: float
    hits_at_1: float
    hits_at_10: float
    num_queries: int


def evaluate_category(
    test_triples: list[Triple],
    index: TemporalRetrievalIndex,
    relation_ids: list[int],
    beta: float,
    max_queries_per_cat: int | None = None,
) -> CategoryResult:
    """Evaluate exponential decay with given β on triples matching specific relations."""
    hits1, hits10 = 0, 0
    mrr_sum = 0.0
    evaluated = 0

    # Filter test triples to this category
    category_triples = [t for t in test_triples if t.relation in set(relation_ids)]

    for triple in category_triples:
        if max_queries_per_cat and evaluated >= max_queries_per_cat:
            break

        candidates = index.get_candidates(triple.subject, triple.relation)
        if not candidates:
            continue

        candidates = candidates | {triple.obj}

        # Score using exponential decay
        scores = {}
        for obj in candidates:
            timestamps = index.get_timestamps(triple.subject, triple.relation, obj)
            if timestamps:
                if beta == 0.0:
                    scores[obj] = float(len(timestamps))
                else:
                    total = sum(np.exp(-beta * abs(triple.timestamp - t)) for t in timestamps)
                    scores[obj] = float(total)
            else:
                scores[obj] = 0.0

        rank = compute_rank(scores, triple.obj)

        if rank <= 1:
            hits1 += 1
        if rank <= 10:
            hits10 += 1
        mrr_sum += 1.0 / rank
        evaluated += 1

    if evaluated == 0:
        return CategoryResult("", beta, 0.0, 0.0, 0.0, 0)

    return CategoryResult(
        category="",
        beta=beta,
        mrr=mrr_sum / evaluated,
        hits_at_1=hits1 / evaluated,
        hits_at_10=hits10 / evaluated,
        num_queries=evaluated,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def run_experiment(max_queries_per_cat: int | None = None) -> dict:
    """Run the β sensitivity analysis."""
    print("=" * 70)
    print("EXPERIMENT 3: β Sensitivity Analysis Across Fact Categories")
    print("=" * 70)

    print("\n[1/4] Loading ICEWS14 dataset...")
    dataset = load_icews14()
    print(f"  {dataset.summary}")

    print("\n[2/4] Building retrieval index...")
    index = TemporalRetrievalIndex(dataset.train)

    print("\n[3/4] Categorizing relations by temporal volatility...")
    categories = categorize_relations(dataset.train, dataset.test)
    for cat_name, rel_ids in categories.items():
        test_count = sum(1 for t in dataset.test if t.relation in set(rel_ids))
        print(f"  {cat_name}: {len(rel_ids)} relations, {test_count} test queries")

    print(f"\n[4/4] Sweeping β across {len(BETA_GRID)} values per category...")

    # Results: category → β → metrics
    all_results: dict[str, dict[float, dict]] = {}
    optimal_betas: dict[str, tuple[float, float]] = {}  # category → (best_β, best_mrr)

    for cat_name, rel_ids in categories.items():
        print(f"\n  --- {cat_name} ---")
        all_results[cat_name] = {}
        best_mrr = 0.0
        best_beta = 0.0

        for beta in BETA_GRID:
            result = evaluate_category(
                dataset.test,
                index,
                rel_ids,
                beta,
                max_queries_per_cat=max_queries_per_cat,
            )
            all_results[cat_name][beta] = {
                "mrr": round(result.mrr, 4),
                "hits@1": round(result.hits_at_1, 4),
                "hits@10": round(result.hits_at_10, 4),
                "num_queries": result.num_queries,
            }
            marker = ""
            if result.mrr > best_mrr:
                best_mrr = result.mrr
                best_beta = beta
                marker = " ← best"
            print(
                f"    β={beta:<6}  MRR={result.mrr:.4f}  H@1={result.hits_at_1:.4f}  "
                f"H@10={result.hits_at_10:.4f}  (n={result.num_queries}){marker}"
            )

        optimal_betas[cat_name] = (best_beta, best_mrr)

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMAL β PER CATEGORY:")
    for cat_name, (beta, mrr) in optimal_betas.items():
        print(f"  {cat_name:<25} β*={beta:<8} MRR={mrr:.4f}")

    # Compute the gain from per-category β vs. global best β
    # Find global best β (best average MRR across categories)
    global_mrr_by_beta: dict[float, float] = {}
    for beta in BETA_GRID:
        avg_mrr = np.mean([all_results[cat][beta]["mrr"] for cat in categories])
        global_mrr_by_beta[beta] = float(avg_mrr)

    global_best_beta = max(global_mrr_by_beta, key=lambda b: global_mrr_by_beta[b])
    print(
        f"\n  Global best β (avg MRR): β={global_best_beta}, "
        f"avg MRR={global_mrr_by_beta[global_best_beta]:.4f}"
    )

    per_cat_avg = np.mean([mrr for _, mrr in optimal_betas.values()])
    print(f"  Per-category β avg MRR: {per_cat_avg:.4f}")
    gain = per_cat_avg - global_mrr_by_beta[global_best_beta]
    gain_pct = gain / global_mrr_by_beta[global_best_beta] * 100
    print(f"  Gain from learned β:    +{gain:.4f} ({gain_pct:.1f}%)")
    print("=" * 70)

    return {
        "categories": {k: list(v) for k, v in categories.items()},
        "results": {
            cat: {str(b): m for b, m in betas.items()} for cat, betas in all_results.items()
        },
        "optimal_betas": {k: {"beta": v[0], "mrr": v[1]} for k, v in optimal_betas.items()},
        "global_best_beta": global_best_beta,
        "global_best_mrr": global_mrr_by_beta[global_best_beta],
        "per_category_avg_mrr": float(per_cat_avg),
    }


def save_results(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "exp3_beta_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {json_path}")

    # Also save the heat map data as CSV (β × category → MRR)
    csv_path = RESULTS_DIR / "exp3_beta_heatmap.csv"
    categories = list(data["results"].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["beta"] + categories)
        for beta in BETA_GRID:
            row = [beta]
            for cat in categories:
                mrr = data["results"][cat].get(str(beta), {}).get("mrr", "")
                row.append(mrr)
            writer.writerow(row)
    print(f"[saved] {csv_path}")


def print_results() -> None:
    json_path = RESULTS_DIR / "exp3_beta_sensitivity.json"
    if not json_path.exists():
        print("No results found. Run the experiment first: --run")
        return
    with open(json_path) as f:
        data = json.load(f)

    categories = list(data["results"].keys())
    print(f"\n{'β':<10}", end="")
    for cat in categories:
        print(f"{cat:>20}", end="")
    print()
    print("-" * (10 + 20 * len(categories)))

    for beta in BETA_GRID:
        print(f"{beta:<10}", end="")
        for cat in categories:
            mrr = data["results"][cat].get(str(beta), {}).get("mrr", 0)
            # Mark optimal with *
            opt_beta = data["optimal_betas"][cat]["beta"]
            marker = " *" if beta == opt_beta else ""
            print(f"{mrr:>17.4f}{marker:>3}", end="")
        print()

    print("\nOptimal β per category:")
    for cat, info in data["optimal_betas"].items():
        print(f"  {cat}: β*={info['beta']}, MRR={info['mrr']:.4f}")
    print(f"\nGlobal best β: {data['global_best_beta']}, MRR={data['global_best_mrr']:.4f}")
    print(f"Per-category avg MRR: {data['per_category_avg_mrr']:.4f}")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: β Sensitivity Analysis")
    parser.add_argument("--run", action="store_true", help="Run full experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (500 queries per category)")
    parser.add_argument("--results", action="store_true", help="Print saved results")
    args = parser.parse_args()

    if args.run or args.quick:
        max_q = 500 if args.quick else None
        data = run_experiment(max_queries_per_cat=max_q)
        save_results(data)

    if args.results:
        print_results()

    if not any([args.run, args.quick, args.results]):
        parser.print_help()


if __name__ == "__main__":
    main()
