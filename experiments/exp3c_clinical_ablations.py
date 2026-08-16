"""Experiment 3c: Clinical Ablation Study

Thorough ablations on MIMIC-IV Demo data:
  1. Fine-grained β grid (focus on 0-0.5 range)
  2. All decay functions compared: exponential, half-life, linear, power-law, Weibull
  3. Per-item breakdown within categories

Usage:
  python experiments/exp3c_clinical_ablations.py --run
  python experiments/exp3c_clinical_ablations.py --quick
  python experiments/exp3c_clinical_ablations.py --results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp3b_clinical_beta import (
    CATEGORIES,
    ClinicalEvent,
    TemporalQuery,
    build_queries,
    evaluate_beta,
    load_clinical_events,
)
from experiments.reproducibility import RANDOM_SEED
from experiments.utils import timestamped_save

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fine-grained β grids
BETA_FINE = [
    0.0,
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
    2.0,
    5.0,
    10.0,
]

# Half-life values (in hours, since clinical data is hourly)
HALF_LIFE_GRID = [1, 2, 4, 6, 12, 24, 48, 72, 168, 336, 720, 2160, 8760]

# Linear rate grid
LINEAR_RATE_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

# Power-law exponent grid
POWER_EXPONENT_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Weibull (scale, shape) configs
WEIBULL_GRID = [
    # Scale in hours, shape parameter
    (6, 0.5),
    (6, 1.0),
    (6, 2.0),
    (24, 0.5),
    (24, 1.0),
    (24, 2.0),
    (168, 0.5),
    (168, 1.0),
    (168, 2.0),
    (720, 0.5),
    (720, 1.0),
    (720, 2.0),
]


# ============================================================
# DECAY FUNCTION EVALUATORS
# ============================================================


def evaluate_half_life(
    queries: list[TemporalQuery],
    half_life: float,
) -> dict[str, float]:
    """Evaluate using half-life decay: 0.5^(Δt/h)."""
    if not queries:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}

    hits1, hits3, mrr_sum, evaluated = 0, 0, 0.0, 0

    for q in queries:
        if not q.history:
            continue
        scored = []
        for ts, val in q.history:
            dt = abs(q.query_time - ts)
            score = 0.5 ** (dt / half_life) if half_life > 0 else 1.0
            scored.append((val, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        rank = next(
            (i + 1 for i, (v, _) in enumerate(scored) if v == q.ground_truth), len(scored) + 1
        )
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        mrr_sum += 1.0 / rank
        evaluated += 1

    if evaluated == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}
    return {
        "mrr": round(mrr_sum / evaluated, 4),
        "hits_at_1": round(hits1 / evaluated, 4),
        "hits_at_3": round(hits3 / evaluated, 4),
        "num_queries": evaluated,
    }


def evaluate_linear(
    queries: list[TemporalQuery],
    rate: float,
) -> dict[str, float]:
    """Evaluate using linear decay: max(0, 1 - α·Δt)."""
    if not queries:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}

    hits1, hits3, mrr_sum, evaluated = 0, 0, 0.0, 0

    for q in queries:
        if not q.history:
            continue
        scored = []
        for ts, val in q.history:
            dt = abs(q.query_time - ts)
            score = max(0.0, 1.0 - rate * dt)
            scored.append((val, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        rank = next(
            (i + 1 for i, (v, _) in enumerate(scored) if v == q.ground_truth), len(scored) + 1
        )
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        mrr_sum += 1.0 / rank
        evaluated += 1

    if evaluated == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}
    return {
        "mrr": round(mrr_sum / evaluated, 4),
        "hits_at_1": round(hits1 / evaluated, 4),
        "hits_at_3": round(hits3 / evaluated, 4),
        "num_queries": evaluated,
    }


def evaluate_power_law(
    queries: list[TemporalQuery],
    exponent: float,
) -> dict[str, float]:
    """Evaluate using power-law decay: (1 + Δt)^(-α)."""
    if not queries:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}

    hits1, hits3, mrr_sum, evaluated = 0, 0, 0.0, 0

    for q in queries:
        if not q.history:
            continue
        scored = []
        for ts, val in q.history:
            dt = abs(q.query_time - ts)
            score = (1.0 + dt) ** (-exponent)
            scored.append((val, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        rank = next(
            (i + 1 for i, (v, _) in enumerate(scored) if v == q.ground_truth), len(scored) + 1
        )
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        mrr_sum += 1.0 / rank
        evaluated += 1

    if evaluated == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}
    return {
        "mrr": round(mrr_sum / evaluated, 4),
        "hits_at_1": round(hits1 / evaluated, 4),
        "hits_at_3": round(hits3 / evaluated, 4),
        "num_queries": evaluated,
    }


def evaluate_weibull(
    queries: list[TemporalQuery],
    scale: float,
    shape: float,
) -> dict[str, float]:
    """Evaluate using Weibull decay: exp(-(Δt/λ)^k)."""
    if not queries:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}

    hits1, hits3, mrr_sum, evaluated = 0, 0, 0.0, 0

    for q in queries:
        if not q.history:
            continue
        scored = []
        for ts, val in q.history:
            dt = abs(q.query_time - ts)
            score = np.exp(-((dt / scale) ** shape))
            scored.append((val, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        rank = next(
            (i + 1 for i, (v, _) in enumerate(scored) if v == q.ground_truth), len(scored) + 1
        )
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        mrr_sum += 1.0 / rank
        evaluated += 1

    if evaluated == 0:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}
    return {
        "mrr": round(mrr_sum / evaluated, 4),
        "hits_at_1": round(hits1 / evaluated, 4),
        "hits_at_3": round(hits3 / evaluated, 4),
        "num_queries": evaluated,
    }


# ============================================================
# PER-ITEM ANALYSIS
# ============================================================


def build_per_item_queries(
    events: list[ClinicalEvent],
    category: str,
    max_per_item: int | None = None,
) -> dict[str, list[TemporalQuery]]:
    """Build queries grouped by item_name within a category."""
    # Group events by (subject_id, item_name)
    groups: dict[tuple[int, str], list[tuple[float, str]]] = defaultdict(list)
    for e in events:
        if e.category == category:
            groups[(e.subject_id, e.item_name)].append((e.timestamp_hours, e.value))

    # Build queries per item
    item_queries: dict[str, list[TemporalQuery]] = defaultdict(list)
    for (sid, item), measurements in groups.items():
        if len(measurements) < 2:
            continue
        measurements.sort(key=lambda x: x[0])
        history = measurements[:-1]
        target_time, target_value = measurements[-1]
        item_queries[item].append(
            TemporalQuery(
                subject_id=sid,
                item_name=item,
                category=category,
                query_time=target_time,
                ground_truth=target_value,
                history=history,
            )
        )

    # Limit per item
    if max_per_item:
        rng = np.random.RandomState(RANDOM_SEED)
        for item in item_queries:
            if len(item_queries[item]) > max_per_item:
                indices = rng.choice(len(item_queries[item]), max_per_item, replace=False)
                item_queries[item] = [item_queries[item][i] for i in indices]

    return dict(item_queries)


# ============================================================
# MAIN ABLATION
# ============================================================


def _sweep(label: str, queries: list[TemporalQuery], eval_fn, param_grid, param_name: str) -> dict:
    """Generic sweep: evaluate a decay function across a parameter grid."""
    results = {}
    best_mrr, best_param = 0.0, None
    for param in param_grid:
        if isinstance(param, tuple):
            metrics = eval_fn(queries, *param)
            key = str(param)
        else:
            metrics = eval_fn(queries, param)
            key = str(param)
        results[key] = metrics
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_param = param
    return {"results": results, "best_param": best_param, "best_mrr": best_mrr}


def run_ablations(max_queries_per_cat: int | None = None) -> dict:
    """Run comprehensive ablation study."""
    print("=" * 70)
    print("EXPERIMENT 3c: Clinical Ablation Study")
    print("=" * 70)

    print("\n[1/5] Loading clinical events...")
    events = load_clinical_events()

    # Build queries per category
    print("\n[2/5] Building queries...")
    all_queries: dict[str, list[TemporalQuery]] = {}
    for cat in CATEGORIES:
        queries = build_queries(events, cat, max_queries_per_cat)
        all_queries[cat] = queries
        print(f"  {cat}: {len(queries)} queries")

    all_results: dict = {}

    # ---- ABLATION 1: Fine-grained β sweep ----
    print("\n[3/5] Ablation 1: Fine-grained exponential β sweep...")
    abl1 = {}
    for cat in CATEGORIES:
        if not all_queries[cat]:
            continue
        print(f"\n  {cat}:")
        sweep = _sweep(cat, all_queries[cat], evaluate_beta, BETA_FINE, "beta")
        abl1[cat] = sweep
        print(f"    Best β={sweep['best_param']}, MRR={sweep['best_mrr']:.4f}")
        # Print top 5
        sorted_results = sorted(sweep["results"].items(), key=lambda x: x[1]["mrr"], reverse=True)
        for k, v in sorted_results[:5]:
            print(f"      β={k:<10} MRR={v['mrr']:.4f}  H@1={v['hits_at_1']:.4f}")
    all_results["fine_beta"] = abl1

    # ---- ABLATION 2: Decay function comparison ----
    print("\n[4/5] Ablation 2: Decay function comparison...")
    abl2 = {}
    for cat in CATEGORIES:
        if not all_queries[cat]:
            continue
        print(f"\n  {cat}:")
        cat_results = {}

        # Exponential (use fine grid best)
        exp_sweep = abl1.get(cat, {})
        cat_results["exponential"] = {
            "best_param": exp_sweep.get("best_param"),
            "best_mrr": exp_sweep.get("best_mrr", 0),
        }
        print(
            f"    Exponential: β*={exp_sweep.get('best_param')}, "
            f"MRR={exp_sweep.get('best_mrr', 0):.4f}"
        )

        # Half-life
        hl_sweep = _sweep(cat, all_queries[cat], evaluate_half_life, HALF_LIFE_GRID, "half_life")
        cat_results["half_life"] = {
            "best_param": hl_sweep["best_param"],
            "best_mrr": hl_sweep["best_mrr"],
            "all": hl_sweep["results"],
        }
        print(f"    Half-life:   h*={hl_sweep['best_param']}, MRR={hl_sweep['best_mrr']:.4f}")

        # Linear
        lin_sweep = _sweep(cat, all_queries[cat], evaluate_linear, LINEAR_RATE_GRID, "rate")
        cat_results["linear"] = {
            "best_param": lin_sweep["best_param"],
            "best_mrr": lin_sweep["best_mrr"],
            "all": lin_sweep["results"],
        }
        print(f"    Linear:      α*={lin_sweep['best_param']}, MRR={lin_sweep['best_mrr']:.4f}")

        # Power-law
        pl_sweep = _sweep(
            cat, all_queries[cat], evaluate_power_law, POWER_EXPONENT_GRID, "exponent"
        )
        cat_results["power_law"] = {
            "best_param": pl_sweep["best_param"],
            "best_mrr": pl_sweep["best_mrr"],
            "all": pl_sweep["results"],
        }
        print(f"    Power-law:   α*={pl_sweep['best_param']}, MRR={pl_sweep['best_mrr']:.4f}")

        # Weibull
        wb_sweep = _sweep(cat, all_queries[cat], evaluate_weibull, WEIBULL_GRID, "weibull")
        cat_results["weibull"] = {
            "best_param": str(wb_sweep["best_param"]),
            "best_mrr": wb_sweep["best_mrr"],
            "all": wb_sweep["results"],
        }
        print(f"    Weibull:     (λ,k)*={wb_sweep['best_param']}, MRR={wb_sweep['best_mrr']:.4f}")

        abl2[cat] = cat_results
    all_results["decay_comparison"] = abl2

    # ---- ABLATION 3: Per-item breakdown ----
    print("\n[5/5] Ablation 3: Per-item breakdown...")
    abl3 = {}
    for cat in ["vital_sign", "lab_result"]:  # Only for high-frequency categories
        print(f"\n  {cat}:")
        item_queries = build_per_item_queries(events, cat, max_per_item=max_queries_per_cat)
        cat_items = {}
        for item_name, queries in sorted(item_queries.items()):
            if len(queries) < 5:
                continue
            sweep = _sweep(item_name, queries, evaluate_beta, BETA_FINE, "beta")
            cat_items[item_name] = {
                "num_queries": len(queries),
                "best_beta": sweep["best_param"],
                "best_mrr": sweep["best_mrr"],
            }
            print(
                f"    {item_name:<30} n={len(queries):<5} "
                f"β*={sweep['best_param']:<10} MRR={sweep['best_mrr']:.4f}"
            )
        abl3[cat] = cat_items
    all_results["per_item"] = abl3

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nFine-grained optimal β per category:")
    for cat in CATEGORIES:
        if cat in abl1:
            print(f"  {cat:<20} β*={abl1[cat]['best_param']:<10} MRR={abl1[cat]['best_mrr']:.4f}")

    print("\nBest decay function per category:")
    for cat in CATEGORIES:
        if cat in abl2:
            best_fn = max(abl2[cat].items(), key=lambda x: x[1].get("best_mrr", 0))
            print(f"  {cat:<20} {best_fn[0]:<15} MRR={best_fn[1]['best_mrr']:.4f}")

    if "vital_sign" in abl3:
        print("\nPer-item β variation (vital signs):")
        betas = [v["best_beta"] for v in abl3["vital_sign"].values() if v["best_beta"] is not None]
        if betas:
            print(f"  β range: [{min(betas)}, {max(betas)}]")
            print(f"  β values: {sorted(set(betas))}")

    print("=" * 70)

    return all_results


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Exp 3c: Clinical Ablations")
    parser.add_argument("--run", action="store_true", help="Run full ablation study")
    parser.add_argument("--quick", action="store_true", help="Quick run (200 queries/category)")
    parser.add_argument("--results", action="store_true", help="Print saved results")
    args = parser.parse_args()

    if args.run or args.quick:
        max_q = 200 if args.quick else None
        data = run_ablations(max_queries_per_cat=max_q)
        timestamped_save(data, "exp3c_clinical_ablations")

    if args.results:
        json_path = RESULTS_DIR / "exp3c_clinical_ablations.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        else:
            print("No results. Run --run first.")

    if not any([args.run, args.quick, args.results]):
        parser.print_help()


if __name__ == "__main__":
    main()
