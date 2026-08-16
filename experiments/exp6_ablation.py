"""Experiment 6: Ablation Study — Layer 1 vs Layer 2 Contribution

Tests whether temporal subspace embedding (Layer 1) adds value beyond
decay-weighted reranking (Layer 2), and vice versa.

Design:
  Using TimE-Lite-News corpus (897 questions) with oracle temporal focus.
  Metric: Gold@5 (does the gold context appear in top-5 retrieved?)

  Variants:
    A) Semantic-only:  cosine_sim(q, ctx)                       — baseline
    B) +Layer 1 only:  cosine_sim([q;t_q], [ctx;t_ctx])         — temporal embedding, no decay
    C) +Layer 2 only:  sim(q,ctx) * exp(-β|t_ctx - t_focus|)    — decay reranking, no temporal emb
    D) Full (L1+L2):   sim([q;t_q],[ctx;t_ctx]) * exp(-β|...)   — both layers

  Layer 1 Implementation:
    Sinusoidal positional encoding of timestamps (like transformer PE),
    concatenated to semantic embedding, re-normalized. This creates a
    temporal subspace in the first t dimensions of the embedding.

  No LLM API calls. TIME-Lite may be downloaded if it is not already cached.

Usage:
  python experiments/exp6_ablation.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproducibility import TIME_LITE_REVISION

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EMBED_CACHE = Path(__file__).resolve().parent / "data" / "time_news_embeddings.npz"

TOP_K = 5
# Best oracle β from Exp 5 sweep (still climbing at 0.05)
BETA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# Temporal subspace dimensionality (number of sinusoidal features)
TEMPORAL_DIMS_VALUES = [4, 8, 16, 32, 64]


# ============================================================
# CORPUS & EMBEDDINGS
# ============================================================


def build_corpus() -> list[dict]:
    """Load TIME-Lite-News corpus, reusing exp5's proven parser."""
    # Import the working parser from exp5 to avoid reimplementation bugs
    from datasets import load_dataset

    from experiments.exp5_rag_endtoend import (
        parse_context_timestamp,
    )

    ds = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    news_rows = [row for row in ds["train"] if row["Dataset Name"] == "TIME-Lite-News"]

    corpus = []
    for i, row in enumerate(news_rows):
        ctx_ts = parse_context_timestamp(row["Context"])
        corpus.append(
            {
                "index": i,
                "context": row["Context"],
                "question": row["Question"],
                "gold_answer": row["Gold Answer"].strip().upper(),
                "task": row["Task"],
                "context_timestamp": ctx_ts,
            }
        )
    return corpus


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load cached semantic embeddings from exp5."""
    if not EMBED_CACHE.exists():
        print(f"ERROR: Embeddings not found at {EMBED_CACHE}")
        print("Run: python experiments/exp5_rag_endtoend.py --embed")
        sys.exit(1)
    data = np.load(EMBED_CACHE)
    return data["context_embeddings"], data["question_embeddings"]


# ============================================================
# LAYER 1: TEMPORAL SUBSPACE ENCODING
# ============================================================


def sinusoidal_timestamp_encoding(
    timestamps: list[datetime | None],
    dim: int,
    ref_date: datetime | None = None,
    scale_days: float = 365.25,
) -> np.ndarray:
    """Encode timestamps as sinusoidal positional features.

    Follows the transformer positional encoding principle:
      PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    where `pos` is the timestamp expressed as fractional days since ref_date,
    normalized by scale_days.

    Args:
        timestamps: List of datetimes (None → zero vector).
        dim: Number of temporal dimensions (must be even).
        ref_date: Reference date for computing positions. Defaults to min timestamp.
        scale_days: Normalization factor. 365.25 means one full sine cycle ≈ 1 year.

    Returns:
        Array of shape (len(timestamps), dim) with L2-normalized rows.
    """
    assert dim % 2 == 0, "dim must be even"

    valid_ts = [t for t in timestamps if t is not None]
    if not valid_ts:
        return np.zeros((len(timestamps), dim), dtype=np.float32)

    if ref_date is None:
        ref_date = min(valid_ts)

    # Compute position = days since ref_date, normalized
    positions = np.zeros(len(timestamps), dtype=np.float64)
    for i, t in enumerate(timestamps):
        if t is not None:
            positions[i] = (t - ref_date).total_seconds() / (86400.0 * scale_days)

    # Sinusoidal encoding
    encodings = np.zeros((len(timestamps), dim), dtype=np.float32)
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float64) * (-np.log(10000.0) / dim))
    encodings[:, 0::2] = np.sin(positions[:, np.newaxis] * div_term[np.newaxis, :])
    encodings[:, 1::2] = np.cos(positions[:, np.newaxis] * div_term[np.newaxis, :])

    # Zero out entries with no timestamp
    for i, t in enumerate(timestamps):
        if t is None:
            encodings[i] = 0.0

    return encodings


def create_temporal_embeddings(
    semantic_embs: np.ndarray,
    timestamps: list[datetime | None],
    temporal_dim: int,
    ref_date: datetime | None = None,
) -> np.ndarray:
    """Concatenate semantic embeddings with temporal subspace features.

    The resulting embedding has structure: [temporal_subspace ; semantic]
    where temporal_subspace occupies the first `temporal_dim` dimensions.

    This ensures that at any Matryoshka truncation level >= temporal_dim,
    the temporal information is structurally preserved.

    Returns:
        Array of shape (N, semantic_dim + temporal_dim), L2-normalized.
    """
    temporal_feats = sinusoidal_timestamp_encoding(timestamps, temporal_dim, ref_date=ref_date)

    # Concatenate: [temporal ; semantic]
    combined = np.concatenate([temporal_feats, semantic_embs], axis=1)

    # L2 normalize each row
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    combined = combined / norms

    return combined.astype(np.float32)


# ============================================================
# RETRIEVAL
# ============================================================


def retrieve_topk(query_emb: np.ndarray, ctx_embs: np.ndarray, k: int = 5) -> list[int]:
    """Retrieve top-k by cosine similarity (embeddings assumed normalized)."""
    scores = ctx_embs @ query_emb
    return np.argsort(-scores)[:k].tolist()


def retrieve_decay(
    query_emb: np.ndarray,
    ctx_embs: np.ndarray,
    ages_days: np.ndarray,
    beta: float,
    k: int = 5,
) -> list[int]:
    """Retrieve top-k by sim * exp(-beta * age)."""
    sims = ctx_embs @ query_emb
    decay = np.exp(-beta * ages_days)
    scores = sims * decay
    return np.argsort(-scores)[:k].tolist()


# ============================================================
# ABLATION RUNNER
# ============================================================


def run_ablation() -> dict:
    """Run the full ablation: 4 variants × multiple beta/dim values."""
    print("=" * 70)
    print("EXPERIMENT 6: Ablation Study — Layer 1 vs Layer 2")
    print("=" * 70)

    # Load corpus and embeddings
    print("\n[1/4] Loading corpus...")
    corpus = build_corpus()
    print(f"  {len(corpus)} questions loaded")

    print("[2/4] Loading semantic embeddings...")
    ctx_embs_sem, q_embs_sem = load_embeddings()
    sem_dim = ctx_embs_sem.shape[1]
    print(f"  Semantic dim: {sem_dim}")

    # Filter to questions with valid timestamps (for oracle proximity)
    ctx_timestamps = [c["context_timestamp"] for c in corpus]
    valid_indices = [i for i, c in enumerate(corpus) if c["context_timestamp"] is not None]
    n_valid = len(valid_indices)
    print(f"  Questions with timestamps: {n_valid}/{len(corpus)}")

    # Reference date for temporal encoding
    valid_ts = [t for t in ctx_timestamps if t is not None]
    ref_date = min(valid_ts) if valid_ts else datetime(2010, 1, 1)
    print(f"  Ref date (earliest context): {ref_date.date()}")

    # ---- Variant A: Semantic-only (baseline) ----
    print("\n[3/4] Running ablation variants...")
    print("  A) Semantic-only (baseline)...")
    gold_hits_a = 0
    for qi in valid_indices:
        idxs = retrieve_topk(q_embs_sem[qi], ctx_embs_sem, TOP_K)
        if qi in idxs:
            gold_hits_a += 1
    rate_a = gold_hits_a / n_valid
    print(f"     Gold@{TOP_K} = {rate_a:.4f} ({gold_hits_a}/{n_valid})")

    results = {
        "semantic_only": {
            "gold_at_k": rate_a,
            "n": n_valid,
            "description": "Cosine similarity only, no temporal information",
        }
    }

    # ---- Variant B: Layer 1 only (temporal embedding, no decay) ----
    print("\n  B) Layer 1 only (temporal embedding, no decay)...")
    l1_results = {}
    for t_dim in TEMPORAL_DIMS_VALUES:
        ctx_embs_temp = create_temporal_embeddings(
            ctx_embs_sem,
            ctx_timestamps,
            t_dim,
            ref_date=ref_date,
        )
        # For queries, we use the gold timestamp as the query's temporal focus
        # (oracle mode — same assumption as Exp 5 oracle)
        q_timestamps_oracle = [
            corpus[i]["context_timestamp"] if i in valid_indices else None
            for i in range(len(corpus))
        ]
        q_embs_temp = create_temporal_embeddings(
            q_embs_sem,
            q_timestamps_oracle,
            t_dim,
            ref_date=ref_date,
        )

        gold_hits = 0
        for qi in valid_indices:
            idxs = retrieve_topk(q_embs_temp[qi], ctx_embs_temp, TOP_K)
            if qi in idxs:
                gold_hits += 1
        rate = gold_hits / n_valid
        l1_results[t_dim] = rate
        print(f"     t_dim={t_dim:3d}: Gold@{TOP_K} = {rate:.4f} ({gold_hits}/{n_valid})")

    best_tdim = max(l1_results, key=l1_results.get)
    results["layer1_only"] = {
        "per_dim": {str(k): v for k, v in l1_results.items()},
        "best_dim": best_tdim,
        "best_gold_at_k": l1_results[best_tdim],
        "n": n_valid,
        "description": "Temporal subspace embedding (sinusoidal PE), no decay reranking",
    }

    # ---- Variant C: Layer 2 only (decay reranking, no temporal embedding) ----
    print("\n  C) Layer 2 only (decay reranking, no temporal embedding)...")
    l2_results = {}
    for beta in BETA_VALUES:
        gold_hits = 0
        for qi in valid_indices:
            t_focus = corpus[qi]["context_timestamp"]
            prox_ages = np.array(
                [abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
            )
            idxs = retrieve_decay(q_embs_sem[qi], ctx_embs_sem, prox_ages, beta, TOP_K)
            if qi in idxs:
                gold_hits += 1
        rate = gold_hits / n_valid
        l2_results[beta] = rate
        print(f"     β={beta:<6}: Gold@{TOP_K} = {rate:.4f} ({gold_hits}/{n_valid})")

    best_beta = max(l2_results, key=l2_results.get)
    results["layer2_only"] = {
        "per_beta": {str(k): v for k, v in l2_results.items()},
        "best_beta": best_beta,
        "best_gold_at_k": l2_results[best_beta],
        "n": n_valid,
        "description": "Semantic cosine sim * exp(-β·|Δt|), no temporal embedding",
    }

    # ---- Variant D: Full (Layer 1 + Layer 2) ----
    print("\n  D) Full pipeline (Layer 1 + Layer 2)...")
    full_results = {}
    for t_dim in TEMPORAL_DIMS_VALUES:
        ctx_embs_temp = create_temporal_embeddings(
            ctx_embs_sem,
            ctx_timestamps,
            t_dim,
            ref_date=ref_date,
        )
        q_timestamps_oracle = [
            corpus[i]["context_timestamp"] if i in valid_indices else None
            for i in range(len(corpus))
        ]
        q_embs_temp = create_temporal_embeddings(
            q_embs_sem,
            q_timestamps_oracle,
            t_dim,
            ref_date=ref_date,
        )

        for beta in BETA_VALUES:
            gold_hits = 0
            for qi in valid_indices:
                t_focus = corpus[qi]["context_timestamp"]
                prox_ages = np.array(
                    [
                        abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6
                        for t in ctx_timestamps
                    ]
                )
                idxs = retrieve_decay(q_embs_temp[qi], ctx_embs_temp, prox_ages, beta, TOP_K)
                if qi in idxs:
                    gold_hits += 1
            rate = gold_hits / n_valid
            key = f"dim={t_dim}_beta={beta}"
            full_results[key] = rate

        # Print best beta for this dim
        dim_results = {k: v for k, v in full_results.items() if k.startswith(f"dim={t_dim}_")}
        best_for_dim = max(dim_results, key=dim_results.get)
        print(
            f"     t_dim={t_dim:3d}, best: {best_for_dim} → "
            f"Gold@{TOP_K} = {full_results[best_for_dim]:.4f}"
        )

    best_full = max(full_results, key=full_results.get)
    results["full_l1_l2"] = {
        "per_config": full_results,
        "best_config": best_full,
        "best_gold_at_k": full_results[best_full],
        "n": n_valid,
        "description": "Temporal embedding + decay reranking (both layers)",
    }

    # ============================================================
    # LAYER 3 ANALYSIS: STL Verification as Quality Gate
    # ============================================================
    print("\n[4/4] Layer 3: STL verification (precision/coverage tradeoff)...")

    # Use best configs from each variant
    best_l2_beta = results["layer2_only"]["best_beta"]
    best_full_config = results["full_l1_l2"]["best_config"]
    # Parse dim and beta from best_full_config like "dim=4_beta=0.2"
    parts = best_full_config.split("_")
    best_full_dim = int(parts[0].split("=")[1])
    best_full_beta = float(parts[1].split("=")[1])

    # STL thresholds to sweep (γ in the paper)
    stl_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # β for decay validity computation (same as retrieval β)
    validity_beta = best_l2_beta

    def compute_stl_for_variant(
        variant_name: str,
        retrieve_fn,
        validity_beta: float,
    ) -> dict:
        """For a variant, compute STL robustness on each question's top-k.

        Returns per-threshold precision/coverage stats.
        """
        per_question = []  # list of (gold_in_topk, robustness)

        for qi in valid_indices:
            t_focus = corpus[qi]["context_timestamp"]
            retrieved_idxs = retrieve_fn(qi)
            gold_in_topk = qi in retrieved_idxs

            # Compute temporal validity of each retrieved context
            validities = []
            for ri in retrieved_idxs:
                t_ctx = ctx_timestamps[ri]
                if t_ctx is not None and t_focus is not None:
                    age_days = abs((t_ctx - t_focus).total_seconds()) / 86400.0
                    v = np.exp(-validity_beta * age_days)
                else:
                    v = 0.0  # no timestamp → worst validity
                validities.append(v)

            # STL robustness = min validity across retrieved set (weakest link)
            min_validity = min(validities) if validities else 0.0
            per_question.append(
                {
                    "qi": qi,
                    "gold_in_topk": gold_in_topk,
                    "min_validity": min_validity,
                }
            )

        # Compute precision/coverage at each threshold
        threshold_results = {}
        for gamma in stl_thresholds:
            answered = [q for q in per_question if q["min_validity"] >= gamma]
            abstained = [q for q in per_question if q["min_validity"] < gamma]
            n_answered = len(answered)
            n_abstained = len(abstained)
            coverage = n_answered / len(per_question) if per_question else 0

            if n_answered > 0:
                precision = sum(q["gold_in_topk"] for q in answered) / n_answered
            else:
                precision = 0.0

            if n_abstained > 0:
                abstain_gold = sum(q["gold_in_topk"] for q in abstained) / n_abstained
            else:
                abstain_gold = 0.0

            threshold_results[str(gamma)] = {
                "gamma": gamma,
                "n_answered": n_answered,
                "n_abstained": n_abstained,
                "coverage": round(coverage, 4),
                "precision_gold_at_k": round(precision, 4),
                "abstain_gold_at_k": round(abstain_gold, 4),
            }

        # Overall (no threshold = γ=0)
        overall_gold = sum(q["gold_in_topk"] for q in per_question) / len(per_question)
        mean_validity = np.mean([q["min_validity"] for q in per_question])

        return {
            "overall_gold_at_k": round(overall_gold, 4),
            "mean_min_validity": round(float(mean_validity), 4),
            "per_threshold": threshold_results,
        }

    # --- Run Layer 3 on each variant's best config ---

    # A) Semantic-only
    print("  Layer 3 on A) Semantic-only...")
    stl_a = compute_stl_for_variant(
        "semantic_only",
        lambda qi: retrieve_topk(q_embs_sem[qi], ctx_embs_sem, TOP_K),
        validity_beta,
    )

    # C) Layer 2 only (best β)
    print(f"  Layer 3 on C) Layer 2 only (β={best_l2_beta})...")

    def _retrieve_l2(qi):
        t_focus = corpus[qi]["context_timestamp"]
        prox_ages = np.array(
            [abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
        )
        return retrieve_decay(q_embs_sem[qi], ctx_embs_sem, prox_ages, best_l2_beta, TOP_K)

    stl_c = compute_stl_for_variant("layer2_only", _retrieve_l2, validity_beta)

    # D) Full (L1+L2, best config)
    print(f"  Layer 3 on D) Full (dim={best_full_dim}, β={best_full_beta})...")
    ctx_embs_full = create_temporal_embeddings(
        ctx_embs_sem,
        ctx_timestamps,
        best_full_dim,
        ref_date=ref_date,
    )
    q_ts_oracle = [
        corpus[i]["context_timestamp"] if i in valid_indices else None for i in range(len(corpus))
    ]
    q_embs_full = create_temporal_embeddings(
        q_embs_sem,
        q_ts_oracle,
        best_full_dim,
        ref_date=ref_date,
    )

    def _retrieve_full(qi):
        t_focus = corpus[qi]["context_timestamp"]
        prox_ages = np.array(
            [abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
        )
        return retrieve_decay(q_embs_full[qi], ctx_embs_full, prox_ages, best_full_beta, TOP_K)

    stl_d = compute_stl_for_variant("full_l1_l2", _retrieve_full, validity_beta)

    results["layer3_stl"] = {
        "validity_beta": validity_beta,
        "thresholds": stl_thresholds,
        "semantic_only": stl_a,
        "layer2_only": stl_c,
        "full_l1_l2": stl_d,
    }

    # Print Layer 3 summary
    print(f"\n{'='*70}")
    print("LAYER 3 (STL) ANALYSIS: Precision/Coverage Tradeoff")
    print(f"{'='*70}")
    print(f"Validity β = {validity_beta}")
    print(
        f"\n{'Variant':<20s}  {'γ':>5s}  {'Coverage':>8s}  {'Gold@5':>7s}  "
        f"{'Abstain Gold@5':>14s}  {'N_ans':>6s}  {'N_abs':>6s}"
    )
    print("-" * 80)

    for variant_name, stl_data in [
        ("Semantic-only", stl_a),
        ("Layer 2 only", stl_c),
        ("Full (L1+L2)", stl_d),
    ]:
        # Print overall (no gate)
        print(
            f"{variant_name:<20s}  {'all':>5s}  {'1.000':>8s}  "
            f"{stl_data['overall_gold_at_k']:7.4f}  {'—':>14s}  "
            f"{n_valid:>6d}  {'0':>6s}"
        )
        # Print selected thresholds
        for gamma_str in ["0.3", "0.5", "0.7", "0.9"]:
            t = stl_data["per_threshold"].get(gamma_str)
            if t:
                print(
                    f"{'':<20s}  {t['gamma']:5.1f}  {t['coverage']:8.4f}  "
                    f"{t['precision_gold_at_k']:7.4f}  "
                    f"{t['abstain_gold_at_k']:14.4f}  "
                    f"{t['n_answered']:>6d}  {t['n_abstained']:>6d}"
                )
        print()

    # ---- Summary (Layers 1+2) ----
    print(f"{'='*70}")
    print("ABLATION SUMMARY (Layers 1 & 2)")
    print(f"{'='*70}")
    print(f"{'Variant':<30s}  {'Gold@5':>7s}  {'Δ vs baseline':>14s}  {'Config'}")
    print("-" * 80)

    baseline = results["semantic_only"]["gold_at_k"]
    rows = [
        ("A) Semantic-only", baseline, "—"),
        (
            "B) +Layer 1 only",
            results["layer1_only"]["best_gold_at_k"],
            f"dim={results['layer1_only']['best_dim']}",
        ),
        (
            "C) +Layer 2 only",
            results["layer2_only"]["best_gold_at_k"],
            f"β={results['layer2_only']['best_beta']}",
        ),
        (
            "D) Full (L1+L2)",
            results["full_l1_l2"]["best_gold_at_k"],
            results["full_l1_l2"]["best_config"],
        ),
    ]
    for name, rate, config in rows:
        delta = rate - baseline
        delta_pct = (delta / baseline * 100) if baseline > 0 else 0
        delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)" if name != "A) Semantic-only" else "—"
        print(f"{name:<30s}  {rate:7.4f}  {delta_str:>14s}  {config}")

    # Print best Layer 3 summary line
    # Find threshold that maximizes precision while keeping coverage > 50%
    for vname, stl_data in [("Full+L3", stl_d)]:
        viable = [(g, t) for g, t in stl_data["per_threshold"].items() if t["coverage"] >= 0.5]
        if viable:
            best_gate = max(viable, key=lambda x: x[1]["precision_gold_at_k"])
            g, t = best_gate
            print(
                f"{'E) Full+Layer 3':<30s}  {t['precision_gold_at_k']:7.4f}  "
                f"{'':>14s}  γ={t['gamma']}, cov={t['coverage']:.1%}"
            )

    # Save
    output = {
        "experiment": "exp6_ablation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": "TIME-Lite-News",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": TOP_K,
            "n_questions": n_valid,
            "n_corpus": len(corpus),
            "temporal_dims_tested": TEMPORAL_DIMS_VALUES,
            "beta_values_tested": BETA_VALUES,
            "ref_date": ref_date.isoformat(),
            "temporal_encoding": "sinusoidal_positional",
        },
        "results": results,
        "summary": {
            "semantic_only": baseline,
            "best_layer1_only": results["layer1_only"]["best_gold_at_k"],
            "best_layer2_only": results["layer2_only"]["best_gold_at_k"],
            "best_full": results["full_l1_l2"]["best_gold_at_k"],
        },
    }

    from experiments.utils import timestamped_save

    timestamped_save(output, "exp6_ablation", save_csv=False)

    return output


if __name__ == "__main__":
    run_ablation()
