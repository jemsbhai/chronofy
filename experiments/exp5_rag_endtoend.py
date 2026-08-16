"""Experiment 5: End-to-End RAG with Temporal Decay (v2 — Temporal Proximity)

THE PROOF EXPERIMENT: Does Chronofy improve actual RAG answer accuracy?

KEY INSIGHT (from v1 smoke test):
  The original design applied RECENCY bias (prefer newest contexts). But TimE
  questions ask about SPECIFIC past events. Recency HURTS because it penalizes
  the gold context for being "old." The correct model is TEMPORAL PROXIMITY:
  prefer contexts whose timestamp is close to the question's temporal focus.

Design:
  Build retrieval corpus from all 897 TIME-Lite-News contexts, each with a
  parsed timestamp. For each question, determine its "temporal focus" (the
  time period the question asks about), then retrieve via:

  Methods:
    A) Vanilla RAG:       rank by cosine_similarity(query, context)
    B) Recency RAG:       rank by sim * exp(-beta * (T_max - t_context))
    C) Chronofy Oracle:   rank by sim * exp(-beta * |t_context - t_gold|)
    D) Chronofy Parsed:   rank by sim * exp(-beta * |t_context - t_parsed|)

  Oracle uses the gold context's timestamp (upper bound — proves the concept).
  Parsed extracts dates from the question text (realistic deployment scenario).

  Two-phase execution for cost efficiency:
    --sweep  : run ALL beta values on retrieval quality (gold_in_topk) — FREE
    --run    : run Gemini LLM on best configs only — costs API credits

Usage:
  python experiments/exp5_rag_endtoend.py --embed           # Build embeddings
  python experiments/exp5_rag_endtoend.py --inspect 10      # Check parsing
  python experiments/exp5_rag_endtoend.py --sweep            # Free beta sweep
  python experiments/exp5_rag_endtoend.py --run --limit 5   # Smoke test
  python experiments/exp5_rag_endtoend.py --run              # Full run
  python experiments/exp5_rag_endtoend.py --results          # Show saved results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.verification.stl import STLVerifier
from experiments.reproducibility import (
    GEMINI_MODEL_ID,
    MINILM_REVISION,
    RANDOM_SEED,
    TIME_LITE_REVISION,
)

# These must be set before the lazy sentence-transformers import in
# compute_embeddings().
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EMBED_CACHE = Path(__file__).resolve().parent / "data" / "time_news_embeddings.npz"

# Retrieval config
TOP_K = 5

# Beta sweep values (per-day decay rates)
BETA_VALUES = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]

# STL config
STL_THRESHOLD = 0.25

# Gemini config
GEMINI_MODEL = GEMINI_MODEL_ID
GEMINI_MAX_TOKENS = 8192
GEMINI_RATE_LIMIT_DELAY = 4.5

SYSTEM_PROMPT = (
    "You are answering a multiple-choice temporal reasoning question.\n\n"
    "INSTRUCTIONS:\n"
    "1. Read ALL provided context passages carefully, paying attention to dates "
    "and temporal relationships.\n"
    "2. Select the best answer from the options given.\n"
    "3. Respond in EXACTLY this JSON format (no other text):\n\n"
    '{"answer": "A", "confidence": 0.85, "reasoning": "Short reason."}\n\n'
    '- "answer": single letter A, B, C, D, or E\n'
    '- "confidence": number 0.0 to 1.0\n'
    '- "reasoning": MAX 20 words. Be extremely brief.\n\n'
    "Output ONLY the JSON object. No markdown. No backticks. No extra text."
)

# ============================================================
# TIMESTAMP PARSING
# ============================================================

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# Context header: "Day: Monday September 14, 2015"
DATE_PATTERN_HEADER = re.compile(r"Day:\s*[A-Z][a-z]+\s+([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})")

# Full date in text: "September 14, 2015" or "14 September, 2015"
DATE_FULL_MDY = re.compile(
    r"(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
)
DATE_FULL_DMY = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September"
    r"|October|November|December),?\s+(\d{4})"
)

# Month + Year: "in March 2015", "during July 2015"
DATE_MONTH_YEAR = re.compile(
    r"(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{4})"
)

# Standalone year with keyword: "in 2015", "during 2014"
DATE_YEAR_KEYWORD = re.compile(r"(?:in|during|from|since|around|by|before|after)\s+(\d{4})")


def parse_context_timestamp(context: str) -> datetime | None:
    """Extract primary date from a TIME-Lite-News context header."""
    match = DATE_PATTERN_HEADER.search(context)
    if match:
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                return datetime(int(year_str), month, int(day_str))
            except ValueError:
                pass
    # Fallback: first full date in text
    match = DATE_FULL_MDY.search(context)
    if match:
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                return datetime(int(year_str), month, int(day_str))
            except ValueError:
                pass
    return None


def parse_question_temporal_focus(question: str) -> datetime | None:
    """Extract the temporal focus from a question's text.

    Returns the best estimate of what time period the question asks about.
    Strategy: collect all date references, return the most specific one.
    """
    candidates: list[tuple[int, datetime]] = []  # (precision, datetime)

    # Full dates (highest precision = 3)
    for match in DATE_FULL_MDY.finditer(question):
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                candidates.append((3, datetime(int(year_str), month, int(day_str))))
            except ValueError:
                pass

    for match in DATE_FULL_DMY.finditer(question):
        day_str, month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                candidates.append((3, datetime(int(year_str), month, int(day_str))))
            except ValueError:
                pass

    # Month + Year (precision = 2)
    for match in DATE_MONTH_YEAR.finditer(question):
        month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        year = int(year_str)
        if month and 1900 < year < 2030:
            candidates.append((2, datetime(year, month, 15)))

    # Year only (precision = 1)
    for match in DATE_YEAR_KEYWORD.finditer(question):
        year = int(match.group(1))
        if 1900 < year < 2030:
            candidates.append((1, datetime(year, 7, 1)))

    if not candidates:
        return None

    # Return highest-precision candidate; break ties by earliest occurrence
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


# ============================================================
# CORPUS BUILDING AND EMBEDDING
# ============================================================


def build_corpus() -> list[dict]:
    """Load TIME-Lite-News and build the retrieval corpus."""
    from datasets import load_dataset

    ds = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    news_rows = [row for row in ds["train"] if row["Dataset Name"] == "TIME-Lite-News"]

    corpus = []
    for i, row in enumerate(news_rows):
        ctx_ts = parse_context_timestamp(row["Context"])
        q_ts = parse_question_temporal_focus(row["Question"])
        corpus.append(
            {
                "index": i,
                "context": row["Context"],
                "question": row["Question"],
                "gold_answer": row["Gold Answer"].strip().upper(),
                "task": row["Task"],
                "setting": row["Setting"],
                "level": row["Level"],
                "context_timestamp": ctx_ts,
                "question_focus": q_ts,
            }
        )

    return corpus


def compute_embeddings(corpus: list[dict]) -> None:
    """Compute and cache embeddings for all contexts and questions."""
    import torch
    from sentence_transformers import SentenceTransformer

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[embed] GPU: {gpu_name} ({vram:.1f} GB) — CUDA")
    else:
        device = "cpu"
        print("[embed] WARNING: No CUDA — running on CPU")
        time_module.sleep(3)

    print(f"[embed] Loading all-MiniLM-L6-v2 on {device}...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
        revision=MINILM_REVISION,
    )

    contexts = [c["context"][:2000] for c in corpus]
    questions = [c["question"] for c in corpus]

    print(f"[embed] Encoding {len(contexts)} contexts...")
    ctx_embs = model.encode(
        contexts, show_progress_bar=True, batch_size=128, normalize_embeddings=True
    )
    print(f"[embed] Encoding {len(questions)} questions...")
    q_embs = model.encode(
        questions, show_progress_bar=True, batch_size=128, normalize_embeddings=True
    )

    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_CACHE, context_embeddings=ctx_embs, question_embeddings=q_embs)
    print(f"[embed] Saved to {EMBED_CACHE}")


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(EMBED_CACHE)
    return data["context_embeddings"], data["question_embeddings"]


# ============================================================
# RETRIEVAL STRATEGIES
# ============================================================


def retrieve_vanilla(query_emb: np.ndarray, ctx_embs: np.ndarray, k: int = 5) -> list[int]:
    """Pure cosine similarity, no temporal signal."""
    scores = ctx_embs @ query_emb
    return np.argsort(-scores)[:k].tolist()


def retrieve_temporal(
    query_emb: np.ndarray,
    ctx_embs: np.ndarray,
    ages_days: np.ndarray,
    beta: float,
    k: int = 5,
) -> list[int]:
    """Score = cosine_sim * exp(-beta * age_days).

    age_days should be pre-computed as:
      - recency mode: (T_max - t_context) in days
      - proximity mode: |t_context - t_focus| in days
    """
    sims = ctx_embs @ query_emb
    decay = np.exp(-beta * ages_days)
    scores = sims * decay
    return np.argsort(-scores)[:k].tolist()


# ============================================================
# LLM INFERENCE
# ============================================================


def _parse_truncated_json(raw: str) -> dict | None:
    answer_match = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', raw)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
    if answer_match:
        return {
            "answer": answer_match.group(1).upper(),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reasoning": "(truncated)",
        }
    return None


def call_gemini(question: str, retrieved_contexts: list[str], api_key: str) -> dict:
    """Call Gemini 2.5 Flash with retrieved contexts."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=GEMINI_MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )

    context_block = "\n\n---\n\n".join(
        f"[Context {i+1}]:\n{ctx[:3000]}" for i, ctx in enumerate(retrieved_contexts)
    )
    user_prompt = (
        f"Retrieved Contexts:\n{context_block}\n\n"
        f"Question:\n{question}\n\nRespond with JSON only."
    )

    raw = ""
    for attempt in range(3):
        try:
            response = model.generate_content(user_prompt)
            raw = response.text.strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                clean = raw
                fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
                if fence:
                    clean = fence.group(1)
                else:
                    brace = re.search(r"(\{[^{}]*\})", clean, re.DOTALL)
                    if brace:
                        clean = brace.group(1)
                try:
                    parsed = json.loads(clean)
                except json.JSONDecodeError:
                    fallback = _parse_truncated_json(raw)
                    if fallback:
                        parsed = fallback
                    else:
                        raise
            return {
                "answer": str(parsed.get("answer", "?")).strip().upper(),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_response": raw,
                "error": None,
            }
        except json.JSONDecodeError:
            if attempt < 2:
                time_module.sleep(2)
                continue
            return {
                "answer": "?",
                "confidence": 0.0,
                "reasoning": "",
                "raw_response": raw,
                "error": "JSON parse error",
            }
        except Exception as e:
            if attempt < 2:
                time_module.sleep(2 * (attempt + 1))
                continue
            return {
                "answer": "?",
                "confidence": 0.0,
                "reasoning": "",
                "raw_response": "",
                "error": f"API error: {e}",
            }


# ============================================================
# PHASE 1: FREE BETA SWEEP (retrieval quality only, no LLM)
# ============================================================


def run_sweep(corpus: list[dict]) -> dict:
    """Evaluate all methods × all betas on retrieval quality (gold_in_topk).

    This is FREE — no LLM calls. Pure numpy.
    """
    print("[sweep] Loading embeddings...")
    ctx_embs, q_embs = load_embeddings()

    # Pre-compute timestamp arrays
    ctx_timestamps = []
    for c in corpus:
        ctx_timestamps.append(c["context_timestamp"])

    # Newest timestamp for the recency baseline.
    valid_ts = [t for t in ctx_timestamps if t is not None]
    t_max = max(valid_ts) if valid_ts else datetime(2019, 1, 1)
    print(f"[sweep] T_max (newest context): {t_max.date()}")
    print(f"[sweep] Contexts with timestamps: {len(valid_ts)}/{len(corpus)}")

    # Pre-compute recency ages (T_max - t_context) in days.
    recency_ages = np.array(
        [(t_max - t).total_seconds() / 86400.0 if t else 0.0 for t in ctx_timestamps]
    )

    # Parse stats
    n_gold_ts = sum(1 for c in corpus if c["context_timestamp"] is not None)
    n_parsed = sum(1 for c in corpus if c["question_focus"] is not None)
    print(f"[sweep] Gold timestamps available: {n_gold_ts}/{len(corpus)}")
    print(
        f"[sweep] Question focus parsed: {n_parsed}/{len(corpus)} "
        f"({100*n_parsed/len(corpus):.1f}%)"
    )

    results = {}

    # --- Vanilla ---
    print("[sweep] Evaluating vanilla...")
    gold_hits = 0
    for qi in range(len(corpus)):
        idxs = retrieve_vanilla(q_embs[qi], ctx_embs, TOP_K)
        if qi in idxs:
            gold_hits += 1
    results["vanilla"] = {
        "gold_in_topk": gold_hits / len(corpus),
        "n_evaluated": len(corpus),
    }
    print(
        f"  vanilla: gold_in_top{TOP_K} = {gold_hits}/{len(corpus)} "
        f"= {gold_hits/len(corpus):.3f}"
    )

    # --- Recency ---
    print("[sweep] Evaluating recency...")
    for beta in BETA_VALUES:
        gold_hits = 0
        for qi in range(len(corpus)):
            idxs = retrieve_temporal(q_embs[qi], ctx_embs, recency_ages, beta, TOP_K)
            if qi in idxs:
                gold_hits += 1
        key = f"recency_beta={beta}"
        rate = gold_hits / len(corpus)
        results[key] = {"gold_in_topk": rate, "n_evaluated": len(corpus)}
        print(f"  {key}: gold_in_top{TOP_K} = {rate:.3f}")

    # --- Oracle Proximity ---
    print("[sweep] Evaluating oracle proximity...")
    # Only evaluate questions where both gold timestamp exists
    oracle_indices = [i for i, c in enumerate(corpus) if c["context_timestamp"] is not None]
    print(f"  ({len(oracle_indices)} questions with gold timestamps)")

    for beta in BETA_VALUES:
        gold_hits = 0
        for qi in oracle_indices:
            t_focus = corpus[qi]["context_timestamp"]
            # Compute proximity ages: |t_context - t_focus| in days
            prox_ages = np.array(
                [abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
            )
            idxs = retrieve_temporal(q_embs[qi], ctx_embs, prox_ages, beta, TOP_K)
            if qi in idxs:
                gold_hits += 1
        key = f"oracle_beta={beta}"
        rate = gold_hits / len(oracle_indices) if oracle_indices else 0
        results[key] = {"gold_in_topk": rate, "n_evaluated": len(oracle_indices)}
        print(f"  {key}: gold_in_top{TOP_K} = {rate:.3f}")

    # --- Parsed Proximity ---
    print("[sweep] Evaluating parsed proximity...")
    parsed_indices = [i for i, c in enumerate(corpus) if c["question_focus"] is not None]
    print(f"  ({len(parsed_indices)} questions with parsed temporal focus)")

    for beta in BETA_VALUES:
        gold_hits = 0
        for qi in parsed_indices:
            t_focus = corpus[qi]["question_focus"]
            prox_ages = np.array(
                [abs((t - t_focus).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
            )
            idxs = retrieve_temporal(q_embs[qi], ctx_embs, prox_ages, beta, TOP_K)
            if qi in idxs:
                gold_hits += 1
        key = f"parsed_beta={beta}"
        rate = gold_hits / len(parsed_indices) if parsed_indices else 0
        results[key] = {"gold_in_topk": rate, "n_evaluated": len(parsed_indices)}
        print(f"  {key}: gold_in_top{TOP_K} = {rate:.3f}")

    # --- Vanilla on subsets (for fair comparison) ---
    print("[sweep] Computing vanilla on subsets for fair comparison...")
    for subset_name, subset_indices in [("oracle", oracle_indices), ("parsed", parsed_indices)]:
        gold_hits = 0
        for qi in subset_indices:
            idxs = retrieve_vanilla(q_embs[qi], ctx_embs, TOP_K)
            if qi in idxs:
                gold_hits += 1
        key = f"vanilla_on_{subset_name}_subset"
        rate = gold_hits / len(subset_indices) if subset_indices else 0
        results[key] = {"gold_in_topk": rate, "n_evaluated": len(subset_indices)}
        print(f"  {key}: gold_in_top{TOP_K} = {rate:.3f} " f"(N={len(subset_indices)})")

    # Find best beta per method
    best = {}
    for method in ["recency", "oracle", "parsed"]:
        method_results = {k: v for k, v in results.items() if k.startswith(method)}
        if method_results:
            best_key = max(method_results, key=lambda k: method_results[k]["gold_in_topk"])
            best[method] = {
                "best_config": best_key,
                "gold_in_topk": method_results[best_key]["gold_in_topk"],
            }

    output = {
        "experiment": "exp5_sweep",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": TOP_K,
            "beta_values": BETA_VALUES,
            "n_corpus": len(corpus),
            "n_oracle_eligible": len(oracle_indices),
            "n_parsed_eligible": len(parsed_indices),
            "T_max": t_max.isoformat(),
        },
        "all_results": results,
        "best_per_method": best,
    }

    # Save
    from experiments.utils import timestamped_save

    timestamped_save(output, "exp5_sweep", save_csv=False)

    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY: Gold-in-Top-K Retrieval Rate")
    print(f"{'='*60}")
    print(f"{'Method':<35s}  {'Gold@k':>7s}  {'N':>6s}  {'Note'}")
    print("-" * 65)
    print(
        f"{'vanilla (all)':<35s}  {results['vanilla']['gold_in_topk']:7.3f}  "
        f"{results['vanilla']['n_evaluated']:6d}"
    )
    for method in ["recency", "oracle", "parsed"]:
        # Show vanilla on the same subset for fair comparison
        subset_key = f"vanilla_on_{method}_subset"
        if subset_key in results:
            v_rate = results[subset_key]["gold_in_topk"]
            v_n = results[subset_key]["n_evaluated"]
            print(
                f"{'  vanilla ('+method+' subset)':<35s}  {v_rate:7.3f}  "
                f"{v_n:6d}  fair baseline"
            )
        if method in best:
            cfg = best[method]["best_config"]
            rate = best[method]["gold_in_topk"]
            n = results[cfg]["n_evaluated"]
            # Compute improvement vs fair baseline
            if subset_key in results and results[subset_key]["gold_in_topk"] > 0:
                v_rate = results[subset_key]["gold_in_topk"]
                delta_pct = (rate - v_rate) / v_rate * 100
                print(
                    f"  {cfg:<33s}  {rate:7.3f}  {n:6d}  "
                    f"best {method} ({delta_pct:+.1f}% vs vanilla)"
                )
            else:
                print(f"  {cfg:<33s}  {rate:7.3f}  {n:6d}  best {method}")
    print()

    return output


# ============================================================
# PHASE 2: LLM EVALUATION (Gemini calls on selected methods)
# ============================================================


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def run_experiment(*, limit: int | None = None) -> dict:
    """Run LLM evaluation on vanilla + best temporal methods.

    Uses deduplication: if two methods retrieve the same context set for a
    question, the Gemini call is made only once.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    # Load sweep results to get best betas
    sweep_path = RESULTS_DIR / "exp5_sweep.json"
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as f:
            sweep = json.load(f)
        best = sweep.get("best_per_method", {})
        print("[run] Loaded sweep results. Best configs:")
        for m, info in best.items():
            print(f"  {info['best_config']}: gold@{TOP_K}={info['gold_in_topk']:.3f}")
    else:
        print("[run] WARNING: No sweep results found. Using default betas.")
        best = {
            "recency": {"best_config": "recency_beta=0.005"},
            "oracle": {"best_config": "oracle_beta=0.01"},
            "parsed": {"best_config": "parsed_beta=0.01"},
        }

    # Parse best betas
    def _extract_beta(config_str: str) -> float:
        return float(config_str.split("=")[1])

    beta_recency = _extract_beta(best.get("recency", {}).get("best_config", "x=0.005"))
    beta_oracle = _extract_beta(best.get("oracle", {}).get("best_config", "x=0.01"))
    beta_parsed = _extract_beta(best.get("parsed", {}).get("best_config", "x=0.01"))

    print(
        f"[run] Beta values: recency={beta_recency}, oracle={beta_oracle}, " f"parsed={beta_parsed}"
    )

    # Build corpus and load embeddings
    print("[run] Building corpus...")
    corpus = build_corpus()
    ctx_embs, q_embs = load_embeddings()

    # Pre-compute timestamps
    ctx_timestamps = [c["context_timestamp"] for c in corpus]
    valid_ts = [t for t in ctx_timestamps if t is not None]
    t_max = max(valid_ts)

    recency_ages = np.array(
        [(t_max - t).total_seconds() / 86400.0 if t else 0.0 for t in ctx_timestamps]
    )

    # Determine eval set
    eval_indices = list(range(len(corpus)))
    if limit:
        eval_indices = eval_indices[:limit]
    print(f"[run] Evaluating {len(eval_indices)} questions")

    # --- Pre-compute ALL retrieval sets ---
    print("[run] Pre-computing retrieval sets for all methods...")
    methods = ["vanilla", "recency", "oracle", "parsed"]

    # retrieval_sets[method][qi] = tuple of retrieved indices
    retrieval_sets: dict[str, dict[int, tuple[int, ...]]] = {m: {} for m in methods}

    for qi in eval_indices:
        # Vanilla
        retrieval_sets["vanilla"][qi] = tuple(retrieve_vanilla(q_embs[qi], ctx_embs, TOP_K))

        # Recency
        retrieval_sets["recency"][qi] = tuple(
            retrieve_temporal(q_embs[qi], ctx_embs, recency_ages, beta_recency, TOP_K)
        )

        # Oracle proximity
        t_gold = corpus[qi]["context_timestamp"]
        if t_gold:
            prox_ages = np.array(
                [abs((t - t_gold).total_seconds()) / 86400.0 if t else 1e6 for t in ctx_timestamps]
            )
            retrieval_sets["oracle"][qi] = tuple(
                retrieve_temporal(q_embs[qi], ctx_embs, prox_ages, beta_oracle, TOP_K)
            )
        else:
            # Fall back to vanilla if no gold timestamp
            retrieval_sets["oracle"][qi] = retrieval_sets["vanilla"][qi]

        # Parsed proximity
        t_parsed = corpus[qi]["question_focus"]
        if t_parsed:
            prox_ages = np.array(
                [
                    abs((t - t_parsed).total_seconds()) / 86400.0 if t else 1e6
                    for t in ctx_timestamps
                ]
            )
            retrieval_sets["parsed"][qi] = tuple(
                retrieve_temporal(q_embs[qi], ctx_embs, prox_ages, beta_parsed, TOP_K)
            )
        else:
            # Fall back to vanilla if no parsed timestamp
            retrieval_sets["parsed"][qi] = retrieval_sets["vanilla"][qi]

    # --- Deduplication ---
    # Map each unique (qi, retrieval_set) to a list of methods
    unique_calls: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for method in methods:
        for qi in eval_indices:
            key = (qi, retrieval_sets[method][qi])
            if key not in unique_calls:
                unique_calls[key] = []
            unique_calls[key].append(method)

    n_total_naive = len(eval_indices) * len(methods)
    n_unique = len(unique_calls)
    savings_pct = (1 - n_unique / n_total_naive) * 100
    print(
        f"[run] Deduplication: {n_total_naive} naive calls -> {n_unique} unique "
        f"({savings_pct:.0f}% savings)"
    )
    est_hours = n_unique * GEMINI_RATE_LIMIT_DELAY / 3600
    print(
        f"[run] Estimated time: {est_hours:.1f} hours "
        f"({n_unique} calls × {GEMINI_RATE_LIMIT_DELAY}s)"
    )

    # --- Load checkpoint ---
    checkpoint_path = RESULTS_DIR / "exp5_v2_checkpoint.json"
    llm_cache: dict[str, dict] = {}  # key_str -> llm_result
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        llm_cache = ckpt.get("llm_cache", {})
        print(f"[run] Resumed from checkpoint: {len(llm_cache)} cached results")

    # --- Run LLM calls ---
    sorted_calls = sorted(unique_calls.keys(), key=lambda x: x[0])
    done = 0
    for call_key in sorted_calls:
        qi, ret_set = call_key
        key_str = f"{qi}_{','.join(map(str, ret_set))}"

        if key_str in llm_cache:
            done += 1
            continue

        if (done + 1) % 25 == 0 or done == 0:
            print(
                f"  LLM call {done+1}/{n_unique} (q={qi}, " f"methods={unique_calls[call_key]})..."
            )

        retrieved_contexts = [corpus[idx]["context"] for idx in ret_set]
        llm_result = call_gemini(corpus[qi]["question"], retrieved_contexts, api_key)
        llm_cache[key_str] = llm_result

        time_module.sleep(GEMINI_RATE_LIMIT_DELAY)
        done += 1

        # Checkpoint every 50
        if done % 50 == 0:
            _save_checkpoint(checkpoint_path, {"llm_cache": llm_cache})
            print(f"    [checkpoint] {done}/{n_unique}")

    # Final checkpoint save
    _save_checkpoint(checkpoint_path, {"llm_cache": llm_cache})

    # --- Assemble results per method ---
    method_results: dict[str, list[dict]] = {m: [] for m in methods}

    for method in methods:
        for qi in eval_indices:
            ret_set = retrieval_sets[method][qi]
            key_str = f"{qi}_{','.join(map(str, ret_set))}"
            llm_result = llm_cache.get(key_str, {})

            gold_letter = corpus[qi]["gold_answer"][0] if corpus[qi]["gold_answer"] else "?"
            predicted = llm_result.get("answer", "?")
            correct = predicted == gold_letter

            method_results[method].append(
                {
                    "query_index": qi,
                    "task": corpus[qi]["task"],
                    "gold_answer": gold_letter,
                    "predicted_answer": predicted,
                    "correct": correct,
                    "confidence": llm_result.get("confidence", 0.0),
                    "gold_in_topk": qi in ret_set,
                    "retrieved_indices": list(ret_set),
                    "reasoning": llm_result.get("reasoning", ""),
                    "error": llm_result.get("error"),
                }
            )

    # --- STL verification on oracle and parsed methods ---
    for method in ["oracle", "parsed"]:
        stl_beta_val = beta_oracle if method == "oracle" else beta_parsed
        decay_fn = ExponentialDecay(beta={"news": stl_beta_val})
        verifier = STLVerifier(decay_fn=decay_fn, threshold=STL_THRESHOLD)

        for r in method_results[method]:
            qi = r["query_index"]
            t_focus = (
                corpus[qi]["context_timestamp"]
                if method == "oracle"
                else corpus[qi]["question_focus"]
            )
            if r.get("error") or not t_focus:
                r["stl_robustness"] = None
                r["stl_satisfied"] = None
                continue

            facts = []
            for idx in r["retrieved_indices"]:
                ts = corpus[idx]["context_timestamp"]
                if ts:
                    facts.append(
                        TemporalFact(
                            content=corpus[idx]["context"][:200],
                            timestamp=ts,
                            fact_type="news",
                            source_quality=0.8,
                        )
                    )
            if facts:
                step = ReasoningStep(step_index=0, content="rag", facts_used=facts)
                trace = ReasoningTrace(steps=[step], query_time=t_focus)
                stl_result = verifier.verify(trace)
                r["stl_robustness"] = stl_result.robustness
                r["stl_satisfied"] = stl_result.satisfied
            else:
                r["stl_robustness"] = None
                r["stl_satisfied"] = None

    # --- Compute metrics ---
    metrics_summary = {}
    for method in methods:
        valid = [r for r in method_results[method] if not r.get("error")]
        if not valid:
            continue
        acc = float(np.mean([r["correct"] for r in valid]))
        conf = float(np.mean([r["confidence"] for r in valid]))
        gold_rate = float(np.mean([r["gold_in_topk"] for r in valid]))

        m = {
            "accuracy": acc,
            "mean_confidence": conf,
            "gold_in_topk_rate": gold_rate,
            "n_valid": len(valid),
            "n_errors": len(method_results[method]) - len(valid),
        }

        # STL metrics for oracle/parsed
        if method in ["oracle", "parsed"]:
            stl_valid = [r for r in valid if r.get("stl_robustness") is not None]
            if stl_valid:
                stl_sat = [r for r in stl_valid if r["stl_satisfied"]]
                stl_unsat = [r for r in stl_valid if not r["stl_satisfied"]]
                m["stl_satisfaction_rate"] = len(stl_sat) / len(stl_valid)
                if stl_sat:
                    m["acc_when_stl_satisfied"] = float(np.mean([r["correct"] for r in stl_sat]))
                if stl_unsat:
                    m["acc_when_stl_unsatisfied"] = float(
                        np.mean([r["correct"] for r in stl_unsat])
                    )

        beta_label = {
            "vanilla": "N/A",
            "recency": str(beta_recency),
            "oracle": str(beta_oracle),
            "parsed": str(beta_parsed),
        }
        m["beta"] = beta_label[method]
        metrics_summary[method] = m

    output = {
        "experiment": "exp5_rag_endtoend_v2",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": GEMINI_MODEL,
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": TOP_K,
            "beta_recency": beta_recency,
            "beta_oracle": beta_oracle,
            "beta_parsed": beta_parsed,
            "dataset": "TIME-Lite-News",
            "num_questions": len(eval_indices),
            "n_unique_llm_calls": n_unique,
            "deduplication_savings_pct": round(savings_pct, 1),
        },
        "metrics_summary": metrics_summary,
        "method_results": {m: method_results[m] for m in methods},
    }

    from experiments.utils import timestamped_save

    timestamped_save(output, "exp5_rag_endtoend", save_csv=False)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print_summary(output)
    return output


# ============================================================
# DISPLAY
# ============================================================


def print_summary(output: dict) -> None:
    """Print formatted results."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 5 v2: End-to-End RAG with Temporal Proximity")
    print(f"{'='*70}")

    cfg = output["config"]
    print(f"\nDataset: {cfg['dataset']} ({cfg['num_questions']} questions)")
    print(f"Embedding: {cfg['embedding_model']}, Top-k: {cfg['top_k']}")
    print(f"LLM: {cfg['model']}")
    if "n_unique_llm_calls" in cfg:
        print(
            f"Unique LLM calls: {cfg['n_unique_llm_calls']} "
            f"(dedup saved {cfg.get('deduplication_savings_pct', 0)}%)"
        )
    print(
        f"Betas: recency={cfg.get('beta_recency')}, "
        f"oracle={cfg.get('beta_oracle')}, parsed={cfg.get('beta_parsed')}"
    )

    print("\n--- RAG Method Comparison ---")
    print(
        f"{'Method':<20s} {'Beta':>6s}  {'Accuracy':>8s}  {'Gold@k':>7s}  "
        f"{'Conf':>6s}  {'N':>5s}"
    )
    print("-" * 60)

    for method in ["vanilla", "recency", "oracle", "parsed"]:
        m = output["metrics_summary"].get(method)
        if not m:
            continue
        print(
            f"{method:<20s} {m['beta']:>6s}  {m['accuracy']:8.3f}  "
            f"{m['gold_in_topk_rate']:7.3f}  {m['mean_confidence']:6.3f}  "
            f"{m['n_valid']:5d}"
        )

    # Improvement vs vanilla
    vanilla_acc = output["metrics_summary"].get("vanilla", {}).get("accuracy", 0)
    for method in ["oracle", "parsed"]:
        m = output["metrics_summary"].get(method)
        if m and vanilla_acc > 0:
            delta = m["accuracy"] - vanilla_acc
            pct = delta / vanilla_acc * 100
            print(f"\n{method} vs vanilla: {delta:+.3f} ({pct:+.1f}%)")

    # STL analysis
    for method in ["oracle", "parsed"]:
        m = output["metrics_summary"].get(method, {})
        if "stl_satisfaction_rate" in m:
            print(f"\n--- STL Verification ({method}) ---")
            print(f"  Satisfaction rate:     {m['stl_satisfaction_rate']:.3f}")
            if "acc_when_stl_satisfied" in m:
                print(f"  Acc when satisfied:   {m['acc_when_stl_satisfied']:.3f}")
            if "acc_when_stl_unsatisfied" in m:
                print(f"  Acc when unsatisfied: {m['acc_when_stl_unsatisfied']:.3f}")

    print()


def show_results() -> None:
    """Load and display saved results."""
    path = RESULTS_DIR / "exp5_rag_endtoend.json"
    if not path.exists():
        print(f"No results at {path}. Run --run first.")
        return
    with open(path, encoding="utf-8") as f:
        output = json.load(f)
    print_summary(output)


# ============================================================
# INSPECT MODE
# ============================================================


def inspect_corpus(n: int = 10) -> None:
    """Show sample questions with their parsed timestamps for verification."""
    corpus = build_corpus()
    n = min(n, len(corpus))

    n_ctx_ts = sum(1 for c in corpus if c["context_timestamp"] is not None)
    n_q_focus = sum(1 for c in corpus if c["question_focus"] is not None)
    print(f"Corpus: {len(corpus)} questions")
    print(
        f"Context timestamps parsed: {n_ctx_ts}/{len(corpus)} " f"({100*n_ctx_ts/len(corpus):.1f}%)"
    )
    print(
        f"Question focus parsed:     {n_q_focus}/{len(corpus)} "
        f"({100*n_q_focus/len(corpus):.1f}%)"
    )

    # Show examples with and without parsed focus
    print(f"\n--- First {n} questions ---")
    for i in range(n):
        c = corpus[i]
        ctx_ts = c["context_timestamp"]
        q_focus = c["question_focus"]
        ctx_preview = c["context"][:100].replace("\n", " ")
        q_preview = c["question"][:150].replace("\n", " ")
        print(f"\n[{i}] Task: {c['task']}")
        print(f"  Context date:  {ctx_ts.date() if ctx_ts else 'NONE'}")
        print(f"  Parsed focus:  {q_focus.date() if q_focus else 'NONE'}")
        if ctx_ts and q_focus:
            delta = abs((ctx_ts - q_focus).days)
            print(f"  Focus error:   {delta} days")
        print(f"  Gold answer:   {c['gold_answer']}")
        print(f"  Context:       {ctx_preview}...")
        print(f"  Question:      {q_preview}...")

    # Show some where parsing failed
    failed = [c for c in corpus if c["question_focus"] is None]
    if failed:
        n_show = min(5, len(failed))
        print(f"\n--- {len(failed)} questions with NO parsed focus " f"(showing {n_show}) ---")
        for c in failed[:n_show]:
            q_preview = c["question"][:200].replace("\n", " ")
            print(f"\n[{c['index']}] {q_preview}...")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5 v2: E2E RAG with Temporal Proximity")
    parser.add_argument("--embed", action="store_true", help="Compute embeddings (local GPU)")
    parser.add_argument(
        "--inspect", type=int, nargs="?", const=10, help="Inspect N questions (default 10)"
    )
    parser.add_argument("--sweep", action="store_true", help="Free beta sweep on retrieval quality")
    parser.add_argument("--run", action="store_true", help="Run LLM evaluation")
    parser.add_argument("--results", action="store_true", help="Show saved results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    if args.embed:
        corpus = build_corpus()
        compute_embeddings(corpus)
    elif args.inspect is not None:
        inspect_corpus(args.inspect)
    elif args.sweep:
        corpus = build_corpus()
        run_sweep(corpus)
    elif args.run:
        run_experiment(limit=args.limit)
    elif args.results:
        show_results()
    else:
        parser.print_help()
