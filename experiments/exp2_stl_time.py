"""Experiment 2: STL Robustness for Knowledge Validity on TimE Benchmark

HEADLINE NOVELTY EXPERIMENT.

Hypothesis: STL robustness over *knowledge temporal validity* (rho) predicts
answer correctness better than raw LLM confidence, especially in cases where
the LLM is confidently wrong due to temporally stale retrieval.

Dataset: TIME-Lite (SylvainWei/TIME-Lite), News subset (897 QA pairs)
  - Multiple-choice temporal reasoning questions
  - Context with explicit timestamps ("Day: Month DD, YYYY")
  - Gold answers (A/B/C/D/E)

Design:
  For each question:
    1. Parse timestamp(s) from Context -> TemporalFact objects
    2. Set query_time = 2019-01-01 (2 years after latest context, giving
       fact ages of ~2-5 years with meaningful spread in validity scores)
    3. Send Question + Context to the configured Gemini API model identifier
       -> get: predicted answer, confidence, reasoning (provider output is not deterministic)
    4. Build ReasoningTrace from parsed facts + LLM reasoning
    5. Compute STL robustness rho via STLVerifier
    6. Record: (predicted, gold, confidence, rho, correct?)

  Analysis:
    - AUROC: rho as predictor of correctness
    - AUROC: LLM confidence as predictor of correctness
    - Joint analysis: does rho add signal beyond confidence?
    - Breakdown by Task type (10 temporal reasoning tasks)
    - "Confidently wrong" analysis: cases LLM is confident but wrong,
      and whether STL catches them

Usage:
  python experiments/exp2_stl_time.py --run              # Run full experiment
  python experiments/exp2_stl_time.py --run --limit 10   # Quick test
  python experiments/exp2_stl_time.py --results          # Print saved results
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace, TemporalFact
from chronofy.verification.stl import STLVerifier
from experiments.reproducibility import GEMINI_MODEL_ID, TIME_LITE_REVISION

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Gemini config
GEMINI_MODEL = GEMINI_MODEL_ID
GEMINI_MAX_TOKENS = 8192  # thinking model uses tokens for internal reasoning
GEMINI_RATE_LIMIT_DELAY = 4.5  # seconds between calls (~13 req/min, under free 15 RPM)


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

# --- News patterns ---
DATE_PATTERN_NEWS = re.compile(r"Day:\s*([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})")
DATE_PATTERN_TEXT = re.compile(
    r"(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
)

# --- Dial patterns ---
DIAL_SESSION_P1 = re.compile(
    r"Session\s+\d+\s+happened\s+at\s+(\d{1,2})[./](\d{1,2})[./](\d{4}),?\s*(\d{1,2}):(\d{2}):?(\d{2})?"
)
DIAL_SESSION_P2 = re.compile(
    r"Session\s+\d+\s+happened\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})"
)
DIAL_MSG_P = re.compile(
    r"\w+\s+\((\d{1,2})[./](\d{1,2})[./](\d{4}),?\s*(\d{1,2}):(\d{2}):?(\d{2})?\)"
)

# --- Wiki patterns ---
WIKI_DATE_FULL1 = re.compile(
    r"(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{1,2}),?\s+(\d{4})"
)
WIKI_DATE_FULL2 = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August"
    r"|September|October|November|December),?\s+(\d{4})"
)
WIKI_MONTH_YEAR = re.compile(
    r"(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{4})"
)
WIKI_YEAR_RANGE = re.compile(r"\*{0,2}(\d{4})\*{0,2}\s*[-\u2013]\s*\*{0,2}(\d{4})\*{0,2}")
WIKI_YEAR_SOLO = re.compile(
    r"(?:in|from|since|during|until|by|after|before)\s+\*{0,2}(\d{4})\*{0,2}"
)
WIKI_BOLD_YEAR = re.compile(r"\*\*(\d{4})\*\*")


def parse_news_timestamp(context: str) -> datetime | None:
    """Extract the primary date from a TIME-Lite-News context header."""
    match = DATE_PATTERN_NEWS.search(context)
    if match:
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                return datetime(int(year_str), month, int(day_str))
            except ValueError:
                pass
    match = DATE_PATTERN_TEXT.search(context)
    if match:
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                return datetime(int(year_str), month, int(day_str))
            except ValueError:
                pass
    return None


def parse_dial_timestamps(context: str) -> list[datetime]:
    """Parse timestamps from a Dial context. Returns list of datetimes."""
    results = []
    for match in DIAL_SESSION_P1.finditer(context):
        day, month, year = match.group(1), match.group(2), match.group(3)
        hour, minute = match.group(4), match.group(5)
        second = match.group(6) or "0"
        try:
            results.append(
                datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            )
        except ValueError:
            pass
    for match in DIAL_SESSION_P2.finditer(context):
        hour, minute, ampm = int(match.group(1)), int(match.group(2)), match.group(3)
        day, month_str, year = int(match.group(4)), match.group(5), int(match.group(6))
        month = MONTH_MAP.get(month_str)
        if month:
            if ampm.lower() == "pm" and hour != 12:
                hour += 12
            elif ampm.lower() == "am" and hour == 12:
                hour = 0
            try:
                results.append(datetime(year, month, day, hour, minute))
            except ValueError:
                pass
    if not results:
        for match in DIAL_MSG_P.finditer(context):
            day, month, year = match.group(1), match.group(2), match.group(3)
            hour, minute = match.group(4), match.group(5)
            second = match.group(6) or "0"
            try:
                results.append(
                    datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                )
            except ValueError:
                pass
    return results


def parse_wiki_timestamps(context: str) -> list[datetime]:
    """Parse timestamps from Wiki biographical prose."""
    results = []
    seen_years: set[int] = set()
    for match in WIKI_DATE_FULL1.finditer(context):
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                dt = datetime(int(year_str), month, int(day_str))
                results.append(dt)
                seen_years.add(int(year_str))
            except ValueError:
                pass
    for match in WIKI_DATE_FULL2.finditer(context):
        day_str, month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                dt = datetime(int(year_str), month, int(day_str))
                results.append(dt)
                seen_years.add(int(year_str))
            except ValueError:
                pass
    for match in WIKI_MONTH_YEAR.finditer(context):
        month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        year = int(year_str)
        if month and 1900 < year < 2030 and year not in seen_years:
            try:
                results.append(datetime(year, month, 1))
                seen_years.add(year)
            except ValueError:
                pass
    for match in WIKI_YEAR_RANGE.finditer(context):
        y1, y2 = int(match.group(1)), int(match.group(2))
        for y in [y1, y2]:
            if 1900 < y < 2030 and y not in seen_years:
                results.append(datetime(y, 6, 15))
                seen_years.add(y)
    for match in WIKI_YEAR_SOLO.finditer(context):
        y = int(match.group(1))
        if 1900 < y < 2030 and y not in seen_years:
            results.append(datetime(y, 6, 15))
            seen_years.add(y)
    for match in WIKI_BOLD_YEAR.finditer(context):
        y = int(match.group(1))
        if 1900 < y < 2030 and y not in seen_years:
            results.append(datetime(y, 6, 15))
            seen_years.add(y)
    return results


# ============================================================
# SUBSET CONFIGURATIONS
# ============================================================

SUBSET_CONFIG = {
    "news": {
        "dataset_name": "TIME-Lite-News",
        "query_time": datetime(2019, 1, 1),
        "beta": {"news": 0.001},
        "stl_threshold": 0.25,
        "source_quality": 0.8,
    },
    "dial": {
        "dataset_name": "TIME-Lite-Dial",
        "query_time": datetime(2025, 1, 1),
        # beta=0.001/day: ages 360-1810d give V=0.42-0.10 (good spread)
        "beta": {"dial": 0.001},
        "stl_threshold": 0.25,
        "source_quality": 0.6,  # informal chat, lower reliability
    },
    "wiki": {
        "dataset_name": "TIME-Lite-Wiki",
        "query_time": datetime(2025, 1, 1),
        "beta": {"wiki": 0.0005},
        "stl_threshold": 0.25,
        "source_quality": 0.9,  # encyclopedic, high reliability
    },
}


def context_to_temporal_facts(
    context: str,
    subset: str = "news",
) -> list[TemporalFact]:
    """Parse a TIME-Lite context into TemporalFact objects.

    Dispatches to the correct parser based on subset.
    For Dial/Wiki: uses the earliest timestamp as the fact's anchor.
    """
    cfg = SUBSET_CONFIG[subset]
    content_preview = context[:500] if len(context) > 500 else context

    if subset == "news":
        ts = parse_news_timestamp(context)
        if ts is None:
            return []
        return [
            TemporalFact(
                content=content_preview,
                timestamp=ts,
                fact_type=subset,
                source_quality=cfg["source_quality"],
                source=cfg["dataset_name"],
            )
        ]
    elif subset == "dial":
        timestamps = parse_dial_timestamps(context)
        if not timestamps:
            return []
        # Use earliest session timestamp as the fact anchor
        return [
            TemporalFact(
                content=content_preview,
                timestamp=min(timestamps),
                fact_type=subset,
                source_quality=cfg["source_quality"],
                source=cfg["dataset_name"],
            )
        ]
    elif subset == "wiki":
        timestamps = parse_wiki_timestamps(context)
        if not timestamps:
            return []
        # Use the most recent timestamp as the fact anchor
        # (biographical articles are as fresh as their latest event)
        return [
            TemporalFact(
                content=content_preview,
                timestamp=max(timestamps),
                fact_type=subset,
                source_quality=cfg["source_quality"],
                source=cfg["dataset_name"],
            )
        ]
    else:
        return []


# ============================================================
# LLM INFERENCE (GEMINI)
# ============================================================

SYSTEM_PROMPT = (
    "You are answering a multiple-choice temporal reasoning question.\n\n"
    "INSTRUCTIONS:\n"
    "1. Read the provided context carefully, paying attention to dates "
    "and temporal relationships.\n"
    "2. Select the best answer from the options given.\n"
    "3. Respond in EXACTLY this JSON format (no other text):\n\n"
    '{"answer": "A", "confidence": 0.85, "reasoning": "Short reason."}\n\n'
    '- "answer": single letter A, B, C, D, or E\n'
    '- "confidence": number 0.0 to 1.0\n'
    '- "reasoning": MAX 20 words. Be extremely brief.\n\n'
    "Output ONLY the JSON object. No markdown. No backticks. No extra text."
)


def build_user_prompt(question: str, context: str) -> str:
    """Build the user prompt for the LLM."""
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nRespond with JSON only."


def _parse_truncated_json(raw: str) -> dict | None:
    """Fallback parser for truncated JSON from token limit.

    Handles cases like: {"answer": "B", "confidence": 0.9, "reasoning": "text cut off
    Extracts answer and confidence even if reasoning is truncated.
    """
    answer_match = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', raw)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
    if answer_match:
        return {
            "answer": answer_match.group(1).upper(),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reasoning": "(truncated)",
        }
    return None


def call_gemini(
    question: str,
    context: str,
    api_key: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Call Gemini API and parse structured response.

    Returns dict: answer, confidence, reasoning, raw_response, error
    """
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
    user_prompt = build_user_prompt(question, context)

    raw = ""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(user_prompt)
            raw = response.text.strip()

            # response_mime_type="application/json" forces valid JSON output
            # but we still add fallback parsing for robustness
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try extracting JSON from fences or bare braces
                clean = raw
                fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
                if fence_match:
                    clean = fence_match.group(1)
                else:
                    brace_match = re.search(r"(\{[^{}]*\})", clean, re.DOTALL)
                    if brace_match:
                        clean = brace_match.group(1)
                try:
                    parsed = json.loads(clean)
                except json.JSONDecodeError:
                    # Last resort: regex extraction
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
            if attempt < max_retries - 1:
                time_module.sleep(retry_delay)
                continue
            return {
                "answer": "?",
                "confidence": 0.0,
                "reasoning": "",
                "raw_response": raw,
                "error": "JSON parse error after retries",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time_module.sleep(retry_delay * (attempt + 1))
                continue
            return {
                "answer": "?",
                "confidence": 0.0,
                "reasoning": "",
                "raw_response": "",
                "error": f"API error: {e}",
            }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================


def _save_checkpoint(path: Path, results: list, parse_failures: int, api_errors: int) -> None:
    """Save incremental checkpoint to resume after crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "parse_failures": parse_failures,
                "api_errors": api_errors,
            },
            f,
            ensure_ascii=False,
        )
    print(f"    [checkpoint] Saved {len(results)} results")


def run_experiment(*, limit: int | None = None, subset: str = "news") -> dict:
    """Run the full Experiment 2 pipeline."""
    from datasets import load_dataset

    if subset not in SUBSET_CONFIG:
        raise ValueError(f"Unknown subset '{subset}'. Choose from: {list(SUBSET_CONFIG)}")
    cfg = SUBSET_CONFIG[subset]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    # Load dataset
    print("[1/5] Loading TIME-Lite dataset...")
    ds = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    all_rows = ds["train"]

    # Filter to requested subset
    subset_rows = [row for row in all_rows if row["Dataset Name"] == cfg["dataset_name"]]
    print(f"  Total rows: {len(all_rows)}, {cfg['dataset_name']}: {len(subset_rows)}")

    if limit:
        subset_rows = subset_rows[:limit]
        print(f"  Limited to first {limit} rows")

    # Setup Chronofy components
    decay_fn = ExponentialDecay(beta=cfg["beta"])
    verifier = STLVerifier(decay_fn=decay_fn, threshold=cfg["stl_threshold"])

    results = []
    parse_failures = 0
    api_errors = 0

    # Resume from checkpoint if available
    checkpoint_path = RESULTS_DIR / f"exp2_stl_time_{subset}_checkpoint.json"
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        results = checkpoint["results"]
        parse_failures = checkpoint["parse_failures"]
        api_errors = checkpoint["api_errors"]
        start_idx = len(results)
        print(f"  Resuming from checkpoint at question {start_idx}/{len(subset_rows)}")

    print(f"\n[2/5] Running inference on {len(subset_rows)} questions...")
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  Subset: {subset}")
    print(f"  Query time: {cfg['query_time'].isoformat()}")
    print(f"  beta: {cfg['beta']}")
    print(f"  STL threshold gamma: {cfg['stl_threshold']}")
    print()

    for i, row in enumerate(subset_rows):
        if i < start_idx:
            continue  # skip already-completed questions
        if (i + 1) % 25 == 0 or i == 0 or i == start_idx:
            print(f"  Processing {i+1}/{len(subset_rows)}...")

        question = row["Question"]
        context = row["Context"]
        gold = row["Gold Answer"].strip().upper()
        task = row["Task"]
        setting = row["Setting"]
        level = row["Level"]

        # Step 1: Parse facts from context
        facts = context_to_temporal_facts(context, subset=subset)
        if not facts:
            parse_failures += 1
            facts_for_trace = []
        else:
            facts_for_trace = facts

        # Step 2: Call LLM
        llm_result = call_gemini(question, context, api_key)
        if llm_result["error"]:
            api_errors += 1
            print(f"    [!] Error on Q{i}: {llm_result['error']}")

        # Step 3: Build ReasoningTrace (single-step for this experiment)
        step = ReasoningStep(
            step_index=0,
            content=llm_result["reasoning"],
            facts_used=facts_for_trace,
        )
        trace = ReasoningTrace(steps=[step], query_time=cfg["query_time"])

        # Step 4: Compute STL robustness
        stl_result = verifier.verify(trace)

        # Step 5: Record result
        predicted = llm_result["answer"]
        gold_letter = gold[0] if gold else "?"
        # Handle multi-word gold answers (Wiki sometimes has full text answers)
        if len(gold) > 1 and gold[0] not in "ABCDE":
            gold_letter = "?"  # can't match, will be marked wrong
        correct = predicted == gold_letter

        record = {
            "index": i,
            "task": task,
            "setting": setting,
            "level": level,
            "gold_answer": gold_letter,
            "predicted_answer": predicted,
            "correct": correct,
            "llm_confidence": llm_result["confidence"],
            "stl_robustness": stl_result.robustness,
            "stl_satisfied": stl_result.satisfied,
            "output_confidence_bound": stl_result.output_confidence_bound,
            "fact_timestamp": facts[0].timestamp.isoformat() if facts else None,
            "fact_age_days": facts[0].age_at(cfg["query_time"]) if facts else None,
            "reasoning": llm_result["reasoning"],
            "error": llm_result["error"],
        }
        results.append(record)

        # Rate limiting
        time_module.sleep(GEMINI_RATE_LIMIT_DELAY)

        # Checkpoint every 50 questions
        if (i + 1) % 50 == 0:
            _save_checkpoint(checkpoint_path, results, parse_failures, api_errors)

    print("\n[3/5] Inference complete.")
    print(
        f"  Total: {len(results)}, Parse failures: {parse_failures}, " f"API errors: {api_errors}"
    )

    # Experiment name includes subset
    exp_name = f"exp2_stl_time_{subset}"

    # Remove checkpoint now that we're done
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Checkpoint removed.")

    # Compute metrics
    print("\n[4/5] Computing metrics...")
    metrics = compute_metrics(results)

    # Save
    print("\n[5/5] Saving results...")
    output = {
        "experiment": "exp2_stl_time",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": GEMINI_MODEL,
            "subset": subset,
            "query_time": cfg["query_time"].isoformat(),
            "beta": cfg["beta"],
            "stl_threshold": cfg["stl_threshold"],
            "dataset": cfg["dataset_name"],
            "num_questions": len(subset_rows),
            "limit": limit,
        },
        "metrics": metrics,
        "parse_failures": parse_failures,
        "api_errors": api_errors,
        "results": results,
    }

    from experiments.utils import timestamped_save

    timestamped_save(output, exp_name, save_csv=False)

    print_summary(metrics, parse_failures, api_errors, len(results), subset)
    return output


# ============================================================
# METRICS
# ============================================================


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute AUROC without sklearn. Returns None if single class."""
    if len(np.unique(labels)) < 2:
        return None
    desc_order = np.argsort(-scores)
    sorted_labels = labels[desc_order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fpr = 0.0
    for label in sorted_labels:
        if label == 1.0:
            tp += 1
        else:
            fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr
    return float(auc)


def compute_metrics(results: list[dict]) -> dict:
    """Compute all evaluation metrics from results."""
    valid = [r for r in results if not r["error"] and r["fact_timestamp"] is not None]
    if not valid:
        return {"error": "No valid results to analyze"}

    correct_flags = np.array([r["correct"] for r in valid], dtype=float)
    confidences = np.array([r["llm_confidence"] for r in valid])
    robustness_scores = np.array([r["stl_robustness"] for r in valid])
    confidence_bounds = np.array([r["output_confidence_bound"] for r in valid])
    ages = np.array([r["fact_age_days"] for r in valid])

    accuracy = float(correct_flags.mean())
    auroc_rho = compute_auroc(correct_flags, robustness_scores)
    auroc_conf = compute_auroc(correct_flags, confidences)
    auroc_bound = compute_auroc(correct_flags, confidence_bounds)

    n = len(valid)
    corr_rho = float(np.corrcoef(correct_flags, robustness_scores)[0, 1]) if n > 2 else 0.0
    corr_conf = float(np.corrcoef(correct_flags, confidences)[0, 1]) if n > 2 else 0.0
    corr_age = float(np.corrcoef(correct_flags, ages)[0, 1]) if n > 2 else 0.0

    stl_sat_rate = float(np.mean([r["stl_satisfied"] for r in valid]))
    sat_results = [r for r in valid if r["stl_satisfied"]]
    unsat_results = [r for r in valid if not r["stl_satisfied"]]
    acc_sat = float(np.mean([r["correct"] for r in sat_results])) if sat_results else None
    acc_unsat = float(np.mean([r["correct"] for r in unsat_results])) if unsat_results else None

    # Task breakdown
    task_metrics = {}
    for task in set(r["task"] for r in valid):
        task_results = [r for r in valid if r["task"] == task]
        task_correct = [r["correct"] for r in task_results]
        task_rho = [r["stl_robustness"] for r in task_results]
        task_metrics[task] = {
            "count": len(task_results),
            "accuracy": float(np.mean(task_correct)),
            "mean_robustness": float(np.mean(task_rho)),
        }

    # "Confidently wrong" analysis
    conf_wrong = [r for r in valid if r["llm_confidence"] > 0.7 and not r["correct"]]
    conf_wrong_caught = [r for r in conf_wrong if not r["stl_satisfied"]]

    # Age-bin analysis: split facts into age tertiles
    age_sorted = sorted(valid, key=lambda r: r["fact_age_days"])
    n_third = len(age_sorted) // 3
    age_bins = {}
    if n_third > 0:
        for label, subset in [
            ("newest_third", age_sorted[:n_third]),
            ("middle_third", age_sorted[n_third : 2 * n_third]),
            ("oldest_third", age_sorted[2 * n_third :]),
        ]:
            age_bins[label] = {
                "count": len(subset),
                "accuracy": float(np.mean([r["correct"] for r in subset])),
                "mean_age_days": float(np.mean([r["fact_age_days"] for r in subset])),
                "mean_robustness": float(np.mean([r["stl_robustness"] for r in subset])),
            }

    return {
        "n_valid": len(valid),
        "accuracy": accuracy,
        "auroc_robustness": auroc_rho,
        "auroc_confidence": auroc_conf,
        "auroc_confidence_bound": auroc_bound,
        "correlation_rho_correct": corr_rho,
        "correlation_conf_correct": corr_conf,
        "correlation_age_correct": corr_age,
        "stl_satisfaction_rate": stl_sat_rate,
        "accuracy_when_stl_satisfied": acc_sat,
        "accuracy_when_stl_unsatisfied": acc_unsat,
        "n_satisfied": len(sat_results),
        "n_unsatisfied": len(unsat_results),
        "n_confidently_wrong": len(conf_wrong),
        "n_confidently_wrong_caught_by_stl": len(conf_wrong_caught),
        "confidently_wrong_catch_rate": (
            len(conf_wrong_caught) / len(conf_wrong) if conf_wrong else None
        ),
        "mean_age_days": float(ages.mean()),
        "mean_confidence": float(confidences.mean()),
        "mean_robustness": float(robustness_scores.mean()),
        "task_breakdown": task_metrics,
        "age_bin_analysis": age_bins,
    }


# ============================================================
# DISPLAY
# ============================================================


def print_summary(
    metrics: dict, parse_failures: int, api_errors: int, total: int, subset: str = "news"
) -> None:
    """Print formatted results summary."""
    cfg = SUBSET_CONFIG.get(subset, {})
    dataset_name = cfg.get("dataset_name", "TIME-Lite")
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: STL Robustness for Knowledge Validity [{subset.upper()}]")
    print("=" * 60)
    print(f"\nDataset: {dataset_name} ({total} questions)")
    print(f"Valid results: {metrics.get('n_valid', 0)}")
    print(f"Parse failures: {parse_failures}, API errors: {api_errors}")

    print("\n--- Overall ---")
    print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"Mean LLM confidence: {metrics.get('mean_confidence', 0):.3f}")
    print(f"Mean STL robustness rho: {metrics.get('mean_robustness', 0):.3f}")
    print(f"Mean fact age (days): {metrics.get('mean_age_days', 0):.0f}")

    print("\n--- Predictive Power (AUROC) ---")
    auroc_r = metrics.get("auroc_robustness")
    auroc_c = metrics.get("auroc_confidence")
    auroc_b = metrics.get("auroc_confidence_bound")
    print(f"AUROC (rho -> correct):          {auroc_r:.3f}" if auroc_r else "AUROC (rho): N/A")
    print(f"AUROC (confidence -> correct):   {auroc_c:.3f}" if auroc_c else "AUROC (conf): N/A")
    print(f"AUROC (Thm1 bound -> correct):   {auroc_b:.3f}" if auroc_b else "AUROC (bound): N/A")

    print("\n--- STL Verification ---")
    print(f"STL satisfaction rate: {metrics.get('stl_satisfaction_rate', 0):.3f}")
    sat = metrics.get("accuracy_when_stl_satisfied")
    unsat = metrics.get("accuracy_when_stl_unsatisfied")
    if sat is not None:
        print(f"Accuracy when STL satisfied:   {sat:.3f} (n={metrics.get('n_satisfied', 0)})")
    else:
        print("  (no satisfied results)")
    if unsat is not None:
        print(f"Accuracy when STL unsatisfied: {unsat:.3f} (n={metrics.get('n_unsatisfied', 0)})")
    else:
        print("  (no unsatisfied results)")

    print("\n--- 'Confidently Wrong' Analysis ---")
    print(f"Cases with confidence > 0.7 but wrong: {metrics.get('n_confidently_wrong', 0)}")
    caught = metrics.get("n_confidently_wrong_caught_by_stl", 0)
    rate = metrics.get("confidently_wrong_catch_rate")
    print(f"Of those, caught by STL (rho < 0): {caught}")
    if rate is not None:
        print(f"Catch rate: {rate:.3f}")

    print("\n--- Correlations ---")
    print(f"Corr(rho, correct):          {metrics.get('correlation_rho_correct', 0):.3f}")
    print(f"Corr(confidence, correct):   {metrics.get('correlation_conf_correct', 0):.3f}")
    print(f"Corr(age, correct):          {metrics.get('correlation_age_correct', 0):.3f}")

    print("\n--- Age Bin Analysis ---")
    for label, ab in sorted(metrics.get("age_bin_analysis", {}).items()):
        print(
            f"  {label:15s}  n={ab['count']:3d}  acc={ab['accuracy']:.3f}  "
            f"age={ab['mean_age_days']:.0f}d  rho={ab['mean_robustness']:.4f}"
        )

    print("\n--- Task Breakdown ---")
    for task, tm in sorted(metrics.get("task_breakdown", {}).items()):
        print(
            f"  {task:25s}  n={tm['count']:3d}  acc={tm['accuracy']:.3f}  "
            f"rho={tm['mean_robustness']:.4f}"
        )
    print()


def show_results(subset: str = "news") -> None:
    """Load and display saved results."""
    results_path = RESULTS_DIR / f"exp2_stl_time_{subset}.json"
    if not results_path.exists():
        # Try legacy name
        results_path = RESULTS_DIR / "exp2_stl_time.json"
    if not results_path.exists():
        print(f"No results found for subset '{subset}'")
        print("Run with --run first.")
        return
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    print_summary(
        data["metrics"],
        data.get("parse_failures", 0),
        data.get("api_errors", 0),
        data["config"]["num_questions"],
        subset,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: STL robustness on TimE")
    parser.add_argument("--run", action="store_true", help="Run the experiment")
    parser.add_argument("--results", action="store_true", help="Show saved results")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument(
        "--subset",
        type=str,
        default="news",
        choices=["news", "dial", "wiki"],
        help="Which TIME-Lite subset to run",
    )
    args = parser.parse_args()

    if args.run:
        run_experiment(limit=args.limit, subset=args.subset)
    elif args.results:
        show_results(subset=args.subset)
    else:
        parser.print_help()
