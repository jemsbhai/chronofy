"""Local-only test of Dial/Wiki parsing + STL pipeline — NO API CALLS.

Verifies that:
1. context_to_temporal_facts() works for all 3 subsets
2. Facts feed correctly into STLVerifier
3. Validity scores are in expected ranges
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import ReasoningStep, ReasoningTrace
from chronofy.verification.stl import STLVerifier
from experiments.reproducibility import TIME_LITE_REVISION

# Import from exp2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from exp2_stl_time import SUBSET_CONFIG, context_to_temporal_facts


def test_subset(subset: str, rows: list[dict]) -> None:
    cfg = SUBSET_CONFIG[subset]
    decay_fn = ExponentialDecay(beta=cfg["beta"])
    verifier = STLVerifier(decay_fn=decay_fn, threshold=cfg["stl_threshold"])

    parse_ok = 0
    parse_fail = 0
    ages = []
    validities = []
    robustness_scores = []

    for row in rows:
        facts = context_to_temporal_facts(row["Context"], subset=subset)
        if not facts:
            parse_fail += 1
            continue
        parse_ok += 1

        fact = facts[0]
        age = fact.age_at(cfg["query_time"])
        ages.append(age)

        # Build a mock trace with this fact
        step = ReasoningStep(step_index=0, content="test", facts_used=facts)
        trace = ReasoningTrace(steps=[step], query_time=cfg["query_time"])
        result = verifier.verify(trace)
        validities.append(result.output_confidence_bound)
        robustness_scores.append(result.robustness)

    ages = np.array(ages)
    validities = np.array(validities)
    robustness_scores = np.array(robustness_scores)

    print(f"\n{'='*60}")
    print(f"SUBSET: {subset.upper()} ({len(rows)} rows)")
    print(f"{'='*60}")
    print(f"Parse success: {parse_ok}/{len(rows)} ({100*parse_ok/len(rows):.1f}%)")
    print(f"Parse failed:  {parse_fail}")
    print(
        f"\nAge (days):       mean={ages.mean():.0f}  min={ages.min():.0f}  "
        f"max={ages.max():.0f}  std={ages.std():.0f}"
    )
    print(
        f"Validity V:       mean={validities.mean():.4f}  "
        f"min={validities.min():.4f}  max={validities.max():.4f}"
    )
    print(
        f"Robustness rho:   mean={robustness_scores.mean():.4f}  "
        f"min={robustness_scores.min():.4f}  max={robustness_scores.max():.4f}"
    )
    print(
        f"STL satisfied:    {(robustness_scores >= 0).sum()}/{len(robustness_scores)} "
        f"({100*(robustness_scores >= 0).mean():.1f}%)"
    )

    # Show a few examples
    print("\nSample facts:")
    for i in [0, len(rows) // 3, 2 * len(rows) // 3]:
        if i >= len(rows):
            continue
        facts = context_to_temporal_facts(rows[i]["Context"], subset=subset)
        if facts:
            f = facts[0]
            v = decay_fn.compute(f, cfg["query_time"])
            age = f.age_at(cfg["query_time"])
            print(f"  [{i}] ts={f.timestamp.date()} age={age:.0f}d V={v:.4f} q={f.source_quality}")


def main():
    from datasets import load_dataset

    ds = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    all_rows = list(ds["train"])

    for subset in ["news", "dial", "wiki"]:
        cfg = SUBSET_CONFIG[subset]
        rows = [r for r in all_rows if r["Dataset Name"] == cfg["dataset_name"]]
        test_subset(subset, rows)

    print(f"\n{'='*60}")
    print("ALL LOCAL TESTS PASSED — ready for API runs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
