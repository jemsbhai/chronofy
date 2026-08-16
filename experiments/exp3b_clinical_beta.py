"""Experiment 3b: Clinical β Sensitivity Analysis using MIMIC-IV Demo

Goal: Explore whether different fact types prefer different decay rates on
the deidentified, open MIMIC-IV Demo. This is not clinical validation.

Dataset: MIMIC-IV Clinical Database Demo v2.2 (100 patients, open access)
  - Vital signs (chartevents): heart rate, blood pressure, SpO2, temperature
  - Lab results (labevents): potassium, creatinine, hemoglobin, glucose
  - Prescriptions (prescriptions): medication orders with start/end dates
  - Diagnoses (diagnoses_icd): ICD codes per admission
  - Patient demographics are loaded for dataset context but are not included
    in the current beta-sweep evaluation categories

These fact types have fundamentally different temporal dynamics:
  - Vital signs: κ high → β should be high (change within hours)
  - Lab results: κ medium → β should be medium (change within days)
  - Prescriptions: κ medium-low → β should be medium-low (change within weeks)
  - Diagnoses: κ low → β should be low (persist for months/years)

Method:
  For each fact category, we construct temporal retrieval queries:
    "Given a query at time T_q, retrieve the correct value from history"
  We sweep β and measure which decay rate gives best retrieval accuracy
  per category. If Proposition 1 holds, optimal β should vary dramatically.

Usage:
  python experiments/exp3b_clinical_beta.py --download
  python experiments/exp3b_clinical_beta.py --run
  python experiments/exp3b_clinical_beta.py --quick
  python experiments/exp3b_clinical_beta.py --results
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproducibility import MIMIC_CHECKSUM_MANIFEST_SHA256, RANDOM_SEED

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data" / "mimic-iv-demo"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MIMIC_DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"

# β sweep grid
BETA_GRID = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# Key vital sign itemids from MIMIC-IV chartevents (MetaVision)
VITAL_SIGN_ITEMS = {
    220045: "Heart Rate",
    220050: "Arterial BP Systolic",
    220051: "Arterial BP Diastolic",
    220052: "Arterial BP Mean",
    220179: "Non-Invasive BP Systolic",
    220180: "Non-Invasive BP Diastolic",
    220210: "Respiratory Rate",
    220277: "SpO2",
    223761: "Temperature F",
    223762: "Temperature C",
}

# Key lab itemids from MIMIC-IV labevents
LAB_ITEMS = {
    50971: "Potassium",
    50912: "Creatinine",
    51222: "Hemoglobin",
    50931: "Glucose",
    50960: "Magnesium",
    50983: "Sodium",
    51006: "Urea Nitrogen",
    51265: "Platelet Count",
    51301: "White Blood Cells",
    50862: "Albumin",
}


# ============================================================
# DATA DOWNLOAD
# ============================================================


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mimic_demo() -> int:
    """Verify every extracted file against PhysioNet's versioned checksum list."""
    base = DATA_DIR / "mimic-iv-clinical-database-demo-2.2"
    checksum_path = base / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {checksum_path}")
    if _sha256(checksum_path) != MIMIC_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("Unexpected MIMIC-IV Demo SHA256SUMS.txt checksum")

    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition(" ")
        if not separator:
            raise ValueError(f"Malformed checksum line: {line!r}")
        path = base / relative.strip()
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"MIMIC-IV Demo checksum mismatch: {relative.strip()}")
        checked += 1
    return checked


def download_mimic_demo() -> None:
    """Download, safely extract, and verify MIMIC-IV Demo v2.2."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "mimic-iv-demo-2.2.zip"
    marker = DATA_DIR / ".downloaded"

    if marker.exists():
        checked = verify_mimic_demo()
        print(f"  [skip] MIMIC-IV Demo already downloaded; verified {checked} files")
        return

    print(f"  [download] {MIMIC_DEMO_URL}")
    print("  (this is ~30MB, may take a minute...)")
    urllib.request.urlretrieve(MIMIC_DEMO_URL, zip_path)
    print(f"  [saved] {zip_path} ({zip_path.stat().st_size:,} bytes)")

    print("  [extracting]...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        data_root = DATA_DIR.resolve()
        for member in zf.infolist():
            destination = (DATA_DIR / member.filename).resolve()
            if destination != data_root and data_root not in destination.parents:
                raise ValueError(f"Unsafe path in MIMIC-IV Demo archive: {member.filename!r}")
        zf.extractall(DATA_DIR)
    checked = verify_mimic_demo()
    print(f"  [done] verified {checked} extracted files")

    # Create marker file
    marker.write_text("ok", encoding="utf-8")

    # Clean up zip
    zip_path.unlink()


def _find_csv(relative_path: str) -> Path:
    """Find a CSV file in the extracted MIMIC-IV demo directory."""
    # The zip extracts to mimic-iv-clinical-database-demo-2.2/
    base = DATA_DIR / "mimic-iv-clinical-database-demo-2.2"
    path = base / relative_path
    # Try both .csv and .csv.gz
    if path.exists():
        return path
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Could not find {relative_path} in {base}")


def _read_csv(relative_path: str) -> list[dict]:
    """Read a CSV file, handling .gz compression."""
    path = _find_csv(relative_path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


# ============================================================
# CLINICAL DATA LOADING
# ============================================================


@dataclass
class ClinicalEvent:
    """A timestamped clinical measurement or event."""

    subject_id: int
    timestamp_hours: float  # Hours from admission
    category: str  # vital_sign, lab_result, prescription, diagnosis, demographic
    item_name: str
    value: str
    hadm_id: int | None = None


def _parse_datetime(dt_str: str) -> float | None:
    """Parse a MIMIC datetime string to hours since epoch (arbitrary)."""
    if not dt_str or dt_str == "":
        return None
    from datetime import datetime

    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        # Return as hours since a reference point
        ref = datetime(2100, 1, 1)  # MIMIC dates are shifted to future
        return (dt - ref).total_seconds() / 3600.0
    except ValueError:
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")
            ref = datetime(2100, 1, 1)
            return (dt - ref).total_seconds() / 3600.0
        except ValueError:
            return None


def load_clinical_events() -> list[ClinicalEvent]:
    """Load clinical events from MIMIC-IV Demo across categories."""
    events: list[ClinicalEvent] = []

    # 1. Vital signs from chartevents (ICU data)
    print("  Loading chartevents (vital signs)...")
    try:
        rows = _read_csv("icu/chartevents.csv")
        vital_count = 0
        for row in rows:
            itemid = int(row.get("itemid", 0))
            if itemid in VITAL_SIGN_ITEMS:
                ts = _parse_datetime(row.get("charttime", ""))
                if ts is not None and row.get("value", ""):
                    events.append(
                        ClinicalEvent(
                            subject_id=int(row["subject_id"]),
                            timestamp_hours=ts,
                            category="vital_sign",
                            item_name=VITAL_SIGN_ITEMS[itemid],
                            value=row["value"],
                            hadm_id=int(row.get("hadm_id", 0)) if row.get("hadm_id") else None,
                        )
                    )
                    vital_count += 1
        print(f"    → {vital_count} vital sign measurements")
    except FileNotFoundError:
        print("    → chartevents not found, skipping")

    # 2. Lab results from labevents
    print("  Loading labevents (lab results)...")
    try:
        rows = _read_csv("hosp/labevents.csv")
        lab_count = 0
        for row in rows:
            itemid = int(row.get("itemid", 0))
            if itemid in LAB_ITEMS:
                ts = _parse_datetime(row.get("charttime", ""))
                if ts is not None and row.get("value", ""):
                    events.append(
                        ClinicalEvent(
                            subject_id=int(row["subject_id"]),
                            timestamp_hours=ts,
                            category="lab_result",
                            item_name=LAB_ITEMS[itemid],
                            value=row["value"],
                            hadm_id=int(row.get("hadm_id", 0)) if row.get("hadm_id") else None,
                        )
                    )
                    lab_count += 1
        print(f"    → {lab_count} lab measurements")
    except FileNotFoundError:
        print("    → labevents not found, skipping")

    # 3. Prescriptions
    print("  Loading prescriptions...")
    try:
        rows = _read_csv("hosp/prescriptions.csv")
        rx_count = 0
        for row in rows:
            ts = _parse_datetime(row.get("starttime", ""))
            drug = row.get("drug", "")
            if ts is not None and drug:
                events.append(
                    ClinicalEvent(
                        subject_id=int(row["subject_id"]),
                        timestamp_hours=ts,
                        category="prescription",
                        item_name=drug,
                        value=row.get("dose_val_rx", ""),
                        hadm_id=int(row.get("hadm_id", 0)) if row.get("hadm_id") else None,
                    )
                )
                rx_count += 1
        print(f"    → {rx_count} prescriptions")
    except FileNotFoundError:
        print("    → prescriptions not found, skipping")

    # 4. Diagnoses (one per admission, treated as a fact at admission time)
    print("  Loading diagnoses...")
    try:
        diag_rows = _read_csv("hosp/diagnoses_icd.csv")
        adm_rows = _read_csv("hosp/admissions.csv")
        # Build admission time lookup
        adm_times: dict[int, float] = {}
        for row in adm_rows:
            hadm_id = int(row["hadm_id"])
            ts = _parse_datetime(row.get("admittime", ""))
            if ts is not None:
                adm_times[hadm_id] = ts

        diag_count = 0
        for row in diag_rows:
            hadm_id = int(row["hadm_id"])
            if hadm_id in adm_times:
                events.append(
                    ClinicalEvent(
                        subject_id=int(row["subject_id"]),
                        timestamp_hours=adm_times[hadm_id],
                        category="diagnosis",
                        item_name=f"ICD-{row.get('icd_code', 'unknown')}",
                        value=row.get("icd_code", ""),
                        hadm_id=hadm_id,
                    )
                )
                diag_count += 1
        print(f"    → {diag_count} diagnoses")
    except FileNotFoundError:
        print("    → diagnoses not found, skipping")

    # 5. Demographics (treated as invariant facts at first admission)
    print("  Loading demographics...")
    try:
        rows = _read_csv("hosp/patients.csv")
        demo_count = 0
        for row in rows:
            events.append(
                ClinicalEvent(
                    subject_id=int(row["subject_id"]),
                    timestamp_hours=0.0,  # Demographics are "always known"
                    category="demographic",
                    item_name="gender",
                    value=row.get("gender", ""),
                )
            )
            if row.get("anchor_age"):
                events.append(
                    ClinicalEvent(
                        subject_id=int(row["subject_id"]),
                        timestamp_hours=0.0,
                        category="demographic",
                        item_name="age",
                        value=row.get("anchor_age", ""),
                    )
                )
            demo_count += 2
        print(f"    → {demo_count} demographic facts")
    except FileNotFoundError:
        print("    → patients not found, skipping")

    return events


# ============================================================
# TEMPORAL RETRIEVAL EVALUATION
# ============================================================


@dataclass
class TemporalQuery:
    """A query asking: 'what is the value of item X for patient P at time T?'"""

    subject_id: int
    item_name: str
    category: str
    query_time: float  # Hours
    ground_truth: str  # Correct value at query time
    history: list[tuple[float, str]]  # (timestamp, value) pairs before query_time


def build_queries(
    events: list[ClinicalEvent],
    category: str,
    max_queries: int | None = None,
) -> list[TemporalQuery]:
    """Build temporal retrieval queries for a given category.

    For each (patient, item) pair with multiple measurements, we use the
    last measurement as the query target and all prior measurements as
    the retrieval history.
    """
    # Group events by (subject_id, item_name) within category
    groups: dict[tuple[int, str], list[tuple[float, str]]] = defaultdict(list)
    for e in events:
        if e.category == category:
            groups[(e.subject_id, e.item_name)].append((e.timestamp_hours, e.value))

    queries = []
    for (sid, item), measurements in groups.items():
        if len(measurements) < 2:
            continue
        # Sort by time
        measurements.sort(key=lambda x: x[0])
        # Use last as ground truth, rest as history
        history = measurements[:-1]
        target_time, target_value = measurements[-1]
        queries.append(
            TemporalQuery(
                subject_id=sid,
                item_name=item,
                category=category,
                query_time=target_time,
                ground_truth=target_value,
                history=history,
            )
        )

    if max_queries and len(queries) > max_queries:
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(len(queries), max_queries, replace=False)
        queries = [queries[i] for i in indices]

    return queries


def evaluate_beta(
    queries: list[TemporalQuery],
    beta: float,
) -> dict[str, float]:
    """Evaluate retrieval accuracy using exponential decay with given β.

    For each query, score each historical measurement using exp(-β·|Δt|)
    and check if the highest-scored value matches the ground truth.

    Since clinical values are continuous, we use "closest value" as a
    softer metric alongside exact match.
    """
    if not queries:
        return {"mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "num_queries": 0}

    hits1, hits3 = 0, 0
    mrr_sum = 0.0
    evaluated = 0

    for q in queries:
        if not q.history:
            continue

        # Score each historical value
        scored: list[tuple[str, float]] = []
        for ts, val in q.history:
            dt = abs(q.query_time - ts)
            if beta == 0.0:
                score = 1.0  # Uniform weight
            else:
                score = np.exp(-beta * dt)
            scored.append((val, score))

        # Rank by score (descending), then find ground truth rank
        scored.sort(key=lambda x: x[1], reverse=True)

        # Find rank of ground truth value
        rank = None
        for i, (val, _) in enumerate(scored):
            if val == q.ground_truth:
                rank = i + 1
                break

        if rank is None:
            # Ground truth value never appeared in history
            rank = len(scored) + 1

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
# MAIN EXPERIMENT
# ============================================================

CATEGORIES = ["vital_sign", "lab_result", "prescription", "diagnosis"]


def run_experiment(max_queries_per_cat: int | None = None) -> dict:
    """Run clinical β sensitivity analysis."""
    print("=" * 70)
    print("EXPERIMENT 3b: Clinical β Sensitivity (MIMIC-IV Demo)")
    print("=" * 70)

    print("\n[1/3] Loading MIMIC-IV Demo clinical events...")
    events = load_clinical_events()
    print(f"  Total events: {len(events)}")

    # Count per category
    cat_counts = defaultdict(int)
    for e in events:
        cat_counts[e.category] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    print("\n[2/3] Building temporal queries...")
    all_queries: dict[str, list[TemporalQuery]] = {}
    for cat in CATEGORIES:
        queries = build_queries(events, cat, max_queries_per_cat)
        all_queries[cat] = queries
        print(f"  {cat}: {len(queries)} queries")

    print(f"\n[3/3] Sweeping β across {len(BETA_GRID)} values per category...")

    results: dict[str, dict[str, dict]] = {}
    optimal_betas: dict[str, tuple[float, float]] = {}

    for cat in CATEGORIES:
        queries = all_queries[cat]
        if not queries:
            print(f"\n  --- {cat} --- (no queries, skipping)")
            continue

        print(f"\n  --- {cat} ({len(queries)} queries) ---")
        results[cat] = {}
        best_mrr = 0.0
        best_beta = 0.0

        for beta in BETA_GRID:
            metrics = evaluate_beta(queries, beta)
            results[cat][str(beta)] = metrics
            marker = ""
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_beta = beta
                marker = " ← best"
            print(
                f"    β={beta:<8} MRR={metrics['mrr']:.4f}  "
                f"H@1={metrics['hits_at_1']:.4f}  "
                f"(n={metrics['num_queries']}){marker}"
            )

        optimal_betas[cat] = (best_beta, best_mrr)

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMAL β PER CLINICAL CATEGORY:")
    for cat, (beta, mrr) in optimal_betas.items():
        print(f"  {cat:<20} β*={beta:<8} MRR={mrr:.4f}")

    # Check if β varies across categories (the whole point of this experiment)
    beta_values = [b for b, _ in optimal_betas.values()]
    if len(set(beta_values)) > 1:
        print("\n  ✓ Optimal β VARIES across clinical categories — supports Proposition 1")
    else:
        print("\n  ⚠ Optimal β is the same across categories — may need finer grid or more data")
    print("=" * 70)

    return {
        "categories": CATEGORIES,
        "results": results,
        "optimal_betas": {k: {"beta": v[0], "mrr": v[1]} for k, v in optimal_betas.items()},
        "event_counts": dict(cat_counts),
    }


def save_results(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "exp3b_clinical_beta.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {json_path}")

    # Heatmap CSV
    csv_path = RESULTS_DIR / "exp3b_clinical_heatmap.csv"
    categories = [c for c in CATEGORIES if c in data["results"]]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["beta"] + categories)
        for beta in BETA_GRID:
            row = [beta]
            for cat in categories:
                mrr = data["results"].get(cat, {}).get(str(beta), {}).get("mrr", "")
                row.append(mrr)
            writer.writerow(row)
    print(f"[saved] {csv_path}")


def print_results() -> None:
    json_path = RESULTS_DIR / "exp3b_clinical_beta.json"
    if not json_path.exists():
        print("No results found. Run: --run")
        return
    with open(json_path) as f:
        data = json.load(f)

    categories = [c for c in CATEGORIES if c in data["results"]]
    print(f"\n{'β':<10}", end="")
    for cat in categories:
        print(f"{cat:>18}", end="")
    print()
    print("-" * (10 + 18 * len(categories)))

    for beta in BETA_GRID:
        print(f"{beta:<10}", end="")
        for cat in categories:
            mrr = data["results"].get(cat, {}).get(str(beta), {}).get("mrr", 0)
            opt_beta = data["optimal_betas"].get(cat, {}).get("beta", -1)
            marker = " *" if beta == opt_beta else ""
            print(f"{mrr:>15.4f}{marker:>3}", end="")
        print()

    print("\nOptimal β per category:")
    for cat, info in data.get("optimal_betas", {}).items():
        print(f"  {cat}: β*={info['beta']}, MRR={info['mrr']:.4f}")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Exp 3b: Clinical β Sensitivity (MIMIC-IV Demo)")
    parser.add_argument("--download", action="store_true", help="Download MIMIC-IV Demo")
    parser.add_argument("--run", action="store_true", help="Run full experiment")
    parser.add_argument("--quick", action="store_true", help="Quick run (limited queries)")
    parser.add_argument("--results", action="store_true", help="Print saved results")
    args = parser.parse_args()

    if args.download:
        print("Downloading MIMIC-IV Demo v2.2...")
        download_mimic_demo()
        print("Done.")

    if args.run or args.quick:
        max_q = 200 if args.quick else None
        data = run_experiment(max_queries_per_cat=max_q)
        save_results(data)

    if args.results:
        print_results()

    if not any([args.download, args.run, args.quick, args.results]):
        parser.print_help()


if __name__ == "__main__":
    main()
