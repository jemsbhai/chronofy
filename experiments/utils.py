"""Shared utilities for Chronofy experiments."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def timestamped_save(
    data: dict,
    experiment_name: str,
    *,
    results_dir: Path | None = None,
    save_csv: bool = True,
    csv_rows: list[dict] | None = None,
    csv_fields: list[str] | None = None,
) -> tuple[Path, Path | None]:
    """Save experiment results with timestamp in filename.

    Saves to both a 'latest' file (for easy loading) and a timestamped
    local archive file. The archive is not a complete reproducibility bundle;
    record the manifest, code revision, environment, and input checksums too.

    Args:
        data: The results dict to save as JSON.
        experiment_name: Base name, e.g. "exp1_tkg_decay".
        results_dir: Override results directory.
        save_csv: Whether to also save a CSV.
        csv_rows: If save_csv, list of dicts for CSV rows.
        csv_fields: If save_csv, column names.

    Returns:
        (json_path, csv_path) of the latest files.
    """
    rdir = results_dir or RESULTS_DIR
    rdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Archive directory
    archive_dir = rdir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Save latest JSON
    json_latest = rdir / f"{experiment_name}.json"
    with open(json_latest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {json_latest}")

    # Save timestamped JSON
    json_archive = archive_dir / f"{experiment_name}_{ts}.json"
    with open(json_archive, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[archived] {json_archive}")

    # CSV
    csv_latest = None
    if save_csv and csv_rows and csv_fields:
        csv_latest = rdir / f"{experiment_name}.csv"
        _write_csv(csv_latest, csv_fields, csv_rows)
        print(f"[saved] {csv_latest}")

        csv_archive = archive_dir / f"{experiment_name}_{ts}.csv"
        _write_csv(csv_archive, csv_fields, csv_rows)
        print(f"[archived] {csv_archive}")

    return json_latest, csv_latest


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
