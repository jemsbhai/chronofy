"""Lightweight, offline validation for the tracked experiment inventory."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from chronofy.decay.exponential import ExponentialDecay
from chronofy.models import TemporalFact
from chronofy.retrieval.filter import EpistemicFilter
from experiments.reproducibility import (
    EVOEXPLORE_REVISION,
    GDELT_REVISION,
    ICEWS14_FILE_SHA256,
    MIMIC_CHECKSUM_MANIFEST_SHA256,
    MINILM_REVISION,
    RANDOM_SEED,
    TIME_LITE_REVISION,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
MANIFEST_PATH = EXPERIMENTS / "manifest.json"
DATASETS_PATH = EXPERIMENTS / "datasets.json"

_LITERAL_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*=\s*" r"['\"][^'\"\n]{8,}['\"]"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.relative_to(ROOT)}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Path escapes repository root: {value!r}") from error
    return candidate


def _validate_source_inventory(manifest: dict[str, Any]) -> int:
    declared_values = manifest.get("tracked_python_sources")
    if not isinstance(declared_values, list) or not declared_values:
        raise ValueError("manifest tracked_python_sources must be a non-empty list")
    if not all(isinstance(value, str) for value in declared_values):
        raise ValueError("manifest tracked_python_sources entries must be strings")

    declared = sorted(declared_values)
    actual = sorted(
        path.relative_to(ROOT).as_posix()
        for path in EXPERIMENTS.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    if declared != actual:
        missing = sorted(set(actual) - set(declared))
        stale = sorted(set(declared) - set(actual))
        raise ValueError(f"Experiment source inventory drift: missing={missing}, stale={stale}")

    for relative in declared:
        path = _repo_path(relative)
        source = path.read_text(encoding="utf-8")
        compile(source, relative, "exec")
        match = _LITERAL_SECRET.search(source)
        if match:
            raise ValueError(f"Possible literal credential in {relative}: {match.group(0)!r}")
    return len(declared)


def _validate_contracts(manifest: dict[str, Any], datasets: dict[str, Any]) -> int:
    if manifest.get("schema_version") != 1 or datasets.get("schema_version") != 1:
        raise ValueError("Unsupported experiment manifest schema")
    if manifest.get("default_seed") != RANDOM_SEED:
        raise ValueError("Manifest seed does not match experiments.reproducibility")

    policies = datasets.get("policy")
    if not isinstance(policies, dict) or any(policies.values()):
        raise ValueError(
            "Dataset policy must keep all raw/restricted/generated artifacts untracked"
        )

    dataset_rows = datasets.get("datasets")
    model_rows = datasets.get("models")
    if not isinstance(dataset_rows, list) or not isinstance(model_rows, list):
        raise ValueError("datasets.json must declare datasets and models lists")

    inputs: dict[str, dict[str, Any]] = {}
    for row in [*dataset_rows, *model_rows]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Every dataset/model entry needs a string id")
        identifier = row["id"]
        if identifier in inputs:
            raise ValueError(f"Duplicate dataset/model id: {identifier}")
        inputs[identifier] = row

    expected_revisions = {
        "icews14-evoexplore": EVOEXPLORE_REVISION,
        "time-lite": TIME_LITE_REVISION,
        "gdelt-tflex": GDELT_REVISION,
        "sentence-transformers/all-MiniLM-L6-v2": MINILM_REVISION,
    }
    for identifier, revision in expected_revisions.items():
        if inputs.get(identifier, {}).get("upstream_revision") != revision:
            raise ValueError(f"Pinned revision drift for {identifier}")
    if inputs["icews14-evoexplore"].get("files_sha256") != ICEWS14_FILE_SHA256:
        raise ValueError("ICEWS checksum declarations have drifted")
    if (
        inputs["mimic-iv-demo-2.2"].get("checksum_manifest_sha256")
        != MIMIC_CHECKSUM_MANIFEST_SHA256
    ):
        raise ValueError("MIMIC checksum declaration has drifted")

    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 8:
        raise ValueError("Manifest must describe the eight canonical experiment entry points")
    ids: set[str] = set()
    for row in experiments:
        if not isinstance(row, dict):
            raise ValueError("Experiment entries must be objects")
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in ids:
            raise ValueError(f"Invalid or duplicate experiment id: {identifier!r}")
        ids.add(identifier)
        entrypoint = _repo_path(row.get("entrypoint"))
        if not entrypoint.is_file():
            raise ValueError(f"Missing entry point for {identifier}")
        commands = row.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"Missing commands for {identifier}")
        if not row.get("output_contract") or not row.get("tolerance") or not row.get("limitations"):
            raise ValueError(f"Incomplete reproducibility contract for {identifier}")
        for input_id in row.get("inputs", []):
            if input_id not in inputs and not str(input_id).startswith("generated "):
                raise ValueError(f"Unknown input {input_id!r} for {identifier}")
    return len(experiments)


def _validate_fixture(manifest: dict[str, Any]) -> int:
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Manifest validation contract is missing")
    fixture_path = _repo_path(validation.get("fixture"))
    if _sha256(fixture_path) != validation.get("fixture_sha256"):
        raise ValueError("Synthetic fixture checksum does not match manifest")

    fixture = _read_object(fixture_path)
    if fixture.get("schema_version") != 1:
        raise ValueError("Unsupported fixture schema")
    reference_time = datetime.fromisoformat(str(fixture["reference_time"]))
    tolerance = float(fixture["absolute_tolerance"])
    beta = float(fixture["beta"])
    decay = ExponentialDecay(
        beta={"synthetic": beta},
        time_unit=str(fixture["time_unit"]),
    )
    epistemic_filter = EpistemicFilter(decay, threshold=float(fixture["threshold"]))

    rows = fixture.get("facts")
    if not isinstance(rows, list):
        raise ValueError("Fixture facts must be a list")
    facts: list[TemporalFact] = []
    expected_inclusion: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Fixture fact entries must be objects")
        identifier = str(row["id"])
        fact = TemporalFact(
            content=identifier,
            timestamp=datetime.fromisoformat(str(row["observed_at"])),
            fact_type="synthetic",
            source_quality=float(row["source_quality"]),
            source="synthetic-fixture",
        )
        actual_score = decay.compute(fact, reference_time)
        expected_score = float(row["expected_score"])
        if not math.isclose(actual_score, expected_score, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Fixture score mismatch for {identifier}: {actual_score} != {expected_score}"
            )
        facts.append(fact)
        expected_inclusion[identifier] = bool(row["expected_included"])

    actual_order = [fact.content for fact in epistemic_filter.filter(facts, reference_time)]
    if actual_order != fixture.get("expected_included_order"):
        raise ValueError(f"Fixture filter order mismatch: {actual_order}")
    for fact in facts:
        if (fact.content in actual_order) != expected_inclusion[fact.content]:
            raise ValueError(f"Fixture inclusion mismatch for {fact.content}")
    if not math.isclose(decay.half_life("synthetic") or 0.0, 1.0, abs_tol=tolerance):
        raise ValueError("Synthetic beta must have a one-day half-life")
    return len(facts)


def _validate_local_datasets(datasets: dict[str, Any]) -> int:
    """Validate acquired bytes when present; clean CI intentionally has none."""
    checked = 0
    rows = datasets.get("datasets", [])
    if not isinstance(rows, list):
        return checked
    by_id = {row.get("id"): row for row in rows if isinstance(row, dict)}

    icews = by_id.get("icews14-evoexplore", {})
    icews_dir = ROOT / str(icews.get("local_path", ""))
    if icews_dir.is_dir():
        checksums = icews.get("files_sha256", {})
        if not isinstance(checksums, dict):
            raise ValueError("ICEWS checksum map is invalid")
        for name, expected in checksums.items():
            path = icews_dir / str(name)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"ICEWS checksum mismatch: {path.relative_to(ROOT)}")
            checked += 1

    mimic = by_id.get("mimic-iv-demo-2.2", {})
    mimic_root = ROOT / str(mimic.get("local_path", ""))
    checksum_relative = mimic.get("checksum_manifest")
    if mimic_root.is_dir() and isinstance(checksum_relative, str):
        checksum_path = mimic_root / checksum_relative
        if _sha256(checksum_path) != mimic.get("checksum_manifest_sha256"):
            raise ValueError("MIMIC-IV Demo checksum manifest mismatch")
        payload_root = checksum_path.parent
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, separator, relative = line.partition(" ")
            if not separator:
                raise ValueError("Malformed MIMIC-IV Demo checksum line")
            path = payload_root / relative.strip()
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"MIMIC-IV Demo checksum mismatch: {relative.strip()}")
            checked += 1
    return checked


def main() -> None:
    manifest = _read_object(MANIFEST_PATH)
    datasets = _read_object(DATASETS_PATH)
    summary = {
        "python_sources": _validate_source_inventory(manifest),
        "experiment_contracts": _validate_contracts(manifest, datasets),
        "synthetic_facts": _validate_fixture(manifest),
        "local_dataset_files_checked": _validate_local_datasets(datasets),
        "network_calls": 0,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
