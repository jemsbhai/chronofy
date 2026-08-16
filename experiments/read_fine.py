"""Summarize an ignored local fine-grained GDELT result artifact."""

from __future__ import annotations

import json
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parent / "results" / "exp4_gdelt_decay_fine.json"


def main() -> None:
    with RESULT_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    betas = []
    for name, relation_data in sorted(data["per_relation_results"].items()):
        betas.append(relation_data["best_beta"])
        print(
            f"{name:40s} best_beta={relation_data['best_beta']:4.1f}  "
            f"best_mrr={relation_data['best_mrr']:.4f}  "
            f"static={relation_data['static']['mrr']:.4f}"
        )
    print(f"\nUnique betas: {sorted(set(betas))}")
    print(f"Count: {len(set(betas))} distinct values")


if __name__ == "__main__":
    main()
