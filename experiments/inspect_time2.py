"""Opt-in inspection of pinned TIME-Lite examples across subsets and tasks."""

from __future__ import annotations

from experiments.reproducibility import TIME_LITE_REVISION


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    train = dataset["train"]
    for index in [0, 100, 300, 500, 700, 900, 1100, 1300, 1548]:
        row = train[index]
        print(f"--- [{index}] {row['Dataset Name']} / {row['Task']} / {row['Setting']} ---")
        print(f"Q: {row['Question'][:200]}")
        print(f"A: {row['Gold Answer']}")
        context = row["Context"][:250] if row["Context"] else "(no context)"
        print(f"C: {context}")
        print()


if __name__ == "__main__":
    main()
