"""Opt-in inspection of the pinned TIME-Lite dataset schema."""

from __future__ import annotations

from collections import Counter

from experiments.reproducibility import TIME_LITE_REVISION


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    train = dataset["train"]

    print(f"Total rows: {len(train)}")
    print(f"Features: {train.features}")
    print()
    for index in range(3):
        print(f"=== Example {index} ===")
        row = train[index]
        for key, value in row.items():
            rendered = str(value)
            if len(rendered) > 300:
                rendered = rendered[:300] + "..."
            print(f"  {key}: {rendered}")
        print()

    for column in ["Level", "Setting", "Task", "Dataset Name"]:
        counts = Counter(train[column])
        print(f"\n{column} distribution:")
        for value, count in counts.most_common():
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
