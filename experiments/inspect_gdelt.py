"""Opt-in schema/count inspection for the pinned GDELT TFLEX dataset."""

from __future__ import annotations

from experiments.reproducibility import GDELT_REVISION


def main() -> None:
    from datasets import load_dataset

    print("Loading pinned GDELT metadata...")
    metadata_dataset = load_dataset(
        "linxy/GDELT",
        "meta",
        revision=GDELT_REVISION,
    )
    metadata = metadata_dataset["train"][0]
    print("\n=== GDELT TKG Dataset ===")
    for key in [
        "entity_count",
        "relation_count",
        "timestamp_count",
        "train_triples_count",
        "valid_triples_count",
        "test_triples_count",
        "triple_count",
    ]:
        print(f"{key}: {metadata[key]}")

    relation_names = metadata["relation2idx"]["name"]
    relation_ids = metadata["relation2idx"]["id"]
    print(f"\n=== Relations ({metadata['relation_count']}) ===")
    for name, relation_id in zip(relation_names, relation_ids):
        print(f"  {relation_id:3d}: {name}")

    timestamp_names = metadata["timestamp2idx"]["name"]
    print(f"Timestamp range: {timestamp_names[0]} .. {timestamp_names[-1]}")

    print("\n=== 'all' config schema ===")
    all_dataset = load_dataset(
        "linxy/GDELT",
        "all",
        revision=GDELT_REVISION,
    )
    for split_name, split_data in all_dataset.items():
        print(f"  {split_name}: {len(split_data)} rows, features: {split_data.features}")

    print("\n=== 'Pe' config schema ===")
    pe_dataset = load_dataset(
        "linxy/GDELT",
        "Pe",
        revision=GDELT_REVISION,
    )
    for split_name, split_data in pe_dataset.items():
        print(f"  {split_name}: {len(split_data)} rows, features: {split_data.features}")

    query_metadata = metadata["query_meta"]
    print(f"Query names: {query_metadata['query_name']}")
    print(f"Query counts: {query_metadata['queries_count']}")


if __name__ == "__main__":
    main()
