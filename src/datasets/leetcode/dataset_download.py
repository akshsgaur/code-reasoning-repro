from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset


def summarize_split(split_name: str, split: Dataset, out_dir: Path) -> None:
    """Print a short summary and persist metadata/sample rows for one split."""

    print(f"Split: {split_name} | rows: {split.num_rows}")
    feature_strings = {name: repr(feature) for name, feature in split.features.items()}
    for name, feature_repr in feature_strings.items():
        print(f"  - {name}: {feature_repr}")

    sample_size = min(3, split.num_rows)
    if sample_size == 0:
        print("  (empty split)")
    else:
        sample_rows = split.select(range(sample_size))
        for idx, row in enumerate(sample_rows):
            print(f"  Sample {idx}:")
            for key, value in row.items():
                preview = value
                if isinstance(value, str) and len(value) > 120:
                    preview = f"{value[:117]}..."
                print(f"    {key}: {preview}")

        summary: Dict[str, Any] = {
            "split": split_name,
            "num_rows": split.num_rows,
            "features": feature_strings,
        }
        summary_path = out_dir / f"{split_name}_summary.json"
        sample_path = out_dir / f"{split_name}_sample.jsonl"
        summary_path.write_text(json.dumps(summary, indent=2))
        sample_rows.to_json(sample_path.as_posix(), orient="records", lines=True)


def save_leetcode_only(dataset: DatasetDict, out_dir: Path) -> None:
    """Filter for LeetCode problems and persist each split to JSONL."""

    leetcode_dir = out_dir / "leetcode_only"
    leetcode_dir.mkdir(parents=True, exist_ok=True)

    leetcode_dataset = dataset.filter(lambda row: row.get("platform") == "leetcode")

    for split_name, split in leetcode_dataset.items():
        target_path = leetcode_dir / f"{split_name}.jsonl"
        split.to_json(target_path.as_posix(), orient="records", lines=True)
        print(f"Saved {split.num_rows} LeetCode rows to {target_path}")


def main() -> None:
    dataset: DatasetDict = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="release_v2",
    )

    output_dir = Path(__file__).resolve().parent / "lcb_codegen_overview"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split in dataset.items():
        summarize_split(split_name, split, output_dir)

    save_leetcode_only(dataset, output_dir)


if __name__ == "__main__":
    main()
