#!/usr/bin/env python3
"""Upload the cleaned LeetCode mutated dataset to HuggingFace."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - network tooling
    from datasets import Dataset
    from huggingface_hub import login
except ImportError:  # pragma: no cover
    raise SystemExit("Install huggingface-hub and datasets: pip install datasets huggingface-hub")


@dataclass
class DatasetStats:
    num_samples: int
    mutated_samples: int
    difficulty_breakdown: Dict[str, int]


def _read_blocks(path: Path) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for idx, block in enumerate(_read_blocks(path), 1):
        try:
            records.append(json.loads(block))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse record #{idx} in {path}: {exc}") from exc
    return records


def compute_stats(records: List[Dict]) -> DatasetStats:
    difficulty_breakdown: Dict[str, int] = {}
    mutated_samples = 0
    for rec in records:
        difficulty = (rec.get("difficulty") or "unknown").lower()
        difficulty_breakdown[difficulty] = difficulty_breakdown.get(difficulty, 0) + 1
        if rec.get("has_mutation") and rec.get("mutated_code"):
            mutated_samples += 1
    return DatasetStats(
        num_samples=len(records),
        mutated_samples=mutated_samples,
        difficulty_breakdown=difficulty_breakdown,
    )


def create_card(repo_id: str, stats: DatasetStats) -> str:
    easy = stats.difficulty_breakdown.get("easy", 0)
    medium = stats.difficulty_breakdown.get("medium", 0)
    hard = stats.difficulty_breakdown.get("hard", 0)
    return f"""---
license: mit
tags:
- leetcode
- python
- code-reasoning
- mutation-testing
- benchmark
---

# LeetCode Contests 431-467 (Mutated)

Mutation-augmented Python solutions for LeetCode weekly contests 431 through 467. Each entry contains:

- Original contest submission (function, inputs, outputs, metadata)
- Automatically generated mutated implementation that changes the output
- `has_mutation`, `mutated_output`, and mutation provenance metadata

## Statistics

- Samples: {stats.num_samples}
- Mutated samples: {stats.mutated_samples}
- Difficulty breakdown → easy: {easy}, medium: {medium}, hard: {hard}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset['train'][0]['code'])
print(dataset['train'][0]['mutated_code'])
```

MIT License.
"""


def upload_dataset(records: List[Dict], repo_id: str, private: bool, token: Optional[str]) -> str:
    if token:
        login(token=token)
    else:  # pragma: no cover
        login()
    ds = Dataset.from_list(records)
    ds.push_to_hub(repo_id=repo_id, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the mutated LeetCode dataset to HuggingFace")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("src/datasets/leetcode/data/datasets/leetcode_contests_431_467_mutated.jsonl"),
        help="Path to the cleaned JSONL dataset",
    )
    parser.add_argument("--repo-id", required=True, help="Destination dataset repo (e.g. asgaur/leetcode-mutations-431-467)")
    parser.add_argument("--token", default=None, help="HuggingFace access token")
    parser.add_argument("--private", action="store_true", help="Create a private dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset)
    if not records:
        raise SystemExit(f"{args.dataset} is empty")

    stats = compute_stats(records)
    url = upload_dataset(records, args.repo_id, args.private, args.token)
    card = create_card(args.repo_id, stats)

    print("\n" + "=" * 60)
    print("Dataset card (copy into README.md on HuggingFace):")
    print("=" * 60)
    print(card)
    print("=" * 60)
    print(f"Dataset uploaded: {url}")


if __name__ == "__main__":
    main()
