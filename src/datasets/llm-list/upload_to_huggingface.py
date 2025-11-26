#!/usr/bin/env python3
"""Upload LLM-List mutated dataset to the HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    from datasets import Dataset
    from huggingface_hub import login
except ImportError:  # pragma: no cover
    print("Error: huggingface packages not installed")
    print("Run: pip install datasets huggingface-hub")
    raise SystemExit(1)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"Failed to parse JSON on line {line_num}: {exc}")
    return records


def compute_stats(records: List[Dict]) -> Dict[str, str]:
    num_samples = len(records)
    num_mutations = sum(1 for r in records if r.get('has_mutation'))
    mutation_types: Dict[str, int] = {}
    for r in records:
        info = r.get('mutation_info') or {}
        mtype = info.get('mutation_type', 'unknown')
        mutation_types[mtype] = mutation_types.get(mtype, 0) + 1
    return {
        'num_samples': str(num_samples),
        'num_mutations': str(num_mutations),
        'mutation_types': ", ".join(f"{k} ({v})" for k, v in sorted(mutation_types.items())),
    }


def create_dataset_card(repo_id: str, stats: Dict[str, str]) -> str:
    return f"""---
license: mit
language:
- en
tags:
- llm-list
- mutation-testing
- code-generation
---

# LLM-List Mutated Dataset (105 samples)

This dataset contains mutated versions of the LLM-List benchmark programs. Each entry includes:

- The original function header, description, and implementation
- A mutated implementation that produces a different output on the provided test cases
- Input/output pairs for both the original and mutated programs

## Dataset Statistics

- **Samples**: {stats['num_samples']}
- **Mutated samples**: {stats['num_mutations']}
- **Mutation types**: {stats['mutation_types']}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
print(dataset['train'][0])
```

## License

MIT License.
"""


def upload_dataset(dataset: Dataset, repo_id: str, private: bool, token: str | None) -> str:
    if token:
        login(token=token)
    else:  # pragma: no cover
        print("Please login to HuggingFace (or pass --token):")
        login()

    dataset.push_to_hub(repo_id=repo_id, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LLM-List mutated dataset to HuggingFace.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("src/datasets/llm-list/final/final_results_mutated_105.jsonl"),
        help="Path to the JSONL dataset (defaults to the 105-sample file).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target HuggingFace dataset repo (e.g. asgaur/llm-list-mutated-105)",
    )
    parser.add_argument("--token", type=str, default=None, help="HuggingFace access token")
    parser.add_argument("--private", action="store_true", help="Upload as a private dataset")
    args = parser.parse_args()

    records = load_jsonl(args.dataset)
    if not records:
        raise SystemExit(f"No data found in {args.dataset}")

    stats = compute_stats(records)
    dataset = Dataset.from_list(records)
    url = upload_dataset(dataset, args.repo_id, args.private, args.token)

    print("\n" + "=" * 60)
    print("Dataset Card Preview")
    print("=" * 60)
    card = create_dataset_card(args.repo_id, stats)
    print(card)
    print("=" * 60)
    print(f"Dataset uploaded: {url}")


if __name__ == "__main__":
    main()
