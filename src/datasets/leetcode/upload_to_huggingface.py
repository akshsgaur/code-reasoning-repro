#!/usr/bin/env python3
"""
Upload LeetCode dataset to HuggingFace Hub
This makes the dataset easily accessible from Google Colab
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

try:
    from datasets import Dataset
    from huggingface_hub import HfApi, login
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Error: huggingface packages not installed")
    print("Run: pip install datasets huggingface-hub")
    exit(1)


def load_leetcode_dataset(jsonl_path: Path) -> List[Dict]:
    """Load dataset from JSONL file"""
    print(f"Loading dataset from {jsonl_path}...")

    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples")
    return data


def create_huggingface_dataset(data: List[Dict]) -> Dataset:
    """Convert to HuggingFace Dataset format"""
    print("Converting to HuggingFace Dataset format...")

    # Convert to HF dataset
    dataset = Dataset.from_list(data)

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Features: {list(dataset.features.keys())}")

    return dataset


def upload_to_hub(
    dataset: Dataset,
    repo_id: str,
    private: bool = False,
    token: str = None
):
    """
    Upload dataset to HuggingFace Hub

    Args:
        dataset: HuggingFace Dataset
        repo_id: Repository ID (e.g., "username/leetcode-contests-431-467")
        private: Make repository private
        token: HuggingFace API token (or use HF_TOKEN env var)
    """
    print(f"\nUploading to HuggingFace Hub: {repo_id}")
    print(f"Private: {private}")

    # Login to HuggingFace
    if token:
        login(token=token)
    else:
        print("\nPlease login to HuggingFace:")
        login()

    # Upload dataset
    print("\nUploading dataset...")
    dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
    )

    print(f"\n✓ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")

    return f"https://huggingface.co/datasets/{repo_id}"


def create_dataset_card(
    repo_id: str,
    num_samples: int,
    num_questions: int,
    num_contests: int,
    difficulty_breakdown: Dict[str, int]
) -> str:
    """Create README.md for the dataset"""

    card = f"""---
license: mit
task_categories:
- text-generation
- code-generation
language:
- en
tags:
- code
- leetcode
- python
- competitive-programming
size_categories:
- n<1K
---

# LeetCode Weekly Contests 431-467 Dataset

A dataset of **{num_samples} Python solutions** from LeetCode weekly contests 431-467 (January 2025 - September 2025).

## Dataset Description

This dataset contains Python code solutions collected from LeetCode weekly contests. Each entry includes:
- **Problem description** (via question_id)
- **Function code** (standalone, no class wrapper)
- **Test input/output** with correct condition
- **Metadata**: difficulty, contest date, LOC metrics

### Intended Use

- **Code generation** evaluation
- **LLM benchmarking** with pass@k metric
- **Competitive programming** research
- **Code reasoning** studies

## Dataset Statistics

- **Total samples**: {num_samples}
- **Unique questions**: {num_questions}
- **Contests covered**: {num_contests} (weekly-contest-431 to 467)
- **Date range**: January 5, 2025 → September 14, 2025

### Difficulty Distribution

- **Easy**: {difficulty_breakdown.get('easy', 0)} samples ({difficulty_breakdown.get('easy', 0)/num_samples*100:.1f}%)
- **Medium**: {difficulty_breakdown.get('medium', 0)} samples ({difficulty_breakdown.get('medium', 0)/num_samples*100:.1f}%)
- **Hard**: {difficulty_breakdown.get('hard', 0)} samples ({difficulty_breakdown.get('hard', 0)/num_samples*100:.1f}%)

## Dataset Structure

```python
{{
  "id": "contest431_q3702_s0",
  "question_id": 3702,
  "function_name": "maxLength",
  "code": "def maxLength(nums: List[int]) -> int:\\n    ...",
  "input": "maxLength(nums=[1,2,1,2,1,1,1])",
  "output": "5",
  "correct_condition": "maxLength(nums=[1,2,1,2,1,1,1]) == 5",
  "contest_id": "weekly-contest-431",
  "contest_date": "2025-01-05T00:00:00.000",
  "difficulty": "easy",
  "metrics": {{"loc": 17, "coverage": null}},
  "submission_id": 1497983046
}}
```

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

### Evaluate with pass@k

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Example: evaluate one sample
sample = dataset['train'][0]
print(f"Problem: {{sample['id']}}")
print(f"Code:\\n{{sample['code']}}")
print(f"Test: {{sample['correct_condition']}}")
```

### Use in Google Colab

```python
!pip install datasets

from datasets import load_dataset
dataset = load_dataset("{repo_id}")

# Access samples
for i in range(5):
    sample = dataset['train'][i]
    print(f"Problem {{i+1}}: {{sample['function_name']}}")
```

## Dataset Creation

**Collection Method**:
- Solutions scraped from LeetCode using authenticated API
- 3 different solutions per question
- Test cases fetched via GraphQL API

**Preprocessing**:
- Class wrapper removed (converted to standalone functions)
- Solutions with class variables filtered out
- Code executed to verify outputs
- LOC calculated using radon library

**Quality Control**:
- Only executable code included
- Verified test input/output pairs
- Filtered incompatible submissions

## Evaluation

This dataset is designed for evaluation using the **pass@k metric** from the HumanEval paper:

```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```

Where:
- n = total samples generated
- c = correct samples
- k = number of samples to consider

See the [repository]() for evaluation code.

## Limitations

- **Language**: Python only
- **Scope**: Competitive programming problems
- **Coverage**: ~30% of collected solutions filtered (class variables)
- **Test cases**: Single test case per problem

## Citation

```bibtex
@dataset{{leetcode_contests_2025,
  title={{LeetCode Weekly Contests 431-467 Dataset}},
  author={{Your Name}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## License

MIT License - Free to use for research and commercial purposes.

## Contact

For questions or issues, please open an issue on [GitHub]().
"""

    return card


def main():
    parser = argparse.ArgumentParser(
        description='Upload LeetCode dataset to HuggingFace Hub'
    )
    parser.add_argument('--dataset', type=str,
                        default='data/datasets/leetcode_contests_431_467.jsonl',
                        help='Path to dataset JSONL file')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='HuggingFace repository ID (e.g., "username/leetcode-dataset")')
    parser.add_argument('--private', action='store_true',
                        help='Make repository private')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token (optional)')

    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        exit(1)

    # Load data
    data = load_leetcode_dataset(dataset_path)

    # Calculate statistics for dataset card
    num_samples = len(data)
    num_questions = len(set(d['question_id'] for d in data))
    num_contests = len(set(d['contest_id'] for d in data))

    difficulty_breakdown = {}
    for d in data:
        diff = d['difficulty']
        difficulty_breakdown[diff] = difficulty_breakdown.get(diff, 0) + 1

    # Create HF dataset
    dataset = create_huggingface_dataset(data)

    # Upload to hub
    url = upload_to_hub(
        dataset=dataset,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )

    # Create and display dataset card
    print("\n" + "="*60)
    print("Dataset Card (README.md)")
    print("="*60)
    card = create_dataset_card(
        repo_id=args.repo_id,
        num_samples=num_samples,
        num_questions=num_questions,
        num_contests=num_contests,
        difficulty_breakdown=difficulty_breakdown
    )

    print(card)

    print("\n" + "="*60)
    print("UPLOAD COMPLETE!")
    print("="*60)
    print(f"Dataset URL: {url}")
    print(f"\nTo use in Python:")
    print(f'  from datasets import load_dataset')
    print(f'  dataset = load_dataset("{args.repo_id}")')
    print("="*60)


if __name__ == '__main__':
    main()
