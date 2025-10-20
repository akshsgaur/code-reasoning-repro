# LeetCode Contest Dataset Collection Pipeline

This pipeline collects data from LeetCode weekly contests (431-467) and builds a dataset following the format used in code reasoning research.

## Overview

The pipeline consists of two main scripts:

1. **`collect_contests.py`** - Scrapes contest data from LeetCode
   - Collects 3 different solutions per question
   - Fetches problem descriptions and test cases
   - Uses Selenium to bypass Cloudflare protection

2. **`build_dataset.py`** - Builds JSONL dataset from collected data
   - Executes code to generate outputs
   - Calculates metrics (LOC, coverage)
   - Creates proper input/output/correct_condition format

## Dataset Format

Each sample in the dataset has the following structure:

```json
{
  "id": "contest467_q3997_s0",
  "question_id": 3997,
  "function_name": "maxSum",
  "code": "class Solution:\n    def maxSum(self, nums, k):\n        ...",
  "input": "maxSum(nums = [1,2,3,4], k = 2)",
  "output": "10",
  "correct_condition": "maxSum(nums = [1,2,3,4], k = 2) == 10",
  "contest_id": "weekly-contest-467",
  "contest_date": "2024-10-13T00:00:00.000",
  "difficulty": "easy",
  "metrics": {
    "coverage": null,
    "loc": 15
  },
  "submission_id": 1769944892
}
```

## Prerequisites

### Python Dependencies

```bash
pip install selenium radon
```

### Chrome WebDriver

The Selenium scripts require Chrome and ChromeDriver:

```bash
# macOS
brew install chromedriver

# Linux
apt-get install chromium-chromedriver

# Or download from: https://chromedriver.chromium.org/
```

### LeetCode Account

You need a LeetCode account to collect contest data.

## Usage

### Step 1: Collect Contest Data

Collect all contests (431-467):

```bash
python3 collect_contests.py --username YOUR_USERNAME --password YOUR_PASSWORD
```

Collect specific contests:

```bash
python3 collect_contests.py --contests "431,432,467"
```

Collect a range:

```bash
python3 collect_contests.py --start 431 --end 440
```

**Output:** Data saved to `data/collected/weekly-contest-{NUM}/`
- `rankings.json` - Contest rankings and metadata
- `question_{ID}_submissions.json` - 3 Python3 solutions per question
- `problems.json` - Problem descriptions and test cases

### Step 2: Build Dataset

Build dataset from all collected contests:

```bash
python3 build_dataset.py
```

Build from specific contests:

```bash
python3 build_dataset.py --contests "431,432,467"
```

Build separate files per contest:

```bash
python3 build_dataset.py --separate
```

**Output:** JSONL dataset files in `data/datasets/`

## Data Collection Details

### Contests Collected

- **Range:** Weekly contests 431-467
- **Period:** January 2024 - October 2024
- **Solutions per question:** 3 (configurable via `SOLUTIONS_PER_QUESTION`)

### What Gets Collected

For each contest:
1. **Rankings:** Top users and their performance
2. **Submissions:** 3 Python3 solutions per question
3. **Problems:** Descriptions, test cases, difficulty
4. **Metadata:** Contest date, question IDs, credits

### Data Structure

```
data/
├── collected/
│   ├── weekly-contest-431/
│   │   ├── rankings.json
│   │   ├── problems.json
│   │   ├── question_3873_submissions.json
│   │   ├── question_3997_submissions.json
│   │   └── ...
│   ├── weekly-contest-432/
│   └── ...
└── datasets/
    ├── leetcode_contests_431_467.jsonl
    ├── weekly-contest-431.jsonl (if --separate)
    └── ...
```

## Dataset Building Details

### Code Execution

The builder executes each solution with test cases to generate outputs:

1. Combines `BASE_IMPORTS` + solution code + test input
2. Executes in isolated namespace
3. Captures output using `repr()` (important for strings!)
4. Builds `correct_condition` for verification

### BASE_IMPORTS

The following imports are added when executing code (but NOT included in the dataset):

```python
from itertools import *
from collections import *
from math import *
from bisect import *
from heapq import *
import numpy as np
import pandas as pd
# ... and more (see build_dataset.py)
```

### Metrics Calculation

- **LOC (Lines of Code):** Calculated using `radon` library
  - Falls back to simple line counting if radon not installed
- **Coverage:** Not calculated (would require running full test suites)
  - Set to `null` in the dataset

### Contest Date Calculation

Contest dates are calculated based on:
- Weekly Contest 431: January 7, 2024 (Sunday)
- Contests happen every Sunday (7-day intervals)

## Troubleshooting

### Login Issues

If login fails:
1. Check your LeetCode username/password
2. Try running without `--headless` mode (comment out in code)
3. Wait for Cloudflare challenge to complete (browser will open)
4. Check if LeetCode has rate-limited your IP

### Missing Test Cases

Some problems may not have test cases fetched:
- The builder creates placeholder entries with `input="functionName()"` and `output="None"`
- You can manually add test cases by editing `problems.json`

### Code Execution Failures

If code execution fails for some solutions:
- Check if solution requires additional imports
- Verify BASE_IMPORTS includes all necessary libraries
- Some solutions may have bugs (these are real contest submissions!)

## Statistics

Expected dataset size (approximate):

- **Contests:** 37 (weekly-contest-431 to 467)
- **Questions per contest:** ~4
- **Solutions per question:** 3
- **Total samples:** ~444 samples

Difficulty distribution:
- Easy: ~30%
- Medium: ~40%
- Hard: ~30%

## Examples

### Example 1: Collect and Build Contest 467

```bash
# Collect data
python3 collect_contests.py --contests "467" --username YOUR_USER

# Build dataset
python3 build_dataset.py --contests "467" --output contest467.jsonl
```

### Example 2: Process All Contests

```bash
# Collect all (takes ~2-3 hours)
python3 collect_contests.py --start 431 --end 467

# Build complete dataset
python3 build_dataset.py --start 431 --end 467
```

### Example 3: Sample Dataset Entry

```python
import json

# Read dataset
with open('data/datasets/leetcode_contests_431_467.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question_id']}")
        print(f"Function: {sample['function_name']}")
        print(f"Difficulty: {sample['difficulty']}")
        print(f"Correct: {sample['correct_condition']}")
        break
```

## Notes

### Rate Limiting

- The scraper includes `sleep()` calls to avoid rate limiting
- Default: 1-2 seconds between requests
- Increase if you encounter 429 errors

### Selenium vs. Curl

- Original `scrape_contest_467.py` uses curl with cookies
- `collect_contests.py` uses Selenium (more reliable, bypasses Cloudflare)
- Selenium is slower but more robust

### Privacy

- Your LeetCode credentials are used ONLY for authentication
- Passwords are never stored or logged
- Use `--username` and `--password` flags or enter interactively

## References

- **LiveCodeBench Paper:** 2403.07974v2.pdf
  - Methodology for code generation benchmarks
  - Dataset format and evaluation scenarios

- **Original Collection Code:** `data_collection/live_code_bench/`
  - Adapted from research code repository

## License

This dataset collection pipeline is for research purposes only. LeetCode contest problems and solutions are property of LeetCode LLC.
