# Weekly Contest 467 Dataset Creation - Summary

## What I Created

I've adapted the code from the `data_collection` repository to create a complete pipeline for scraping and building dataset entries from LeetCode weekly-contest-467.

### Files Created

1. **`scrape_contest_467.py`** - Contest data scraper
   - Fetches contest rankings and metadata
   - Downloads Python3 submissions
   - Saves structured data to `data/site/weekly-contest-467/`

2. **`build_dataset_467.py`** - Dataset builder
   - Processes scraped submissions
   - Executes code to generate outputs
   - Calculates code metrics (LOC, coverage)
   - Outputs JSONL format matching the paper's specification

3. **`README_contest_467.md`** - Complete documentation
   - Problem: LeetCode blocks unauthenticated requests
   - Solutions: How to get browser cookies
   - Manual and automated approaches
   - Expected output structure

## The Challenge: LeetCode API Blocks Automated Requests

LeetCode's API returns **403 Forbidden** for unauthenticated requests. This is a common anti-scraping measure.

### Solution Options

#### Option 1: Use Browser Cookies (Recommended)
```bash
# 1. Get cookies from your browser (see README)
# 2. Run scraper with authentication:
python3 lc_rankings.py weekly-contest-467 \
  --cookie "YOUR_COOKIE_STRING" \
  --max-pages 10 \
  -o data/site/weekly-contest-467/rankings.json
```

#### Option 2: Use the Original data_collection Repo
```bash
cd data_collection/live_code_bench

# Edit config.py:
# WEEKLY_CONTEST_ID_START = 467
# WEEKLY_CONTEST_ID_END = 467

# Run:
python3 commands/leetcode_get_contest_ranking_by_lang.py
```

#### Option 3: Manual Download
1. Visit https://leetcode.com/contest/api/ranking/weekly-contest-467/?pagination=1
2. Save JSON responses manually
3. Place in `data/site/weekly-contest-467/rankings.json`

## Dataset Entry Format

Following the specification from `ResponseToCMUTeam.docx`:

```json
{
  "id": "contest467_q1234_s0",
  "question_id": 1234,
  "function_name": "solutionFunction",
  "code": "def solutionFunction(...):\\n    ...",
  "input": "solutionFunction(arg1, arg2)",
  "output": "expected_result",
  "correct_condition": "solutionFunction(arg1, arg2) == expected_result",
  "contest_id": "weekly-contest-467",
  "contest_date": "2024-10-13T00:00:00.000",
  "difficulty": "easy",
  "metrics": {
    "coverage": 100.0,
    "loc": 7
  }
}
```

## Complete Workflow

### Step 1: Scrape Contest Data

```bash
# Option A: With cookies
python3 lc_rankings.py weekly-contest-467 --cookie "..." -o data/site/weekly-contest-467/rankings.json

# Option B: Use data_collection repo (if set up with cookies)
cd ~/data_collection/live_code_bench
python3 commands/leetcode_get_contest_ranking_by_lang.py
```

Expected output:
```
data/site/weekly-contest-467/
├── rankings.json                           # Full ranking data
├── index.json                              # Contest metadata
├── top_users.json                          # User rankings
├── question_1234_submissions.json          # Submissions for Q1
├── question_1235_submissions.json          # Submissions for Q2
└── ...
```

### Step 2: Build Dataset Entries

```bash
python3 build_dataset_467.py
```

This will:
1. Load contest data and submissions
2. Extract function names from code
3. Execute code with test inputs to generate outputs
4. Calculate code metrics (LOC using radon)
5. Save to `data/weekly-contest-467.jsonl`

### Step 3: Apply Mutations (Optional)

Using the modified `mutmut` in `mutmut_src/`:

```bash
# For each code sample:
# 1. Save code to temp file
# 2. Run: mutmut run --paths-to-mutate temp_file.py
# 3. Get mutations: mutmut results
# 4. Apply each: mutmut apply <id>
# 5. Backup file and .mutmut-cache between applies
```

## Key Implementation Details

### From ResponseToCMUTeam.docx

1. **BASE_IMPORTS**: Large import block used when executing code
2. **Output Format**: Must use `repr(output)` for proper string formatting
3. **Metrics**:
   - LOC via `radon` library
   - Coverage via `coverage.py` command-line tool
4. **Mutation**: Modified mutmut in `mutmut_src/` directory

### Code Execution

```python
full_code = f"{BASE_IMPORTS}\\n{code}\\noutput = {input}"
namespace = {}
exec(full_code, namespace)
result = repr(namespace['output'])  # Important: use repr()!
```

### Metrics Calculation

```python
from radon.raw import analyze
analysis = analyze(code)
loc = analysis.loc

# Coverage requires running the code with coverage.py
# coverage run --source=. code.py
# coverage json -o coverage.json
```

## Files in This Directory

```
code-reasoning-repro/src/datasets/leetcode/
├── scrape_contest_467.py          # New: Contest 467 scraper
├── build_dataset_467.py            # New: Dataset builder
├── README_contest_467.md           # New: Complete documentation
├── DATASET_467_SUMMARY.md          # New: This file
├── lc_rankings.py                  # Existing: Scraper with cookie support
├── scrape470.py                    # Existing: Example scraper
├── dataset_download.py             # Existing: Download utilities
└── data/
    └── site/
        └── weekly-contest-467/     # Output directory
```

## Next Steps

1. **Get LeetCode cookies** from your browser (see README_contest_467.md)
2. **Run scraper** with authentication
3. **Build dataset** using build_dataset_467.py
4. **Verify output** matches the expected format
5. **Add test cases** by scraping problem details
6. **Calculate coverage** by running tests
7. **Apply mutations** using mutmut_src

## References

- Original methodology: `ResponseToCMUTeam.docx`
- Data collection code: `data_collection/` directory
- Mutation framework: `mutmut_src/` directory
- LiveCodeBench dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite

## Troubleshooting

**Q: Getting 403 Forbidden errors?**
A: You need valid browser cookies. See README_contest_467.md Section "Step 1: Get Your Browser Cookies"

**Q: No submissions found?**
A: Increase `RANKING_PAGE_END` or check if contest data is available

**Q: Code execution fails?**
A: Some code may have dependencies not in BASE_IMPORTS. You can skip these or add imports.

**Q: Where do I get test cases?**
A: Scrape from LeetCode's problem API or contest API (requires separate implementation)

## Contact

For questions about this implementation, refer to:
- ResponseToCMUTeam.docx for methodology
- data_collection repo for working examples
- LeetCode API documentation
