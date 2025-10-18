# Data Collection for Weekly Contest 467

## Overview
This guide explains how to collect data from LeetCode weekly-contest-467 following the methodology from the `data_collection` repository.

## Problem: LeetCode Blocks Automated Requests

LeetCode's API returns `403 Forbidden` for unauthenticated requests. You need to provide valid browser cookies to scrape the data.

## Solution: Use Browser Cookies

### Step 1: Get Your Browser Cookies

1. Open your browser and navigate to: https://leetcode.com/contest/weekly-contest-467/ranking/
2. Open Developer Tools (F12 or Cmd+Option+I)
3. Go to the Network tab
4. Refresh the page
5. Click on any `ranking` API call
6. In the Headers section, find `Cookie:` under Request Headers
7. Copy the entire cookie string

### Step 2: Run the Scraper with Cookies

Use the `lc_rankings.py` script which supports cookies:

```bash
python3 lc_rankings.py weekly-contest-467 \
  --cookie "PASTE_YOUR_COOKIE_HERE" \
  --max-pages 10 \
  -o data/site/weekly-contest-467/rankings.json
```

### Step 3: Alternative - Manual Data Download

If you have access to a working LeetCode account, you can manually download:

1. **Contest Info**: Visit `https://leetcode.com/contest/api/info/weekly-contest-467/`
2. **Rankings**: Visit `https://leetcode.com/contest/api/ranking/weekly-contest-467/?pagination=1`
3. Save the JSON responses to the `data/site/weekly-contest-467/` directory

## Expected Output Structure

```
data/site/weekly-contest-467/
├── rankings.json          # Full ranking data with submissions
├── index.json             # Contest metadata and questions
└── top_users.json         # Top user details
```

## Data Format

The ranking JSON should contain:
- `contest_slug`: "weekly-contest-467"
- `problems`: Array of problem metadata (id, title, title_slug)
- `users`: Array of user rankings
- `submissions`: Array of submission details per user/problem

## Next Steps

Once you have the ranking data:

1. **Extract Problem IDs**: Parse `problems` array from the ranking data
2. **Filter Python3 Submissions**: From the `submissions` array, filter for `lang: "python3"`
3. **Download Submission Code**: For each submission_id, fetch from:
   - US: `https://leetcode.com/api/submissions/{submission_id}/`
   - CN: `https://leetcode.cn/api/submissions/{submission_id}/`

4. **Build Dataset Entries**: Create JSONL entries following this format:
   ```json
   {
     "id": "sample_0",
     "question_id": 1234,
     "function_name": "functionName",
     "code": "def functionName(...):\\n    ...",
     "input": "functionName(arg1, arg2)",
     "output": "expected_result",
     "correct_condition": "functionName(arg1, arg2) == expected_result",
     "contest_id": "weekly-contest-467",
     "contest_date": "2024-10-13T00:00:00.000",
     "difficulty": "easy",
     "metrics": {
       "coverage": 100.0,
       "loc": 7
     }
   }
   ```

## Automated Approach (from data_collection repo)

If you have the `data_collection` repository set up:

```bash
# 1. Update config.py
cd data_collection/live_code_bench
# Edit config.py:
# WEEKLY_CONTEST_ID_START = 467
# WEEKLY_CONTEST_ID_END = 467

# 2. Run the scraper
python3 commands/leetcode_get_contest_ranking_by_lang.py
```

This will:
- Fetch rankings (pages 1-10)
- Download Python3 submissions
- Save to `data/0.0.1/leetcode/` directory

## Troubleshooting

**403 Forbidden**: Your cookies expired or you're not logged in
**No submissions found**: Increase `RANKING_PAGE_END` in the scraper
**Rate limiting**: Add longer `sleep()` delays between requests

## References

- Original data_collection repo: https://github.com/Naman-ntc/data_collection
- ResponseToCMUTeam.docx: Contains detailed methodology
- mutmut_src/: Modified mutation testing framework
