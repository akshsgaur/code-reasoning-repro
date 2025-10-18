# How to Run the Contest 467 Data Collector

## Quick Start

This code is **exactly adapted** from the `data_collection` repository's approach using Selenium.

### Step 1: Install Dependencies

```bash
pip install selenium
```

### Step 2: Install ChromeDriver

**macOS (using Homebrew):**
```bash
brew install chromedriver
```

**Or download manually:**
- Visit https://chromedriver.chromium.org/downloads
- Download the version matching your Chrome browser
- Add to PATH

**Check installation:**
```bash
chromedriver --version
```

### Step 3: Run the Collector

```bash
python3 collect_467_selenium.py
```

You'll be prompted for:
- **LeetCode username**
- **LeetCode password**

The script will:
1. Open Chrome browser (you'll see it)
2. Login to LeetCode
3. Fetch ranking data (pages 1-10)
4. Download Python3 submissions
5. Save everything to `data/collected/weekly-contest-467/`

### Expected Output

```
Logging in to LeetCode...
Login successful!

[1/3] Fetching contest rankings...
  Page 1: 25 users
  Page 2: 25 users
  ...
Total users collected: 250

[2/3] Fetching Python3 submissions...
  Q3519: 1 submissions (total: 1)
  Q3519: 2 submissions (total: 2)
  ...

[3/3] Saving submissions...
  Q3519: 15 submissions
  Q3520: 15 submissions
  ...

Collection complete!
Questions with submissions: 4
Total submissions collected: 60
Output: .../data/collected/weekly-contest-467
```

## How It Works (Adapted from data_collection repo)

### 1. Login with Selenium
```python
# From: data_collection/live_code_bench/leetcode/login.py
browser = webdriver.Chrome()
browser.get("https://leetcode.com/accounts/login/")
# Enter credentials
# Get LEETCODE_SESSION cookie
```

### 2. Fetch Rankings
```python
# From: data_collection/live_code_bench/leetcode/contest_ranking.py
# Use authenticated browser to fetch JSON from API
```

### 3. Get Submissions
```python
# From: data_collection/live_code_bench/leetcode/submission.py
# Use authenticated browser to get submission code
```

## Troubleshooting

**"selenium not found"**
```bash
pip install selenium
```

**"chromedriver not found"**
- Install chromedriver (see Step 2 above)
- Or update the script to use Firefox (like original repo does)

**"Login failed"**
- Check username/password
- Try logging in manually first in a browser
- LeetCode might have CAPTCHA - solve it manually if browser opens

**Browser doesn't open**
- Remove `--headless` from line 26 to see what's happening
- Check Chrome is installed

## Files Created

After running, you'll have:

```
data/collected/weekly-contest-467/
├── rankings.json                          # All ranking data
├── question_3519_submissions.json         # Python3 submissions for Q1
├── question_3520_submissions.json         # Python3 submissions for Q2
├── question_3521_submissions.json         # Python3 submissions for Q3
└── question_3522_submissions.json         # Python3 submissions for Q4
```

## Next Step: Build Dataset

Once data is collected, run:
```bash
python3 build_dataset_467.py
```

This will create the final JSONL dataset from the collected data.

## Code Source

This implementation is adapted from:
- `data_collection/live_code_bench/leetcode/login.py` (Selenium login)
- `data_collection/live_code_bench/leetcode/contest_ranking.py` (Ranking API)
- `data_collection/live_code_bench/leetcode/submission.py` (Submission API)
- `data_collection/live_code_bench/commands/leetcode_get_contest_ranking_by_lang.py` (Main logic)

The approach is **identical** - using Selenium to bypass Cloudflare and authenticate.
