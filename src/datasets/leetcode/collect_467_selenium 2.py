#!/usr/bin/env python3
"""
Collect data for weekly-contest-467 using Selenium
EXACT adaptation from data_collection repo:
- data_collection/live_code_bench/leetcode/login.py (for Selenium login)
- data_collection/live_code_bench/commands/leetcode_get_contest_ranking_by_lang.py (for scraping logic)
"""

import json
from pathlib import Path
from time import sleep

# Selenium imports (matching data_collection/live_code_bench/leetcode/login.py)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions

# Configuration
CONTEST_ID = "weekly-contest-467"
RANKING_PAGE_START = 1
RANKING_PAGE_END = 10
LANGS = ["python3"]
OUTPUT_DIR = Path(__file__).parent / "data" / "collected" / CONTEST_ID


def login_leetcode(username: str, password: str):
    """
    Login to LeetCode using Selenium
    Adapted from: data_collection/live_code_bench/leetcode/login.py:16
    """
    print(f"Logging in as {username}...")

    options = ChromeOptions()
    # Comment out headless for debugging if needed
    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    browser = webdriver.Chrome(options=options)
    browser.get("https://leetcode.com/accounts/login/")

    # Wait for page load and solve Cloudflare challenge
    sleep(10)  # Increased from 5 to handle Cloudflare

    # Login
    browser.find_element(By.ID, "id_login").send_keys(username)
    browser.find_element(By.ID, "id_password").send_keys(password)
    browser.find_element(By.ID, "signin_btn").click()

    # Wait for login to complete
    sleep(10)

    # Get cookies
    cookies = browser.get_cookies()

    # Find LEETCODE_SESSION cookie
    leetcode_session_cookies = [
        cookie for cookie in cookies if cookie["name"] == "LEETCODE_SESSION"
    ]

    if not leetcode_session_cookies:
        browser.quit()
        raise Exception("Login failed! No LEETCODE_SESSION cookie found")

    print("Login successful!")

    # Return browser and session cookie
    return browser, leetcode_session_cookies[0]["value"]


def fetch_with_browser(browser, url):
    """Fetch URL using authenticated browser session"""
    browser.get(url)
    sleep(2)

    # Get the JSON from the page
    pre = browser.find_element(By.TAG_NAME, "pre")
    return json.loads(pre.text)


def get_contest_ranking(browser, contest_id: str, ranking_page: int):
    """
    Fetch contest ranking using Selenium
    Adapted from: data_collection/live_code_bench/leetcode/contest_ranking.py:92
    """
    url = f'https://leetcode.com/contest/api/ranking/{contest_id}/?pagination={ranking_page}'
    print(f"Fetching page {ranking_page}...")
    return fetch_with_browser(browser, url)


def get_submission(browser, submission_id: int, data_region: str):
    """
    Fetch submission using Selenium
    Adapted from: data_collection/live_code_bench/leetcode/submission.py:61
    """
    if data_region == "CN":
        url = f"https://leetcode.cn/api/submissions/{submission_id}/"
    else:
        url = f"https://leetcode.com/api/submissions/{submission_id}/"

    print(f"  Fetching submission {submission_id}")
    return fetch_with_browser(browser, url)


def save_json(data, filename):
    """Save data to JSON file"""
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def main():
    """
    Main logic adapted from:
    data_collection/live_code_bench/commands/leetcode_get_contest_ranking_by_lang.py:10
    """
    print(f"Collecting data for {CONTEST_ID}")
    print("=" * 60)

    # Get credentials
    print("\nEnter your LeetCode credentials:")
    print("(These are used ONLY for logging into LeetCode via Selenium)")
    print("(Your password is NOT stored anywhere)")
    username = input("Username: ").strip()
    password = input("Password: ").strip()

    if not username or not password:
        print("Error: Username and password required!")
        return

    # Login with Selenium
    print("\n[0/3] Logging in to LeetCode...")
    try:
        browser, session_cookie = login_leetcode(username, password)
    except Exception as e:
        print(f"Login failed: {e}")
        return

    try:
        # Step 1: Get contest rankings
        print("\n[1/3] Fetching contest rankings...")
        all_rankings = {
            "contest_id": CONTEST_ID,
            "is_past": None,
            "submissions": [],
            "questions": [],
            "total_rank": [],
            "user_num": 0,
        }

        for ranking_page in range(RANKING_PAGE_START, RANKING_PAGE_END + 1):
            try:
                sleep(1)
                page_result = get_contest_ranking(browser, CONTEST_ID, ranking_page)

                if ranking_page == 1:
                    all_rankings["is_past"] = page_result.get("is_past")
                    all_rankings["questions"] = page_result.get("questions", [])
                    all_rankings["user_num"] = page_result.get("user_num", 0)

                all_rankings["total_rank"].extend(page_result.get("total_rank", []))
                all_rankings["submissions"].extend(page_result.get("submissions", []))

                print(f"  Page {ranking_page}: {len(page_result.get('total_rank', []))} users")

            except Exception as e:
                print(f"  Error on page {ranking_page}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Save rankings
        save_json(all_rankings, "rankings.json")
        print(f"Total users collected: {len(all_rankings['total_rank'])}")
        print(f"Total questions: {len(all_rankings['questions'])}")

        # Step 2: Get Python3 submissions
        print("\n[2/3] Fetching Python3 submissions...")
        all_submissions = {}  # question_id -> list of submissions

        submission_count = 0
        for submission_dict in all_rankings["submissions"]:
            for question_id_str, submission_info in submission_dict.items():
                question_id = int(question_id_str)

                # Filter by language
                lang = submission_info.get("lang")
                if lang is None or lang not in LANGS:
                    continue

                submission_id = submission_info.get("submission_id")
                data_region = submission_info.get("data_region", "US")

                try:
                    # Initialize question entry
                    if question_id not in all_submissions:
                        all_submissions[question_id] = []

                    # Stop if we have enough for this question
                    if len(all_submissions[question_id]) >= 15:
                        continue

                    # Fetch submission code
                    sleep(1)
                    submission_data = get_submission(browser, submission_id, data_region)

                    # Check language again
                    if submission_data.get("lang") not in LANGS:
                        continue

                    # Add submission
                    all_submissions[question_id].append({
                        "submission_id": submission_id,
                        "code": submission_data.get("code"),
                        "lang": submission_data.get("lang"),
                        "question_id": question_id,
                        "contest_submission": submission_data.get("contest_submission"),
                    })

                    submission_count += 1
                    print(f"  Q{question_id}: {len(all_submissions[question_id])} submissions (total: {submission_count})")

                except Exception as e:
                    print(f"  Error fetching submission {submission_id}: {e}")
                    continue

        # Save submissions by question
        print("\n[3/3] Saving submissions...")
        for question_id, submissions in all_submissions.items():
            save_json(submissions, f"question_{question_id}_submissions.json")
            print(f"  Q{question_id}: {len(submissions)} submissions")

        print("\n" + "=" * 60)
        print(f"Collection complete!")
        print(f"Questions with submissions: {len(all_submissions)}")
        print(f"Total submissions collected: {submission_count}")
        print(f"Output: {OUTPUT_DIR}")

    finally:
        browser.quit()
        print("\nBrowser closed.")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LeetCode Contest Data Collector (with Selenium)")
    print("Adapted from: data_collection/live_code_bench/leetcode/")
    print("=" * 60)
    main()
