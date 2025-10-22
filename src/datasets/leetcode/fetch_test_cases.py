#!/usr/bin/env python3
"""
Fetch test cases for collected contests
This script fetches problem descriptions and test cases that may have been missed
during initial collection.
"""

import json
import argparse
from pathlib import Path
from time import sleep
from typing import Dict, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions

BASE_DATA_DIR = Path(__file__).parent / "data" / "collected"


def login_leetcode(username: str, password: str):
    """Login to LeetCode using Selenium"""
    print(f"Logging in as {username}...")

    options = ChromeOptions()
    # Comment out headless for debugging
    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    browser = webdriver.Chrome(options=options)
    browser.get("https://leetcode.com/accounts/login/")

    # Wait for page load and Cloudflare
    sleep(10)

    # Login
    browser.find_element(By.ID, "id_login").send_keys(username)
    browser.find_element(By.ID, "id_password").send_keys(password)
    browser.find_element(By.ID, "signin_btn").click()

    sleep(10)

    # Check for session cookie
    cookies = browser.get_cookies()
    leetcode_session_cookies = [
        cookie for cookie in cookies if cookie["name"] == "LEETCODE_SESSION"
    ]

    if not leetcode_session_cookies:
        browser.quit()
        raise Exception("Login failed! No LEETCODE_SESSION cookie found")

    print("Login successful!")
    return browser, leetcode_session_cookies[0]["value"]


def get_question_slug_from_id(browser, question_id: int) -> Optional[str]:
    """Get question slug from question ID using GraphQL"""
    try:
        script = f"""
        return fetch('https://leetcode.com/graphql', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                query: `query {{ question(questionId: "{question_id}") {{ titleSlug }} }}`
            }})
        }}).then(r => r.json());
        """

        result = browser.execute_script(script)
        sleep(1)

        if result and "data" in result and "question" in result["data"]:
            return result["data"]["question"]["titleSlug"]

        return None

    except Exception as e:
        print(f"  Error getting slug for Q{question_id}: {e}")
        return None


def get_problem_data(browser, question_slug: str) -> Optional[Dict]:
    """Fetch problem description and test cases using GraphQL"""
    try:
        graphql_query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                title
                content
                difficulty
                exampleTestcases
                sampleTestCase
            }
        }
        """

        # Clean query for JavaScript
        clean_query = graphql_query.replace('\n', ' ').replace('"', '\\"')

        script = f"""
        return fetch('https://leetcode.com/graphql', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                query: "{clean_query}",
                variables: {{ titleSlug: "{question_slug}" }}
            }})
        }}).then(r => r.json());
        """

        result = browser.execute_script(script)
        sleep(2)

        if result and "data" in result and "question" in result["data"]:
            return result["data"]["question"]

        return None

    except Exception as e:
        print(f"  Error fetching problem data for {question_slug}: {e}")
        return None


def fetch_test_cases_for_contest(browser, contest_num: int):
    """Fetch test cases for a single contest"""
    contest_id = f"weekly-contest-{contest_num}"
    contest_dir = BASE_DATA_DIR / contest_id

    print(f"\n{'='*60}")
    print(f"Fetching test cases for {contest_id}")
    print(f"{'='*60}")

    if not contest_dir.exists():
        print(f"  ERROR: Contest directory not found: {contest_dir}")
        return False

    # Load rankings to get question IDs
    rankings_file = contest_dir / "rankings.json"
    if not rankings_file.exists():
        print(f"  ERROR: rankings.json not found")
        return False

    with open(rankings_file, 'r') as f:
        rankings = json.load(f)

    questions = rankings.get("questions", [])
    print(f"  Found {len(questions)} questions")

    problems_data = {}

    for q in questions:
        question_id = q.get("question_id")
        question_slug = q.get("title_slug")

        if not question_slug:
            print(f"  Q{question_id}: Getting slug from ID...")
            question_slug = get_question_slug_from_id(browser, question_id)

        if not question_slug:
            print(f"  Q{question_id}: ERROR - Could not get slug")
            continue

        print(f"  Q{question_id}: Fetching {question_slug}...")
        problem_data = get_problem_data(browser, question_slug)

        if problem_data:
            problems_data[question_id] = {
                "question_id": question_id,
                "title": problem_data.get("title"),
                "content": problem_data.get("content"),
                "difficulty": problem_data.get("difficulty"),
                "exampleTestcases": problem_data.get("exampleTestcases"),
                "sampleTestCase": problem_data.get("sampleTestCase"),
                "title_slug": question_slug,
            }
            print(f"    ✓ Got test cases")
        else:
            print(f"    ✗ Failed to get data")

    # Save problems.json
    if problems_data:
        output_file = contest_dir / "problems.json"
        with open(output_file, 'w') as f:
            json.dump(problems_data, f, indent=2)
        print(f"\n  Saved test cases for {len(problems_data)} questions to {output_file}")
        return True
    else:
        print(f"\n  No test cases fetched")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fetch test cases for collected contests')
    parser.add_argument('--start', type=int, default=431,
                        help='First contest to process')
    parser.add_argument('--end', type=int, default=467,
                        help='Last contest to process')
    parser.add_argument('--contests', type=str, default=None,
                        help='Specific contests (comma-separated, e.g., "431,467")')
    parser.add_argument('--username', type=str, default=None,
                        help='LeetCode username')
    parser.add_argument('--password', type=str, default=None,
                        help='LeetCode password')

    args = parser.parse_args()

    # Get credentials
    username = args.username
    password = args.password

    if not username:
        print("\nEnter your LeetCode credentials:")
        username = input("Username: ").strip()

    if not password:
        import getpass
        password = getpass.getpass("Password: ")

    if not username or not password:
        print("Error: Username and password required!")
        return

    # Determine which contests
    if args.contests:
        contest_numbers = [int(x.strip()) for x in args.contests.split(',')]
    else:
        contest_numbers = list(range(args.start, args.end + 1))

    print(f"\n{'='*60}")
    print(f"LeetCode Test Case Fetcher")
    print(f"{'='*60}")
    print(f"Contests: {len(contest_numbers)}")
    print(f"{'='*60}")

    # Login
    print("\nLogging in to LeetCode...")
    try:
        browser, session_cookie = login_leetcode(username, password)
    except Exception as e:
        print(f"Login failed: {e}")
        return

    try:
        successful = 0
        failed = 0

        for contest_num in contest_numbers:
            success = fetch_test_cases_for_contest(browser, contest_num)
            if success:
                successful += 1
            else:
                failed += 1

            if contest_num != contest_numbers[-1]:
                sleep(3)

        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(contest_numbers)}")
        print(f"Failed: {failed}/{len(contest_numbers)}")
        print(f"{'='*60}")

    finally:
        browser.quit()
        print("\nBrowser closed.")


if __name__ == '__main__':
    main()
