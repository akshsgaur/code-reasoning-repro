#!/usr/bin/env python3
"""
Generalized LeetCode contest data collector
Collects data for weekly contests 431-467 using Selenium
- Scrapes 3 different solutions per question
- Fetches problem descriptions and test cases
- Saves data for building final dataset
"""

import json
import argparse
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import NoSuchElementException

# Configuration
FIRST_WEEKLY_CONTEST = 431
LAST_WEEKLY_CONTEST = 467
SOLUTIONS_PER_QUESTION = 3  # Changed from 15 to 3
RANKING_PAGES = 10
LANGS = ["python3"]
BASE_OUTPUT_DIR = Path(__file__).parent / "data" / "collected"


def login_leetcode(username: str, password: str):
    """
    Login to LeetCode using Selenium
    Returns browser and session cookie
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
    sleep(10)

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

    return browser, leetcode_session_cookies[0]["value"]


def fetch_json_api(browser, url):
    """Fetch JSON from API endpoint using authenticated browser"""
    browser.get(url)
    sleep(2)

    # Get the JSON from the page
    pre = browser.find_element(By.TAG_NAME, "pre")
    return json.loads(pre.text)


def get_contest_ranking(browser, contest_id: str, ranking_page: int):
    """Fetch contest ranking page"""
    url = f'https://leetcode.com/contest/api/ranking/{contest_id}/?pagination={ranking_page}'
    print(f"  Fetching ranking page {ranking_page}...")
    return fetch_json_api(browser, url)


def get_submission(browser, submission_id: int, data_region: str):
    """Fetch submission code"""
    if data_region == "CN":
        url = f"https://leetcode.cn/api/submissions/{submission_id}/"
    else:
        url = f"https://leetcode.com/api/submissions/{submission_id}/"

    return fetch_json_api(browser, url)


def get_problem_description(browser, question_slug: str) -> Optional[Dict]:
    """
    Fetch problem description and test cases using GraphQL API
    """
    try:
        # Navigate to problem page
        url = f"https://leetcode.com/problems/{question_slug}/"
        browser.get(url)
        sleep(3)

        # Execute GraphQL query to get problem data
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

        # Use browser console to execute GraphQL
        script = f"""
        return fetch('https://leetcode.com/graphql', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{
                query: `{graphql_query}`,
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
        print(f"    Warning: Could not fetch problem description for {question_slug}: {e}")
        return None


def get_question_slug_from_id(browser, question_id: int) -> Optional[str]:
    """
    Get question slug (URL-friendly name) from question ID
    """
    try:
        # Use GraphQL to get slug from ID
        script = f"""
        return fetch('https://leetcode.com/graphql', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify({{
                query: `query {{ question(questionId: "{question_id}") {{ titleSlug }} }}`,
            }})
        }}).then(r => r.json());
        """

        result = browser.execute_script(script)
        sleep(1)

        if result and "data" in result and "question" in result["data"]:
            return result["data"]["question"]["titleSlug"]

        return None

    except Exception as e:
        print(f"    Warning: Could not get slug for question {question_id}: {e}")
        return None


def save_json(data, filepath):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file if it exists (to avoid permission issues)
    if filepath.exists():
        try:
            filepath.unlink()
        except Exception as e:
            print(f"    Warning: Could not remove existing file {filepath}: {e}")

    # Write new file
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"    ERROR: Could not save to {filepath}: {e}")
        # Try alternative approach: write to temp file first
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        try:
            json.dump(data, temp_file, indent=2)
            temp_file.close()
            # Move temp file to target
            import shutil
            shutil.move(temp_file.name, str(filepath))
        except Exception as e2:
            print(f"    ERROR: Could not save even with temp file: {e2}")
            raise


def collect_contest(browser, contest_num: int):
    """
    Collect data for a single contest
    Returns True if successful, False otherwise
    """
    contest_id = f"weekly-contest-{contest_num}"
    output_dir = BASE_OUTPUT_DIR / contest_id

    print(f"\n{'='*60}")
    print(f"Collecting {contest_id}")
    print(f"{'='*60}")

    try:
        # Step 1: Get contest rankings
        print("\n[1/4] Fetching contest rankings...")
        all_rankings = {
            "contest_id": contest_id,
            "is_past": None,
            "submissions": [],
            "questions": [],
            "total_rank": [],
            "user_num": 0,
        }

        for ranking_page in range(1, RANKING_PAGES + 1):
            try:
                sleep(1)
                page_result = get_contest_ranking(browser, contest_id, ranking_page)

                if ranking_page == 1:
                    all_rankings["is_past"] = page_result.get("is_past")
                    all_rankings["questions"] = page_result.get("questions", [])
                    all_rankings["user_num"] = page_result.get("user_num", 0)

                all_rankings["total_rank"].extend(page_result.get("total_rank", []))
                all_rankings["submissions"].extend(page_result.get("submissions", []))

                print(f"    Page {ranking_page}: {len(page_result.get('total_rank', []))} users")

            except Exception as e:
                print(f"    Error on page {ranking_page}: {e}")
                break

        # Save rankings
        save_json(all_rankings, output_dir / "rankings.json")
        print(f"  Total users: {len(all_rankings['total_rank'])}")
        print(f"  Total questions: {len(all_rankings['questions'])}")

        if not all_rankings['questions']:
            print(f"  WARNING: No questions found for {contest_id}, skipping...")
            return False

        # Step 2: Get Python3 submissions (3 per question)
        print("\n[2/4] Fetching Python3 submissions (3 per question)...")
        all_submissions = {}  # question_id -> list of submissions

        submission_count = 0
        for submission_dict in all_rankings["submissions"]:
            # Check if we have enough submissions for all questions
            if all(len(subs) >= SOLUTIONS_PER_QUESTION for subs in all_submissions.values()):
                if len(all_submissions) >= len(all_rankings['questions']):
                    print(f"  Collected {SOLUTIONS_PER_QUESTION} submissions for all questions!")
                    break

            for question_id_str, submission_info in submission_dict.items():
                question_id = int(question_id_str)

                # Filter by language
                lang = submission_info.get("lang")
                if lang is None or lang not in LANGS:
                    continue

                # Initialize question entry
                if question_id not in all_submissions:
                    all_submissions[question_id] = []

                # Stop if we have enough for this question
                if len(all_submissions[question_id]) >= SOLUTIONS_PER_QUESTION:
                    continue

                submission_id = submission_info.get("submission_id")
                data_region = submission_info.get("data_region", "US")

                try:
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
                    print(f"    Q{question_id}: {len(all_submissions[question_id])}/{SOLUTIONS_PER_QUESTION} submissions")

                except Exception as e:
                    print(f"    Error fetching submission {submission_id}: {e}")
                    continue

        # Save submissions by question
        print("\n[3/4] Saving submissions...")
        for question_id, submissions in all_submissions.items():
            save_json(submissions, output_dir / f"question_{question_id}_submissions.json")
            print(f"    Q{question_id}: {len(submissions)} submissions")

        # Step 3: Fetch problem descriptions and test cases
        print("\n[4/4] Fetching problem descriptions and test cases...")
        problems_data = {}

        for q in all_rankings["questions"]:
            question_id = q.get("question_id")
            question_slug = q.get("title_slug")

            if not question_slug:
                # Try to get slug from ID
                question_slug = get_question_slug_from_id(browser, question_id)

            if question_slug:
                print(f"    Fetching Q{question_id}: {question_slug}")
                problem_data = get_problem_description(browser, question_slug)

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
            else:
                print(f"    Warning: Could not find slug for Q{question_id}")

        # Save problem descriptions
        if problems_data:
            save_json(problems_data, output_dir / "problems.json")
            print(f"  Saved descriptions for {len(problems_data)} problems")

        print(f"\n{'='*60}")
        print(f"{contest_id} complete!")
        print(f"Questions: {len(all_submissions)}")
        print(f"Submissions: {submission_count}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\nERROR collecting {contest_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Collect LeetCode contest data')
    parser.add_argument('--start', type=int, default=FIRST_WEEKLY_CONTEST,
                        help=f'First contest to collect (default: {FIRST_WEEKLY_CONTEST})')
    parser.add_argument('--end', type=int, default=LAST_WEEKLY_CONTEST,
                        help=f'Last contest to collect (default: {LAST_WEEKLY_CONTEST})')
    parser.add_argument('--contests', type=str, default=None,
                        help='Specific contests to collect (comma-separated, e.g., "431,432,467")')
    parser.add_argument('--username', type=str, default=None,
                        help='LeetCode username (will prompt if not provided)')
    parser.add_argument('--password', type=str, default=None,
                        help='LeetCode password (will prompt if not provided)')

    args = parser.parse_args()

    # Get credentials
    username = args.username
    password = args.password

    if not username:
        print("\nEnter your LeetCode credentials:")
        print("(These are used ONLY for logging into LeetCode via Selenium)")
        print("(Your password is NOT stored anywhere)")
        username = input("Username: ").strip()

    if not password:
        import getpass
        password = getpass.getpass("Password: ")

    if not username or not password:
        print("Error: Username and password required!")
        return

    # Determine which contests to collect
    if args.contests:
        contest_numbers = [int(x.strip()) for x in args.contests.split(',')]
    else:
        contest_numbers = list(range(args.start, args.end + 1))

    print(f"\n{'='*60}")
    print(f"LeetCode Contest Data Collector")
    print(f"{'='*60}")
    print(f"Contests to collect: {len(contest_numbers)}")
    print(f"Range: {min(contest_numbers)} - {max(contest_numbers)}")
    print(f"Solutions per question: {SOLUTIONS_PER_QUESTION}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"{'='*60}")

    # Login with Selenium
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
            success = collect_contest(browser, contest_num)
            if success:
                successful += 1
            else:
                failed += 1

            # Sleep between contests to be nice to LeetCode
            if contest_num != contest_numbers[-1]:
                sleep(5)

        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(contest_numbers)}")
        print(f"Failed: {failed}/{len(contest_numbers)}")
        print(f"Output: {BASE_OUTPUT_DIR}")
        print(f"{'='*60}")

    finally:
        browser.quit()
        print("\nBrowser closed.")


if __name__ == '__main__':
    main()
