#!/usr/bin/env python3
"""
Scrape data from LeetCode weekly-contest-467
Adapted from data_collection/leetcode repository
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from pprint import pprint
from time import sleep
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

CONTEST_ID = "weekly-contest-467"
OUTPUT_DIR = Path(__file__).parent / "data" / "site" / CONTEST_ID
RANKING_PAGE_START = 1
RANKING_PAGE_END = 10
LANGS = ["python3"]

# You MUST replace this with your own cookies from your browser!
# See README_contest_467.md for instructions
COOKIE_STRING = r"gr_user_id=1d8589af-a6ba-44b6-aa58-0cb8d41a09ad; __stripe_mid=792985e9-115a-4b8f-b673-0c9ee469a9a6a6ef66; _gcl_au=1.1.1342357061.1759255411; _ga_DKXQ03QCVK=GS2.1.s1759274666$o3$g0$t1759274666$j60$l0$h0; INGRESSCOOKIE=914312d48185e6b025d80840347eeae8|8e0876c7c1464cc0ac96bc2edceabd27; _gid=GA1.2.392269615.1760657928; 87b5a3c3f1a55520_gr_last_sent_cs1=akshitgaur997; cf_clearance=JMUW_iZco88qFOKdkPUS0ABjLen2RY9W0YiuII.LkNw-1760720220-1.2.1.1-e5pXzn1n.wqB86QbeH01_ySIHKipz6wCLS53JX.tFi4ytH0xIAR5HZ6BIWTEcwp0RKiaFQ.4YWWwQ1pldyArdQ.JNkce8bUHJy64JFSHyCuhms_X4oydijY91TcP0B_r3kaC5yFClx_lyC84ES7kEJdkFXgcn8JOgV9pBaeKB1mCAD4fwyWDvK1JvdlcCQxC7P2Y5u41gRJcMzUp_7Ld3kO08NMs8EWTOHDKHCzjTfQ; csrftoken=CReKMpGvQVmTQWczt4SM3UDd4XZlGv8uPhvGhWJQXf1zjLt1cwv99mKoCTIeIqIG; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFTitXBpTy4NDkZKJJWmpNTCdOSmaeQWKyQmF2ckVmSnlhaZGlpDjQiFgDP7icI:1v9nlK:fYUywXX7J6DsIOMwwN0OOAxaQfhlsSDI2ot9pcCyqh8; LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjczOTIwNCIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImFsbGF1dGguYWNjb3VudC5hdXRoX2JhY2tlbmRzLkF1dGhlbnRpY2F0aW9uQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImQyMTAyNGE3NGJjYjFjODg4ZDFlMGFmMDY3MWZlOWQyZjdjNTgwMWZhNzYxYzUxYWFjNjZmM2ZhMmZlMzYxN2MiLCJzZXNzaW9uX3V1aWQiOiI1ZWE1ZGYzMiIsImlkIjoyNzM5MjA0LCJlbWFpbCI6ImFrc2hpdGdhdXI5OTdAZ21haWwuY29tIiwidXNlcm5hbWUiOiJha3NoaXRnYXVyOTk3IiwidXNlcl9zbHVnIjoiYWtzaGl0Z2F1cjk5NyIsImF2YXRhciI6Imh0dHBzOi8vYXNzZXRzLmxlZXRjb2RlLmNvbS91c2Vycy9ha3NoaXRnYXVyOTk3L2F2YXRhcl8xNTg0MDQ5NjY0LnBuZyIsInJlZnJlc2hlZF9hdCI6MTc2MDgwNjk5NSwiaXAiOiIyNjAxOjY0Nzo0ODAwOjI5NzA6ZDE3YzoyNTllOjVmZjE6OWFjYiIsImlkZW50aXR5IjoiZGY0YmYwNGY5YmY3YjZhZjA5ZTNlOTQxNzk3MzM3NzAiLCJkZXZpY2Vfd2l0aF9pcCI6WyJmNjlkNzAwYWQ0NGNlZTAzNDFiODU5OWY5NWVlMmM4ZCIsIjI2MDE6NjQ3OjQ4MDA6Mjk3MDpkMTdjOjI1OWU6NWZmMTo5YWNiIl0sIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.2V4lTPKpcvd0O84hLZlT7XMp1-Fhmdl3qA2dpXIScNM; ip_check=(false, \"2601:647:4800:2970:d17c:259e:5ff1:9acb\"); 87b5a3c3f1a55520_gr_session_id=403925eb-a5f6-402d-b9a0-d830f29da33c; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=403925eb-a5f6-402d-b9a0-d830f29da33c; 87b5a3c3f1a55520_gr_session_id_sent_vst=403925eb-a5f6-402d-b9a0-d830f29da33c; __stripe_sid=be78e734-00a6-459e-9f9c-3451fbd6b6cf1b8397; _ga=GA1.1.1832363036.1756255369; 87b5a3c3f1a55520_gr_cs1=akshitgaur997; _ga_CDRWKZTDEX=GS2.1.s1760806997$o21$g1$t1760808535$j45$l0$h0"


def subprocess_curl(url: str) -> str:
    """
    Execute curl command with cookies to fetch data from LeetCode

    NOTE: You must update COOKIE_STRING above with your browser cookies!
    To get cookies:
    1. Go to https://leetcode.com/contest/weekly-contest-467/ranking/
    2. Open DevTools (F12) -> Network tab
    3. Refresh page
    4. Click on a 'ranking' request
    5. Copy the 'Cookie:' header value
    6. Paste it into COOKIE_STRING above
    """
    cmd = f"""curl '{url}' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cache-control: max-age=0' \
  -H 'cookie: {COOKIE_STRING}' \
  -H 'dnt: 1' \
  -H 'priority: u=0, i' \
  -H 'sec-ch-ua: "Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"' \
  -H 'sec-ch-ua-mobile: ?1' \
  -H 'sec-ch-ua-platform: "Android"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: none' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36'"""

    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return output.stdout


def get_contest_ranking(contest_id: str, ranking_page: int) -> Dict:
    """Fetch contest ranking page"""
    url = f"https://leetcode.com/contest/api/ranking/{contest_id}/?pagination={ranking_page}"

    print(f"Fetching ranking page {ranking_page}...")

    if COOKIE_STRING == "REPLACE_WITH_YOUR_COOKIES":
        print("\n" + "="*60)
        print("ERROR: You must update COOKIE_STRING in this script!")
        print("="*60)
        print("\nTo get cookies:")
        print("1. Visit https://leetcode.com/contest/weekly-contest-467/ranking/")
        print("2. Open DevTools (F12) -> Network tab")
        print("3. Refresh the page")
        print("4. Click on a 'ranking' API request")
        print("5. In Headers, find 'Cookie:' under Request Headers")
        print("6. Copy the entire cookie string")
        print("7. Replace COOKIE_STRING in scrape_contest_467.py")
        print("\nSee README_contest_467.md for detailed instructions.")
        sys.exit(1)

    res = json.loads(subprocess_curl(url))
    sleep(2)  # Be nice to LeetCode servers

    return res


def get_submission(submission_id: int, data_region: str) -> Dict:
    """Fetch individual submission code"""
    if data_region == "CN":
        url = f"https://leetcode.cn/api/submissions/{submission_id}/"
    else:
        url = f"https://leetcode.com/api/submissions/{submission_id}/"

    print(f"  Fetching submission {submission_id}...")
    res = json.loads(subprocess_curl(url))
    sleep(2)  # Be nice to LeetCode servers

    return res


def scrape_contest_rankings() -> Dict:
    """Scrape all ranking pages for the contest"""
    all_data = {
        "contest_id": CONTEST_ID,
        "is_past": None,
        "questions": [],
        "total_rank": [],
        "submissions": [],
        "user_num": 0,
    }

    for page in range(RANKING_PAGE_START, RANKING_PAGE_END + 1):
        try:
            page_data = get_contest_ranking(CONTEST_ID, page)

            # Store metadata from first page
            if page == 1:
                all_data["is_past"] = page_data.get("is_past")
                all_data["questions"] = page_data.get("questions", [])
                all_data["user_num"] = page_data.get("user_num", 0)

            # Append ranking data
            all_data["total_rank"].extend(page_data.get("total_rank", []))
            all_data["submissions"].extend(page_data.get("submissions", []))

            print(f"  Fetched {len(page_data.get('total_rank', []))} users from page {page}")

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    return all_data


def scrape_submissions(ranking_data: Dict) -> Dict[int, List[Dict]]:
    """Scrape Python3 submissions for each problem"""
    submissions_by_question = {}

    # Iterate through submissions from ranking data
    for submission_dict in ranking_data["submissions"]:
        for question_id_str, submission_info in submission_dict.items():
            question_id = int(question_id_str)

            # Filter by language
            lang = submission_info.get("lang")
            if lang is None or lang not in LANGS:
                continue

            submission_id = submission_info.get("submission_id")
            data_region = submission_info.get("data_region", "US")

            try:
                # Fetch the actual code
                submission_data = get_submission(submission_id, data_region)

                # Check language again after fetching
                if submission_data.get("lang") not in LANGS:
                    continue

                # Initialize question entry if needed
                if question_id not in submissions_by_question:
                    submissions_by_question[question_id] = []

                # Add submission
                submissions_by_question[question_id].append({
                    "submission_id": submission_id,
                    "code": submission_data.get("code"),
                    "lang": submission_data.get("lang"),
                    "question_id": question_id,
                })

                print(f"  Saved Python3 submission for Q{question_id} (total: {len(submissions_by_question[question_id])})")

                # Stop after collecting enough submissions per question
                if all(len(subs) >= 15 for subs in submissions_by_question.values()):
                    if len(submissions_by_question) >= len(ranking_data["questions"]):
                        print("Collected sufficient submissions for all questions!")
                        return submissions_by_question

            except Exception as e:
                print(f"  Error fetching submission {submission_id}: {e}")
                continue

    return submissions_by_question


def save_data(ranking_data: Dict, submissions_data: Dict):
    """Save scraped data to files"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save contest info and rankings
    index_file = OUTPUT_DIR / "index.json"
    with open(index_file, "w") as f:
        json.dump({
            "contest_id": ranking_data["contest_id"],
            "is_past": ranking_data["is_past"],
            "questions": ranking_data["questions"],
            "user_num": ranking_data["user_num"],
        }, f, indent=2)
    print(f"Saved contest info to {index_file}")

    # Save top users
    top_users_file = OUTPUT_DIR / "top_users.json"
    with open(top_users_file, "w") as f:
        json.dump(ranking_data["total_rank"], f, indent=2)
    print(f"Saved {len(ranking_data['total_rank'])} users to {top_users_file}")

    # Save submissions by question
    for question_id, submissions in submissions_data.items():
        question_file = OUTPUT_DIR / f"question_{question_id}_submissions.json"
        with open(question_file, "w") as f:
            json.dump(submissions, f, indent=2)
        print(f"Saved {len(submissions)} submissions for Q{question_id} to {question_file}")


def main():
    print(f"Starting data collection for {CONTEST_ID}...")
    print("=" * 60)

    # Step 1: Scrape contest rankings
    print("\n[1/2] Scraping contest rankings...")
    ranking_data = scrape_contest_rankings()
    print(f"Collected data for {len(ranking_data['total_rank'])} users")
    print(f"Found {len(ranking_data['questions'])} questions")

    # Step 2: Scrape submissions
    print("\n[2/2] Scraping Python3 submissions...")
    submissions_data = scrape_submissions(ranking_data)
    print(f"Collected submissions for {len(submissions_data)} questions")

    # Step 3: Save data
    print("\nSaving data...")
    save_data(ranking_data, submissions_data)

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
