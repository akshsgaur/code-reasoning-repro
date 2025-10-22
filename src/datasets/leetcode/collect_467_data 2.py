#!/usr/bin/env python3
"""
Collect data for weekly-contest-467 using the exact approach from data_collection repo
Adapted from: data_collection/live_code_bench/commands/leetcode_get_contest_ranking_by_lang.py
"""

import json
import subprocess
import sys
from pathlib import Path
from pprint import pprint
from time import sleep

# Configuration (matching data_collection/live_code_bench/config.py)
CONTEST_ID = "weekly-contest-467"
RANKING_PAGE_START = 1
RANKING_PAGE_END = 10
LANGS = ["python3"]

OUTPUT_DIR = Path(__file__).parent / "data" / "collected" / CONTEST_ID


def subprocess_curl(url: str) -> str:
    """
    Execute curl with cookies - EXACT copy from data_collection repo
    From: data_collection/live_code_bench/leetcode/contest_ranking.py:69
    """
    # NOTE: You MUST update this cookie string with your own!
    # The original repo had hardcoded cookies that worked at the time
    # Get fresh cookies from your browser for this to work
    cookie = r"gr_user_id=1d8589af-a6ba-44b6-aa58-0cb8d41a09ad; __stripe_mid=792985e9-115a-4b8f-b673-0c9ee469a9a6a6ef66; _gcl_au=1.1.1342357061.1759255411; _ga_DKXQ03QCVK=GS2.1.s1759274666$o3$g0$t1759274666$j60$l0$h0; INGRESSCOOKIE=914312d48185e6b025d80840347eeae8|8e0876c7c1464cc0ac96bc2edceabd27; _gid=GA1.2.392269615.1760657928; 87b5a3c3f1a55520_gr_last_sent_cs1=akshitgaur997; cf_clearance=JMUW_iZco88qFOKdkPUS0ABjLen2RY9W0YiuII.LkNw-1760720220-1.2.1.1-e5pXzn1n.wqB86QbeH01_ySIHKipz6wCLS53JX.tFi4ytH0xIAR5HZ6BIWTEcwp0RKiaFQ.4YWWwQ1pldyArdQ.JNkce8bUHJy64JFSHyCuhms_X4oydijY91TcP0B_r3kaC5yFClx_lyC84ES7kEJdkFXgcn8JOgV9pBaeKB1mCAD4fwyWDvK1JvdlcCQxC7P2Y5u41gRJcMzUp_7Ld3kO08NMs8EWTOHDKHCzjTfQ; csrftoken=CReKMpGvQVmTQWczt4SM3UDd4XZlGv8uPhvGhWJQXf1zjLt1cwv99mKoCTIeIqIG; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFTitXBpTy4NDkZKJJWmpNTCdOSmaeQWKyQmF2ckVmSnlhaZGlpDjQiFgDP7icI:1v9nlK:fYUywXX7J6DsIOMwwN0OOAxaQfhlsSDI2ot9pcCyqh8; LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjczOTIwNCIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImFsbGF1dGguYWNjb3VudC5hdXRoX2JhY2tlbmRzLkF1dGhlbnRpY2F0aW9uQmFja2VuZCIsIl9hdXRoX3VzZXJfaGFzaCI6ImQyMTAyNGE3NGJjYjFjODg4ZDFlMGFmMDY3MWZlOWQyZjdjNTgwMWZhNzYxYzUxYWFjNjZmM2ZhMmZlMzYxN2MiLCJzZXNzaW9uX3V1aWQiOiI1ZWE1ZGYzMiIsImlkIjoyNzM5MjA0LCJlbWFpbCI6ImFrc2hpdGdhdXI5OTdAZ21haWwuY29tIiwidXNlcm5hbWUiOiJha3NoaXRnYXVyOTk3IiwidXNlcl9zbHVnIjoiYWtzaGl0Z2F1cjk5NyIsImF2YXRhciI6Imh0dHBzOi8vYXNzZXRzLmxlZXRjb2RlLmNvbS91c2Vycy9ha3NoaXRnYXVyOTk3L2F2YXRhcl8xNTg0MDQ5NjY0LnBuZyIsInJlZnJlc2hlZF9hdCI6MTc2MDgwNjk5NSwiaXAiOiIyNjAxOjY0Nzo0ODAwOjI5NzA6ZDE3YzoyNTllOjVmZjE6OWFjYiIsImlkZW50aXR5IjoiZGY0YmYwNGY5YmY3YjZhZjA5ZTNlOTQxNzk3MzM3NzAiLCJkZXZpY2Vfd2l0aF9pcCI6WyJmNjlkNzAwYWQ0NGNlZTAzNDFiODU5OWY5NWVlMmM4ZCIsIjI2MDE6NjQ3OjQ4MDA6Mjk3MDpkMTdjOjI1OWU6NWZmMTo5YWNiIl0sIl9zZXNzaW9uX2V4cGlyeSI6MTIwOTYwMH0.2V4lTPKpcvd0O84hLZlT7XMp1-Fhmdl3qA2dpXIScNM; ip_check=(false, \"2601:647:4800:2970:d17c:259e:5ff1:9acb\"); 87b5a3c3f1a55520_gr_session_id=403925eb-a5f6-402d-b9a0-d830f29da33c; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=403925eb-a5f6-402d-b9a0-d830f29da33c; 87b5a3c3f1a55520_gr_session_id_sent_vst=403925eb-a5f6-402d-b9a0-d830f29da33c; __stripe_sid=be78e734-00a6-459e-9f9c-3451fbd6b6cf1b8397; _ga=GA1.1.1832363036.1756255369; 87b5a3c3f1a55520_gr_cs1=akshitgaur997; _ga_CDRWKZTDEX=GS2.1.s1760806997$o21$g1$t1760808535$j45$l0$h0"

    cmd = f"""curl '{url}' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cache-control: max-age=0' \
  -H 'cookie: {cookie}' \
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


def get_contest_ranking(contest_id: str, ranking_page: int):
    """
    From: data_collection/live_code_bench/leetcode/contest_ranking.py:92
    """
    url = f'https://leetcode.com/contest/api/ranking/{contest_id}/?pagination={ranking_page}'
    print(f"Fetching {url}")

    res = json.loads(subprocess_curl(url))
    sleep(2)

    return res


def get_submission(submission_id: int, data_region: str):
    """
    From: data_collection/live_code_bench/leetcode/submission.py:61
    """
    if data_region == "CN":
        url = f"https://leetcode.cn/api/submissions/{submission_id}/"
    else:
        url = f"https://leetcode.com/api/submissions/{submission_id}/"

    print(f"  Fetching submission {submission_id}")
    res = json.loads(subprocess_curl(url))
    sleep(2)

    return res


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

    # Step 1: Get contest rankings
    print("\n[1/2] Fetching contest rankings...")
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
            page_result = get_contest_ranking(CONTEST_ID, ranking_page)

            if ranking_page == 1:
                all_rankings["is_past"] = page_result.get("is_past")
                all_rankings["questions"] = page_result.get("questions", [])
                all_rankings["user_num"] = page_result.get("user_num", 0)

            all_rankings["total_rank"].extend(page_result.get("total_rank", []))
            all_rankings["submissions"].extend(page_result.get("submissions", []))

            print(f"  Page {ranking_page}: {len(page_result.get('total_rank', []))} users")

        except Exception as e:
            print(f"  Error on page {ranking_page}: {e}")
            break

    # Save rankings
    save_json(all_rankings, "rankings.json")
    print(f"Total users collected: {len(all_rankings['total_rank'])}")

    # Step 2: Get Python3 submissions
    print("\n[2/2] Fetching Python3 submissions...")
    all_submissions = {}  # question_id -> list of submissions

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
                # Fetch submission code
                sleep(1)
                submission_data = get_submission(submission_id, data_region)

                # Check language again
                if submission_data.get("lang") not in LANGS:
                    continue

                # Initialize question entry
                if question_id not in all_submissions:
                    all_submissions[question_id] = []

                # Add submission
                all_submissions[question_id].append({
                    "submission_id": submission_id,
                    "code": submission_data.get("code"),
                    "lang": submission_data.get("lang"),
                    "question_id": question_id,
                    "contest_submission": submission_data.get("contest_submission"),
                })

                print(f"  Q{question_id}: {len(all_submissions[question_id])} submissions")

                # Stop after 15 submissions per question (matching original repo logic)
                if len(all_submissions[question_id]) >= 15:
                    continue

            except Exception as e:
                print(f"  Error fetching submission {submission_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save submissions by question
    for question_id, submissions in all_submissions.items():
        save_json(submissions, f"question_{question_id}_submissions.json")
        print(f"Saved {len(submissions)} submissions for Q{question_id}")

    print("\n" + "=" * 60)
    print(f"Collection complete!")
    print(f"Questions: {len(all_submissions)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
