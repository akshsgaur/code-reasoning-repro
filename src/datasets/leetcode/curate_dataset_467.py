#!/usr/bin/env python3
"""
Curate dataset from weekly-contest-467 following LiveCodeBench methodology
Based on: LiveCodeBench paper (2403.07974v2.pdf)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from radon.raw import analyze
    HAS_RADON = True
except ImportError:
    HAS_RADON = False

# Configuration
CONTEST_ID = "weekly-contest-467"
DATA_DIR = Path(__file__).parent / "data" / "collected" / CONTEST_ID
OUTPUT_FILE = Path(__file__).parent / "data" / f"{CONTEST_ID}.jsonl"

def load_rankings() -> Dict:
    """Load contest rankings and metadata"""
    rankings_file = DATA_DIR / "rankings.json"

    if not rankings_file.exists():
        print(f"Error: {rankings_file} not found!")
        print("Please run collect_467_selenium.py first to collect the data.")
        sys.exit(1)

    with open(rankings_file, "r") as f:
        return json.load(f)

def load_submissions(question_id: int) -> List[Dict]:
    """Load submissions for a specific question"""
    submissions_file = DATA_DIR / f"question_{question_id}_submissions.json"

    if not submissions_file.exists():
        print(f"Warning: {submissions_file} not found")
        return []

    with open(submissions_file, "r") as f:
        return json.load(f)

def calculate_loc(code: str) -> int:
    """Calculate lines of code using radon"""
    if HAS_RADON:
        try:
            analysis = analyze(code)
            return analysis.loc
        except:
            pass

    # Fallback to simple line count
    return len([line for line in code.split('\n') if line.strip()])

def extract_function_signature(code: str) -> Optional[str]:
    """Extract the function signature from code"""
    import re

    # Match class Solution with method definition
    match = re.search(r'class Solution:.*?def\s+(\w+)\s*\([^)]*\)', code, re.DOTALL)
    if match:
        # Extract just the def line
        def_match = re.search(r'def\s+(\w+)\s*\([^)]*\)', match.group(0))
        if def_match:
            return def_match.group(0)

    # Fallback: just match def
    match = re.search(r'def\s+\w+\s*\([^)]*\)', code)
    if match:
        return match.group(0)

    return None

def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from Python code"""
    import re

    # Match method inside class Solution
    match = re.search(r'class Solution:.*?def\s+(\w+)\s*\(', code, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: just match def
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)

    return None

def build_code_generation_entry(
    sample_id: str,
    question_id: int,
    question_title: str,
    question_content: str,
    code: str,
    contest_date: str,
    difficulty: str,
    submission_id: int
) -> Dict:
    """
    Build a dataset entry for code generation scenario
    Following LiveCodeBench format from paper Section 3.3
    """

    function_name = extract_function_name(code)
    function_signature = extract_function_signature(code)
    loc = calculate_loc(code)

    return {
        "id": sample_id,
        "scenario": "code_generation",
        "question_id": question_id,
        "question_title": question_title,
        "question_content": question_content,
        "function_name": function_name,
        "function_signature": function_signature,
        "code": code,
        "submission_id": submission_id,
        "contest_id": CONTEST_ID,
        "contest_date": contest_date,
        "difficulty": difficulty,
        "metrics": {
            "loc": loc,
        },
        "platform": "leetcode"
    }

def get_question_info(question_id: int, questions_metadata: List[Dict]) -> Dict:
    """Extract question information from metadata"""
    for q in questions_metadata:
        if q.get("question_id") == question_id:
            return {
                "title": q.get("title", f"Question {question_id}"),
                "credit": q.get("credit", 0),
            }
    return {
        "title": f"Question {question_id}",
        "credit": 0
    }

def map_credit_to_difficulty(credit: int) -> str:
    """Map LeetCode credit score to difficulty"""
    if credit <= 3:
        return "easy"
    elif credit <= 5:
        return "medium"
    else:
        return "hard"

def main():
    print(f"Curating dataset for {CONTEST_ID}...")
    print("=" * 60)

    # Load rankings data
    print("\n[1/3] Loading collected data...")
    rankings_data = load_rankings()

    questions = rankings_data.get("questions", [])
    print(f"Found {len(questions)} questions")

    # Extract question IDs
    question_ids = set()
    for q in questions:
        qid = q.get("question_id")
        if qid:
            question_ids.add(qid)

    print(f"Question IDs: {sorted(question_ids)}")

    # Process each question
    print("\n[2/3] Processing submissions...")
    dataset_entries = []
    total_submissions = 0

    # Estimate contest date from metadata (weekly-contest-467 was around Oct 2024)
    contest_date = "2024-10-13T00:00:00.000"

    for question_id in sorted(question_ids):
        question_info = get_question_info(question_id, questions)
        question_title = question_info["title"]
        difficulty = map_credit_to_difficulty(question_info["credit"])

        print(f"\nProcessing Q{question_id}: {question_title} ({difficulty})")

        # Load submissions for this question
        submissions = load_submissions(question_id)
        print(f"  Found {len(submissions)} Python3 submissions")

        # Process each submission (limit to 15 per question as per LiveCodeBench)
        for idx, submission in enumerate(submissions[:15]):
            code = submission.get("code")
            submission_id = submission.get("submission_id")

            if not code:
                continue

            sample_id = f"contest467_q{question_id}_s{idx}"

            # Note: question_content would ideally be scraped from LeetCode
            # For now, we use a placeholder
            question_content = f"[Problem description for {question_title} - Question ID: {question_id}]"

            entry = build_code_generation_entry(
                sample_id=sample_id,
                question_id=question_id,
                question_title=question_title,
                question_content=question_content,
                code=code,
                contest_date=contest_date,
                difficulty=difficulty,
                submission_id=submission_id
            )

            dataset_entries.append(entry)
            total_submissions += 1

        print(f"  Processed {min(len(submissions), 15)} submissions for Q{question_id}")

    # Save dataset
    print(f"\n[3/3] Saving dataset...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(dataset_entries)} entries to {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("Dataset curation complete!")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total questions: {len(question_ids)}")
    print(f"Total submissions: {total_submissions}")

    # Count by difficulty
    difficulty_counts = {}
    for entry in dataset_entries:
        diff = entry["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print("\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    # Print sample entry
    if dataset_entries:
        print("\n=== Sample Entry ===")
        sample = dataset_entries[0]
        print(json.dumps({
            "id": sample["id"],
            "question_id": sample["question_id"],
            "question_title": sample["question_title"],
            "function_name": sample["function_name"],
            "difficulty": sample["difficulty"],
            "metrics": sample["metrics"],
            "code_preview": sample["code"][:200] + "..."
        }, indent=2))

if __name__ == "__main__":
    main()
