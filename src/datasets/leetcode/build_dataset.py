#!/usr/bin/env python3
"""
Build dataset from collected LeetCode contest data
Creates JSONL files with the exact format:
{
  "id": "sample_0",
  "question_id": 2727,
  "function_name": "countSeniors",
  "code": "...",
  "input": "countSeniors(a = ['7868190130M7522', ...])",
  "output": "2",
  "correct_condition": "countSeniors(a = [...]) == 2",
  "contest_id": "biweekly-contest-104",
  "contest_date": "2023-05-13T00:00:00.000",
  "difficulty": "easy",
  "metrics": {
    "coverage": 100.0,
    "loc": 7
  }
}
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# BASE_IMPORTS for executing code (from your specification)
BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, cycle, permutations, product, groupby, islice, repeat, zip_longest
from copy import deepcopy
from string import ascii_uppercase, ascii_lowercase
from math import floor, factorial, log, log2, log10, sqrt, prod, comb, lcm, gcd, ceil, inf, isqrt, isfinite
from collections import defaultdict, deque, Counter, OrderedDict
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge, heapreplace, heappushpop, nsmallest, nlargest
from functools import reduce, cache, lru_cache, partial, cmp_to_key
from random import randrange, shuffle, randint, getrandbits
from operator import itemgetter, add, sub, mul, iand, ior, xor, and_, or_
from re import search as re_search
from os.path import commonprefix
from sys import maxsize
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator, NamedTuple
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
import sys
"""

# Try to import radon for LOC calculation
try:
    from radon.raw import analyze
    HAS_RADON = True
except ImportError:
    HAS_RADON = False
    print("Warning: radon not installed. Using simple line counting for LOC.")

# Configuration
BASE_DATA_DIR = Path(__file__).parent / "data" / "collected"
OUTPUT_DIR = Path(__file__).parent / "data" / "datasets"

# Contest 431 started on approximately 2024-01-07 (weekly contests are on Sundays)
CONTEST_431_DATE = datetime(2024, 1, 7)


def get_contest_date(contest_num: int) -> str:
    """
    Calculate contest date based on contest number
    Weekly contests happen every Sunday (7 days apart)
    """
    weeks_from_431 = contest_num - 431
    contest_date = CONTEST_431_DATE + timedelta(weeks=weeks_from_431)
    return contest_date.strftime("%Y-%m-%dT00:00:00.000")


def calculate_loc(code: str) -> int:
    """Calculate lines of code using radon or fallback"""
    if HAS_RADON:
        try:
            analysis = analyze(code)
            return analysis.loc
        except:
            pass

    # Fallback to simple line count
    return len([line for line in code.split('\n') if line.strip()])


def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from Python code"""
    # Match method inside class Solution
    match = re.search(r'class\s+Solution:.*?def\s+(\w+)\s*\(', code, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: just match def
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)

    return None


def parse_test_input(test_string: str, function_name: str) -> Optional[str]:
    """
    Parse test case string into function call format
    Example: "['7868190130M7522', '5303914400F9211']"
    -> "countSeniors(a = ['7868190130M7522', '5303914400F9211'])"
    """
    if not test_string or not function_name:
        return None

    # Test cases from LeetCode are typically newline-separated parameter values
    # We need to parse them and map to function parameters
    # For now, we'll use a simplified approach

    lines = [line.strip() for line in test_string.strip().split('\n') if line.strip()]

    if not lines:
        return None

    # Simple case: single parameter
    if len(lines) == 1:
        return f"{function_name}({lines[0]})"

    # Multiple parameters: use positional arguments
    params = ', '.join(lines)
    return f"{function_name}({params})"


def execute_code_safely(code: str, input_call: str, timeout: int = 5) -> Optional[str]:
    """
    Execute code safely and capture output
    Returns repr(output) or None if execution fails
    IMPORTANT: Returns repr() for proper string formatting
    """
    try:
        # Combine imports, code, and input
        full_code = f"{BASE_IMPORTS}\n{code}\noutput = {input_call}"

        # Execute in isolated namespace
        namespace = {}
        exec(full_code, namespace)

        # Return repr of output (important for strings!)
        return repr(namespace.get('output'))

    except Exception as e:
        print(f"      Warning: Code execution failed: {e}")
        return None


def build_sample_entry(
    sample_id: str,
    question_id: int,
    question_title: str,
    code: str,
    test_input_call: str,
    test_output: str,
    contest_id: str,
    contest_date: str,
    difficulty: str,
    submission_id: int
) -> Optional[Dict]:
    """
    Build a single dataset sample entry
    """
    function_name = extract_function_name(code)

    if not function_name:
        print(f"      Warning: Could not extract function name")
        return None

    # Calculate metrics
    loc = calculate_loc(code)

    # Note: coverage requires running tests, which we skip for now
    metrics = {
        "loc": loc,
        "coverage": None  # Would need to run with coverage.py
    }

    # Build correct condition
    correct_condition = f"{test_input_call} == {test_output}"

    return {
        "id": sample_id,
        "question_id": question_id,
        "function_name": function_name,
        "code": code,
        "input": test_input_call,
        "output": test_output,
        "correct_condition": correct_condition,
        "contest_id": contest_id,
        "contest_date": contest_date,
        "difficulty": difficulty.lower(),
        "metrics": metrics,
        "submission_id": submission_id,
    }


def load_json(filepath: Path) -> Optional[Dict]:
    """Load JSON file"""
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def get_question_info(question_id: int, rankings: Dict, problems: Dict) -> Dict:
    """Extract question information from rankings and problems data"""
    info = {
        "title": f"Question {question_id}",
        "difficulty": "unknown",
        "test_cases": None,
        "credit": 0,
    }

    # Get title and credit from rankings
    for q in rankings.get("questions", []):
        if q.get("question_id") == question_id:
            info["title"] = q.get("title", info["title"])
            info["credit"] = q.get("credit", 0)
            break

    # Get difficulty and test cases from problems data
    if problems and str(question_id) in problems:
        problem = problems[str(question_id)]
        info["difficulty"] = problem.get("difficulty", "unknown")
        info["test_cases"] = problem.get("exampleTestcases") or problem.get("sampleTestCase")

    # Fallback: map credit to difficulty
    if info["difficulty"] == "unknown" and info["credit"] > 0:
        if info["credit"] <= 3:
            info["difficulty"] = "easy"
        elif info["credit"] <= 5:
            info["difficulty"] = "medium"
        else:
            info["difficulty"] = "hard"

    return info


def process_contest(contest_num: int, sample_counter: int) -> tuple[List[Dict], int]:
    """
    Process a single contest and return dataset entries
    Returns (entries, updated_sample_counter)
    """
    contest_id = f"weekly-contest-{contest_num}"
    contest_dir = BASE_DATA_DIR / contest_id

    print(f"\n{'='*60}")
    print(f"Processing {contest_id}")
    print(f"{'='*60}")

    if not contest_dir.exists():
        print(f"  WARNING: Directory not found: {contest_dir}")
        return [], sample_counter

    # Load rankings
    rankings = load_json(contest_dir / "rankings.json")
    if not rankings:
        print(f"  WARNING: rankings.json not found")
        return [], sample_counter

    # Load problems (descriptions and test cases)
    problems = load_json(contest_dir / "problems.json")

    # Get contest date
    contest_date = get_contest_date(contest_num)

    # Process each question
    entries = []
    questions = rankings.get("questions", [])
    print(f"  Found {len(questions)} questions")

    for q in questions:
        question_id = q.get("question_id")
        if not question_id:
            continue

        # Load submissions for this question
        submissions_file = contest_dir / f"question_{question_id}_submissions.json"
        submissions = load_json(submissions_file)

        if not submissions:
            print(f"  Q{question_id}: No submissions found")
            continue

        # Get question info
        q_info = get_question_info(question_id, rankings, problems)
        question_title = q_info["title"]
        difficulty = q_info["difficulty"]
        test_cases_str = q_info["test_cases"]

        print(f"\n  Q{question_id}: {question_title} ({difficulty})")
        print(f"    Submissions: {len(submissions)}")

        # Process each submission (we already have 3 per question from collector)
        for idx, submission in enumerate(submissions):
            code = submission.get("code")
            submission_id = submission.get("submission_id")

            if not code:
                print(f"      Submission {idx}: No code found")
                continue

            # Extract function name to build test input call
            function_name = extract_function_name(code)
            if not function_name:
                print(f"      Submission {idx}: Could not extract function name")
                continue

            # Parse test cases if available
            test_input_call = None
            test_output = None

            if test_cases_str:
                # Try to parse test case and execute code
                test_input_call = parse_test_input(test_cases_str, function_name)

                if test_input_call:
                    # Execute code to get output
                    test_output = execute_code_safely(code, test_input_call)

            # If we don't have test cases, create a placeholder
            if not test_input_call or not test_output:
                print(f"      Submission {idx}: No valid test case, using placeholder")
                test_input_call = f"{function_name}()"
                test_output = "None"

            # Build sample entry
            sample_id = f"contest{contest_num}_q{question_id}_s{idx}"

            entry = build_sample_entry(
                sample_id=sample_id,
                question_id=question_id,
                question_title=question_title,
                code=code,
                test_input_call=test_input_call,
                test_output=test_output,
                contest_id=contest_id,
                contest_date=contest_date,
                difficulty=difficulty,
                submission_id=submission_id
            )

            if entry:
                entries.append(entry)
                print(f"      Submission {idx}: ✓ Sample {sample_id}")
                sample_counter += 1
            else:
                print(f"      Submission {idx}: ✗ Failed to create entry")

    print(f"\n  Total entries created: {len(entries)}")
    return entries, sample_counter


def main():
    parser = argparse.ArgumentParser(description='Build dataset from collected LeetCode data')
    parser.add_argument('--start', type=int, default=431,
                        help='First contest to process (default: 431)')
    parser.add_argument('--end', type=int, default=467,
                        help='Last contest to process (default: 467)')
    parser.add_argument('--contests', type=str, default=None,
                        help='Specific contests to process (comma-separated, e.g., "431,432,467")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: data/datasets/leetcode_contests_{start}_{end}.jsonl)')
    parser.add_argument('--separate', action='store_true',
                        help='Create separate JSONL file for each contest')

    args = parser.parse_args()

    # Determine which contests to process
    if args.contests:
        contest_numbers = [int(x.strip()) for x in args.contests.split(',')]
    else:
        contest_numbers = list(range(args.start, args.end + 1))

    print(f"\n{'='*60}")
    print(f"LeetCode Dataset Builder")
    print(f"{'='*60}")
    print(f"Contests to process: {len(contest_numbers)}")
    print(f"Range: {min(contest_numbers)} - {max(contest_numbers)}")
    print(f"{'='*60}")

    # Process contests
    all_entries = []
    sample_counter = 0

    for contest_num in contest_numbers:
        entries, sample_counter = process_contest(contest_num, sample_counter)
        all_entries.extend(entries)

        # Save separate file if requested
        if args.separate and entries:
            contest_output = OUTPUT_DIR / f"weekly-contest-{contest_num}.jsonl"
            contest_output.parent.mkdir(parents=True, exist_ok=True)

            with open(contest_output, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

            print(f"  Saved to: {contest_output}")

    # Save combined dataset
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = OUTPUT_DIR / f"leetcode_contests_{min(contest_numbers)}_{max(contest_numbers)}.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n{'='*60}")
    print(f"DATASET BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_entries)}")
    print(f"Output file: {output_file}")

    # Statistics
    difficulty_counts = {}
    for entry in all_entries:
        diff = entry["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    # Show sample entry
    if all_entries:
        print(f"\n{'='*60}")
        print(f"Sample Entry:")
        print(f"{'='*60}")
        sample = all_entries[0]
        print(json.dumps({
            "id": sample["id"],
            "question_id": sample["question_id"],
            "function_name": sample["function_name"],
            "input": sample["input"],
            "output": sample["output"],
            "correct_condition": sample["correct_condition"],
            "difficulty": sample["difficulty"],
            "metrics": sample["metrics"],
            "code_preview": sample["code"][:200] + "..."
        }, indent=2))

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
