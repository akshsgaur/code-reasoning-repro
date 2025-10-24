#!/usr/bin/env python3
"""
Build dataset entries from weekly-contest-467 data
This script processes scraped data and creates JSONL dataset entries
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# BASE_IMPORTS for executing code (from ResponseToCMUTeam.docx)
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

CONTEST_ID = "weekly-contest-467"
DATA_DIR = Path(__file__).parent / "data" / "site" / CONTEST_ID
OUTPUT_FILE = Path(__file__).parent / "data" / f"{CONTEST_ID}.jsonl"


def load_contest_data() -> Dict:
    """Load contest metadata and rankings"""
    rankings_file = DATA_DIR / "rankings.json"

    if not rankings_file.exists():
        print(f"Error: {rankings_file} not found!")
        print("Please run the scraper first to download contest data.")
        print("See README_contest_467.md for instructions.")
        sys.exit(1)

    with open(rankings_file, "r") as f:
        return json.load(f)


def load_submissions(question_id: int) -> List[Dict]:
    """Load submissions for a specific question"""
    submissions_file = DATA_DIR / f"question_{question_id}_submissions.json"

    if not submissions_file.exists():
        return []

    with open(submissions_file, "r") as f:
        return json.load(f)


def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from Python code"""
    import re

    # Match def function_name(
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    return None


def execute_code_safely(code: str, input_call: str) -> Optional[str]:
    """
    Execute code safely and capture output
    Returns repr(output) or None if execution fails
    """
    try:
        # Combine imports, code, and input
        full_code = f"{BASE_IMPORTS}\n{code}\noutput = {input_call}"

        # Execute in isolated namespace
        namespace = {}
        exec(full_code, namespace)

        # Return repr of output (important for strings)
        return repr(namespace.get('output'))

    except Exception as e:
        print(f"  Warning: Code execution failed: {e}")
        return None


def calculate_metrics(code: str) -> Dict:
    """
    Calculate code metrics using radon and coverage
    """
    try:
        from radon.metrics import mi_visit
        from radon.raw import analyze

        # Lines of code
        analysis = analyze(code)
        loc = analysis.loc

        # Coverage would require running tests - skip for now
        coverage = None

        return {
            "loc": loc,
            "coverage": coverage,
        }
    except ImportError:
        print("  Warning: radon not installed, skipping metrics")
        return {"loc": len(code.split('\n')), "coverage": None}


def build_dataset_entry(
    sample_id: str,
    question_id: int,
    question_title: str,
    code: str,
    test_input: str,
    test_output: str,
    contest_date: str,
    difficulty: str = "unknown"
) -> Dict:
    """Build a single dataset entry"""

    function_name = extract_function_name(code)

    if not function_name:
        print(f"  Warning: Could not extract function name for Q{question_id}")
        return None

    # Build input call (simplified - you may need to parse test_input properly)
    input_call = f"{function_name}({test_input})"

    # Execute code to get output (if test_output not provided)
    if not test_output:
        test_output = execute_code_safely(code, input_call)

    # Build correct condition
    correct_condition = f"{input_call} == {test_output}"

    # Calculate metrics
    metrics = calculate_metrics(code)

    return {
        "id": sample_id,
        "question_id": question_id,
        "question_title": question_title,
        "function_name": function_name,
        "code": code,
        "input": input_call,
        "output": test_output,
        "correct_condition": correct_condition,
        "contest_id": CONTEST_ID,
        "contest_date": contest_date,
        "difficulty": difficulty,
        "metrics": metrics,
    }


def main():
    print(f"Building dataset for {CONTEST_ID}...")
    print("=" * 60)

    # Load contest data
    print("\n[1/3] Loading contest data...")
    contest_data = load_contest_data()

    problems = contest_data.get("problems", {})
    print(f"Found {len(problems)} problems")

    # Process each problem
    print("\n[2/3] Processing submissions...")
    dataset_entries = []
    sample_counter = 0

    for question_id_str, problem_info in problems.items():
        question_id = int(question_id_str) if isinstance(question_id_str, str) else question_id_str
        question_title = problem_info.get("title", f"Question {question_id}")

        print(f"\nProcessing Q{question_id}: {question_title}")

        # Load submissions for this question
        submissions = load_submissions(question_id)
        print(f"  Found {len(submissions)} Python3 submissions")

        # Process each submission
        for idx, submission in enumerate(submissions[:15]):  # Limit to 15 per question
            code = submission.get("code")

            if not code:
                continue

            sample_id = f"contest467_q{question_id}_s{idx}"

            # For now, we don't have test cases - you'd need to scrape these separately
            # or use LeetCode's problem API
            test_input = ""  # TODO: Get from problem test cases
            test_output = ""  # TODO: Get from problem test cases

            entry = build_dataset_entry(
                sample_id=sample_id,
                question_id=question_id,
                question_title=question_title,
                code=code,
                test_input=test_input,
                test_output=test_output,
                contest_date="2024-10-13T00:00:00.000",  # Update with actual date
                difficulty="unknown",  # Update with actual difficulty
            )

            if entry:
                dataset_entries.append(entry)
                sample_counter += 1

    # Save dataset
    print(f"\n[3/3] Saving dataset...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(dataset_entries)} entries to {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("Dataset build complete!")

    # Print sample entry
    if dataset_entries:
        print("\nSample entry:")
        print(json.dumps(dataset_entries[0], indent=2))


if __name__ == "__main__":
    main()
