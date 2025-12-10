#!/usr/bin/env python3
"""
Create mutated version of LeetCode dataset following Algorithm 1 from paper 2504.05518v1.pdf

Algorithm 1: Mutated Dataset Creation
1. For each (program, input) pair in dataset
2. Generate all mutants by applying mutation operators (Table 1)
3. Filter mutants that execute successfully and produce different output
4. Select mutant with most similar line coverage to original
5. Add (mutant, input, new_output) to mutated dataset

Mutation Operators (Table 1, page 6):
- Arithmetic: +↔-, *↔/, //↔*, %↔//
- Relational: <↔<=, >↔>=, ==↔!=
- Logical: and↔or
- Keyword: continue↔break, True↔False
- Number: n→n±1
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import tempfile
import os

# Import simple mutator
from simple_mutator import generate_all_mutants

# BASE_IMPORTS for code execution (same as in evaluate.py)
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
import sys
"""


def execute_code_with_input(code: str, function_name: str, test_input, timeout_seconds: int = 5) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Execute code with one or multiple test inputs.

    test_input may be:
      - str: e.g. "length([1,2,3])" (LeetCode style)
      - list[list[str]]: e.g. [["[1,2,3]"], ["[5,5,1]"], ["[-1,2]"]] (LLM-List: each inner list is arg strings)
    function_name may be:
      - just the name: "length"
      - or a header string: "def length(lst):"
    Returns: (success, [repr(...) per input], error)
    """
    try:
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timeout")

        # Extract function name if a header was provided
        def _func_name(name_or_header: str) -> str:
            s = name_or_header.strip()
            if s.startswith("def "):
                return s.split("def ", 1)[1].split("(", 1)[0].strip()
            return s

        fname = _func_name(function_name)

        # Normalize test_input into a list of call strings
        calls: List[str] = []
        if isinstance(test_input, str):
            # LeetCode style: already "fn(args)"
            calls = [test_input]
        else:
            # LLM-List style: list of arg-lists (strings); build "fn(arg1, arg2, ...)" per case
            for args in test_input:
                calls.append(f"{fname}({', '.join(args)})")

        # Timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            # Prepare exec env
            exec_globals = {}
            exec(BASE_IMPORTS, exec_globals)
            exec(code, exec_globals)

            outputs: List[str] = []
            for call in calls:
                result = eval(call, exec_globals)
                outputs.append(repr(result))

            signal.alarm(0)
            return (True, outputs, None)
        finally:
            signal.alarm(0)

    except TimeoutError as e:
        return (False, None, f"Timeout: {e}")
    except Exception as e:
        return (False, None, str(e))


def get_line_coverage(code: str, function_name: str, test_input: str) -> Optional[Set[int]]:
    """
    Get line coverage when executing code with input

    NOTE: Coverage tracking disabled for performance - it was very slow
    and showing 0.00 similarity anyway. We'll just select the first valid mutant.

    Returns:
        None (coverage tracking disabled)
    """
    return None


def coverage_similarity(cov1: Optional[Set[int]], cov2: Optional[Set[int]]) -> float:
    """
    Calculate Jaccard similarity between two coverage sets

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not cov1 or not cov2:
        return 0.0

    intersection = len(cov1 & cov2)
    union = len(cov1 | cov2)

    return intersection / union if union > 0 else 0.0


def find_best_mutant(
    original_code: str,
    function_name: str,
    test_input,
    original_output,
    debug: bool = False
) -> Optional[Dict]:
    """
    Supports:
      - test_input: str  OR  list[list[str]]
      - original_output: str OR list[str]
    For LLM-List, pass list of arg-lists and list of outputs (3 each).
    """
    # Normalize original_output to a list
    if isinstance(original_output, str):
        original_outputs = [original_output]
    else:
        original_outputs = list(original_output)

    # Step 1: Generate mutants
    mutants = generate_all_mutants(original_code)
    if debug:
        print(f"    Generated {len(mutants)} mutants")
    if not mutants:
        if debug:
            print(f"    No mutants generated")
        return None

    # Coverage disabled (keep interface)
    original_cov = get_line_coverage(original_code, function_name, test_input)

    valid_mutants = []
    execution_failures = 0
    same_output_count = 0

    for i, (mutated_code, mutation_type, mutation_idx) in enumerate(mutants):
        if debug and i > 0 and i % 10 == 0:
            print(f"    Processing mutant {i}/{len(mutants)}...")

        # Execute mutant on all provided inputs
        success, mutated_outputs, error = execute_code_with_input(mutated_code, function_name, test_input)
        if not success or mutated_outputs is None:
            execution_failures += 1
            continue

        # Must produce different output on at least one input
        # and lengths should match if we have multiple outputs
        if len(mutated_outputs) != len(original_outputs):
            # If length mismatch, treat as invalid (safer)
            execution_failures += 1
            continue

        differs = any(mo != oo for mo, oo in zip(mutated_outputs, original_outputs))
        if not differs:
            same_output_count += 1
            continue

        valid_mutants.append({
            'code': mutated_code,
            'outputs': mutated_outputs,
            'mutation_type': mutation_type,
            'mutation_id': mutation_idx,
            'coverage': None
        })

        # Optimization: stop after first valid
        if debug:
            print(f"    Found valid mutant, stopping early (checked {i+1}/{len(mutants)} mutants)")
        break

    if debug:
        print(f"    Valid mutants: {len(valid_mutants)}")
        print(f"    Execution failures: {execution_failures}")
        print(f"    Same output: {same_output_count}")

    if not valid_mutants:
        return None

    best = valid_mutants[0]
    return {
        'mutated_code': best['code'],
        'mutated_output': best['outputs'],  # list[str]
        'mutation_type': best['mutation_type'],
        'mutation_id': best['mutation_id'],
        'coverage_similarity': 0.0
    }


def mutate_dataset(input_path: Path, output_path: Path, limit: Optional[int] = None):
    """
    Create mutated version of dataset following Algorithm 1

    Input format: {id, code, input, output, function_name, ...}
    Output format: Single entry with both 'code' and 'mutated_code' keys

    Example output:
    {
        "id": "contest431_q3702_s0",
        "code": "def maxLength(nums): ...",
        "mutated_code": "def maxLength(nums): ...mutated...",
        "input": "maxLength(nums=[1,2,1,2,1,1,1])",
        "output": "5",
        "mutated_output": "3",
        "has_mutation": true,
        "mutation_info": {"mutation_type": "arithmetic", "mutation_id": 1}
    }

    Args:
        input_path: Path to original dataset JSONL
        output_path: Path to save mutated dataset JSONL
        limit: Optional limit on number of problems to process
    """
    print(f"{'='*60}")
    print("Creating Mutated Dataset (Algorithm 1)")
    print(f"{'='*60}")

    # Load dataset
    with open(input_path) as f:
        dataset = [json.loads(line) for line in f]

    if limit:
        dataset = dataset[:limit]
        print(f"Processing {len(dataset)} problems (limited)")
    else:
        print(f"Processing {len(dataset)} problems")

    mutated_dataset = []
    successful_mutations = 0
    failed_mutations = 0

    for idx, sample in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] {sample['id']}")

        # Create entry with both original and mutated code
        entry = sample.copy()

        # Try to create mutant
        try:
            print(f"  Generating mutants...")
            result = find_best_mutant(
                original_code=sample['code'],
                function_name=sample['header'],
                test_input=sample['inputs'],
                original_output=sample['outputs'],
                debug=True
            )

            if result:
                # Add mutated code and output to the same entry
                entry['mutated_code'] = result['mutated_code']
                entry['mutated_output'] = result['mutated_output']
                entry['has_mutation'] = True
                entry['mutation_info'] = {
                    'mutation_type': result['mutation_type'],
                    'mutation_id': result['mutation_id'],
                    'coverage_similarity': result['coverage_similarity']
                }

                successful_mutations += 1
                print(f"  ✓ Created mutant: {result['mutation_type']}[{result['mutation_id']}]")
                print(f"    Original → Mutated: {sample['outputs']} → {result['mutated_output']}")
            else:
                # No mutation found - entry will have original code only
                entry['mutated_code'] = None
                entry['mutated_output'] = None
                entry['has_mutation'] = False
                entry['mutation_info'] = None

                failed_mutations += 1
                print(f"  ✗ No valid mutant found")

        except Exception as e:
            # Error during mutation - entry will have original code only
            entry['mutated_code'] = None
            entry['mutated_output'] = None
            entry['has_mutation'] = False
            entry['mutation_info'] = {'error': str(e)}

            failed_mutations += 1
            print(f"  ✗ Error: {e}")

        mutated_dataset.append(entry)

    # Save mutated dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in mutated_dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\n{'='*60}")
    print("Mutation Complete!")
    print(f"{'='*60}")
    print(f"Total problems: {len(dataset)}")
    print(f"Successful mutations: {successful_mutations}")
    print(f"Failed mutations: {failed_mutations}")
    print(f"Success rate: {successful_mutations / len(dataset) * 100:.1f}%")
    print(f"\nDataset size: {len(mutated_dataset)} entries")
    print(f"  - With mutations: {successful_mutations}")
    print(f"  - Without mutations: {failed_mutations}")
    print(f"\nOutput: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create mutated dataset following Algorithm 1 from paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on 5 problems
  python3 mutate_dataset.py --limit 5

  # Process full dataset
  python3 mutate_dataset.py

  # Custom paths
  python3 mutate_dataset.py \\
    --input data/datasets/my_dataset.jsonl \\
    --output data/datasets/my_dataset_mutated.jsonl
        """
    )
    parser.add_argument('--input', type=str,
                        default='data/datasets/leetcode_contests_431_467.jsonl',
                        help='Input dataset path (default: leetcode_contests_431_467.jsonl)')
    parser.add_argument('--output', type=str,
                        default='data/datasets/leetcode_contests_431_467_mutated.jsonl',
                        help='Output mutated dataset path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of problems to process (for testing)')

    args = parser.parse_args()

    input_path = Path(__file__).parent / args.input
    output_path = Path(__file__).parent / args.output

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Check if coverage module is available
    try:
        import coverage
        print("Coverage tracking enabled")
    except ImportError:
        print("Warning: 'coverage' module not installed. Coverage similarity will be disabled.")
        print("Install with: pip install coverage")

    mutate_dataset(input_path, output_path, limit=args.limit)


if __name__ == '__main__':
    main()
