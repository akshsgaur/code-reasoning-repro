#!/usr/bin/env python3
"""
Evaluation script for LeetCode dataset
Implements pass@k metric from HumanEval paper (2107.03374v2.pdf)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from math import comb

# BASE_IMPORTS for code execution
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


def check_predicted_output(predicted_output: str, expected_output: str) -> Tuple[bool, Optional[str]]:
    """
    Compare predicted output with expected output

    For execution prediction task, the model predicts what the output will be.
    We simply compare the predicted value with the actual expected value.

    Args:
        predicted_output: Model's predicted output value (extracted from [ANSWER] tags)
        expected_output: Expected output as repr string from dataset

    Returns:
        (is_correct, error_message)
    """
    try:
        # Normalize both strings for comparison
        predicted = predicted_output.strip()
        expected = expected_output.strip()

        # Direct string comparison
        if predicted == expected:
            return (True, None)

        # Try evaluating both as Python literals and compare
        try:
            import ast
            predicted_val = ast.literal_eval(predicted)
            expected_val = ast.literal_eval(expected)

            if predicted_val == expected_val:
                return (True, None)
        except (ValueError, SyntaxError):
            # If we can't parse as literals, fall back to string comparison
            pass

        # Not equal
        return (False, f"Predicted: {predicted}, Expected: {expected}")

    except Exception as e:
        return (False, str(e))


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using unbiased estimator from HumanEval paper

    Formula: pass@k = E[1 - (n-c choose k) / (n choose k)]

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        pass@k score (0.0 to 1.0)
    """
    if n - c < k:
        return 1.0

    return 1.0 - (comb(n - c, k) / comb(n, k))


def evaluate_model_results(
    dataset_path: Path,
    inference_results_path: Path,
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    """
    Evaluate model inference results

    Args:
        dataset_path: Path to original dataset
        inference_results_path: Path to inference results JSONL
        k_values: List of k values for pass@k metric

    Returns:
        Dictionary with evaluation results
    """
    print(f"{'='*60}")
    print(f"Evaluating: {inference_results_path.name}")
    print(f"{'='*60}")

    # Load dataset
    with open(dataset_path) as f:
        dataset = {json.loads(line)['id']: json.loads(line) for line in f}

    # Load inference results
    with open(inference_results_path) as f:
        results = [json.loads(line) for line in f]

    print(f"Dataset problems: {len(dataset)}")
    print(f"Inference results: {len(results)}")

    # Group results by problem
    results_by_problem = defaultdict(list)
    for result in results:
        results_by_problem[result['problem_id']].append(result)

    # Evaluate each problem
    evaluation = {
        'total_problems': len(results_by_problem),
        'total_attempts': len(results),
        'correct_count': 0,
        'problems': {},
        'pass_at_k': {},
    }

    for problem_id, problem_results in results_by_problem.items():
        if problem_id not in dataset:
            print(f"Warning: {problem_id} not in dataset, skipping")
            continue

        problem_data = dataset[problem_id]
        test_input = problem_data['input']
        expected_output = problem_data['output']

        # Evaluate each generated solution
        problem_eval = {
            'total_samples': len(problem_results),
            'correct_samples': 0,
            'attempts': []
        }

        for result in problem_results:
            if not result['success']:
                problem_eval['attempts'].append({
                    'correct': False,
                    'error': result.get('error', 'Inference failed')
                })
                continue

            # Check predicted output against expected output
            is_correct, error = check_predicted_output(
                result['generated_code'],  # This now contains the predicted output
                expected_output
            )

            if is_correct:
                problem_eval['correct_samples'] += 1
                evaluation['correct_count'] += 1

            problem_eval['attempts'].append({
                'correct': is_correct,
                'error': error
            })

        evaluation['problems'][problem_id] = problem_eval

    # Calculate pass@k metrics
    print(f"\n{'='*60}")
    print("Calculating pass@k metrics...")
    print(f"{'='*60}")

    for k in k_values:
        total_pass_at_k = 0.0
        valid_problems = 0

        for problem_id, problem_eval in evaluation['problems'].items():
            n = problem_eval['total_samples']
            c = problem_eval['correct_samples']

            if n >= k:
                pass_k = calculate_pass_at_k(n, c, k)
                total_pass_at_k += pass_k
                valid_problems += 1

        if valid_problems > 0:
            avg_pass_at_k = total_pass_at_k / valid_problems
            evaluation['pass_at_k'][f'pass@{k}'] = avg_pass_at_k
            print(f"pass@{k}: {avg_pass_at_k:.2%} (over {valid_problems} problems)")
        else:
            evaluation['pass_at_k'][f'pass@{k}'] = 0.0
            print(f"pass@{k}: Not enough samples")

    # Overall statistics
    print(f"\n{'='*60}")
    print("Overall Statistics")
    print(f"{'='*60}")
    print(f"Total problems: {evaluation['total_problems']}")
    print(f"Total attempts: {evaluation['total_attempts']}")
    print(f"Correct solutions: {evaluation['correct_count']}")
    print(f"Success rate: {evaluation['correct_count'] / evaluation['total_attempts']:.2%}")

    return evaluation


def main():
    parser = argparse.ArgumentParser(description='Evaluate model inference results')
    parser.add_argument('--dataset', type=str,
                        default='../datasets/leetcode/data/datasets/leetcode_contests_431_467.jsonl',
                        help='Path to dataset JSONL file')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to inference results JSONL file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation results (JSON)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 5, 10],
                        help='k values for pass@k metric')

    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset
    results_path = Path(args.results)

    # Run evaluation
    evaluation = evaluate_model_results(
        dataset_path=dataset_path,
        inference_results_path=results_path,
        k_values=args.k_values
    )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)

        print(f"\nEvaluation results saved to: {output_path}")


if __name__ == '__main__':
    main()
