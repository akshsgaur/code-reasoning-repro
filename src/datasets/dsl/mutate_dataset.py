#!/usr/bin/env python3
"""Mutate DSL-generated programs following the LeetCode mutation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .simple_mutator import generate_all_mutants

MAX_MUTATED = 100


def mutate_records(records: list[Dict], limit: int = MAX_MUTATED) -> list[Dict]:
    """Return up to `limit` mutated dataset entries."""

    mutated: list[Dict] = []
    for entry in records:
        if len(mutated) >= limit:
            break
        code = entry["code"]
        call_examples = entry["input"]
        outputs = entry["output"]

        best_mutation = None

        mutants = generate_all_mutants(code)
        for mutated_candidate, mutation_type, mutation_id in mutants:
            candidate_outputs = []
            valid = True
            for call_expr, base_output in zip(call_examples, outputs):
                success, mutated_output = _execute_code(mutated_candidate, entry["function_name"], call_expr)
                if not success or mutated_output == base_output:
                    valid = False
                    break
                candidate_outputs.append(mutated_output)
            if valid and len(set(candidate_outputs)) > 1:
                best_mutation = {
                    "mutated_code": mutated_candidate,
                    "mutated_output": candidate_outputs,
                    "mutation_type": mutation_type,
                    "mutation_id": mutation_id,
                }
                break

        if not best_mutation:
            continue

        entry = entry.copy()
        entry["has_mutation"] = True
        entry["mutated_code"] = best_mutation["mutated_code"]
        entry["mutated_output"] = best_mutation["mutated_output"]
        entry["mutation_info"] = {
            "mutation_type": best_mutation["mutation_type"],
            "mutation_id": best_mutation["mutation_id"],
            "coverage_similarity": 0.0,
        }
        mutated.append(entry)
    return mutated


def _find_mutation(code: str, fn_name: str, call_expr: str, output_literal: str) -> Optional[Dict]:
    """Return a valid mutation if one exists."""

    original_output = output_literal
    mutants = generate_all_mutants(code)
    for mutated_code, mutation_type, mutation_id in mutants:
        success, mutated_output = _execute_code(mutated_code, fn_name, call_expr)
        if not success or mutated_output == original_output:
            continue
        return {
            "mutated_code": mutated_code,
            "mutated_output": mutated_output,
            "mutation_type": mutation_type,
            "mutation_id": mutation_id,
        }
    return None


def _execute_code(code: str, function_name: str, call_expr: str) -> tuple[bool, str]:
    """Execute code and return (success, repr(result))."""

    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("Execution timeout")

    exec_globals: Dict[str, object] = {}
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        exec(code, exec_globals)  # noqa: S102
        result = eval(call_expr, exec_globals)  # noqa: S307
        signal.alarm(0)
        return True, repr(result)
    except Exception:
        signal.alarm(0)
        return False, ""


def mutate_dataset(json_path: Path, jsonl_path: Path, limit: int = MAX_MUTATED) -> None:
    """Load programs_structured.json, apply mutations, and rewrite in place."""

    data = json.loads(json_path.read_text())
    mutated = mutate_records(data, limit=limit)
    if len(mutated) < limit:
        print(f"Warning: only {len(mutated)} mutated programs generated (requested {limit}).")
    json_path.write_text(json.dumps(mutated, indent=2))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for entry in mutated:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    target = Path(__file__).parent / "data" / "programs_structured.json"
    jsonl_target = Path(__file__).parent / "data" / "programs.jsonl"
    mutate_dataset(target, jsonl_target)
