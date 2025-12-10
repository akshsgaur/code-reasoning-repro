"""Execution Choice benchmark via Claude Sonnet on Bedrock."""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .aws_client import BedrockClaudeClient, BedrockInvocationParams
from .prompts import (
    NormalizedSample,
    build_execution_choice_prompt,
    check_predicted_output,
    extract_output_from_assertion,
    format_output_value,
    is_boolean_output,
    normalize_sample,
    parse_execution_choice_response,
)


@dataclass
class ExecutionChoiceConfig:
    num_problems: Optional[int] = 20
    start_index: int = 0
    runs_per_problem: int = 2
    reasoning_effort: str = "medium"
    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95
    skip_boolean_for_reversion: bool = True
    seed: int = 123
    enable_thinking: bool = False
    thinking_budget_tokens: Optional[int] = None
    latency: Optional[str] = None
    log_galileo_metrics: bool = False


@dataclass
class ExecutionChoiceResult:
    summary: Dict[str, Optional[float]]
    counts: Dict[str, Dict[str, int]]
    results: List[Dict]


def run_execution_choice(
    dataset_split,
    client: BedrockClaudeClient,
    config: ExecutionChoiceConfig,
) -> ExecutionChoiceResult:
    if config.runs_per_problem not in (1, 2):
        raise ValueError("runs_per_problem must be 1 or 2 to support order swapping.")

    random.seed(config.seed)

    if config.num_problems is None:
        indices = list(range(len(dataset_split)))
    else:
        stop = min(len(dataset_split), config.start_index + config.num_problems)
        indices = list(range(config.start_index, stop))

    if not indices:
        raise ValueError("No problems selected for execution choice.")

    execution_choice_counts = {
        "preference": {"original": 0, "mutated": 0, "total": 0},
        "OC": {"correct": 0, "total": 0, "reversion_correct": 0, "reversion_total": 0},
        "MC": {"correct": 0, "total": 0, "reversion_correct": 0, "reversion_total": 0},
        "invalid_runs": 0,
    }

    execution_choice_results: List[Dict] = []
    execution_choice_latencies: List[float] = []
    reversion_skip_count = 0
    orderings = [True, False]
    selected_orderings = orderings[: config.runs_per_problem]

    for idx in tqdm(indices, desc="Execution Choice"):
        sample = dataset_split[idx]
        normalized: NormalizedSample = normalize_sample(sample)
        original_outputs = normalized.original_outputs
        mutated_outputs = normalized.mutated_outputs

        include_reversion = True
        if config.skip_boolean_for_reversion:
            def _has_boolean(values):
                return any(is_boolean_output(format_output_value(v)) for v in values)

            if _has_boolean(original_outputs) or _has_boolean(mutated_outputs):
                include_reversion = False
                reversion_skip_count += 1

        base_seed = config.seed + idx * 1000

        def _trace_metadata(run_index: int, original_first_flag: bool) -> Dict[str, str]:
            return {
                "benchmark": "execution_choice",
                "problem_id": str(sample.get("id", idx)),
                "problem_index": str(int(idx)),
                "run_index": str(run_index),
                "original_first": str(original_first_flag),
                "include_reversion": str(include_reversion),
            }

        for run_offset, original_first in enumerate(selected_orderings):
            params = BedrockInvocationParams(
                reasoning_effort=config.reasoning_effort,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=base_seed + run_offset,
                enable_thinking=config.enable_thinking,
                thinking_budget_tokens=config.thinking_budget_tokens,
                latency=config.latency,
            )

            oc_all_correct = True
            or_all_reversion = True
            mc_all_correct = True
            mr_all_reversion = True

            run_testcases: List[Dict] = []

            for case_idx, test_input in enumerate(normalized.test_inputs):
                prompt, mapping = build_execution_choice_prompt(
                    normalized, original_first=original_first, test_input=test_input
                )
                try:
                    generation = client.invoke(prompt, params)
                except RuntimeError as exc:
                    # Bedrock content filter / invocation failure – treat as invalid run
                    run_record = {
                        "problem_index": int(idx),
                        "problem_id": sample.get("id", idx),
                        "function_name": normalized.function_name,
                        "run_index": run_offset,
                        "original_first": original_first,
                        "test_input": str(test_input),
                        "response": "",  # No model text
                        "latency_s": 0.0,
                        "include_reversion": None,
                        "chosen_program_letter": None,
                        "chosen_program_type": None,
                        "assertion": None,
                        "prediction": None,
                        "correct_for_chosen_program": None,
                        "reversion_for_other_program": None,
                        "correctness_error": f"Invocation failed: {exc}",
                        "reversion_error": None,
                    }
                    execution_choice_counts["invalid_runs"] += 1
                    run_testcases.append(run_record)
                    continue
                
                execution_choice_latencies.append(generation.latency_s)

                run_record = {
                    "problem_index": int(idx),
                    "problem_id": sample.get("id", idx),
                    "function_name": normalized.function_name,
                    "run_index": run_offset,
                    "original_first": original_first,
                    "test_input": str(test_input),
                    "response": generation.text,
                    "latency_s": generation.latency_s,
                    "include_reversion": None,
                    "chosen_program_letter": None,
                    "chosen_program_type": None,
                    "assertion": None,
                    "prediction": None,
                    "correct_for_chosen_program": None,
                    "reversion_for_other_program": None,
                    "correctness_error": None,
                    "reversion_error": None,
                }

                try:
                    parsed = parse_execution_choice_response(generation.text)
                    chosen_letter = parsed.get("chosen_program")
                    assertion_text = parsed.get("assertion", "")
                    outputs_text = parsed.get("outputs")
                    if not chosen_letter or chosen_letter not in mapping:
                        raise ValueError("Missing/invalid chosen_program in response.")
                    chosen_type = mapping[chosen_letter]
                except Exception as exc:
                    run_record["correctness_error"] = f"Failed to parse response: {exc}"
                    execution_choice_counts["invalid_runs"] += 1
                    run_testcases.append(run_record)
                    continue

                if outputs_text is not None:
                    outputs_value = outputs_text
                    if isinstance(outputs_text, str):
                        try:
                            outputs_value = ast.literal_eval(outputs_text)
                        except Exception:
                            outputs_value = [outputs_text]
                    if isinstance(outputs_value, list) and len(outputs_value) == 1:
                        predicted_output = format_output_value(outputs_value[0])
                    elif isinstance(outputs_value, list):
                        predicted_output = format_output_value(outputs_value)
                    else:
                        predicted_output = format_output_value(outputs_value)
                else:
                    predicted_output = extract_output_from_assertion(assertion_text)

                expected_original = format_output_value(original_outputs[case_idx])
                expected_mutated = format_output_value(mutated_outputs[case_idx])

                chosen_output = expected_original if chosen_type == "original" else expected_mutated
                other_output = expected_mutated if chosen_type == "original" else expected_original

                case_include_reversion = True
                if expected_original == expected_mutated:
                    case_include_reversion = False
                if config.skip_boolean_for_reversion and (
                    is_boolean_output(expected_original) or is_boolean_output(expected_mutated)
                ):
                    case_include_reversion = False
                    reversion_skip_count += 1

                run_record["include_reversion"] = case_include_reversion

                is_correct, correctness_error = check_predicted_output(predicted_output, chosen_output)
                is_correct_flag = bool(is_correct)

                if case_include_reversion:
                    is_reversion_raw, reversion_error = check_predicted_output(predicted_output, other_output)
                    is_reversion_flag = (not is_correct_flag) and bool(is_reversion_raw)
                else:
                    is_reversion_flag, reversion_error = None, None

                if chosen_type == "original":
                    if not is_correct_flag:
                        oc_all_correct = False
                    if case_include_reversion and not is_reversion_flag:
                        or_all_reversion = False
                else:
                    if not is_correct_flag:
                        mc_all_correct = False
                    if case_include_reversion and not is_reversion_flag:
                        mr_all_reversion = False

                execution_choice_counts["preference"]["total"] += 1
                if chosen_type == "original":
                    execution_choice_counts["preference"]["original"] += 1
                    bucket = execution_choice_counts["OC"]
                else:
                    execution_choice_counts["preference"]["mutated"] += 1
                    bucket = execution_choice_counts["MC"]

                bucket["total"] += 1
                if is_correct_flag:
                    bucket["correct"] += 1
                if case_include_reversion:
                    bucket["reversion_total"] += 1
                    if is_reversion_flag:
                        bucket["reversion_correct"] += 1

                run_record.update(
                    {
                        "chosen_program_letter": chosen_letter,
                        "chosen_program_type": chosen_type,
                        "assertion": assertion_text,
                        "prediction": predicted_output,
                        "expected_output": chosen_output,
                        "other_output": other_output,
                        "correct_for_chosen_program": is_correct_flag,
                        "reversion_for_other_program": is_reversion_flag if case_include_reversion else None,
                        "correctness_error": correctness_error,
                        "reversion_error": reversion_error,
                    }
                )

                run_testcases.append(run_record)

            if oc_all_correct and mc_all_correct:
                pass
            execution_choice_results.extend(run_testcases)

    def _safe_ratio(num: int, denom: int) -> Optional[float]:
        return None if denom == 0 else num / denom

    preference_total = execution_choice_counts["preference"]["total"]
    summary = {
        "dataset": "LeetCode",
        "problems_evaluated": len(indices),
        "runs_per_problem": config.runs_per_problem,
        "preference_original": _safe_ratio(
            execution_choice_counts["preference"]["original"], preference_total
        ),
        "preference_mutated": _safe_ratio(
            execution_choice_counts["preference"]["mutated"], preference_total
        ),
        "oc_correct": _safe_ratio(
            execution_choice_counts["OC"]["correct"], execution_choice_counts["OC"]["total"]
        ),
        "or_reversion": _safe_ratio(
            execution_choice_counts["OC"]["reversion_correct"],
            execution_choice_counts["OC"]["reversion_total"],
        ),
        "mc_correct": _safe_ratio(
            execution_choice_counts["MC"]["correct"], execution_choice_counts["MC"]["total"]
        ),
        "mr_reversion": _safe_ratio(
            execution_choice_counts["MC"]["reversion_correct"],
            execution_choice_counts["MC"]["reversion_total"],
        ),
        "avg_latency_s": (
            sum(execution_choice_latencies) / len(execution_choice_latencies)
            if execution_choice_latencies
            else None
        ),
        "invalid_runs": execution_choice_counts["invalid_runs"],
        "reversion_skipped_problems": reversion_skip_count if config.skip_boolean_for_reversion else 0,
    }

    return ExecutionChoiceResult(summary=summary, counts=execution_choice_counts, results=execution_choice_results)


__all__ = [
    "ExecutionChoiceConfig",
    "ExecutionChoiceResult",
    "run_execution_choice",
]
