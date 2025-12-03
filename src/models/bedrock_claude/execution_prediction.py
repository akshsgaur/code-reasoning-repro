"""Execution Prediction benchmark using Claude Sonnet via AWS Bedrock."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .aws_client import BedrockClaudeClient, BedrockInvocationParams
from .prompts import (
    NormalizedSample,
    build_execution_prediction_prompt,
    check_predicted_output,
    extract_answer_from_response,
    format_output_value,
    is_boolean_output,
    normalize_sample,
)


@dataclass
class ExecutionPredictionConfig:
    num_problems: Optional[int] = 20
    start_index: int = 0
    num_generations: int = 5
    reasoning_effort: str = "medium"
    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95
    skip_boolean_for_reversion: bool = True
    seed: int = 42
    enable_thinking: bool = False
    thinking_budget_tokens: Optional[int] = None
    latency: Optional[str] = None
    log_galileo_metrics: bool = False


@dataclass
class ExecutionPredictionResult:
    metrics_summary: Dict[str, Optional[float]]
    benchmark_summary: Dict[str, Optional[float]]
    metrics_counts: Dict[str, Dict[str, int]]
    results: List[Dict]


def _compute_pass(counts: Dict[str, int]) -> Optional[float]:
    total = counts["total"]
    if total == 0:
        return None
    return counts["success"] / total


def run_execution_prediction(
    dataset_split,
    client: BedrockClaudeClient,
    config: ExecutionPredictionConfig,
) -> ExecutionPredictionResult:
    random.seed(config.seed)

    if config.num_problems is None:
        indices = list(range(len(dataset_split)))
    else:
        stop = min(len(dataset_split), config.start_index + config.num_problems)
        indices = list(range(config.start_index, stop))

    if not indices:
        raise ValueError("No problems selected. Adjust start_index/num_problems.")

    metrics_counts = {
        "OC": {"success": 0, "total": 0},
        "OR": {"success": 0, "total": 0},
        "MC": {"success": 0, "total": 0},
        "MR": {"success": 0, "total": 0},
    }
    reversion_skip_count = 0
    all_latencies: List[float] = []
    results: List[Dict] = []

    for idx in tqdm(indices, desc="Execution Prediction"):
        sample = dataset_split[idx]
        normalized: NormalizedSample = normalize_sample(sample)
        original_outputs = normalized.original_outputs
        mutated_outputs = normalized.mutated_outputs
        has_mutation = sample.get("has_mutation", normalized.mutated_code != normalized.original_code)

        if normalized.mutated_code is None or not str(normalized.mutated_code).strip():
            results.append(
                {
                    "problem_index": int(idx),
                    "problem_id": sample.get("id", idx),
                    "function_name": normalized.function_name,
                    "skipped": True,
                    "skip_reason": "Missing mutated_code.",
                }
            )
            continue

        include_reversion = True
        if config.skip_boolean_for_reversion:
            def _has_boolean(values):
                return any(is_boolean_output(format_output_value(v)) for v in values)

            if _has_boolean(original_outputs) or _has_boolean(mutated_outputs):
                include_reversion = False
                reversion_skip_count += 1

        oc_successes = or_successes = mc_successes = mr_successes = 0
        original_predictions: List[Dict] = []
        mutated_predictions: List[Dict] = []
        seed_base = config.seed + idx * 1000

        for gen_idx in range(config.num_generations):
            params = BedrockInvocationParams(
                reasoning_effort=config.reasoning_effort,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=seed_base + gen_idx,
                enable_thinking=config.enable_thinking,
                thinking_budget_tokens=config.thinking_budget_tokens,
                latency=config.latency,
            )
            params_mut = BedrockInvocationParams(
                reasoning_effort=config.reasoning_effort,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=seed_base + 500 + gen_idx,
                enable_thinking=config.enable_thinking,
                thinking_budget_tokens=config.thinking_budget_tokens,
                latency=config.latency,
            )

            oc_all_correct = True
            or_all_reversion = True
            mc_all_correct = True
            mr_all_reversion = True

            gen_original_records: List[Dict] = []
            gen_mutated_records: List[Dict] = []

            for case_idx, test_input in enumerate(normalized.test_inputs):
                expected_orig = format_output_value(original_outputs[case_idx])
                expected_mut = format_output_value(mutated_outputs[case_idx])

                original_prompt = build_execution_prediction_prompt(
                    normalized, use_mutated=False, test_input=test_input
                )
                response = client.invoke(original_prompt, params)
                prediction = extract_answer_from_response(response.text)
                gen_original_records.append(
                    {
                        "test_input": str(test_input),
                        "expected_output": expected_orig,
                        "mutated_expected_output": expected_mut,
                        "prediction": prediction,
                        "response": response.text,
                        "latency_s": response.latency_s,
                    }
                )
                all_latencies.append(response.latency_s)

                is_correct, _ = check_predicted_output(prediction, expected_orig)
                if not is_correct:
                    oc_all_correct = False
                if include_reversion:
                    is_reversion, _ = check_predicted_output(prediction, expected_mut)
                    if not is_reversion:
                        or_all_reversion = False

                mutated_prompt = build_execution_prediction_prompt(
                    normalized, use_mutated=True, test_input=test_input
                )
                response_mut = client.invoke(mutated_prompt, params_mut)
                prediction_mut = extract_answer_from_response(response_mut.text)
                gen_mutated_records.append(
                    {
                        "test_input": str(test_input),
                        "expected_output": expected_mut,
                        "original_expected_output": expected_orig,
                        "prediction": prediction_mut,
                        "response": response_mut.text,
                        "latency_s": response_mut.latency_s,
                    }
                )
                all_latencies.append(response_mut.latency_s)

                is_mut_correct, _ = check_predicted_output(prediction_mut, expected_mut)
                if not is_mut_correct:
                    mc_all_correct = False
                if include_reversion:
                    is_mut_reversion, _ = check_predicted_output(prediction_mut, expected_orig)
                    if not is_mut_reversion:
                        mr_all_reversion = False

            if oc_all_correct:
                oc_successes += 1
            if include_reversion and or_all_reversion:
                or_successes += 1
            if mc_all_correct:
                mc_successes += 1
            if include_reversion and mr_all_reversion:
                mr_successes += 1

            original_predictions.append(
                {
                    "generation_index": gen_idx,
                    "testcases": gen_original_records,
                    "all_correct": oc_all_correct,
                    "all_reversion": or_all_reversion if include_reversion else None,
                }
            )
            mutated_predictions.append(
                {
                    "generation_index": gen_idx,
                    "testcases": gen_mutated_records,
                    "all_correct": mc_all_correct,
                    "all_reversion": mr_all_reversion if include_reversion else None,
                }
            )

        metrics_counts["OC"]["success"] += oc_successes
        metrics_counts["OC"]["total"] += config.num_generations
        metrics_counts["MC"]["success"] += mc_successes
        metrics_counts["MC"]["total"] += config.num_generations

        if include_reversion:
            metrics_counts["OR"]["success"] += or_successes
            metrics_counts["OR"]["total"] += config.num_generations
            metrics_counts["MR"]["success"] += mr_successes
            metrics_counts["MR"]["total"] += config.num_generations

        results.append(
            {
                "problem_index": int(idx),
                "problem_id": sample.get("id", idx),
                "function_name": normalized.function_name,
                "difficulty": sample.get("difficulty"),
                "has_mutation": has_mutation,
                "include_reversion": include_reversion,
                "original_output": [format_output_value(v) for v in original_outputs],
                "mutated_output": [format_output_value(v) for v in mutated_outputs],
                "test_inputs": normalized.test_inputs,
                "oc_successes": oc_successes,
                "or_successes": or_successes if include_reversion else None,
                "mc_successes": mc_successes,
                "mr_successes": mr_successes if include_reversion else None,
                "original_predictions": original_predictions,
                "mutated_predictions": mutated_predictions,
            }
        )

    metrics_summary = {metric: _compute_pass(counts) for metric, counts in metrics_counts.items()}
    avg_latency = (sum(all_latencies) / len(all_latencies)) if all_latencies else None
    benchmark_summary = {
        "dataset": "LeetCode",
        "problems_evaluated": len(indices),
        "generations_per_problem": config.num_generations,
        "oc_pass_at_1": metrics_summary["OC"],
        "or_pass_at_1": metrics_summary["OR"],
        "mc_pass_at_1": metrics_summary["MC"],
        "mr_pass_at_1": metrics_summary["MR"],
        "avg_latency_s": avg_latency,
        "reversion_skipped_problems": reversion_skip_count if config.skip_boolean_for_reversion else 0,
    }

    return ExecutionPredictionResult(
        metrics_summary=metrics_summary,
        benchmark_summary=benchmark_summary,
        metrics_counts=metrics_counts,
        results=results,
    )


__all__ = [
    "ExecutionPredictionConfig",
    "ExecutionPredictionResult",
    "run_execution_prediction",
]
