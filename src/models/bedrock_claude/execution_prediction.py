"""Execution Prediction benchmark using Claude Sonnet via AWS Bedrock."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .aws_client import BedrockClaudeClient, BedrockInvocationParams
from .prompts import (
    build_execution_prediction_prompt,
    check_predicted_output,
    extract_answer_from_response,
    is_boolean_output,
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
        original_output = sample["output"]
        mutated_output = sample.get("mutated_output") or original_output
        has_mutation = sample.get("has_mutation", mutated_output != original_output)

        original_prompt = build_execution_prediction_prompt(sample, use_mutated=False)
        mutated_prompt = build_execution_prediction_prompt(sample, use_mutated=True)

        include_reversion = True
        if config.skip_boolean_for_reversion and (
            is_boolean_output(original_output) or is_boolean_output(mutated_output)
        ):
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
            response = client.invoke(original_prompt, params)
            prediction = extract_answer_from_response(response.text)
            original_pred = {
                "prediction": prediction,
                "response": response.text,
                "latency_s": response.latency_s,
            }
            original_predictions.append(original_pred)
            all_latencies.append(response.latency_s)

            is_correct, _ = check_predicted_output(prediction, original_output)
            if is_correct:
                oc_successes += 1

            if include_reversion:
                is_reversion, _ = check_predicted_output(prediction, mutated_output)
                if is_reversion:
                    or_successes += 1

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
            response_mut = client.invoke(mutated_prompt, params_mut)
            prediction_mut = extract_answer_from_response(response_mut.text)
            mutated_pred = {
                "prediction": prediction_mut,
                "response": response_mut.text,
                "latency_s": response_mut.latency_s,
            }
            mutated_predictions.append(mutated_pred)
            all_latencies.append(response_mut.latency_s)

            is_mut_correct, _ = check_predicted_output(prediction_mut, mutated_output)
            if is_mut_correct:
                mc_successes += 1

            if include_reversion:
                is_mut_reversion, _ = check_predicted_output(prediction_mut, original_output)
                if is_mut_reversion:
                    mr_successes += 1

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
                "problem_id": sample["id"],
                "function_name": sample["function_name"],
                "difficulty": sample.get("difficulty"),
                "has_mutation": has_mutation,
                "include_reversion": include_reversion,
                "original_output": original_output,
                "mutated_output": mutated_output,
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
