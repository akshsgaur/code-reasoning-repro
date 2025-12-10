"""Execution Choice benchmark via Claude Sonnet on Bedrock."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .aws_client import BedrockClaudeClient, BedrockInvocationParams
from .prompts import (
    build_execution_choice_prompt,
    check_predicted_output,
    extract_output_from_assertion,
    is_boolean_output,
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
        original_output = sample["output"]
        mutated_output = sample.get("mutated_output") or original_output

        include_reversion = True
        if config.skip_boolean_for_reversion and (
            is_boolean_output(original_output) or is_boolean_output(mutated_output)
        ):
            include_reversion = False
            reversion_skip_count += 1

        base_seed = config.seed + idx * 1000

        def _trace_metadata(run_index: int, original_first_flag: bool) -> Dict[str, str]:
            return {
                "benchmark": "execution_choice",
                "problem_id": str(sample["id"]),
                "problem_index": str(int(idx)),
                "run_index": str(run_index),
                "original_first": str(original_first_flag),
                "include_reversion": str(include_reversion),
            }

        for run_offset, original_first in enumerate(selected_orderings):
            prompt, mapping = build_execution_choice_prompt(sample, original_first=original_first)
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
            choice_meta_cache: Dict[str, object] = {}

            def _choice_metadata_builder(response_text: str) -> Dict[str, object]:
                metadata: Dict[str, object] = {}
                try:
                    parsed = parse_execution_choice_response(response_text)
                except Exception as exc:
                    choice_meta_cache["error"] = f"Failed to parse response: {exc}"
                    metadata["choice_parse_error"] = str(exc)
                    return metadata
                assertion_text = parsed.get("assertion", "") or ""
                chosen_letter = parsed.get("chosen_program")
                choice_meta_cache.update(
                    {
                        "parsed": parsed,
                        "assertion": assertion_text,
                        "chosen_letter": chosen_letter,
                    }
                )
                if not chosen_letter or chosen_letter not in mapping:
                    error_msg = "Missing/invalid chosen_program in response."
                    choice_meta_cache["error"] = error_msg
                    metadata["choice_parse_error"] = error_msg
                    return metadata

                chosen_type = mapping[chosen_letter]
                predicted_output = extract_output_from_assertion(assertion_text)
                chosen_output = original_output if chosen_type == "original" else mutated_output
                other_output = mutated_output if chosen_type == "original" else original_output
                is_correct, _ = check_predicted_output(predicted_output, chosen_output)
                metadata_key = "metric_oc" if chosen_type == "original" else "metric_mc"
                metadata[metadata_key] = bool(is_correct)

                if include_reversion:
                    is_reversion, _ = check_predicted_output(predicted_output, other_output)
                    metadata_key_rev = "metric_or" if chosen_type == "original" else "metric_mr"
                    metadata[metadata_key_rev] = is_reversion
                else:
                    metadata_key_rev = "metric_or" if chosen_type == "original" else "metric_mr"
                    metadata[metadata_key_rev] = "skipped"

                choice_meta_cache.update(
                    {
                        "chosen_type": chosen_type,
                        "prediction": predicted_output,
                        "is_correct": bool(is_correct),
                        "is_reversion": metadata[metadata_key_rev]
                        if include_reversion
                        else None,
                    }
                )
                metadata["choice_variant"] = chosen_type
                metadata["chosen_letter"] = chosen_letter
                return metadata

            generation = client.invoke(
                prompt,
                params,
                trace_metadata=_trace_metadata(run_offset, original_first)
                if config.log_galileo_metrics
                else None,
                metadata_postprocessor=_choice_metadata_builder if config.log_galileo_metrics else None,
            )
            execution_choice_latencies.append(generation.latency_s)

            run_record = {
                "problem_index": int(idx),
                "problem_id": sample["id"],
                "function_name": sample["function_name"],
                "run_index": run_offset,
                "original_first": original_first,
                "response": generation.text,
                "latency_s": generation.latency_s,
                "include_reversion": include_reversion,
                "chosen_program_letter": None,
                "chosen_program_type": None,
                "assertion": None,
                "prediction": None,
                "correct_for_chosen_program": None,
                "reversion_for_other_program": None,
                "correctness_error": None,
                "reversion_error": None,
            }

            cache_error = choice_meta_cache.get("error")
            if cache_error:
                run_record["correctness_error"] = str(cache_error)
                execution_choice_counts["invalid_runs"] += 1
                execution_choice_results.append(run_record)
                continue

            correctness_error = None
            reversion_error = None
            if choice_meta_cache:
                chosen_letter = choice_meta_cache.get("chosen_letter")
                chosen_type = choice_meta_cache.get("chosen_type")
                assertion_text = str(choice_meta_cache.get("assertion", ""))
                predicted_output = str(choice_meta_cache.get("prediction", ""))
                if not chosen_letter or not isinstance(chosen_type, str):
                    run_record["correctness_error"] = "Missing/invalid chosen_program in response."
                    execution_choice_counts["invalid_runs"] += 1
                    execution_choice_results.append(run_record)
                    continue
                is_correct = bool(choice_meta_cache.get("is_correct"))
                reversion_cache = choice_meta_cache.get("is_reversion")
                is_reversion = bool(reversion_cache) if isinstance(reversion_cache, bool) else None
            else:
                try:
                    parsed = parse_execution_choice_response(generation.text)
                    chosen_letter = parsed.get("chosen_program")
                    assertion_text = parsed.get("assertion", "")
                    if not chosen_letter or chosen_letter not in mapping:
                        raise ValueError("Missing/invalid chosen_program in response.")
                    chosen_type = mapping[chosen_letter]
                except Exception as exc:
                    run_record["correctness_error"] = f"Failed to parse response: {exc}"
                    execution_choice_counts["invalid_runs"] += 1
                    execution_choice_results.append(run_record)
                    continue

                predicted_output = extract_output_from_assertion(assertion_text)
                is_correct, correctness_error = check_predicted_output(
                    predicted_output, original_output if chosen_type == "original" else mutated_output
                )
                if include_reversion:
                    is_reversion, reversion_error = check_predicted_output(
                        predicted_output, mutated_output if chosen_type == "original" else original_output
                    )
                else:
                    is_reversion, reversion_error = None, None

                is_correct = bool(is_correct)
                is_reversion = bool(is_reversion) if isinstance(is_reversion, bool) else None

            execution_choice_counts["preference"]["total"] += 1
            if chosen_type == "original":
                execution_choice_counts["preference"]["original"] += 1
                bucket = execution_choice_counts["OC"]
            else:
                execution_choice_counts["preference"]["mutated"] += 1
                bucket = execution_choice_counts["MC"]

            bucket["total"] += 1
            if is_correct:
                bucket["correct"] += 1
            if include_reversion:
                bucket["reversion_total"] += 1
                if is_reversion:
                    bucket["reversion_correct"] += 1

            run_record.update(
                {
                    "chosen_program_letter": chosen_letter,
                    "chosen_program_type": chosen_type,
                    "assertion": assertion_text,
                    "prediction": predicted_output,
                    "correct_for_chosen_program": bool(is_correct),
                    "reversion_for_other_program": bool(is_reversion)
                    if include_reversion
                    else None,
                    "correctness_error": correctness_error,
                    "reversion_error": reversion_error,
                }
            )

            execution_choice_results.append(run_record)

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
