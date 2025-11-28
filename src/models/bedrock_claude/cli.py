"""Command-line runner for Claude Sonnet 4.5 evaluations via AWS Bedrock."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from .aws_client import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    BedrockClaudeClient,
    BedrockClientConfig,
    BedrockInvocationParams,
)
from .execution_choice import ExecutionChoiceConfig, run_execution_choice
from .execution_prediction import ExecutionPredictionConfig, run_execution_prediction
from .galileo_logger import GalileoTraceConfig, GalileoTracer
from .prompts import build_execution_prediction_prompt, check_predicted_output, extract_answer_from_response


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _serialize_prediction_payload(result, config: ExecutionPredictionConfig, client_config: BedrockClientConfig) -> Dict:
    return {
        "model": "claude-3.5-sonnet-bedrock",
        "bedrock_model_id": client_config.model_id,
        "aws_region": client_config.region,
        "config": asdict(config),
        "metrics_summary": result.metrics_summary,
        "benchmark_summary": result.benchmark_summary,
        "metrics_counts": result.metrics_counts,
        "results": result.results,
    }


def _serialize_choice_payload(result, config: ExecutionChoiceConfig) -> Dict:
    return {
        "execution_choice_summary": result.summary,
        "execution_choice_counts": result.counts,
        "execution_choice_results": result.results,
        "execution_choice_config": asdict(config),
    }


def _compare_reasoning_levels(
    dataset_split,
    client: BedrockClaudeClient,
    comparison_samples: int,
    base_params: BedrockInvocationParams,
) -> Dict[str, Dict[str, float]]:
    reasoning_levels = ["low", "medium", "high"]
    summary: Dict[str, Dict[str, float]] = {}

    for level in reasoning_levels:
        correct = 0
        total_latency = 0.0
        for idx in range(min(comparison_samples, len(dataset_split))):
            sample = dataset_split[idx]
            prompt = build_execution_prediction_prompt(sample)
            params = BedrockInvocationParams(
                reasoning_effort=level,
                max_tokens=base_params.max_tokens,
                temperature=base_params.temperature,
                top_p=base_params.top_p,
                seed=base_params.seed + idx if base_params.seed is not None else None,
                enable_thinking=base_params.enable_thinking,
                thinking_budget_tokens=base_params.thinking_budget_tokens,
                latency=base_params.latency,
            )
            response = client.invoke(prompt, params)
            predicted_output = extract_answer_from_response(response.text)
            is_correct, _ = check_predicted_output(predicted_output, sample["output"])
            if is_correct:
                correct += 1
            total_latency += response.latency_s

        summary[level] = {
            "pass_at_1": correct / comparison_samples,
            "avg_latency_s": total_latency / comparison_samples,
        }

    return summary


def _filter_missing_mutations(dataset_split: Dataset) -> Tuple[Dataset, int, List[str]]:
    dropped_preview: List[str] = []
    dropped_count = 0

    def _has_mutation(example, idx):  # type: ignore[override]
        nonlocal dropped_count
        mutated_code = example.get("mutated_code")
        if mutated_code is None or (isinstance(mutated_code, str) and not mutated_code.strip()):
            dropped_count += 1
            if len(dropped_preview) < 5:
                dropped_preview.append(str(example.get("id", idx)))
            return False
        return True

    filtered = dataset_split.filter(_has_mutation, with_indices=True)
    return filtered, dropped_count, dropped_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Claude Sonnet 4.5 evaluations via AWS Bedrock")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset repo id (e.g., user/leetcode-contests-431-467)")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/bedrock_claude"))
    parser.add_argument("--task", choices=["prediction", "choice", "both"], default="prediction")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--region", default=None, help="AWS region (defaults to AWS_REGION env)")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Claude extended thinking mode")
    parser.add_argument("--thinking-budget-tokens", type=int, default=None, help="Budget tokens for thinking mode")
    parser.add_argument("--latency-profile", default=None, help="Performance config latency hint (e.g., standard)")

    # Prediction options
    parser.add_argument("--pred-num-problems", type=int, default=20)
    parser.add_argument("--pred-start-index", type=int, default=0)
    parser.add_argument("--pred-generations", type=int, default=5)
    parser.add_argument("--pred-reasoning", default="medium")
    parser.add_argument("--pred-max-tokens", type=int, default=1000)
    parser.add_argument("--pred-temperature", type=float, default=0.6)
    parser.add_argument("--pred-top-p", type=float, default=0.95)
    parser.add_argument("--pred-seed", type=int, default=42)
    parser.add_argument("--pred-skip-boolean", action="store_true")

    # Choice options
    parser.add_argument("--choice-num-problems", type=int, default=20)
    parser.add_argument("--choice-start-index", type=int, default=0)
    parser.add_argument("--choice-runs", type=int, default=2)
    parser.add_argument("--choice-reasoning", default="medium")
    parser.add_argument("--choice-max-tokens", type=int, default=1000)
    parser.add_argument("--choice-temperature", type=float, default=0.6)
    parser.add_argument("--choice-top-p", type=float, default=0.95)
    parser.add_argument("--choice-seed", type=int, default=123)
    parser.add_argument("--choice-skip-boolean", action="store_true")

    parser.add_argument("--compare-reasoning", action="store_true", help="Run optional low/medium/high comparison")
    parser.add_argument("--comparison-samples", type=int, default=5)

    parser.add_argument(
        "--include-missing-mutations",
        action="store_true",
        help="Keep problems without mutated_code entries (default skips them)",
    )

    parser.add_argument(
        "--enable-galileo",
        action="store_true",
        help="Log Bedrock traces to Galileo (requires GALILEO_API_KEY)",
    )
    parser.add_argument("--galileo-project", default=None, help="Override Galileo project name")
    parser.add_argument("--galileo-log-stream", default=None, help="Override Galileo log stream name")

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset_dict = load_dataset(args.dataset)
    if args.split not in dataset_dict:
        raise ValueError(f"Split '{args.split}' not found in dataset {args.dataset}")
    dataset_split = dataset_dict[args.split]
    if not args.include_missing_mutations:
        dataset_split, dropped_count, dropped_preview = _filter_missing_mutations(dataset_split)
        if dropped_count:
            preview_msg = ", ".join(dropped_preview)
            if dropped_count > len(dropped_preview):
                preview_msg += ", ..."
            print(
                f"⚠️  Skipped {dropped_count} problems missing mutated_code: {preview_msg}"
            )

    region = args.region or DEFAULT_REGION
    client_config = BedrockClientConfig(
        model_id=args.model_id,
        region=region,
    )
    galileo_tracer = None
    if args.enable_galileo:
        dataset_slug = args.dataset.replace("/", "-")
        project = (
            args.galileo_project
            or os.environ.get("GALILEO_PROJECT")
            or dataset_slug
        )
        log_stream = (
            args.galileo_log_stream
            or os.environ.get("GALILEO_LOG_STREAM")
            or f"{dataset_slug}-{args.task}"
        )
        galileo_tracer = GalileoTracer(
            GalileoTraceConfig(
                enabled=True,
                project=project,
                log_stream=log_stream,
            )
        )

    client = BedrockClaudeClient(client_config, galileo_tracer=galileo_tracer)

    outputs: Dict[str, Dict] = {}

    if args.task in {"prediction", "both"}:
        pred_config = ExecutionPredictionConfig(
            num_problems=None if args.pred_num_problems <= 0 else args.pred_num_problems,
            start_index=args.pred_start_index,
            num_generations=args.pred_generations,
            reasoning_effort=args.pred_reasoning,
            max_new_tokens=args.pred_max_tokens,
            temperature=args.pred_temperature,
            top_p=args.pred_top_p,
            skip_boolean_for_reversion=args.pred_skip_boolean,
            seed=args.pred_seed,
            enable_thinking=args.enable_thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            latency=args.latency_profile,
        )
        if (
            pred_config.enable_thinking
            and pred_config.thinking_budget_tokens is not None
            and pred_config.max_new_tokens is not None
            and pred_config.thinking_budget_tokens >= pred_config.max_new_tokens
        ):
            raise ValueError("pred-max-tokens must exceed thinking-budget-tokens when thinking is enabled")
        prediction_result = run_execution_prediction(dataset_split, client, pred_config)
        outputs["prediction"] = _serialize_prediction_payload(prediction_result, pred_config, client_config)

    if args.task in {"choice", "both"}:
        choice_config = ExecutionChoiceConfig(
            num_problems=None if args.choice_num_problems <= 0 else args.choice_num_problems,
            start_index=args.choice_start_index,
            runs_per_problem=args.choice_runs,
            reasoning_effort=args.choice_reasoning,
            max_new_tokens=args.choice_max_tokens,
            temperature=args.choice_temperature,
            top_p=args.choice_top_p,
            skip_boolean_for_reversion=args.choice_skip_boolean,
            seed=args.choice_seed,
            enable_thinking=args.enable_thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            latency=args.latency_profile,
        )
        choice_result = run_execution_choice(dataset_split, client, choice_config)
        outputs["choice"] = _serialize_choice_payload(choice_result, choice_config)

    if args.compare_reasoning:
        params = BedrockInvocationParams(
            reasoning_effort="medium",
            max_tokens=args.pred_max_tokens,
            temperature=args.pred_temperature,
            top_p=args.pred_top_p,
            seed=args.pred_seed,
            enable_thinking=args.enable_thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            latency=args.latency_profile,
        )
        comparison = _compare_reasoning_levels(
            dataset_split,
            client,
            comparison_samples=args.comparison_samples,
            base_params=params,
        )
        outputs["reasoning_comparison"] = comparison

    if not outputs:
        raise RuntimeError("No tasks executed. Check --task option.")

    out_path = args.output_dir / f"claude_sonnet_bedrock_{_timestamp()}.json"
    _write_json(outputs, out_path)
    print(f"✓ Saved evaluation summary to {out_path}")

    if galileo_tracer and galileo_tracer.is_enabled:
        galileo_tracer.flush()
        links = galileo_tracer.get_console_links() or {}
        print("\n🚀 Galileo trace logging enabled:")
        project_url = links.get("project")
        log_stream_url = links.get("log_stream")
        if project_url:
            print(f"  Project   : {project_url}")
        if log_stream_url:
            print(f"  Log Stream: {log_stream_url}")


if __name__ == "__main__":
    main()
