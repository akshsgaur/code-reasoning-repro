"""Command-line runner for GPT-OSS evaluations on GPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from datasets import load_dataset

from .gpu_client import DEFAULT_MODEL_ID, GPTOSSClient, GPTOSSClientConfig
from .execution_choice import ExecutionChoiceConfig, run_execution_choice
from .execution_prediction import ExecutionPredictionConfig, run_execution_prediction


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS evaluations on GPU")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/gpt_oss"))
    parser.add_argument("--task", choices=["prediction", "choice", "both"], default="prediction")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # Prediction options
    parser.add_argument("--pred-num-problems", type=int, default=20)
    parser.add_argument("--pred-start-index", type=int, default=0)
    parser.add_argument("--pred-generations", type=int, default=5)
    parser.add_argument("--pred-max-tokens", type=int, default=512)
    parser.add_argument("--pred-temperature", type=float, default=0.6)
    parser.add_argument("--pred-top-p", type=float, default=0.95)
    parser.add_argument("--pred-seed", type=int, default=42)
    parser.add_argument("--pred-skip-boolean", action="store_true")

    # Choice options
    parser.add_argument("--choice-num-problems", type=int, default=20)
    parser.add_argument("--choice-start-index", type=int, default=0)
    parser.add_argument("--choice-runs", type=int, default=2)
    parser.add_argument("--choice-max-tokens", type=int, default=512)
    parser.add_argument("--choice-temperature", type=float, default=0.6)
    parser.add_argument("--choice-top-p", type=float, default=0.95)
    parser.add_argument("--choice-seed", type=int, default=123)
    parser.add_argument("--choice-skip-boolean", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dict = load_dataset(args.dataset)
    if args.split not in dataset_dict:
        raise ValueError(f"Split '{args.split}' not found in dataset {args.dataset}")
    dataset_split = dataset_dict[args.split]

    client_config = GPTOSSClientConfig(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    client = GPTOSSClient(client_config)

    outputs: Dict[str, Dict] = {}

    if args.task in {"prediction", "both"}:
        pred_config = ExecutionPredictionConfig(
            num_problems=None if args.pred_num_problems <= 0 else args.pred_num_problems,
            start_index=args.pred_start_index,
            num_generations=args.pred_generations,
            max_new_tokens=args.pred_max_tokens,
            temperature=args.pred_temperature,
            top_p=args.pred_top_p,
            skip_boolean_for_reversion=args.pred_skip_boolean,
            seed=args.pred_seed,
        )
        prediction_result = run_execution_prediction(dataset_split, client, pred_config)
        outputs["prediction"] = {
            "model": "gpt-oss",
            "model_id": client_config.model_id,
            "config": asdict(pred_config),
            "metrics_summary": prediction_result.metrics_summary,
            "benchmark_summary": prediction_result.benchmark_summary,
            "metrics_counts": prediction_result.metrics_counts,
            "results": prediction_result.results,
        }

    if args.task in {"choice", "both"}:
        choice_config = ExecutionChoiceConfig(
            num_problems=None if args.choice_num_problems <= 0 else args.choice_num_problems,
            start_index=args.choice_start_index,
            runs_per_problem=args.choice_runs,
            max_new_tokens=args.choice_max_tokens,
            temperature=args.choice_temperature,
            top_p=args.choice_top_p,
            skip_boolean_for_reversion=args.choice_skip_boolean,
            seed=args.choice_seed,
        )
        choice_result = run_execution_choice(dataset_split, client, choice_config)
        outputs["choice"] = {
            "execution_choice_summary": choice_result.summary,
            "execution_choice_counts": choice_result.counts,
            "execution_choice_results": choice_result.results,
            "execution_choice_config": asdict(choice_config),
        }

    if not outputs:
        raise RuntimeError("No tasks executed. Check --task option.")

    out_path = args.output_dir / f"gpt_oss_{_timestamp()}.json"
    _write_json(outputs, out_path)
    print(f"✓ Saved evaluation summary to {out_path}")


if __name__ == "__main__":
    main()