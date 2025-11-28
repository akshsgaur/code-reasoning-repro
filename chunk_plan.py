"""Helper utilities for chunked Bedrock evaluation runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ChunkConfig:
    start_index: int
    num_problems: int
    output_path: str


def _plan_chunks(total: int, chunk_size: int, out_dir: Path, prefix: str) -> List[ChunkConfig]:
    if chunk_size <= 0:
        chunk_name = f"{prefix}_{_now()}_chunk_all.json"
        return [ChunkConfig(0, total, str(out_dir / chunk_name))]

    chunks: List[ChunkConfig] = []
    start = 0
    while start < total:
        num = min(chunk_size, total - start)
        chunk_name = f"{prefix}_{_now()}_chunk_{start:04d}.json"
        chunks.append(ChunkConfig(start, num, str(out_dir / chunk_name)))
        start += num
    return chunks


def chunk_dataset_config(
    *,
    total_predictions: int,
    total_choices: int,
    chunk_size: int,
    output_dir: str,
    prefix: str,
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "prediction_chunks": [asdict(cfg) for cfg in _plan_chunks(total_predictions, chunk_size, out_dir, f"{prefix}_prediction")],
        "choice_chunks": [asdict(cfg) for cfg in _plan_chunks(total_choices, chunk_size, out_dir, f"{prefix}_choice")],
        "metadata": {
            "total_prediction_problems": total_predictions,
            "total_choice_problems": total_choices,
            "chunk_size": chunk_size,
            "output_dir": str(out_dir),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    plan_path = out_dir / f"chunk_plan_{_now()}.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan


def merge_chunk_results(chunk_files: List[str], output_path: str) -> None:
    merged: Dict[str, Dict] = {}
    for file_path in chunk_files:
        data = json.loads(Path(file_path).read_text())
        for key in ("prediction", "choice"):
            if key not in data:
                continue
            merged.setdefault(key, {"results": []})
            if key == "prediction":
                merged[key].setdefault("metrics_counts", data[key].get("metrics_counts"))
                merged[key].setdefault("benchmark_summary", data[key].get("benchmark_summary"))
                merged[key]["results"].extend(data[key].get("results", []))
            else:
                merged[key].setdefault("execution_choice_counts", data[key].get("execution_choice_counts"))
                merged[key]["results"].extend(data[key].get("execution_choice_results", []))

    Path(output_path).write_text(json.dumps(merged, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk planning/merging utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Create chunk plan")
    plan.add_argument("--total-predictions", type=int, required=True)
    plan.add_argument("--total-choices", type=int, required=True)
    plan.add_argument("--chunk-size", type=int, default=100)
    plan.add_argument("--output-dir", default="chunk_plans")
    plan.add_argument("--prefix", default="claude_bedrock")

    merge = sub.add_parser("merge", help="Merge chunk outputs")
    merge.add_argument("output", help="Output JSON path")
    merge.add_argument("chunks", nargs="+", help="Chunk JSON files")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.command == "plan":
        plan = chunk_dataset_config(
            total_predictions=args.total_predictions,
            total_choices=args.total_choices,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
        print(json.dumps(plan, indent=2))
    else:
        merge_chunk_results(args.chunks, args.output)
        print(f"Merged {len(args.chunks)} chunk files into {args.output}")
