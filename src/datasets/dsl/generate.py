"""Dataset generation pipeline for DSL-List programs."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import List

from . import inputs, sampler, transpile, validator
from .ast import Ty, Var
from .spec import DSLGenerationConfig, DSLProgramRecord, GeneratedDataset


def _sample_program(
    cfg: DSLGenerationConfig,
    rng: random.Random,
) -> DSLProgramRecord | None:
    """Sample a single program and return it if it passes validation."""

    depth = rng.randint(1, cfg.max_depth)
    scope = tuple(Var(f"a{i+1}") for i in range(cfg.arity))
    expr = sampler.sample_expr(Ty.LIST_INT, depth, scope, rng)
    py_src = transpile.transpile_to_python(expr, cfg.arity)
    program_inputs = inputs.sample_inputs(cfg.arity, cfg.input_trials, rng)
    ok, outputs = validator.is_valid(py_src, program_inputs, expr, cfg.arity)
    if not ok:
        return None
    return DSLProgramRecord(
        arity=cfg.arity,
        depth=depth,
        dsl=repr(expr),
        python=py_src,
        inputs=program_inputs,
        outputs=outputs,
    )


def generate_dataset(cfg: DSLGenerationConfig, out_dir: Path) -> GeneratedDataset:
    """Generate `cfg.num_programs` valid programs and persist artefacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    records: List[DSLProgramRecord] = []
    attempts = 0

    while len(records) < cfg.num_programs and attempts < cfg.max_attempts:
        attempts += 1
        record = _sample_program(cfg, rng)
        if record is None:
            continue
        records.append(record)

    if len(records) < cfg.num_programs:
        raise RuntimeError(
            f"Failed to sample {cfg.num_programs} programs within "
            f"{cfg.max_attempts} attempts (collected {len(records)})."
        )

    dataset = GeneratedDataset(config=cfg, records=records, output_dir=out_dir)
    persist_dataset(dataset)
    return dataset


def persist_dataset(dataset: GeneratedDataset) -> None:
    """Write the dataset (JSONL) and config metadata to `dataset.output_dir`."""

    dataset.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = dataset.output_dir / "programs.jsonl"
    metadata_path = dataset.output_dir / "metadata.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in dataset.records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    metadata = {
        "num_programs": len(dataset.records),
        "config": asdict(dataset.config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

