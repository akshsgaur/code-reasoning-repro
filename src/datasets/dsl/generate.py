"""Dataset generation pipeline for DSL-List programs."""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import List

from .neural_pcfg import DeepSynthNeuralSampler
from .spec import DSLGenerationConfig, DSLProgramRecord, GeneratedDataset


def generate_dataset(cfg: DSLGenerationConfig, out_dir: Path) -> GeneratedDataset:
    """Generate `cfg.num_programs` valid programs and persist artefacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.use_neural_pcfg:
        raise RuntimeError("Only neural PCFG generation is supported. Set use_neural_pcfg=true in the config.")

    rng = random.Random(cfg.seed)
    records: List[DSLProgramRecord] = []

    neural_sampler = DeepSynthNeuralSampler(cfg, rng)
    while len(records) < cfg.num_programs:
        record = neural_sampler.sample_record()
        if record is None:
            break
        records.append(record)

    if len(records) < cfg.num_programs:
        raise RuntimeError(
            f"Neural PCFG sampler terminated early after collecting {len(records)} "
            f"programs (target {cfg.num_programs}). Consider increasing "
            "`neural_max_attempts` or relaxing constraints."
        )

    dataset = GeneratedDataset(config=cfg, records=records, output_dir=out_dir)
    persist_dataset(dataset)
    return dataset


def persist_dataset(dataset: GeneratedDataset) -> None:
    """Write the dataset (JSONL) and config metadata to `dataset.output_dir`."""

    dataset.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = dataset.output_dir / "programs.jsonl"
    metadata_path = dataset.output_dir / "metadata.json"
    structured_path = dataset.output_dir / "programs_structured.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in dataset.records:
            f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    _persist_structured(dataset.records, structured_path)

    metadata = {
        "num_programs": len(dataset.records),
        "config": asdict(dataset.config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _persist_structured(records: List[DSLProgramRecord], path: Path) -> None:
    """Emit records in the evaluation schema with placeholders for mutations."""

    entries = []
    for idx, rec in enumerate(records):
        params = [f"a{i+1}" for i in range(rec.arity)]
        fn_name = f"dsl_prog_{idx}"
        header = f"def {fn_name}({', '.join(params)}):"
        body_lines = rec.python.splitlines()[1:]
        body = textwrap.dedent("\n".join(body_lines)).strip("\n")
        indented_body = (
            "\n".join("    " + line for line in body.splitlines()) if body else "    pass"
        )
        code = header + "\n" + indented_body + "\n"

        call_exprs = []
        outputs = []
        conditions = []
        for example_input, example_output in zip(rec.inputs, rec.outputs):
            call_args = ", ".join(
                f"{params[i]}={example_input[i]}" for i in range(len(params))
            )
            call_expr = f"{fn_name}({call_args})"
            output_literal = json.dumps(example_output)
            call_exprs.append(call_expr)
            outputs.append(output_literal)
            conditions.append(f"{call_expr} == {output_literal}")

        entries.append(
            {
                "id": f"dsl_neural_{idx}",
                "function_name": fn_name,
                "code": code,
                "input": call_exprs,
                "output": outputs,
                "correct_condition": conditions,
                "mutated_code": "",
                "mutated_output": "",
                "has_mutation": False,
                "mutation_info": {
                    "mutation_type": "",
                    "mutation_id": -1,
                    "coverage_similarity": 0.0,
                },
                "dsl": rec.dsl,
                "arity": rec.arity,
                "depth": rec.depth,
                "metadata": rec.metadata or {"source": "deepsynth_neural"},
            }
        )

    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
