from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence


@dataclass
class DSLGenerationConfig:
    """Configuration for sampling DSL-List programs."""

    num_programs: int
    max_depth: int
    arity: int
    input_trials: int = 3
    max_attempts: int = 50_000
    seed: Optional[int] = None
    use_neural_pcfg: bool = False
    deepsynth_root: Optional[str] = None
    neural_max_attempts: int = 5_000


@dataclass
class DSLProgramRecord:
    """A single dataset entry containing the DSL program and its execution data."""

    arity: int
    depth: int
    dsl: str
    python: str
    inputs: Sequence[Sequence[Any]]
    outputs: Sequence[Any]
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Return a JSON-serialisable representation."""

        return {
            "arity": self.arity,
            "depth": self.depth,
            "dsl": self.dsl,
            "python": self.python,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedDataset:
    """Container for the artefacts emitted by dataset generation."""

    config: DSLGenerationConfig
    records: List[DSLProgramRecord]
    output_dir: Path
