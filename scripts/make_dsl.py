from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dsl import DSLGenerationConfig, generate_dataset


def load_config(path: Path) -> DSLGenerationConfig:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    return DSLGenerationConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSL-List dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file describing DSLGenerationConfig.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target directory for the generated dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse the config file and print it.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        print(json.dumps(cfg.__dict__, indent=2))
        return

    generate_dataset(cfg, args.out)


if __name__ == "__main__":
    main()
