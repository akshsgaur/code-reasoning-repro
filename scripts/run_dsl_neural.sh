#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${REPO_ROOT}/src/datasets/dsl/dsl_neural.yaml"
OUTPUT_DIR="${REPO_ROOT}/src/datasets/dsl/data"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/make_dsl.py" \
  --config "${CONFIG_PATH}" \
  --out "${OUTPUT_DIR}" \
  "$@"
