#!/bin/bash
# Wrapper script to run the dataset builder with the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Use the venv Python
PYTHON="$PROJECT_ROOT/venv/bin/python3"

# Run the builder
"$PYTHON" "$SCRIPT_DIR/build_dataset.py" "$@"
