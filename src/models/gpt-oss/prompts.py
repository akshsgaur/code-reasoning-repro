"""Prompt builders - reuses Bedrock prompt logic."""

# Import all prompt functions from bedrock_claude
import sys
from pathlib import Path

# Add parent directory to path to import from bedrock_claude
sys.path.insert(0, str(Path(__file__).parent.parent))

from bedrock_claude.prompts import (
    build_execution_prediction_prompt,
    build_execution_choice_prompt,
    parse_execution_choice_response,
    extract_output_from_assertion,
    extract_answer_from_response,
    check_predicted_output,
    is_boolean_output,
)

__all__ = [
    "build_execution_prediction_prompt",
    "build_execution_choice_prompt",
    "parse_execution_choice_response",
    "extract_output_from_assertion",
    "extract_answer_from_response",
    "check_predicted_output",
    "is_boolean_output",
]