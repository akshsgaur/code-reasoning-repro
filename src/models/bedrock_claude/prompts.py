"""Prompt builders and output parsers for execution prediction/choice."""

from __future__ import annotations

import ast
import json
import re
from typing import Dict, Tuple


def build_execution_prediction_prompt(sample: Dict, use_mutated: bool = False) -> str:
    function_name = sample["function_name"]

    if use_mutated:
        program = sample.get("mutated_code")
        if program is None:
            raise KeyError("Sample missing 'mutated_code'. Did you load the mutated dataset?")
        if not str(program).strip():
            raise ValueError("Sample has empty 'mutated_code'.")
    else:
        program = sample["code"]

    test_input = sample["input"]

    if test_input and test_input.startswith(f"{function_name}(") and test_input.endswith(")"):
        input_args = test_input[len(function_name) + 1 : -1]
    else:
        input_args = test_input

    prompt = f"""You are given a Python program and an assertion containing an input to a function. Replace the ?? in the assertion with a literal (no unsimplified expressions, no function calls) representing the function's return value for the given input. Execute the program exactly as written, even if it is incorrect or incomplete. For your final answer, provide the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{program}
assert {function_name}({input_args}) == ??
[/PYTHON]"""

    return prompt


def build_execution_choice_prompt(sample: Dict, original_first: bool = True) -> Tuple[str, Dict[str, str]]:
    function_name = sample["function_name"]
    original_code = sample["code"]
    mutated_code = sample.get("mutated_code") or original_code
    test_input = sample["input"]

    if test_input and test_input.startswith(f"{function_name}(") and test_input.endswith(")"):
        input_args = test_input[len(function_name) + 1 : -1]
    else:
        input_args = test_input

    if original_first:
        program_a, program_b = original_code, mutated_code
        mapping = {"A": "original", "B": "mutated"}
    else:
        program_a, program_b = mutated_code, original_code
        mapping = {"A": "mutated", "B": "original"}

    prompt_template = (
        "You are given two Python programs below and an assertion containing an input to a function. "
        "First, choose either program, whichever one you are more confident in reasoning about. "
        "Then, replace the ?? in the assertion with a literal (no unsimplified expressions, no function calls) "
        "representing the function's return value for the given input on your chosen program. Execute the program "
        "exactly as written, even if it is incorrect or incomplete. For your final answer, output the letter of your "
        "chosen program (A or B) and the full assertion in the following json format:\n\n"
        "{\n"
        '  "chosen_program": "A or B",\n'
        '  "assertion": "full_assertion"\n'
        "}\n\n"
        "[PROGRAM_A]\n{program_a}\n[/PROGRAM_A]\n"
        "[PROGRAM_B]\n{program_b}\n[/PROGRAM_B]\n"
        "[ASSERTION]\nassert {function_name}({input_args}) == ??\n[/ASSERTION]"
    )

    prompt = prompt_template.format(
        program_a=program_a,
        program_b=program_b,
        function_name=function_name,
        input_args=input_args,
    )

    return prompt, mapping


def parse_execution_choice_response(response: str) -> Dict[str, str]:
    json_match = re.search(r"\{\s*\"chosen_program\"\s*:.*?\}", response, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find JSON payload in the response.")

    json_text = json_match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        chosen_match = re.search(r'"chosen_program"\s*:\s*"?([A-Za-z])"?', json_text)
        assertion_match = re.search(r'"assertion"\s*:\s*("(?:[^"\\]|\\.)*")', json_text)
        if not chosen_match or not assertion_match:
            raise ValueError("Failed to parse execution choice JSON response.")

        chosen_program = chosen_match.group(1)
        assertion_literal = assertion_match.group(1)
        try:
            assertion = ast.literal_eval(assertion_literal)
        except Exception:
            assertion = assertion_literal.strip('"')

        return {
            "chosen_program": chosen_program,
            "assertion": assertion,
        }


def extract_output_from_assertion(assertion: str) -> str:
    if not assertion:
        return ""

    text = assertion.strip()
    text = re.sub(r"^\[ASSERTION\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\[/ASSERTION\]$", "", text, flags=re.IGNORECASE)

    match = re.search(r"assert\s+[\w\.]+\([^)]*\)\s*==\s*(.+)", text)
    if match:
        return match.group(1).strip()

    return text


def extract_answer_from_response(response: str) -> str:
    pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        assertion = matches[0].strip()
        match = re.search(r"assert\s+\w+\([^)]*\)\s*==\s*(.+)", assertion)
        if match:
            return match.group(1).strip()
        return assertion

    pattern = r"assert\s+\w+\([^)]*\)\s*==\s*(.+?)(?:\n|$)"
    matches = re.findall(pattern, response, re.MULTILINE)
    if matches:
        return matches[0].strip()

    return response.strip()


def check_predicted_output(predicted_output: str, expected_output: str) -> Tuple[bool, str | None]:
    try:
        predicted = (predicted_output or "").strip()
        expected = (expected_output or "").strip()

        if predicted == expected:
            return True, None

        try:
            predicted_val = ast.literal_eval(predicted)
            expected_val = ast.literal_eval(expected)
            if predicted_val == expected_val:
                return True, None
        except (ValueError, SyntaxError):
            pass

        return False, f"Predicted: {predicted}, Expected: {expected}"
    except Exception as exc:
        return False, str(exc)


def is_boolean_output(value: str) -> bool:
    if value is None:
        return False

    try:
        parsed = ast.literal_eval(value.strip())
        return isinstance(parsed, bool)
    except (ValueError, SyntaxError, AttributeError):
        lowered = value.strip().lower()
        return lowered in {"true", "false"}


__all__ = [
    "build_execution_prediction_prompt",
    "build_execution_choice_prompt",
    "parse_execution_choice_response",
    "extract_output_from_assertion",
    "extract_answer_from_response",
    "check_predicted_output",
    "is_boolean_output",
]
