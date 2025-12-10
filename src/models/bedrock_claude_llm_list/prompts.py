"""Prompt builders and output parsers for execution prediction/choice."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class NormalizedSample:
    """Unified view of dataset samples.

    Some datasets contain a single input/output string per problem, others (like
    the DSL example) provide multiple test cases. This object normalizes both
    shapes so downstream code can build prompts and compare outputs without
    worrying about the original schema.
    """

    raw_sample: Dict
    function_name: str
    original_code: str
    mutated_code: str
    test_inputs: List[str]
    original_outputs: List[Any]
    mutated_outputs: List[Any]


def _infer_function_name(sample: Dict) -> str:
    if "function_name" in sample and sample["function_name"]:
        return str(sample["function_name"])

    code = sample.get("code") or sample.get("original_code") or ""
    match = re.search(r"def\s+([A-Za-z_][\w]*)\s*\(", str(code))
    if match:
        return match.group(1)

    raise KeyError("Could not determine function name from sample.")


def _ensure_list(value: Any) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_literal_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip()


def _parse_conditions(conditions: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Extract function calls and expected outputs from `correct_condition` entries."""

    inputs: List[str] = []
    outputs: List[str] = []
    for cond in conditions:
        if not cond:
            continue
        match = re.match(r"\s*([\w\.]+\(.*\))\s*==\s*(.+)", cond)
        if not match:
            continue
        inputs.append(match.group(1).strip())
        outputs.append(match.group(2).strip())

    return inputs, outputs


def _extract_test_inputs(sample: Dict, function_name: str) -> List[str]:
    inputs_field = sample.get("input") or sample.get("inputs")
    inputs = _ensure_list(inputs_field)

    if not inputs:
        conditions = _ensure_list(sample.get("correct_condition") or sample.get("conditions"))
        if conditions:
            inputs, _ = _parse_conditions(conditions)

    if not inputs:
        raise ValueError("Sample missing input test cases.")

    normalized: List[str] = []
    for call in inputs:
        if call is None:
            continue

        if isinstance(call, list):
            # Join arg strings into "arg1, arg2, arg3"
            text = ", ".join(str(a) for a in call).strip()
        else:
            text = str(call).strip()

        # Strip any 'assert ... == ??' wrapper if present
        text = re.sub(r"^assert\s+", "", text)
        text = re.sub(r"\s*==\s*\?\?$", "", text)

        # If we only see arguments (no function call), wrap with function_name(...)
        if "(" not in text and ")" not in text:
            text = f"{function_name}({text})"

        normalized.append(text)

    return normalized


def _extract_outputs(sample: Dict, *, mutated: bool, expected_len: int) -> List[Any]:
    key = "mutated_output" if mutated else "output"
    alt_key = "mutated_outputs" if mutated else "outputs"
    raw_outputs = sample.get(key)
    if raw_outputs is None:
        raw_outputs = sample.get(alt_key)

    outputs = _ensure_list(raw_outputs)
    if not outputs:
        conditions = _ensure_list(sample.get("correct_condition") or sample.get("conditions"))
        if conditions:
            _, outputs = _parse_conditions(conditions)

    if not outputs:
        return []

    if len(outputs) != expected_len:
        if len(outputs) == 1:
            outputs = outputs * expected_len
        else:
            raise ValueError(
                f"Number of outputs ({len(outputs)}) does not match number of inputs ({expected_len})."
            )

    return [_parse_literal_value(val) for val in outputs]


def normalize_sample(sample: Dict) -> NormalizedSample:
    """Convert heterogeneous dataset records into a consistent structure."""

    function_name = _infer_function_name(sample)
    original_code = sample.get("code") or sample.get("original_code")
    if not original_code:
        raise ValueError("Sample missing program code.")

    mutated_code = sample.get("mutated_code") or original_code
    test_inputs = _extract_test_inputs(sample, function_name)

    original_outputs = _extract_outputs(sample, mutated=False, expected_len=len(test_inputs))
    if not original_outputs:
        raise ValueError("Sample missing expected outputs.")

    mutated_outputs = _extract_outputs(sample, mutated=True, expected_len=len(test_inputs))
    if not mutated_outputs:
        mutated_outputs = original_outputs

    return NormalizedSample(
        raw_sample=sample,
        function_name=function_name,
        original_code=str(original_code),
        mutated_code=str(mutated_code),
        test_inputs=test_inputs,
        original_outputs=original_outputs,
        mutated_outputs=mutated_outputs,
    )


def format_expected_outputs(values: Sequence[Any]) -> str:
    """Return a Python-literal string for the expected outputs list."""

    return repr(list(values))


def format_output_value(value: Any) -> str:
    """Return a Python-literal string for a single expected output."""

    try:
        return repr(value)
    except Exception:
        return str(value)


def build_execution_prediction_prompt(
    sample: Dict | NormalizedSample,
    *,
    use_mutated: bool = False,
    test_input: str | None = None,
) -> str:
    normalized = sample if isinstance(sample, NormalizedSample) else normalize_sample(sample)
    program = normalized.mutated_code if use_mutated else normalized.original_code

    call = test_input or (normalized.test_inputs[0] if normalized.test_inputs else "")
    call_str = str(call).strip()

    if call_str and not call_str.startswith(normalized.function_name):
        call_str = f"{normalized.function_name}({call_str})"

    if call_str.startswith(f"{normalized.function_name}(") and call_str.endswith(")"):
        input_args = call_str[len(normalized.function_name) + 1 : -1]
    else:
        input_args = call_str

    prompt = f"""You are given a Python program and an assertion containing an input to a function. Replace the ?? in the assertion with a literal (no unsimplified expressions, no function calls) representing the function's return value for the given input. Execute the program exactly as written, even if it is incorrect or incomplete. For your final answer, provide the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{program}
assert {normalized.function_name}({input_args}) == ??
[/PYTHON]"""

    return prompt


def build_execution_choice_prompt(
    sample: Dict | NormalizedSample,
    *,
    original_first: bool = True,
    test_input: str | None = None,
) -> Tuple[str, Dict[str, str]]:
    normalized = sample if isinstance(sample, NormalizedSample) else normalize_sample(sample)

    if original_first:
        program_a, program_b = normalized.original_code, normalized.mutated_code
        mapping = {"A": "original", "B": "mutated"}
    else:
        program_a, program_b = normalized.mutated_code, normalized.original_code
        mapping = {"A": "mutated", "B": "original"}

    call = test_input or (normalized.test_inputs[0] if normalized.test_inputs else "")
    call_str = str(call).strip()
    if call_str and not call_str.startswith(normalized.function_name):
        call_str = f"{normalized.function_name}({call_str})"

    if call_str.startswith(f"{normalized.function_name}(") and call_str.endswith(")"):
        input_args = call_str[len(normalized.function_name) + 1 : -1]
    else:
        input_args = call_str

    json_schema = (
        '{\n'
        '  "chosen_program": "A or B",\n'
        '  "assertion": "full_assertion"\n'
        '}'
    )

    prompt = (
        "You are given two Python programs and an assertion containing an input to a function."
        " First, choose the program (A or B) you are more confident in. Then replace the ?? with the literal"
        " return value of your chosen program for this input. Execute the chosen program exactly as written, even"
        " if it is incorrect or incomplete. Respond strictly in this JSON format:\n\n"
        f"{json_schema}\n\n"
        f"[PROGRAM_A]\n{program_a}\n[/PROGRAM_A]\n"
        f"[PROGRAM_B]\n{program_b}\n[/PROGRAM_B]\n"
        f"[ASSERTION]\nassert {normalized.function_name}({input_args}) == ??\n[/ASSERTION]"
    )

    return prompt, mapping


def parse_execution_choice_response(response: str) -> Dict[str, Any]:
    json_match = re.search(r"\{\s*\"chosen_program\"\s*:.*?\}", response, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find JSON payload in the response.")

    json_text = json_match.group(0)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        chosen_match = re.search(r'"chosen_program"\s*:\s*"?([A-Za-z])"?', json_text)
        assertion_match = re.search(r'"assertion"\s*:\s*("(?:[^"\\]|\\.)*")', json_text)
        outputs_match = re.search(r'"outputs"\s*:\s*(\[[^\]]*\])', json_text, re.DOTALL)
        if not chosen_match:
            raise ValueError("Failed to parse execution choice JSON response.")

        parsed = {"chosen_program": chosen_match.group(1)}
        if outputs_match:
            parsed["outputs"] = ast.literal_eval(outputs_match.group(1))
        elif assertion_match:
            try:
                parsed["assertion"] = ast.literal_eval(assertion_match.group(1))
            except Exception:
                parsed["assertion"] = assertion_match.group(1).strip('"')
        else:
            raise ValueError("Execution choice response missing outputs/assertion.")

    return parsed


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
        if isinstance(parsed, bool):
            return True
        if isinstance(parsed, list) and parsed and all(isinstance(v, bool) for v in parsed):
            return True
        return False
    except (ValueError, SyntaxError, AttributeError):
        lowered = value.strip().lower()
        return lowered in {"true", "false"}


__all__ = [
    "NormalizedSample",
    "normalize_sample",
    "format_expected_outputs",
    "format_output_value",
    "build_execution_prediction_prompt",
    "build_execution_choice_prompt",
    "parse_execution_choice_response",
    "extract_output_from_assertion",
    "extract_answer_from_response",
    "check_predicted_output",
    "is_boolean_output",
]
