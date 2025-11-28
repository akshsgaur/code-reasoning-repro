#!/usr/bin/env python3
"""
Model inference script for LeetCode dataset
Generates code solutions using various LLMs
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import API clients
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Run: pip install openai")

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    print("Warning: google-generativeai package not installed. Run: pip install google-generativeai")

from .models_config import get_model_config, list_available_models


@dataclass
class InferenceResult:
    """Result of a single inference"""
    problem_id: str
    model_name: str
    generated_code: str
    prompt: str
    success: bool
    error: Optional[str] = None
    reasoning: Optional[str] = None  # For reasoning models
    latency_ms: float = 0.0


class ModelInference:
    """Handle inference for different model providers"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_model_config(model_name)

        # Setup API key
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Set {self.config.api_key_env} environment variable")

        # Initialize client based on provider
        if self.config.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package required for OpenAI models")
            self.client = openai.OpenAI(api_key=api_key)

        elif self.config.provider == "deepseek":
            if not HAS_OPENAI:
                raise ImportError("openai package required for DeepSeek models")
            # DeepSeek uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )

        elif self.config.provider == "google":
            if not HAS_GOOGLE:
                raise ImportError("google-generativeai package required for Google models")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.config.identifier)

    def build_prompt(self, problem_data: Dict[str, Any]) -> str:
        """
        Build prompt for execution prediction
        Uses the exact format from the paper (Execution Prediction Prompt - Zero-Shot)

        Format:
        You are given a Python program and an assertion containing an input to a function.
        Replace the ?? in the assertion with a literal representing the function's return
        value for the given input. Provide the full assertion in [ANSWER] and [/ANSWER] tags.

        [PYTHON]
        {program}
        assert {function_name}({input}) == ??
        [/PYTHON]
        """
        function_name = problem_data['function_name']
        code = problem_data['code']  # Use the collected solution as the program
        test_input = problem_data['input']  # e.g., "nums=[1,2,3], k=2"

        # Extract just the input part (remove function name and parentheses if present)
        if test_input.startswith(f"{function_name}(") and test_input.endswith(")"):
            input_args = test_input[len(function_name)+1:-1]
        else:
            input_args = test_input

        prompt = f"""You are given a Python program and an assertion containing an input to a function. Replace the ?? in the assertion with a literal (no unsimplified expressions, no function calls) representing the function's return value for the given input. Execute the program exactly as written, even if it is incorrect or incomplete. For your final answer, provide the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert {function_name}({input_args}) == ??
[/PYTHON]"""

        return prompt

    def generate_code(self, problem_data: Dict[str, Any]) -> InferenceResult:
        """Generate code for a single problem"""
        prompt = self.build_prompt(problem_data)
        problem_id = problem_data['id']

        start_time = time.time()

        try:
            if self.config.provider in ["openai", "deepseek"]:
                response = self.client.chat.completions.create(
                    model=self.config.identifier,
                    messages=[
                        {"role": "system", "content": "You are an expert Python programmer. Generate clean, efficient code."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                latency_ms = (time.time() - start_time) * 1000
                full_response = response.choices[0].message.content

                # Extract predicted answer from [ANSWER] tags
                predicted_output = self._extract_answer_from_response(full_response)

                return InferenceResult(
                    problem_id=problem_id,
                    model_name=self.model_name,
                    generated_code=predicted_output,  # Store predicted value
                    prompt=prompt,
                    success=True,
                    latency_ms=latency_ms
                )

            elif self.config.provider == "google":
                response = self.client.generate_content(prompt)
                latency_ms = (time.time() - start_time) * 1000

                full_response = response.text
                predicted_output = self._extract_answer_from_response(full_response)

                return InferenceResult(
                    problem_id=problem_id,
                    model_name=self.model_name,
                    generated_code=predicted_output,  # Store predicted value
                    prompt=prompt,
                    success=True,
                    latency_ms=latency_ms
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                problem_id=problem_id,
                model_name=self.model_name,
                generated_code="",
                prompt=prompt,
                success=False,
                error=str(e),
                latency_ms=latency_ms
            )

    def _extract_answer_from_response(self, text: str) -> str:
        """
        Extract predicted answer from [ANSWER] tags

        Expected format: [ANSWER] assert function_name(input) == output [/ANSWER]
        We want to extract just the output value
        """
        import re

        # Look for [ANSWER] tags
        pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            assertion = matches[0].strip()

            # Parse the assertion to extract the predicted value
            # Format: "assert function_name(input) == value"
            match = re.search(r'assert\s+\w+\([^)]*\)\s*==\s*(.+)', assertion)
            if match:
                predicted_value = match.group(1).strip()
                return predicted_value

            # If we can't parse it, return the whole assertion
            return assertion

        # Fallback: try to find "assert ... == VALUE" pattern anywhere
        pattern = r'assert\s+\w+\([^)]*\)\s*==\s*(.+?)(?:\n|$)'
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return matches[0].strip()

        # Return as-is if no answer tags found
        return text.strip()


def run_inference(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    num_samples: Optional[int] = None,
    start_idx: int = 0
):
    """
    Run inference on dataset

    Args:
        dataset_path: Path to JSONL dataset
        model_name: Name of model to use
        output_dir: Directory to save results
        num_samples: Number of samples to process (None = all)
        start_idx: Starting index in dataset
    """
    print(f"{'='*60}")
    print(f"Running inference with {model_name}")
    print(f"{'='*60}")

    # Load dataset
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]

    print(f"Loaded {len(dataset)} problems")

    # Slice dataset if needed
    if num_samples:
        dataset = dataset[start_idx:start_idx + num_samples]
        print(f"Processing {len(dataset)} problems (from index {start_idx})")

    # Initialize model
    print(f"Initializing {model_name}...")
    inferencer = ModelInference(model_name)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_results.jsonl"

    # Run inference
    results = []
    for idx, problem in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] Processing {problem['id']}...")

        result = inferencer.generate_code(problem)

        if result.success:
            print(f"  ✓ Generated code ({result.latency_ms:.0f}ms)")
        else:
            print(f"  ✗ Failed: {result.error}")

        # Save result
        result_dict = {
            "problem_id": result.problem_id,
            "model_name": result.model_name,
            "generated_code": result.generated_code,
            "success": result.success,
            "error": result.error,
            "latency_ms": result.latency_ms,
        }
        results.append(result_dict)

        # Write incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')

        # Rate limiting
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"{'='*60}")
    print(f"Total problems: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run model inference on LeetCode dataset')
    parser.add_argument('--dataset', type=str,
                        default='../datasets/leetcode/data/datasets/leetcode_contests_431_467.jsonl',
                        help='Path to dataset JSONL file')
    parser.add_argument('--model', type=str, required=True,
                        choices=list_available_models(),
                        help='Model to use for inference')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/inference_results',
                        help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in dataset')

    args = parser.parse_args()

    dataset_path = Path(__file__).parent / args.dataset
    output_dir = Path(__file__).parent / args.output_dir

    run_inference(
        dataset_path=dataset_path,
        model_name=args.model,
        output_dir=output_dir,
        num_samples=args.num_samples,
        start_idx=args.start_idx
    )


if __name__ == '__main__':
    main()
