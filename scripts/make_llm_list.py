# Pipeline:
#   1) Brainstorm 100 function headers + descriptions
#   2) Add manually specified headers and descriptions for ten sorting algorithms and two search algorithms (112 progs total)
#   2) Generate code implementations based on brainstormed headers + descriptions (use GPT-5)
#   3) Generate 3 inputs per function
#       a) Execute inputs 
#       b) Re-generate any failed inputs
#   4) Obtain 112 programs with three inputs for each program

import json
from pathlib import Path
import ast
import re

DATA_DIR = Path("code-reasoning-repro/src/datasets/llm-list")
PROMPTS_DIR = Path("code-reasoning-repro/src/datasets/llm-list/prompts")

BRAINSTORM_JSON = DATA_DIR / "brainstorm" / "brainstorm.json"
FUNCTIONS_JSON   = DATA_DIR / "generated_code" / "functions.json"
FINAL_JSONL   = DATA_DIR / "final" / "final_results.jsonl"

# Parsing functions
def parse_brainstorm_response(results):
    res = []
    pattern = re.compile(
        r'^\s*\d+\.\s*"\s*([A-Za-z_]\w*\s*\([^)]*\))\s*"\s*:\s*"\s*([^"]+)\s*"\s*$'
    )
    for ln in results.splitlines():
        ln = ln.strip()
        if ln:
            m = pattern.match(ln)
            if m:
                header = m.group(1).strip()
                desc = m.group(2).strip()
                if not desc.endswith('.'):
                    desc += '.'
                header = f"def {header}:"
                res.append((header, desc))
    return res

def parse_code_response(results):
    code = results.strip()
    if code.startswith("```"):
        code = code.strip("`")
        lines = [l for l in code.splitlines() if not l.lower().startswith("python")]
        code = "\n".join(lines).strip()
    if "def" not in code:
        raise ValueError("No function definition found in code response.")
    # sanity check valid python
    ast.parse(code)
    return code

def parse_input_response(results):
    res = []
    lines = [ln.strip() for ln in results.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError("Expected 3 input lines.")
    for ln in lines[:3]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            raise ValueError("Empty input line.")
        res.append(parts)
    return res

# Function to call LLM
def call_llm(prompt):
    from openai import OpenAI
    client = OpenAI()

    result = client.responses.create(
        model="gpt-5",
        input=prompt,
        reasoning={ "effort": "low" },
        text={ "verbosity": "low" },
    )

# Call LLM with brainstorm prompt to get function headers and descriptions 
def brainstorm():
    prompt = (PROMPTS_DIR / "brainstorm.txt").read_text()
    results = call_llm(prompt)
    headers_descriptions = parse_brainstorm_response(results)
    if len(headers_descriptions) < 100:
        raise RuntimeError(f"Brainstorm returned only {len(headers_descriptions)} items instead of 100.")
    # take the first 100 if > 100
    headers_descriptions = headers_descriptions[:100]
    BRAINSTORM_JSON.write_text(json.dumps(headers_descriptions, indent = 2))
    return headers_descriptions

# Manually specified headers + description
MANUAL_12 = [
    ('def bubble_sort(lst):',        'Return a new list with the elements of lst sorted ascending using bubble sort.'),
    ('def selection_sort(lst):',     'Return a new list with the elements of lst sorted ascending using selection sort.'),
    ('def insertion_sort(lst):',     'Return a new list with the elements of lst sorted ascending using insertion sort.'),
    ('def merge_sort(lst):',         'Return a new list with the elements of lst sorted ascending using merge sort.'),
    ('def quick_sort(lst):',         'Return a new list with the elements of lst sorted ascending using quick sort (deterministic pivot).'),
    ('def heap_sort(lst):',          'Return a new list with the elements of lst sorted ascending using heap sort implemented explicitly.'),
    ('def shell_sort(lst):',         'Return a new list with the elements of lst sorted ascending using shell sort (gap sequence halving).'),
    ('def counting_sort(lst):',      'Return a new list with the elements of lst sorted ascending using counting sort; assume all values are non-negative and small.'),
    ('def radix_sort(lst):',         'Return a new list with the elements of lst sorted ascending using LSD radix sort for non-negative integers.'),
    ('def cocktail_shaker_sort(lst):','Return a new list sorted ascending using cocktail shaker sort (bidirectional bubble).'),
    ('def linear_search(lst, target):','Return the index of target in lst using linear search, or -1 if not found.'),
    ('def binary_search(sorted_lst, target):','Return the index of target in sorted_lst using binary search, or -1 if not found; sorted_lst is ascending.')
]

# Call LLM with code gen prompt to get programs
def code_generation(headers_descriptions):
    functions = []
    # hd should contain header, desc
    for i, hd in enumerate(headers_descriptions):
        prompt = (PROMPTS_DIR / "codegen.txt").read_text()
        results = call_llm(prompt)
        code = parse_code_response(results)
        functions.append({
            "id": f"llmlist_{i}",
            "header": hd.header,
            "description": hd.desc,
            "code": code,
        })
    FUNCTIONS_JSON.write_text(json.dumps(functions, indent=2))
    return functions

# Validate generated inputs for constraints
def validate_inputs(inputs):
    #TODO
    pass

# Call LLM with input gen prompt to get inputs
def input_generation(functions, max_retries):
    final_programs = []
    # f should contain id, header, desc, code
    for f in functions:
        inputs = []
        outputs = []
        for i in range(max_retries):
            prompt = (PROMPTS_DIR / "inputgen.txt").read_text()
            results = call_llm(prompt)
            try:
                possible_inputs = parse_input_response(results)
                # Execute + validate inputs
                valid, outs = validate_inputs(possible_inputs)
                if valid:
                    inputs = possible_inputs
                    outputs = outs
                    break
            except Exception:
                pass
        # Generate final program
        final_programs.append({
            **f,
            "inputs": inputs,
            "outputs": outputs
        })
    return final_programs

def main():
    # Brainstorm
    headers_and_descriptions = brainstorm()
    # Add headers + descriptions
    headers_and_descriptions_112 = headers_and_descriptions + MANUAL_12
    if len(headers_and_descriptions_112) != 112:
        raise RuntimeError(f"Expected 112 ideas, got {len(headers_and_descriptions_112)}.")
    # Code Generation
    code = code_generation(headers_and_descriptions_112)
    # Input Generation
    final_results = input_generation(code, max_retries=3)
    # Ensure we have 112 programs
    if len(final_results) != 112:
        print(f"[WARNING] Produced only {len(final_results)} records instead of 112.")
    else:
        FINAL_JSONL.write_text("\n".join(json.dumps(r) for r in final_results))

if __name__ == "__main__":
    main()
