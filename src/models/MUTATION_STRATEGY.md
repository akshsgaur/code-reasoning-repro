# Mutation Strategy: Paper vs mutmut_src Implementation

**Date**: October 20, 2025
**Paper**: 2504.05518v1.pdf - "Code Reasoning"
**Framework**: Modified MutMut (located at `/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src/`)

---

## Overview

The paper describes **Algorithm 1: Mutated Dataset Creation** (pages 5-6) which creates semantically different programs by applying local mutation operators. The `mutmut_src` directory contains a modified version of the MutMut mutation testing framework that implements these operators.

---

## Paper's Mutation Strategy (Algorithm 1)

### Algorithm Steps

For each program-input pair (P, x) in dataset D:

1. **Generate all mutants**: Apply mutation operators to create set S of mutated programs
2. **Filter valid mutants**: Keep only mutants P' where:
   - P'(x) executes without error
   - P'(x) produces different output than P(x)
3. **Select best mutant**: Choose P' with most similar line coverage to P
4. **Add to dataset**: Include (P', x) in mutated dataset D'

### Mutation Operators (Table 1, page 6)

| Operator Type | Original | Mutated | Example |
|--------------|----------|---------|---------|
| Arithmetic Operator | `a + b` | `a - b` | `sum + 1` → `sum - 1` |
| Relational Operator | `a < b` | `a <= b` | `x < 10` → `x <= 10` |
| Logical Operator | `a and b` | `a or b` | `x and y` → `x or y` |
| Keyword | `continue` | `break` | `continue` → `break` |
| Numerical Literal | `1` | `0` | `return 1` → `return 0` |

### Key Properties

- **Local mutations**: Single operator change per mutant
- **Semantic difference**: Output must differ from original
- **Syntactic similarity**: Similar line coverage preferred
- **Error-free execution**: Mutants must execute without runtime errors

---

## mutmut_src Implementation

Located at: `/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src/`

### Core Files

1. **`__init__.py`** (43KB): Core mutation logic
2. **`__main__.py`** (19KB): CLI interface

### Mutation Operators Implemented

#### 1. Arithmetic Operators (`operator_mutation`, lines 330-380)

```python
def operator_mutation(value, node, **_):
    return {
        '+': '-',
        '-': '+',
        '*': '//',  # Modified from standard mutmut
        '/': '*',
        '//': '*',
        '%': '//',
        '<<': '>>',
        '>>': '<<',
        '&': '|',
        '|': '&',
        '^': '&',
        '**': '*',
        '~': '',

        # Compound assignment operators
        '+=': '-=',
        '-=': '+=',
        '*=': '//=',
        '/=': '*=',
        # ... more operators
    }.get(value)
```

**Paper Correlation**: ✅ Matches "Arithmetic Operator" from Table 1

#### 2. Relational Operators (`operator_mutation`, lines 373-380)

```python
{
    '<': '<=',
    '<=': '<',
    '>': '>=',
    '>=': '>',
    '==': '!=',
    '!=': '==',
    '<>': '==',
}
```

**Paper Correlation**: ✅ Matches "Relational Operator" from Table 1

#### 3. Logical Operators (`and_or_test_mutation`, lines 421-427)

```python
def and_or_test_mutation(children, node, **_):
    children = children[:]
    children[1] = Keyword(
        value={'and': ' or', 'or': ' and'}[children[1].value],
        start_pos=node.start_pos,
    )
    return children
```

**Paper Correlation**: ✅ Matches "Logical Operator" from Table 1

#### 4. Keyword Mutations (`keyword_mutation`, lines 305-321)

```python
def keyword_mutation(value, context, **_):
    return {
        'not': '\b',
        'is': 'is not',
        'in': 'not in',
        'break': 'continue',
        'continue': 'break',
        'True': 'False',
        'False': 'True',
    }.get(value)
```

**Paper Correlation**: ✅ Matches "Keyword" from Table 1 (`continue` ↔ `break`)

#### 5. Numerical Literals (`number_mutation`, lines 195-232)

```python
def number_mutation(value, **_):
    # ... parsing logic ...
    parsed = int(value, base=base)  # or float
    result = [repr(parsed - 1), repr(parsed + 1)]
    return result
```

**Paper Correlation**: ✅ Matches "Numerical Literal" from Table 1 (1 → 0, etc.)

### Additional Mutations (Not in Paper)

The mutmut_src framework includes additional mutations that are **commented out** in `mutations_by_type`:

```python
mutations_by_type = {
    'operator': dict(value=operator_mutation),
    'keyword': dict(value=keyword_mutation),
    'number': dict(value=number_mutation),
    'or_test': dict(children=and_or_test_mutation),
    'and_test': dict(children=and_or_test_mutation),

    # COMMENTED OUT (not used for paper):
    # 'name': dict(value=name_mutation),
    # 'string': dict(value=string_mutation),
    # 'fstring': dict(children=fstring_mutation),
    # 'argument': dict(children=argument_mutation),
    # 'lambdef': dict(children=lambda_mutation),
    # 'expr_stmt': dict(children=expression_mutation),
    # 'decorator': dict(children=decorator_mutation),
    # 'annassign': dict(children=expression_mutation),
}
```

This suggests the framework was **customized** to match the paper's mutation operators exactly.

---

## How to Apply Mutations to LeetCode Dataset

### Step 1: Understanding the Workflow

The standard MutMut workflow:

1. **Run**: `mutmut run` - generates all mutants and tests them
2. **Show**: `mutmut show <id>` - display specific mutant
3. **Apply**: `mutmut apply <id>` - apply a mutant to source code

### Step 2: Adapting for Dataset Creation

For the LeetCode dataset, we need to:

1. **For each problem's code**:
   - Generate all possible mutants
   - Execute each mutant with the test input
   - Filter mutants that:
     - Execute successfully (no errors)
     - Produce different output than original
   - Select mutant with similar coverage
   - Save mutated code + original input + new output

### Step 3: Implementation Plan

Create a new script: `/src/datasets/leetcode/mutate_dataset.py`

```python
#!/usr/bin/env python3
"""
Create mutated version of LeetCode dataset following Algorithm 1 from paper
"""

import json
import sys
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import coverage

# Add mutmut_src to path
sys.path.insert(0, '/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src')
from mutmut import mutate, Context, RelativeMutationID

# Import base imports for code execution
BASE_IMPORTS = """..."""  # Same as in evaluate.py


def generate_all_mutants(code: str, filename: str = "temp.py") -> List[Tuple[str, RelativeMutationID]]:
    """Generate all possible mutants of the code"""
    mutants = []

    # Mutate with each possible mutation_id
    mutation_index = 0
    while True:
        context = Context(
            source=code,
            filename=filename,
            mutation_id=RelativeMutationID(
                filename=filename,
                line='',  # Will be set by mutate()
                index=mutation_index,
                line_number=0
            ),
        )

        try:
            mutated_code, num_mutations = mutate(context)
            if num_mutations == 0:
                break
            mutants.append((mutated_code, context.mutation_id))
            mutation_index += 1
        except Exception:
            break

    return mutants


def execute_code_with_input(code: str, function_name: str, test_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Execute code with test input and return (success, output, error)
    """
    try:
        # Prepare execution environment
        exec_globals = {}
        exec(BASE_IMPORTS, exec_globals)
        exec(code, exec_globals)

        # Execute test input
        result = eval(test_input, exec_globals)
        return (True, repr(result), None)

    except Exception as e:
        return (False, None, str(e))


def get_line_coverage(code: str, function_name: str, test_input: str) -> Optional[Set[int]]:
    """Get line coverage when executing code with input"""
    try:
        cov = coverage.Coverage()
        cov.start()

        exec_globals = {}
        exec(BASE_IMPORTS, exec_globals)
        exec(code, exec_globals)
        eval(test_input, exec_globals)

        cov.stop()

        # Get executed lines
        analysis = cov.analysis2(filename='<string>')
        return set(analysis[1])  # Executed lines

    except Exception:
        return None


def coverage_similarity(cov1: Set[int], cov2: Set[int]) -> float:
    """Calculate Jaccard similarity between two coverage sets"""
    if not cov1 or not cov2:
        return 0.0
    intersection = len(cov1 & cov2)
    union = len(cov1 | cov2)
    return intersection / union if union > 0 else 0.0


def find_best_mutant(
    original_code: str,
    function_name: str,
    test_input: str,
    original_output: str
) -> Optional[Dict]:
    """
    Implement Algorithm 1 from paper

    Returns:
        Dict with mutated_code, mutated_output, mutation_id, or None if no valid mutant
    """
    print(f"  Generating mutants for {function_name}...")

    # Step 1: Generate all mutants
    mutants = generate_all_mutants(original_code)
    print(f"  Generated {len(mutants)} mutants")

    if not mutants:
        return None

    # Get original coverage
    original_cov = get_line_coverage(original_code, function_name, test_input)

    # Step 2: Filter valid mutants
    valid_mutants = []
    for mutated_code, mutation_id in mutants:
        success, output, error = execute_code_with_input(mutated_code, function_name, test_input)

        # Must execute successfully
        if not success:
            continue

        # Must produce different output
        if output == original_output:
            continue

        # Get coverage for this mutant
        mutant_cov = get_line_coverage(mutated_code, function_name, test_input)

        valid_mutants.append({
            'code': mutated_code,
            'output': output,
            'mutation_id': mutation_id,
            'coverage': mutant_cov
        })

    print(f"  {len(valid_mutants)} valid mutants (produce different output)")

    if not valid_mutants:
        return None

    # Step 3: Select mutant with most similar coverage
    best_mutant = max(
        valid_mutants,
        key=lambda m: coverage_similarity(original_cov, m['coverage']) if original_cov and m['coverage'] else 0
    )

    return {
        'mutated_code': best_mutant['code'],
        'mutated_output': best_mutant['output'],
        'mutation_id': str(best_mutant['mutation_id']),
        'coverage_similarity': coverage_similarity(original_cov, best_mutant['coverage']) if original_cov and best_mutant['coverage'] else 0
    }


def mutate_dataset(input_path: Path, output_path: Path):
    """
    Create mutated version of dataset following Algorithm 1

    Input format: {id, code, input, output, function_name, ...}
    Output format: {id, code, input, output, is_mutated, original_id, mutation_id, ...}
    """
    print(f"{'='*60}")
    print("Creating Mutated Dataset (Algorithm 1)")
    print(f"{'='*60}")

    with open(input_path) as f:
        dataset = [json.loads(line) for line in f]

    print(f"Loaded {len(dataset)} problems")

    mutated_dataset = []
    successful_mutations = 0

    for idx, sample in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] Processing {sample['id']}...")

        # Add original to mutated dataset
        original_entry = sample.copy()
        original_entry['is_mutated'] = False
        mutated_dataset.append(original_entry)

        # Try to create mutant
        result = find_best_mutant(
            original_code=sample['code'],
            function_name=sample['function_name'],
            test_input=sample['input'],
            original_output=sample['output']
        )

        if result:
            # Create mutated entry
            mutated_entry = sample.copy()
            mutated_entry['id'] = sample['id'] + '_mutated'
            mutated_entry['code'] = result['mutated_code']
            mutated_entry['output'] = result['mutated_output']
            mutated_entry['is_mutated'] = True
            mutated_entry['original_id'] = sample['id']
            mutated_entry['mutation_info'] = {
                'mutation_id': result['mutation_id'],
                'coverage_similarity': result['coverage_similarity']
            }

            mutated_dataset.append(mutated_entry)
            successful_mutations += 1

            print(f"  ✓ Created mutant (coverage similarity: {result['coverage_similarity']:.2f})")
        else:
            print(f"  ✗ No valid mutant found")

    # Save mutated dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in mutated_dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\n{'='*60}")
    print("Mutation Complete!")
    print(f"{'='*60}")
    print(f"Original problems: {len(dataset)}")
    print(f"Successful mutations: {successful_mutations}")
    print(f"Total dataset size: {len(mutated_dataset)} ({successful_mutations} original + {successful_mutations} mutated)")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create mutated dataset')
    parser.add_argument('--input', type=str,
                        default='data/datasets/leetcode_contests_431_467.jsonl',
                        help='Input dataset path')
    parser.add_argument('--output', type=str,
                        default='data/datasets/leetcode_contests_431_467_mutated.jsonl',
                        help='Output mutated dataset path')

    args = parser.parse_args()

    input_path = Path(__file__).parent / args.input
    output_path = Path(__file__).parent / args.output

    mutate_dataset(input_path, output_path)
```

---

## Key Differences: Paper vs MutMut

### Similarities

✅ **Mutation Operators**: Same 5 operator types
✅ **Local Mutations**: Single change per mutant
✅ **Syntax-Based**: Uses AST parsing (parso library)

### Differences

| Aspect | Paper Algorithm 1 | MutMut Framework |
|--------|------------------|------------------|
| **Purpose** | Create mutated dataset | Mutation testing (find bugs in tests) |
| **Selection** | Most similar coverage | All killed/survived mutants |
| **Output** | One best mutant per program | Multiple mutants per program |
| **Filtering** | Must change output | Tests must kill mutant |
| **Usage** | Dataset augmentation | Software testing |

### Adaptation Required

To use mutmut_src for dataset creation, we need to:

1. **Bypass test suite**: MutMut expects pytest tests, we execute manually
2. **Filter by output**: Keep only mutants that change output
3. **Coverage-based selection**: Pick mutant with most similar coverage
4. **Generate dataset**: Save original + mutant pairs

---

## Dataset Format After Mutation

### Original Entry
```json
{
  "id": "contest431_q3702_s0",
  "function_name": "maxLength",
  "code": "def maxLength(nums: List[int]) -> int:\n    ...",
  "input": "maxLength(nums=[1,2,1,2,1,1,1])",
  "output": "5",
  "is_mutated": false
}
```

### Mutated Entry
```json
{
  "id": "contest431_q3702_s0_mutated",
  "function_name": "maxLength",
  "code": "def maxLength(nums: List[int]) -> int:\n    ... (mutated) ...",
  "input": "maxLength(nums=[1,2,1,2,1,1,1])",
  "output": "3",
  "is_mutated": true,
  "original_id": "contest431_q3702_s0",
  "mutation_info": {
    "mutation_id": "line 5, index 2",
    "coverage_similarity": 0.95
  }
}
```

---

## Usage Instructions

### Step 1: Install Dependencies

```bash
cd /Users/akshitgaur/Desktop/CMU/IDL/code-reasoning-repro
source venv/bin/activate

# Install coverage library
pip install coverage
```

### Step 2: Run Mutation

```bash
cd src/datasets/leetcode

# Create mutated dataset (will take ~1-2 hours for 347 problems)
python3 mutate_dataset.py \
  --input data/datasets/leetcode_contests_431_467.jsonl \
  --output data/datasets/leetcode_contests_431_467_mutated.jsonl
```

### Step 3: Verify Output

```bash
# Count entries
wc -l data/datasets/leetcode_contests_431_467_mutated.jsonl

# Should be ~694 lines (347 original + up to 347 mutated)
```

### Step 4: Run Evaluation

The existing inference and evaluation scripts will work with the mutated dataset:

```bash
cd /Users/akshitgaur/Desktop/CMU/IDL/code-reasoning-repro/src/models

# Run inference on mutated dataset
python3 inference.py \
  --dataset ../datasets/leetcode/data/datasets/leetcode_contests_431_467_mutated.jsonl \
  --model gpt-4o-mini \
  --num-samples 10

# Evaluate
python3 evaluate.py \
  --dataset ../datasets/leetcode/data/datasets/leetcode_contests_431_467_mutated.jsonl \
  --results outputs/inference_results/gpt-4o-mini_results.jsonl
```

---

## Measuring Reversion (Paper Metric)

The paper measures **reversion**: when given mutated code, does model predict the ORIGINAL output instead of the mutated output?

### Add Reversion Metric to Evaluation

In `evaluate.py`, add:

```python
def calculate_reversion_rate(dataset, results):
    """
    Calculate reversion: % of mutated problems where model predicted original output
    """
    reversion_count = 0
    mutated_count = 0

    for result in results:
        sample = dataset[result['problem_id']]

        if not sample.get('is_mutated', False):
            continue  # Skip original samples

        mutated_count += 1

        # Get original sample's output
        original_id = sample['original_id']
        original_output = dataset[original_id]['output']

        # Did model predict original output?
        predicted_output = result['generated_code']
        if predicted_output == original_output:
            reversion_count += 1

    reversion_rate = reversion_count / mutated_count if mutated_count > 0 else 0
    return reversion_rate
```

**High reversion** = Model relies on pattern matching
**Low reversion** = Model performs genuine reasoning

---

## Summary

### What We Found

1. **mutmut_src location**: `/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src/`
2. **Framework**: Modified MutMut 2.5.1 with customized operators
3. **Operators match paper**: All 5 mutation types from Table 1 implemented
4. **Adaptation needed**: Need to wrap mutmut_src for dataset creation

### Next Steps

1. ✅ Create `mutate_dataset.py` script (see implementation above)
2. ⏳ Test on small subset (5-10 problems)
3. ⏳ Run on full dataset (347 problems)
4. ⏳ Add reversion metric to evaluation
5. ⏳ Run all models on mutated dataset
6. ⏳ Compare original vs mutated accuracy + reversion rates

---

## References

- **Paper**: 2504.05518v1.pdf (Algorithm 1, pages 5-6; Table 1, page 6)
- **MutMut Framework**: https://github.com/boxed/mutmut
- **Modified Version**: `/Users/akshitgaur/Desktop/CMU/IDL/mutmut_src/`
- **Dataset**: `/src/datasets/leetcode/data/datasets/leetcode_contests_431_467.jsonl`
