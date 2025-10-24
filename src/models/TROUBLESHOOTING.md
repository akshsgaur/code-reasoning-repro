# Troubleshooting Guide

## Issue: Model Doesn't Provide Answer in [ANSWER] Tags

### Symptom
```
Generated Response:
analysisWe need to run the program with given input and find the result...
[reasoning text continues but gets cut off]

Extracted Predicted Output:
[full reasoning text instead of just the answer]

Test Result: ✗ INCORRECT
Error: Predicted: analysisWe need to run..., Expected: 5
```

### Root Cause
The model's response was **truncated** before it could provide the `[ANSWER]` tags due to `max_new_tokens` limit.

### Solution

**For Colab (gpt-oss)**:
Increase `max_new_tokens` in generation cells:

```python
# OLD
generated = model.generate(**inputs, max_new_tokens=500)

# NEW
generated = model.generate(**inputs, max_new_tokens=1000)
```

Updated cells: **11, 13, 17**

**For Local Inference (other models)**:
The `models_config.py` already has reasonable limits:
- DeepSeek-R1: 4096 tokens
- GPT-4o: 2048 tokens
- o3-mini: 4096 tokens
- Gemini 2.5 Pro: 4096 tokens

---

## Issue: Model Doesn't Follow [ANSWER] Format

### Symptom
```
Generated Response:
The answer is 5.

Test Result: ✗ INCORRECT
Error: Predicted: The answer is 5., Expected: 5
```

### Root Cause
Model didn't follow the instruction to use `[ANSWER]` tags.

### Solution

Our extraction function has **fallback logic**:

1. **First**: Look for `[ANSWER]...[/ANSWER]` tags
2. **Second**: Look for `assert function_name(...) == VALUE` pattern
3. **Third**: Return full text (will likely fail)

**Improve prompt clarity** (already done):
```python
prompt = f"""...For your final answer, provide the full assertion in [ANSWER] and [/ANSWER] tags.

[PYTHON]
{code}
assert {function_name}({input}) == ??
[/PYTHON]"""
```

The phrase "For your final answer" emphasizes the requirement.

---

## Issue: Model Reasoning Is Too Verbose

### Symptom
Model spends 500+ tokens analyzing but doesn't reach the answer.

### Solution Options

### Option 1: Use Lower Reasoning Effort
```python
reasoning_effort="low"  # Less thinking, faster response
```

### Option 2: Increase Token Limit
```python
max_new_tokens=2000  # Allow more space for reasoning
```

### Option 3: Add Token Budget Hint to Prompt
```python
prompt = f"""...Execute the program exactly as written. Provide a concise analysis (max 200 words) followed by your answer in [ANSWER] tags.

[PYTHON]
{code}
assert {function_name}({input}) == ??
[/PYTHON]"""
```

---

## Issue: Extraction Returns Full Reasoning Text

### Symptom
```
Extracted Predicted Output:
analysisWe need to run the program...
```

### Root Cause
The `extract_answer_from_response()` function couldn't find `[ANSWER]` tags OR assertion pattern, so it returned the full text.

### Debug Steps

1. **Check if response was complete**:
```python
print(f"Response length: {len(response)} characters")
print(f"Max tokens: {max_new_tokens}")
print(f"\nLast 100 chars: {response[-100:]}")
```

If response ends abruptly (no `[/ANSWER]` or period), it was truncated.

2. **Check for [ANSWER] tags manually**:
```python
if "[ANSWER]" in response:
    print("✓ Found [ANSWER] tag")
else:
    print("✗ No [ANSWER] tag - model didn't follow format")
```

3. **Check for assertion pattern**:
```python
import re
pattern = r'assert\s+\w+\([^)]*\)\s*==\s*(.+?)(?:\n|$)'
matches = re.findall(pattern, response, re.MULTILINE)
if matches:
    print(f"✓ Found assertion with value: {matches[-1]}")
else:
    print("✗ No assertion pattern found")
```

### Fix

If model consistently fails to use `[ANSWER]` tags:

**Update extraction to be more lenient**:
```python
def extract_answer_from_response(response: str) -> str:
    # Try [ANSWER] tags first
    pattern = r'\[ANSWER\](.*?)\[/ANSWER\]'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        assertion = matches[0].strip()
        match = re.search(r'assert\s+\w+\([^)]*\)\s*==\s*(.+)', assertion)
        if match:
            return match.group(1).strip()

    # Try assertion pattern anywhere
    pattern = r'assert\s+\w+\([^)]*\)\s*==\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, response, re.MULTILINE)
    if matches:
        return matches[-1].strip()

    # Try to find "the answer is X" pattern
    pattern = r'(?:answer|result|output)\s+(?:is|=)\s+(\S+)'
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Last resort
    return response.strip()
```

---

## Issue: Predicted Output Format Doesn't Match Expected

### Symptom
```
Test Result: ✗ INCORRECT
Error: Predicted: [1, 2, 3], Expected: [1,2,3]
```

### Root Cause
Whitespace differences in list/tuple formatting.

### Solution

Already handled in `check_predicted_output()`:
```python
def check_predicted_output(predicted_output: str, expected_output: str):
    # Direct string comparison (handles exact matches)
    if predicted == expected:
        return (True, None)

    # Try evaluating as Python literals (handles formatting differences)
    try:
        import ast
        if ast.literal_eval(predicted) == ast.literal_eval(expected):
            return (True, None)
    except:
        pass
```

The `ast.literal_eval()` normalizes formatting differences.

---

## Issue: Model Gets Wrong Answer

### Symptom
```
Test Result: ✗ INCORRECT
Error: Predicted: 7, Expected: 5
```

### Root Cause
Model **incorrectly reasoned** about the code execution.

### Analysis Steps

1. **Manually verify expected output**:
```python
# Execute the code yourself to verify
exec(BASE_IMPORTS)
exec(sample['code'])
result = eval(sample['input'])
print(f"Actual result: {result}")
print(f"Expected in dataset: {sample['output']}")
```

2. **Check if code has bugs**:
Some collected solutions may be incorrect! Remember: we're testing if the model can predict the OUTPUT, even if the code is wrong.

3. **Review model's reasoning**:
Read the generated response to see WHERE the model made a mistake.

### Expectations

- **Easy problems**: ~70-80% accuracy expected
- **Medium problems**: ~50-60% accuracy expected
- **Hard problems**: ~30-40% accuracy expected

If accuracy is much lower, check:
- Prompt format is correct
- Model is following instructions
- Token limit is sufficient

---

## Configuration Checklist

✅ **Colab Notebook**:
- [ ] `DATASET_REPO_ID` updated with your HuggingFace username
- [ ] `max_new_tokens=1000` (or higher for complex problems)
- [ ] `reasoning_effort="medium"` (or adjust based on needs)
- [ ] Runtime has GPU enabled (T4)

✅ **Local Inference**:
- [ ] API keys set in environment:
  - `OPENAI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `GOOGLE_API_KEY`
- [ ] Dataset path is correct
- [ ] Output directories exist: `outputs/inference_results/`, `outputs/evaluation_results/`

✅ **Dataset**:
- [ ] Has `code` field (the program to analyze)
- [ ] Has `input` field (test input)
- [ ] Has `output` field (expected output)
- [ ] Has `function_name` field

---

## Quick Fixes Summary

| Problem | Quick Fix |
|---------|-----------|
| Response truncated | Increase `max_new_tokens` to 1000+ |
| No [ANSWER] tags | Check response completion, add fallback extraction |
| Too verbose reasoning | Use `reasoning_effort="low"` |
| Format mismatch | Already handled by `ast.literal_eval()` |
| Wrong answer | Expected! Model may not reason correctly |
| Model slow | Reduce `max_new_tokens` or use `reasoning_effort="low"` |

---

## Getting Help

If issues persist:

1. **Share debug output**:
   - Full prompt
   - Generated response (first 500 chars and last 500 chars)
   - Extracted prediction
   - Expected output

2. **Check model-specific docs**:
   - gpt-oss: OpenAI documentation
   - DeepSeek-R1: DeepSeek API docs
   - Gemini: Google AI docs

3. **Verify dataset entry**:
```python
import json
with open('dataset.jsonl') as f:
    sample = json.loads(f.readline())
    print(json.dumps(sample, indent=2))
```

Make sure `code`, `input`, `output`, `function_name` are all present and valid.
