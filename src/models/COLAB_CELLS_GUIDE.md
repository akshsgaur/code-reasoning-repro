# Google Colab Cells - Visual Guide

This guide shows exactly what each cell looks like in your Colab notebook.

---

## 📋 Cell Overview

| Cell # | Type | What It Does | Time |
|--------|------|-------------|------|
| 1 | Text | Title and intro | - |
| 2 | Text | Setup instructions | - |
| 3 | Code | **Install packages** | 2-3 min |
| 4 | Text | ⚠️ Restart reminder | - |
| 5 | Text | Model loading instructions | - |
| 6 | Code | **Load gpt-oss-20b** | 3-5 min |
| 7 | Text | Dataset instructions | - |
| 8 | Code | **Load dataset from HuggingFace** | 30 sec |
| 9 | Text | Helper functions header | - |
| 10 | Code | **Define helper functions** | Instant |
| 11 | Text | Test one sample header | - |
| 12 | Code | **Test generation on 1 problem** | 5-10 sec |
| 13 | Text | Batch evaluation header | - |
| 14 | Code | **Run evaluation (10+ samples)** | 30 sec - 2 hrs |
| 15 | Text | Save results header | - |
| 16 | Code | **Save and download results** | Instant |
| 17 | Text | Optional comparison header | - |
| 18 | Code | **Compare reasoning efforts** | 1-2 min |

---

## 📱 Cell-by-Cell Preview

### Cell 1: Title (Markdown)
```
┌────────────────────────────────────────────────────────┐
│ # GPT-OSS 20B Evaluation on LeetCode Dataset          │
│                                                        │
│ This notebook evaluates OpenAI's gpt-oss-20b model    │
│ on the LeetCode contests dataset.                     │
│                                                        │
│ **Requirements**:                                      │
│ - Free Google Colab (T4 GPU)                          │
│ - HuggingFace account (to load dataset)               │
└────────────────────────────────────────────────────────┘
```

---

### Cell 2: Setup Instructions (Markdown)
```
┌────────────────────────────────────────────────────────┐
│ ## Step 1: Setup Environment                          │
│                                                        │
│ Install required packages for mxfp4 quantization      │
│ support.                                               │
└────────────────────────────────────────────────────────┘
```

---

### Cell 3: Install Packages (Code) ⚙️
```python
┌────────────────────────────────────────────────────────┐
│ # Install bleeding-edge PyTorch and transformers      │
│ !pip install -q --upgrade torch                       │
│ !pip install -q transformers triton==3.4 kernels      │
│ !pip uninstall -q torchvision torchaudio -y           │
│                                                        │
│ # Install datasets library                            │
│ !pip install -q datasets                              │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
Installing...
Successfully installed torch-2.x.x
Successfully installed transformers-4.x.x
Successfully installed triton-3.4
Successfully installed kernels-x.x.x
Successfully installed datasets-x.x.x
```

---

### Cell 4: Restart Reminder (Markdown) ⚠️
```
┌────────────────────────────────────────────────────────┐
│ ⚠️ **IMPORTANT**: Please restart your Colab runtime   │
│ after running the cell above.                          │
│                                                        │
│ Click: **Runtime → Restart runtime**                  │
└────────────────────────────────────────────────────────┘
```

**ACTION REQUIRED**:
1. Click "Runtime" in menu
2. Click "Restart runtime"
3. Wait for session to restart
4. **DO NOT** re-run Cell 3

---

### Cell 5: Model Loading Header (Markdown)
```
┌────────────────────────────────────────────────────────┐
│ ## Step 2: Load GPT-OSS 20B Model                     │
└────────────────────────────────────────────────────────┘
```

---

### Cell 6: Load Model (Code) 🤖
```python
┌────────────────────────────────────────────────────────┐
│ from transformers import AutoModelForCausalLM,         │
│                          AutoTokenizer                 │
│ import torch                                           │
│                                                        │
│ print("Loading gpt-oss-20b model...")                 │
│ print(f"CUDA available: {torch.cuda.is_available()}")│
│ print(f"CUDA device: {torch.cuda.get_device_name(0)}")│
│                                                        │
│ model_id = "openai/gpt-oss-20b"                       │
│                                                        │
│ tokenizer = AutoTokenizer.from_pretrained(model_id)   │
│ model = AutoModelForCausalLM.from_pretrained(         │
│     model_id,                                          │
│     torch_dtype="auto",                                │
│     device_map="cuda",                                 │
│ )                                                      │
│                                                        │
│ print("✓ Model loaded successfully!")                 │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
Loading gpt-oss-20b model...
CUDA available: True
CUDA device: Tesla T4

Downloading model files...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Loading checkpoint shards: 100% 2/2 [01:30<00:00, 45.2s/it]

✓ Model loaded successfully!
```

⏱️ **Time**: 3-5 minutes (first time only)

---

### Cell 7: Dataset Header (Markdown)
```
┌────────────────────────────────────────────────────────┐
│ ## Step 3: Load LeetCode Dataset from HuggingFace     │
└────────────────────────────────────────────────────────┘
```

---

### Cell 8: Load Dataset (Code) 📊
```python
┌────────────────────────────────────────────────────────┐
│ from datasets import load_dataset                      │
│                                                        │
│ # TODO: Replace with your HuggingFace dataset repo ID │
│ DATASET_REPO_ID = "YOUR_USERNAME/leetcode-contests-431-467" │
│                                                        │
│ print(f"Loading dataset from {DATASET_REPO_ID}...")   │
│ dataset = load_dataset(DATASET_REPO_ID)               │
│                                                        │
│ print(f"\n✓ Dataset loaded!")                         │
│ print(f"Total samples: {len(dataset['train'])}")      │
│ print(f"\nFirst sample:")                             │
│ sample = dataset['train'][0]                          │
│ print(f"  ID: {sample['id']}")                        │
│ print(f"  Function: {sample['function_name']}")       │
│ print(f"  Difficulty: {sample['difficulty']}")        │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**⚠️ IMPORTANT**: Change this line:
```python
# CHANGE THIS:
DATASET_REPO_ID = "YOUR_USERNAME/leetcode-contests-431-467"

# TO THIS (with your HF username):
DATASET_REPO_ID = "akshitgaur/leetcode-contests-431-467"
```

**Expected Output:**
```
Loading dataset from akshitgaur/leetcode-contests-431-467...
Downloading data files: 100%
Generating train split: 347/347 [00:00<00:00, 12345.67 examples/s]

✓ Dataset loaded!
Total samples: 347

First sample:
  ID: contest431_q3702_s0
  Function: maxLength
  Difficulty: easy
  Input: maxLength(nums=[1,2,1,2,1,1,1])...
```

⏱️ **Time**: 10-30 seconds

---

### Cell 9-10: Helper Functions
```python
┌────────────────────────────────────────────────────────┐
│ # Helper functions for code generation and testing    │
│                                                        │
│ def build_prompt(sample: Dict) -> str:                │
│     """Build prompt for code generation"""            │
│     # ... code ...                                     │
│                                                        │
│ def extract_code_from_response(response: str) -> str: │
│     """Extract Python code from model response"""     │
│     # ... code ...                                     │
│                                                        │
│ def execute_code_with_test(...) -> Tuple[bool, str]:  │
│     """Execute generated code and check correctness"""│
│     # ... code ...                                     │
│                                                        │
│ print("✓ Helper functions defined")                   │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
✓ Helper functions defined
```

⏱️ **Time**: Instant

---

### Cell 11-12: Test One Sample (Code) 🧪
```python
┌────────────────────────────────────────────────────────┐
│ # Test with one sample                                 │
│ test_sample = dataset['train'][0]                     │
│                                                        │
│ # Build prompt and generate code                      │
│ messages = [...]                                       │
│ inputs = tokenizer.apply_chat_template(               │
│     messages,                                          │
│     reasoning_effort="medium",  # ← CONFIGURABLE      │
│ ).to(model.device)                                    │
│                                                        │
│ print("Generating code...")                           │
│ generated = model.generate(**inputs, max_new_tokens=500)│
│ response = tokenizer.decode(...)                      │
│                                                        │
│ # Test correctness                                     │
│ is_correct, error = execute_code_with_test(...)       │
│ print(f"Test Result: {'✓' if is_correct else '✗'}")   │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
Prompt:
Write a Python function to solve this LeetCode problem:

Function to implement: maxLength
...

============================================================
Generating code...

Generated Response:
Here's a solution:

```python
def maxLength(nums):
    n = len(nums)
    ans = 0
    for l in range(n):
        # ... implementation
    return ans
```

============================================================
Extracted Code:
def maxLength(nums):
    n = len(nums)
    ans = 0
    ...

============================================================
Test Result: ✓ CORRECT
Expected: 5
```

⏱️ **Time**: 5-10 seconds per sample

---

### Cell 13-14: Batch Evaluation (Code) 🚀
```python
┌────────────────────────────────────────────────────────┐
│ # Configuration                                        │
│ NUM_SAMPLES = 10  # ← START SMALL, INCREASE LATER    │
│ REASONING_EFFORT = "medium"  # low/medium/high        │
│                                                        │
│ results = []                                           │
│ correct_count = 0                                      │
│                                                        │
│ for idx in tqdm(range(NUM_SAMPLES)):                 │
│     sample = dataset['train'][idx]                    │
│                                                        │
│     # Generate code                                    │
│     # ... generation logic ...                         │
│                                                        │
│     # Test correctness                                 │
│     is_correct, error = execute_code_with_test(...)   │
│     if is_correct:                                     │
│         correct_count += 1                             │
│                                                        │
│ # Print results                                        │
│ print(f"pass@1: {correct_count/NUM_SAMPLES*100:.2f}%")│
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
Evaluating 10 samples with reasoning_effort=medium...

100%|████████████████████████| 10/10 [00:45<00:00, 4.5s/it]

============================================================
EVALUATION RESULTS
============================================================
Model: gpt-oss-20b
Reasoning effort: medium
Total samples: 10
Correct: 7
pass@1: 70.00%
Average latency: 3.45s
============================================================

By Difficulty:
  Easy: 3/3 (100.0%)
  Medium: 3/5 (60.0%)
  Hard: 1/2 (50.0%)
```

⏱️ **Time**:
- 10 samples: ~30-60 seconds
- 50 samples: ~3-5 minutes
- 347 samples: ~1-2 hours

**💡 TIP**: Start with `NUM_SAMPLES = 10` to test!

---

### Cell 15-16: Save Results (Code) 💾
```python
┌────────────────────────────────────────────────────────┐
│ import json                                            │
│ from datetime import datetime                          │
│                                                        │
│ output_filename = f"gpt_oss_20b_results_{...}.json"   │
│                                                        │
│ output_data = {                                        │
│     "model": "gpt-oss-20b",                           │
│     "pass_at_1": correct_count / NUM_SAMPLES,         │
│     "results": results                                 │
│ }                                                      │
│                                                        │
│ with open(output_filename, 'w') as f:                 │
│     json.dump(output_data, f, indent=2)               │
│                                                        │
│ # Download the file                                    │
│ from google.colab import files                        │
│ files.download(output_filename)                       │
│                                                        │
│ ▶ [Run this cell]                                     │
└────────────────────────────────────────────────────────┘
```

**Expected Output:**
```
✓ Results saved to: gpt_oss_20b_results_medium_20250120_143022.json

[Browser download popup appears]
⬇️ Downloading: gpt_oss_20b_results_medium_20250120_143022.json
```

File is saved to your Downloads folder!

---

## 🎯 Quick Action Checklist

When you open the notebook:

- [ ] **Cell 3**: Run install packages → Wait for completion
- [ ] **IMPORTANT**: Runtime → Restart runtime
- [ ] **Cell 6**: Run to load model (3-5 min wait)
- [ ] **Cell 8**: **EDIT** `DATASET_REPO_ID` with your username, then run
- [ ] **Cell 10**: Run to define helpers
- [ ] **Cell 12**: Run to test 1 sample
- [ ] **Cell 14**: **EDIT** `NUM_SAMPLES = 10`, then run
- [ ] **Cell 16**: Run to download results

---

## ⚙️ Customization Points

### Change Number of Samples
```python
# In Cell 14
NUM_SAMPLES = 10    # Quick test
NUM_SAMPLES = 50    # Medium test
NUM_SAMPLES = 347   # Full dataset (1-2 hours!)
```

### Change Reasoning Effort
```python
# In Cell 14
REASONING_EFFORT = "low"     # Faster, less accurate
REASONING_EFFORT = "medium"  # Balanced (recommended)
REASONING_EFFORT = "high"    # Slower, more accurate
```

### Change Max Tokens
```python
# In Cell 12 or 14
generated = model.generate(**inputs, max_new_tokens=500)  # Default
generated = model.generate(**inputs, max_new_tokens=300)  # Shorter
generated = model.generate(**inputs, max_new_tokens=1000) # Longer
```

---

## 🐛 Common Issues

### "Runtime disconnected"
**Solution**: Keep tab active or use Colab Pro

### "CUDA out of memory"
**Solution**: Reduce `max_new_tokens` to 300

### "Dataset not found"
**Solution**: Check `DATASET_REPO_ID` matches your HuggingFace repo

### "Model download stuck"
**Solution**: Wait 5 minutes, or restart runtime and try again

---

## 📊 Expected Performance

Based on similar models:

| Reasoning | pass@1 | Time/sample |
|-----------|--------|-------------|
| Low | 55-65% | ~2s |
| Medium | 60-70% | ~3s |
| High | 65-75% | ~5s |

---

This is what your Colab notebook will look like! Ready to run? 🚀
