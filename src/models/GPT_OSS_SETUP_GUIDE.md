# GPT-OSS Evaluation Setup Guide

Complete step-by-step guide to evaluate OpenAI's gpt-oss-20b on Google Colab using the LeetCode dataset.

## Overview

1. **Upload dataset** to HuggingFace Hub
2. **Open Colab notebook** and load dataset
3. **Run evaluation** with gpt-oss-20b (free T4 GPU)
4. **Download results** and analyze

---

## Part 1: Upload Dataset to HuggingFace

### Step 1.1: Install Required Packages

```bash
cd /Users/akshitgaur/Desktop/CMU/IDL/code-reasoning-repro/src/datasets/leetcode

pip install datasets huggingface-hub
```

### Step 1.2: Get HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "leetcode-dataset-upload"
4. Type: "Write"
5. Copy the token

### Step 1.3: Upload Dataset

```bash
python3 upload_to_huggingface.py \
  --dataset data/datasets/leetcode_contests_431_467.jsonl \
  --repo-id YOUR_USERNAME/leetcode-contests-431-467 \
  --token YOUR_HF_TOKEN
```

**Example**:
```bash
python3 upload_to_huggingface.py \
  --dataset data/datasets/leetcode_contests_431_467.jsonl \
  --repo-id akshitgaur/leetcode-contests-431-467 \
  --token hf_xxxxxxxxxxxxxx
```

**Notes**:
- Replace `YOUR_USERNAME` with your HuggingFace username
- Add `--private` flag if you want a private dataset
- The script will guide you through the login process

### Step 1.4: Verify Upload

Visit: `https://huggingface.co/datasets/YOUR_USERNAME/leetcode-contests-431-467`

You should see:
- ✓ Dataset with 347 samples
- ✓ README with statistics
- ✓ Data viewer working

---

## Part 2: Run Evaluation on Google Colab

### Step 2.1: Open Colab Notebook

1. **Upload notebook to Google Drive**:
   - Go to: https://drive.google.com
   - Upload: `src/models/gpt_oss_evaluation_colab.ipynb`

2. **Open in Colab**:
   - Right-click the notebook
   - Select: "Open with → Google Colaboratory"

3. **Enable GPU**:
   - Click: "Runtime → Change runtime type"
   - Hardware accelerator: **GPU** (T4)
   - Click "Save"

### Step 2.2: Update Dataset Repository ID

In the notebook, find this cell:

```python
# TODO: Replace with your HuggingFace dataset repo ID
DATASET_REPO_ID = "YOUR_USERNAME/leetcode-contests-431-467"
```

Replace with your actual repository ID:

```python
DATASET_REPO_ID = "akshitgaur/leetcode-contests-431-467"
```

### Step 2.3: Run Setup

**Run Cell 1**: Install packages
```python
!pip install -q --upgrade torch
!pip install -q transformers triton==3.4 kernels
!pip uninstall -q torchvision torchaudio -y
!pip install -q datasets
```

⚠️ **IMPORTANT**: After this cell completes:
- Click: **Runtime → Restart runtime**
- Wait for the session to restart
- Do NOT run this cell again after restarting

### Step 2.4: Load Model

**Run Cell 2**: Load gpt-oss-20b

This will download ~10GB model (takes 2-3 minutes on free Colab).

Expected output:
```
Loading gpt-oss-20b model...
CUDA available: True
CUDA device: Tesla T4
✓ Model loaded successfully!
```

### Step 2.5: Load Dataset

**Run Cell 3**: Load dataset from HuggingFace

Expected output:
```
Loading dataset from akshitgaur/leetcode-contests-431-467...
✓ Dataset loaded!
Total samples: 347
```

### Step 2.6: Test with One Sample

**Run Cell 5**: Generate code for one problem

This tests that everything is working correctly.

Expected output:
```
Generating code...
Generated Response: [code here]
Test Result: ✓ CORRECT (or ✗ INCORRECT)
```

### Step 2.7: Run Full Evaluation

**Run Cell 6**: Evaluate multiple samples

**Configuration**:
```python
NUM_SAMPLES = 10  # Start with 10, increase later
REASONING_EFFORT = "medium"  # "low", "medium", or "high"
```

**For full dataset** (347 samples):
```python
NUM_SAMPLES = 347
```

⚠️ **Note**: Full dataset evaluation takes ~1-2 hours on free Colab.

**Expected Output**:
```
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

### Step 2.8: Download Results

**Run Cell 7**: Save and download results

This creates a JSON file with all results and automatically downloads it.

File format:
```json
{
  "model": "gpt-oss-20b",
  "reasoning_effort": "medium",
  "num_samples": 10,
  "correct_count": 7,
  "pass_at_1": 0.7,
  "results": [...]
}
```

---

## Part 3: Analyze Results Locally

### Step 3.1: Move Results to Project

```bash
cd /Users/akshitgaur/Desktop/CMU/IDL/code-reasoning-repro/src/models

# Move downloaded results
mv ~/Downloads/gpt_oss_20b_results_*.json outputs/evaluation_results/
```

### Step 3.2: Compare with Other Models

Create a comparison script:

```python
import json
from pathlib import Path

results_dir = Path("outputs/evaluation_results")

models = {
    "GPT-OSS 20B": "gpt_oss_20b_results_medium_*.json",
    "GPT-4o": "gpt-4o_evaluation.json",
    "DeepSeek-R1": "deepseek-r1_evaluation.json",
}

print(f"{'Model':<20} {'pass@1':<10}")
print("-" * 30)

for model_name, pattern in models.items():
    files = list(results_dir.glob(pattern))
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        pass_1 = data.get('pass_at_1', data.get('pass_at_k', {}).get('pass@1', 0))
        print(f"{model_name:<20} {pass_1*100:>6.2f}%")
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size or max_new_tokens
```python
generated = model.generate(**inputs, max_new_tokens=300)  # Reduce from 500
```

**Solution 2**: Use Colab Pro (more VRAM)

### Issue: "Dataset not found"

**Solution**: Check repository ID is correct
```python
# Make sure it matches your HuggingFace repo
DATASET_REPO_ID = "YOUR_USERNAME/leetcode-contests-431-467"
```

### Issue: "Colab disconnects during evaluation"

**Solution**: Keep the tab active or use Colab Pro

To resume from interruption:
```python
# In Cell 6, modify to skip completed problems
START_IDX = 50  # Start from problem 50
for idx in range(START_IDX, NUM_SAMPLES):
    # ... rest of code
```

### Issue: "Model loading takes forever"

**Solution**: This is normal first time (downloads ~10GB). Subsequent runs are faster.

---

## Tips for Better Results

### 1. Reasoning Effort Comparison

Run comparison (Cell 8) to see impact:
```
Reasoning       pass@1      Avg Latency
------------------------------------------------------------
low              65.0%        2.34s
medium           70.0%        3.45s
high             75.0%        5.12s
```

**Recommendation**: Use "medium" for good balance of accuracy and speed.

### 2. Incremental Evaluation

For the full 347 samples, evaluate in batches:

```python
# Batch 1: Easy problems
easy_samples = [s for s in dataset['train'] if s['difficulty'] == 'easy']
evaluate_samples(easy_samples)

# Batch 2: Medium problems
medium_samples = [s for s in dataset['train'] if s['difficulty'] == 'medium']
evaluate_samples(medium_samples)

# Batch 3: Hard problems
hard_samples = [s for s in dataset['train'] if s['difficulty'] == 'hard']
evaluate_samples(hard_samples)
```

### 3. Save Progress Frequently

Add this after every batch:
```python
# Save intermediate results
with open(f'results_batch_{batch_num}.json', 'w') as f:
    json.dump(results, f)
```

---

## Cost Estimate

**Using Free Google Colab**:
- Model: gpt-oss-20b (free, runs on T4 GPU)
- Dataset: 347 samples
- Time: ~1-2 hours
- Cost: **$0** (completely free!)

**Comparison with API models**:
- GPT-4o via API: ~$35 for 347 samples
- DeepSeek-R1 via API: ~$2 for 347 samples
- gpt-oss on Colab: **FREE**

---

## Expected Performance

Based on similar models:

| Model | Expected pass@1 |
|-------|----------------|
| gpt-oss-20b (low) | ~55-65% |
| gpt-oss-20b (medium) | ~60-70% |
| gpt-oss-20b (high) | ~65-75% |

Compare with:
- GPT-4o: ~75-85%
- DeepSeek-R1: ~70-80%
- GPT-4o-mini: ~60-70%

---

## Next Steps

After completing evaluation:

1. **Analyze results** by difficulty, problem type
2. **Compare** with other models
3. **Share findings** with team
4. **Write paper** section on gpt-oss performance

## Questions?

Contact: [Your email]
