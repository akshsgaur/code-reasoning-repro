# LeetCode Dataset - Model Evaluation Guide

This guide explains how to run inference and evaluation on the LeetCode dataset with various LLMs.

## Table of Contents
- [Quick Start](#quick-start)
- [Models](#models)
- [Setup](#setup)
- [Running Evaluation](#running-evaluation)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_eval.txt

# 2. Set API keys
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
export GOOGLE_API_KEY="your_google_key"

# 3. Run complete evaluation pipeline
./run_evaluation.sh
```

## Models

The evaluation supports the following models:

| Model Name | Provider | Identifier | Reasoning |
|------------|----------|------------|-----------|
| DeepSeek-R1 | DeepSeek | deepseek-ai/DeepSeek-R1 | ✓ |
| GPT-4o-mini | OpenAI | gpt-4o-mini-2024-07-18 | ✗ |
| GPT-4o | OpenAI | gpt-4o-2024-08-06 | ✗ |
| o3-mini | OpenAI | o3-mini-2025-01-31 | ✓ |
| Gemini 2.5 Pro | Google | gemini-2.5-pro | ✓ (Deep Think) |

**Note**: GPT-5 will be added when available.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_eval.txt
```

This installs:
- `openai` - OpenAI and DeepSeek API client
- `google-generativeai` - Google Gemini API client
- `numpy`, `pandas` - For data processing
- `selenium`, `radon` - For dataset building

### 2. Get API Keys

**OpenAI** (for GPT-4o, GPT-4o-mini, o3-mini):
- Visit: https://platform.openai.com/api-keys
- Create new API key
- Set: `export OPENAI_API_KEY="sk-..."`

**DeepSeek** (for DeepSeek-R1):
- Visit: https://platform.deepseek.com/api_keys
- Create new API key
- Set: `export DEEPSEEK_API_KEY="sk-..."`

**Google** (for Gemini 2.5 Pro):
- Visit: https://makersuite.google.com/app/apikey
- Create new API key
- Set: `export GOOGLE_API_KEY="AI..."`

### 3. Verify Setup

```bash
python3 -c "from models_config import list_available_models; print(list_available_models())"
```

Should output:
```
['deepseek-r1', 'gpt-4o-mini', 'gpt-4o', 'o3-mini', 'gemini-2.5-pro']
```

## Running Evaluation

### Option 1: Complete Pipeline (Recommended)

Run inference + evaluation for all models:

```bash
./run_evaluation.sh
```

This will:
1. Run inference with each model on the full dataset
2. Evaluate all results using pass@k metric
3. Generate summary report

### Option 2: Individual Models

**Run inference for a single model:**

```bash
python3 inference.py --model gpt-4o-mini --dataset data/datasets/leetcode_contests_431_467.jsonl
```

**Evaluate results:**

```bash
python3 evaluate.py \
  --dataset data/datasets/leetcode_contests_431_467.jsonl \
  --results data/inference_results/gpt-4o-mini_results.jsonl \
  --output data/evaluation_results/gpt-4o-mini_evaluation.json
```

### Option 3: Subset Testing

Test on a small subset first:

```bash
# Test with first 10 problems
python3 inference.py \
  --model gpt-4o-mini \
  --dataset data/datasets/leetcode_contests_431_467.jsonl \
  --num-samples 10

# Evaluate
python3 evaluate.py \
  --dataset data/datasets/leetcode_contests_431_467.jsonl \
  --results data/inference_results/gpt-4o-mini_results.jsonl
```

## Understanding Results

### Inference Results

Saved to: `data/inference_results/{model}_results.jsonl`

Each line contains:
```json
{
  "problem_id": "contest431_q3702_s0",
  "model_name": "gpt-4o-mini",
  "generated_code": "def maxLength(nums):\n    ...",
  "success": true,
  "error": null,
  "latency_ms": 1234.5
}
```

### Evaluation Results

Saved to: `data/evaluation_results/{model}_evaluation.json`

Contains:
```json
{
  "total_problems": 347,
  "total_attempts": 347,
  "correct_count": 187,
  "pass_at_k": {
    "pass@1": 0.539,
    "pass@5": 0.712,
    "pass@10": 0.823
  },
  "problems": {
    "contest431_q3702_s0": {
      "total_samples": 1,
      "correct_samples": 1,
      "attempts": [...]
    },
    ...
  }
}
```

### pass@k Metric

The **pass@k** metric (from HumanEval paper) measures:
> "The probability that at least one sample passes the tests when generating k samples"

Formula: `pass@k = E[1 - C(n-c, k) / C(n, k)]`

Where:
- `n` = total samples generated
- `c` = correct samples
- `k` = number of samples to consider

**Interpretation**:
- **pass@1** (≈ 54%): Probability that a single generation is correct
- **pass@5** (≈ 71%): Probability that at least 1 of 5 generations is correct
- **pass@10** (≈ 82%): Probability that at least 1 of 10 generations is correct

Higher is better. Typical ranges:
- Excellent: >80%
- Good: 60-80%
- Fair: 40-60%
- Poor: <40%

## Advanced Usage

### Custom Prompts

Edit `inference.py` to customize the prompt:

```python
def build_prompt(self, problem_data: Dict[str, Any]) -> str:
    # Modify this function to change the prompt template
    pass
```

### Multiple Samples per Problem

Generate multiple solutions per problem (for better pass@k estimation):

```python
# In inference.py, modify to generate multiple samples
# Use temperature > 0 for diversity
```

### Custom Evaluation Metrics

Add to `evaluate.py`:

```python
def calculate_custom_metric(results):
    # Your custom metric
    pass
```

### Filtering by Difficulty

```bash
# Only evaluate on "hard" problems
python3 -c "
import json
with open('data/datasets/leetcode_contests_431_467.jsonl') as f:
    hard = [line for line in f if json.loads(line)['difficulty'] == 'hard']
with open('data/datasets/hard_only.jsonl', 'w') as f:
    f.writelines(hard)
"

python3 inference.py --dataset data/datasets/hard_only.jsonl --model gpt-4o
```

## Troubleshooting

### Rate Limits

If you hit rate limits:

```bash
# Add delay between requests (edit inference.py)
time.sleep(1.0)  # Increase from 0.5

# Or process in batches
python3 inference.py --num-samples 50 --start-idx 0
python3 inference.py --num-samples 50 --start-idx 50
# ... continue
```

### API Errors

```bash
# Test API connection
python3 -c "
import openai
client = openai.OpenAI()
print(client.models.list())
"
```

### Memory Issues

For large datasets:

```python
# Process incrementally in inference.py
# Results are already saved incrementally to JSONL
```

## Dataset Statistics

- **Total samples**: 347
- **Unique questions**: 137
- **Contests**: 37 (weekly-contest-431 to 467)
- **Date range**: Jan 5, 2025 - Sep 14, 2025
- **Difficulty**: Easy (21%), Medium (51%), Hard (28%)

## Citation

If you use this dataset or evaluation code, please cite:

```bibtex
@dataset{leetcode_contests_2025,
  title={LeetCode Weekly Contests 431-467 Dataset},
  author={Your Name},
  year={2025},
  note={Dataset of 347 Python solutions from LeetCode weekly contests}
}
```

## License

[Your license here]

## Contact

For questions or issues, contact: [your email]
