# Model Inference and Evaluation

This directory contains model inference and evaluation code for the LeetCode dataset.

## Directory Structure

```
src/models/
├── __init__.py                # Package initialization
├── models_config.py           # Model configurations (DeepSeek, OpenAI, Google)
├── inference.py               # Code generation inference
├── evaluate.py                # Evaluation with pass@k metric
├── run_evaluation.sh          # Complete evaluation pipeline script
├── requirements_eval.txt      # Python dependencies
├── EVALUATION_README.md       # Detailed documentation
├── closed_access.py           # Placeholder for closed-source models
├── open_access.py             # Placeholder for open-source models
└── outputs/                   # Generated outputs (created on first run)
    ├── inference_results/     # Model inference results
    └── evaluation_results/    # Evaluation metrics
```

## Quick Start

See [EVALUATION_README.md](EVALUATION_README.md) for detailed instructions.

### 1. Install Dependencies

```bash
pip install -r requirements_eval.txt
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
```

### 3. Run Evaluation

```bash
# Complete pipeline for all models
./run_evaluation.sh

# Or single model
python3 inference.py --model gpt-4o-mini --num-samples 10
python3 evaluate.py --results outputs/inference_results/gpt-4o-mini_results.jsonl
```

## Supported Models

| Model | Provider | Reasoning Capable |
|-------|----------|-------------------|
| DeepSeek-R1 | DeepSeek | ✓ |
| GPT-4o-mini | OpenAI | ✗ |
| GPT-4o | OpenAI | ✗ |
| o3-mini | OpenAI | ✓ |
| Gemini 2.5 Pro | Google | ✓ |

## Files Description

- **models_config.py**: Configuration for all supported models (API endpoints, parameters)
- **inference.py**: Generate code solutions using LLMs (zero-shot prompting)
- **evaluate.py**: Evaluate generated code using pass@k metric from HumanEval paper
- **run_evaluation.sh**: Automated pipeline to run all models and generate summary
- **closed_access.py**: Future implementation for closed-source model APIs
- **open_access.py**: Future implementation for open-source model inference

## Usage Examples

### Run inference on specific problems

```bash
python3 inference.py \
  --model gpt-4o-mini \
  --num-samples 50 \
  --start-idx 0
```

### Evaluate results

```bash
python3 evaluate.py \
  --results outputs/inference_results/gpt-4o-mini_results.jsonl \
  --k-values 1 3 5 10 \
  --output outputs/evaluation_results/gpt-4o-mini_eval.json
```

### Custom dataset path

```bash
python3 inference.py \
  --model deepseek-r1 \
  --dataset /path/to/custom/dataset.jsonl
```

## Output Format

### Inference Results (JSONL)
```json
{
  "problem_id": "contest431_q3702_s0",
  "model_name": "gpt-4o-mini",
  "generated_code": "def maxLength(nums):\n    ...",
  "success": true,
  "latency_ms": 1234.5
}
```

### Evaluation Results (JSON)
```json
{
  "total_problems": 347,
  "pass_at_k": {
    "pass@1": 0.539,
    "pass@5": 0.712,
    "pass@10": 0.823
  }
}
```

## Citation

Based on the HumanEval evaluation methodology:
```bibtex
@article{chen2021evaluating,
  title={Evaluating Large Language Models Trained on Code},
  author={Chen, Mark and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}
```

## Notes

- All models use **zero-shot prompting** (no few-shot examples)
- Code is extracted from markdown code blocks automatically
- Results are saved incrementally to prevent data loss
- See EVALUATION_README.md for troubleshooting and advanced usage
