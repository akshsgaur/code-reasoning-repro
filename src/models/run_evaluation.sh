#!/bin/bash
# Complete evaluation pipeline for LeetCode dataset
# Runs inference with all models and evaluates results

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Use the venv Python
PYTHON="$SCRIPT_DIR/../../venv/bin/python3"

# Configuration
DATASET="../datasets/leetcode/data/datasets/leetcode_contests_431_467.jsonl"
OUTPUT_DIR="outputs/inference_results"
EVAL_OUTPUT_DIR="outputs/evaluation_results"

# Models to evaluate
MODELS=(
    "deepseek-r1"
    "gpt-4o-mini"
    "gpt-4o"
    "o3-mini"
    "gemini-2.5-pro"
)

echo "==========================================================="
echo "LeetCode Dataset Evaluation Pipeline"
echo "==========================================================="
echo "Dataset: $DATASET"
echo "Models: ${MODELS[@]}"
echo "==========================================================="

# Check API keys
echo ""
echo "Checking API keys..."
missing_keys=()

if [ -z "$OPENAI_API_KEY" ]; then
    missing_keys+=("OPENAI_API_KEY")
fi

if [ -z "$DEEPSEEK_API_KEY" ]; then
    missing_keys+=("DEEPSEEK_API_KEY")
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    missing_keys+=("GOOGLE_API_KEY")
fi

if [ ${#missing_keys[@]} -gt 0 ]; then
    echo "WARNING: Missing API keys: ${missing_keys[@]}"
    echo "Some models may not work. Set the following environment variables:"
    for key in "${missing_keys[@]}"; do
        echo "  export $key=your_api_key_here"
    done
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "==========================================================="
echo "Step 1: Running Inference"
echo "==========================================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "-----------------------------------------------------------"
    echo "Running inference with $model..."
    echo "-----------------------------------------------------------"

    if "$PYTHON" inference.py \
        --dataset "$DATASET" \
        --model "$model" \
        --output-dir "$OUTPUT_DIR"; then
        echo "✓ $model inference complete"
    else
        echo "✗ $model inference failed (continuing with other models)"
    fi
done

echo ""
echo "==========================================================="
echo "Step 2: Evaluating Results"
echo "==========================================================="

mkdir -p "$EVAL_OUTPUT_DIR"

for model in "${MODELS[@]}"; do
    results_file="$OUTPUT_DIR/${model}_results.jsonl"

    if [ -f "$results_file" ]; then
        echo ""
        echo "-----------------------------------------------------------"
        echo "Evaluating $model..."
        echo "-----------------------------------------------------------"

        if "$PYTHON" evaluate.py \
            --dataset "$DATASET" \
            --results "$results_file" \
            --output "$EVAL_OUTPUT_DIR/${model}_evaluation.json" \
            --k-values 1 3 5 10; then
            echo "✓ $model evaluation complete"
        else
            echo "✗ $model evaluation failed"
        fi
    else
        echo "⊘ Skipping $model (no results file found)"
    fi
done

echo ""
echo "==========================================================="
echo "Step 3: Summary"
echo "==========================================================="

echo ""
echo "Generating summary report..."

"$PYTHON" - <<EOF
import json
from pathlib import Path

eval_dir = Path("$EVAL_OUTPUT_DIR")
models = ["${MODELS[@]}".split()]

print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"\n{'Model':<20} {'pass@1':<10} {'pass@5':<10} {'pass@10':<10}")
print("-"*60)

for model in models:
    eval_file = eval_dir / f"{model}_evaluation.json"
    if eval_file.exists():
        with open(eval_file) as f:
            data = json.load(f)

        pass_at_k = data.get('pass_at_k', {})
        p1 = pass_at_k.get('pass@1', 0)
        p5 = pass_at_k.get('pass@5', 0)
        p10 = pass_at_k.get('pass@10', 0)

        print(f"{model:<20} {p1*100:>6.2f}%   {p5*100:>6.2f}%   {p10*100:>6.2f}%")
    else:
        print(f"{model:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("="*60)
print(f"\nDetailed results saved to: {eval_dir}")
print("="*60)
EOF

echo ""
echo "==========================================================="
echo "PIPELINE COMPLETE"
echo "==========================================================="
echo "Results saved to:"
echo "  Inference: $OUTPUT_DIR"
echo "  Evaluation: $EVAL_OUTPUT_DIR"
echo "==========================================================="
