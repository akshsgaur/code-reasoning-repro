#!/bin/bash
set -e

echo "========================================"
echo "CODE REASONING EVALUATION - FULL RUN"
echo "Started: $(date)"
echo "========================================"

GREEN='\033[0;32m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

# Check AWS credentials for Bedrock
print_status "Checking AWS credentials..."
aws sts get-caller-identity > /dev/null 2>&1
print_status "✓ AWS credentials OK"

# Check GPU
print_status "Checking GPU..."
nvidia-smi > /dev/null 2>&1
print_status "✓ GPU detected"

# Install dependencies
print_status "Installing dependencies..."
pip install -q -r requirements.txt
print_status "✓ Dependencies installed"

# Create results directories
mkdir -p experiments/results/{bedrock_claude,gpt_oss,deepseek_coder}
mkdir -p logs

# Run Claude (Bedrock) - in background
print_status "Starting Claude evaluation (background)..."
python -m src.models.bedrock_claude.cli \
  --dataset asgaur/leetcode-contests-431-467-mutated \
  --task both \
  --output-dir experiments/results/bedrock_claude \
  --pred-num-problems 340 \
  --pred-generations 5 \
  --choice-num-problems 340 \
  --choice-runs 2 \
  --enable-thinking \
  --thinking-budget-tokens 1024 \
  > logs/bedrock_claude.log 2>&1 &
CLAUDE_PID=$!
print_status "Claude PID: $CLAUDE_PID"

# Run GPT-OSS
print_status "Starting GPT-OSS evaluation..."
python -m src.models.gpt_oss.cli \
  --dataset asgaur/leetcode-contests-431-467-mutated \
  --task both \
  --output-dir experiments/results/gpt_oss \
  --pred-num-problems 340 \
  --pred-generations 5 \
  --choice-num-problems 340 \
  --choice-runs 2 \
  2>&1 | tee logs/gpt_oss.log

# Run DeepSeek-Coder (modify cli.py model_id)
print_status "Starting DeepSeek-Coder evaluation..."
python -m src.models.gpt_oss.cli \
  --dataset asgaur/leetcode-contests-431-467-mutated \
  --task both \
  --output-dir experiments/results/deepseek_coder \
  --model-id deepseek-ai/deepseek-coder-33b-instruct \
  --pred-num-problems 340 \
  --pred-generations 5 \
  --choice-num-problems 340 \
  --choice-runs 2 \
  2>&1 | tee logs/deepseek_coder.log

# Wait for Claude
print_status "Waiting for Claude to complete..."
wait $CLAUDE_PID

print_status "All evaluations complete!"
print_status "Results in experiments/results/"
print_status "Logs in logs/"
EOF

chmod +x run_all.sh