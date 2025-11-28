# Claude Sonnet 4.5 (AWS Bedrock) Evaluation

This module reproduces the execution-prediction and execution-choice evaluations with Anthropic's **Claude 3.5 Sonnet (October 2024 / Sonnet 4.5)** served through AWS Bedrock.

## Components

- `aws_client.py` – thin wrapper over the Bedrock runtime API plus invocation dataclasses.
- `prompts.py` – prompt builders, response parsers, metric helpers.
- `execution_prediction.py` – OC/OR/MC/MR benchmark loop.
- `execution_choice.py` – preference/correctness/reversion benchmark loop.
- `cli.py` – command-line entry point that stitches everything together, including optional reasoning-effort comparison.

## Usage

```bash
python -m src.models.bedrock_claude.cli \
  --dataset YOUR_USERNAME/leetcode-contests-431-467 \
  --task both \
  --output-dir experiments/results/bedrock_claude \
  --pred-num-problems 40 \
  --choice-num-problems 20
```

Flags mirror the notebook workflow:

- `--task` – `prediction`, `choice`, or `both`.
- `--pred-*` / `--choice-*` – knobs for reasoning effort, temperature, number of problems, etc.
- `--compare-reasoning` – optional low/medium/high ablation on a small subset.
- `--model-id`, `--latency-profile`, `--enable-thinking`, `--thinking-budget-tokens` – Bedrock-specific tuning (inference profile ARN, performance config, and Claude extended thinking options).
- Entries without a corresponding `mutated_code` are skipped automatically to avoid crashes; pass `--include-missing-mutations` if you need to keep them.

Outputs are written as timestamped JSON blobs containing the config, counts, summaries, and per-problem traces.

## AWS Requirements

1. **Model availability** – In November 2024 Amazon announced that `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (Sonnet 4.5) is generally available via Bedrock inference profiles, e.g. `arn:aws:bedrock:us-east-1:216874796537:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0`. Confirm the model (or a region-specific alias) is enabled for your account in the Bedrock console.
2. **Access + IAM** – Grant the executing IAM user/role `bedrock:InvokeModel` (and optionally `bedrock:InvokeModelWithResponseStream`) permissions for the desired region. Make sure Bedrock service access is activated inside the account.
3. **Credentials in runtime** – Export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and `AWS_REGION`. You can also override the region via `--region` and the model via `--model-id`.
4. **Dependencies** – Install `boto3`, `datasets`, `pandas`, and `tqdm` (either via repo `requirements.txt` or `pip install boto3 datasets pandas tqdm`).
5. **Thinking mode** – The CLI exposes `--enable-thinking` plus `--thinking-budget-tokens` to match AWS’s extended-thinking API described here: <https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html>.
6. **Cost monitoring** – Bedrock API calls are billable; keep an eye on CloudWatch metrics / Cost Explorer when running large batches.

### Raw AWS CLI example

```bash
aws bedrock-runtime converse \
  --model-id arn:aws:bedrock:us-east-1:216874796537:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --messages '[{"role":"user","content":[{"text":"hello."}]},{"role":"assistant","content":[{"text":"Hello! How can I help you today?"}]},{"role":"user","content":[{"text":""}]}]' \
  --inference-config '{}' \
  --additional-model-request-fields '{"thinking":{"type":"enabled","budget_tokens":2000}}' \
  --performance-config '{"latency":"standard"}' \
  --region us-east-1
```

This mirrors what the Python CLI does under the hood (via the Bedrock `converse` API) and shows how to enable extended thinking with explicit budgets/latency settings.

## Notes

- The CLI relies on HuggingFace datasets. Set `HF_HOME` or log in if the dataset is private.
- Increase `--pred-num-problems` / `--choice-num-problems` gradually; a full sweep over ~350 DSL/LeetCode tasks can take hours and incur meaningful API spend.
- The `BedrockInvocationParams` object supports seeding through `metadata.user_id`, which provides light stochastic control when temperature sampling is enabled.
- Problems missing mutated variants are filtered out by default (they often have `mutated_code=null` in the dataset). This prevents mid-run failures when building prompts.

## Galileo tracing (optional)

Enable the `--enable-galileo` flag to mirror the Galileo sample workflow and capture every Bedrock invocation as a trace:

1. Install the extra dependencies (already listed in `requirements.txt`): `pip install galileo python-dotenv`.
2. Create a `.env` file with your `GALILEO_API_KEY` (and optionally `GALILEO_CONSOLE_URL`, `GALILEO_PROJECT`, `GALILEO_LOG_STREAM`).
3. Provide AWS credentials via the usual environment variables or CLI config, then run for example:

```bash
python -m src.models.bedrock_claude.cli \
  --dataset YOUR_USERNAME/leetcode-contests-431-467 \
  --task prediction \
  --enable-galileo \
  --galileo-project MyFirstEvaluation \
  --galileo-log-stream aws-bedrock-sonnet
```

When tracing is enabled the CLI prints the project/log-stream URLs after the run so you can inspect the captured spans in Galileo.

### Extended thinking tips

- AWS enforces `temperature=1` whenever thinking mode is enabled.
- `--thinking-budget-tokens` must be strictly less than the corresponding `--pred-max-tokens` / `--choice-max-tokens`.
- Expect higher latency and cost when raising the thinking budget; start with ~2000–3000 tokens and tune from there.
