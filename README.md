# Code-Reasoning Reproduction (CMU 11-785)

Reproduction of a recent code-reasoning paper with **two evaluations**—*Execution Prediction* and *Execution Choice*—using **coverage-guided mutations** over three sources: **DSL-List**, **LLM-List**, and **LeetCode**. Fully config-driven runs, ≥3 seeds, logs/plots, and at least one extension (ablation or robustness).

---

## 📌 What’s inside

* **Datasets**: deterministic DSL generator → Python; LLM-seeded problems; LeetCode contest solutions.
* **Mutations**: local code edits filtered by execution & line-coverage similarity.
* **Evals**:

  * **Exec Prediction**: predict program output; metrics: **OC/MC/OR/MR**.
  * **Exec Choice**: choose A/B then predict; metrics: **Preference, Correctness, Reversion**.
* **Reproducibility**: configs, fixed seeds, one-command runs, artifacts on disk.

---

## 🗂️ Repo structure

```
code-reasoning-repro/
  README.md
  RUNME.md
  env.yml
  LICENSE
  CITATION.cff
  configs/
    model_qwen25_7b.yaml
    model_deepseek_6.7b.yaml
    dataset_dsl_small.yaml
  experiments/
    logs/
    results/
  scripts/
    make_dsl.py
    make_llm_list.py
    make_leetcode.py
    run_exec_pred.sh
    run_exec_choice.sh
  src/
    datasets/
      dsl/
      llm_list/
      leetcode/
      mutate.py
    eval/
      exec_prediction.py
      exec_choice.py
    models/
      open_access.py
      closed_access.py
    utils/
      run_python_safely.py
      coverage.py
      io.py
  prompts/
```

---

## 📖 Target paper

```
@article{yang2025evaluating,
  title   = {Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning},
  author  = {Yang, Rem and Dai, Julian and Vasilakis, Nikos and Rinard, Martin},
  journal = {arXiv preprint arXiv:2504.05518},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.05518}
}
```
[2504.05518v1.pdf](https://github.com/user-attachments/files/22913347/2504.05518v1.pdf)

---

## 🚀 Quickstart

### 1) Create environment

```bash
# Option A: conda (recommended)
conda env create -f env.yml
conda activate code-reasoning-repro

# Option B: from scratch (example versions)
conda create -n code-reasoning-repro python=3.10 -y
conda activate code-reasoning-repro
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate datasets evaluate pandas numpy tqdm pyyaml rich jsonlines coverage
pip install matplotlib scipy
# optional trackers
pip install wandb tensorboard
```

### 2) Get/open models

* Use **vLLM** for open models (e.g., Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B).
* For closed models, set API keys and use `src/models/closed_access.py`.

### 2.5) Download the LeetCode dataset (local only)

GitHub rejects the raw LeetCode dump because it exceeds the 100 MB file limit, so you must materialize it on your own machine whenever you need it.

```bash
# activate the repo environment first
python src/datasets/leetcode/dataset_download.py
```

What this script does:

- streams `livecodebench/code_generation_lite` from Hugging Face (≈4 GB download); the data stays in the HF cache under `~/.cache/huggingface/` for reuse
- prints a schema/sample summary for each split
- writes lightweight metadata + sample rows to `src/datasets/leetcode/lcb_codegen_overview/`
- creates LeetCode-only JSONL files in `src/datasets/leetcode/lcb_codegen_overview/leetcode_only/`

Those outputs are **ignored by git** (see `.gitignore`), so keep them locally or upload to your own storage if you want to share them.

### 3) Generate a tiny DSL split (MVP)

```bash
python scripts/make_dsl.py \
  --config configs/dataset_dsl_small.yaml \
  --out experiments/datasets/dsl_small
```

### 4) Create mutants

```bash
python -m src.datasets.mutate \
  --in_dir experiments/datasets/dsl_small \
  --out_dir experiments/datasets/dsl_small_mut \
  --max_mutants 1
```

### 5) Run **Execution Prediction**

```bash
bash scripts/run_exec_pred.sh \
  configs/model_qwen25_7b.yaml \
  experiments/datasets/dsl_small \
  experiments/datasets/dsl_small_mut \
  experiments/results/exec_pred_qwen25_7b
```

### 6) Run **Execution Choice**

```bash
bash scripts/run_exec_choice.sh \
  configs/model_qwen25_7b.yaml \
  experiments/datasets/dsl_small \
  experiments/datasets/dsl_small_mut \
  experiments/results/exec_choice_qwen25_7b
```

### 7) Plot & summarize

```bash
python -m src.eval.summarize \
  --in_dir experiments/results/exec_pred_qwen25_7b \
  --plots_dir experiments/results/plots
```

---

## ⚙️ Configuration

All experiments are **config-first**. Example fields:

```yaml
# configs/model_qwen25_7b.yaml
model:
  name: Qwen2.5-Coder-7B
  backend: vllm
  dtype: float16
  max_new_tokens: 512
  temperature: 0.2
  top_p: 0.95
  seed: 123

eval:
  type: exec_prediction   # or exec_choice
  samples_per_item: 5     # for pass@1 estimate
  timeout_s: 6

logging:
  save_generations: true
  save_prompts: true
  save_traces: true
```

```yaml
# configs/dataset_dsl_small.yaml
dataset:
  name: dsl_list
  n_programs: 40
  depth_min: 3
  depth_max: 5
  inputs_per_program: 3
  int_min: -1
  int_max: 5
  list_len_min: 3
  list_len_max: 5
mutate:
  enable: true
  ops: [arith, relop, logic, keyword, literal]
  coverage_similarity: true
```

---

## 🧪 Evaluations & Metrics

### Execution Prediction

* Input: single program + input(s); model outputs final value.
* Metrics:

  * **OC**: correct on original.
  * **MC**: correct on mutated.
  * **OR/MR** (*reversion*): model changes a prior correct answer to wrong on re-query; boolean outputs excluded from reversion.

### Execution Choice

* Input: pair (A: original, B: mutated). Model chooses one and predicts.
* Metrics: **Preference** for original, **Correctness** on chosen, **Reversion** across repeats with A/B swapped.

> All metrics reported as mean ± 95% CI over ≥3 seeds.

---

## 📊 Reproducing tables & figures

1. Set seeds: `--seed {123, 456, 789}` or via config list.
2. Run both evals on **DSL-List** first.
3. Add **LLM-List** (seed with open model) and a filtered **LeetCode** slice.
4. Use `src/eval/summarize.py` to produce:

   * OC/MC/OR/MR vs. LOC
   * Preference histograms
   * Seed-wise tables (CSV/JSON)

---

## 🧩 Extensions (pick ≥1)

* **Ablation**: disable coverage-similarity selection and compare OC/MC.
* **Generalization**: increase DSL depth or operator priors (e.g., more `map`/`if`).
* **Evaluation+**: add intermediate state checks for a subset.

---

## 📦 Artifacts & Logging

* Prompts, generations, raw predictions → `experiments/results/**/artifacts/`
* Coverage traces → `experiments/results/**/coverage/`
* Optional Weights & Biases: set `WANDB_PROJECT` and run any script.

---

## 🔁 One-command runs (for CI/repro)

Add to `RUNME.md`:

```bash
make dsl-small        # data + mutants
make pred-qwen        # Exec Prediction (Qwen2.5-7B)
make choice-qwen      # Exec Choice (Qwen2.5-7B)
make summarize        # tables + plots
```

---

## 🖥️ System & Environment

Record in the report & commit:

* GPUs/CPUs, RAM, OS
* CUDA/cuDNN
* `conda env export > env.yml`
* Git tag: `v1.0-repro`

---

## 🤝 Contributing

PRs welcome for:

* Additional mutation ops
* New open models/configs
* Dataset adapters (other DSLs)

---

## 🔒 License

Choose a permissive license (e.g., MIT). Update `LICENSE`.

---

## 🧾 Citation

If you use this repo, please cite the target paper and this reproduction (CFF provided).

---

## 🙏 Acknowledgements

Thanks to the original authors of the target paper and the open-model communities (Qwen, DeepSeek, etc.).
