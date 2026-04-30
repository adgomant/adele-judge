# ADeLe Distilled Judge

A configurable training and evaluation pipeline for an **ADeLe-suite-specific distilled judge**. The judge learns to score model responses on the ADeLe benchmark suite as an ordinal value from `1` to `5`, then derives binary correctness only for reporting:

```text
score >= 3 -> CORRECT
score <= 2 -> INCORRECT
```

The project is designed for out-of-model generalization: train on responses from one set of models and evaluate on held-out models from the same ADeLe suite.

## What This Project Provides

- Configurable parquet loading with YAML column mappings
- Judge-disagreement and tokenizer-based response-length filtering
- Ordinal target construction from two judge scores
- Fixed model-based splits and leave-one-model-out workflows
- Chat-style supervised fine-tuning with loss only on the score digit
- Optional restricted 5-way score cross-entropy aligned with continuation scoring
- Optional discriminative 5-way sequence-classification training
- QLoRA training support through Unsloth/TRL/Transformers/PEFT-compatible tooling
- Batched restricted continuation log-probability inference over `"1"` through `"5"`
- Ordinal, binary, grouped, and baseline metrics
- Typer/Rich command-line interface with progress bars and tables
- Reproducible run directories with configs, reports, predictions, and model artifacts

## Repository Layout

```text
.
├── configs/
│   └── adele_judge_qwen3_8b.yaml
├── data/
│   └── processed/
│       └── response_scores.parquet
├── scripts/
│   ├── prepare_dataset.py
│   ├── debug_tokenization.py
│   ├── train_judge.py
│   ├── predict_judge.py
│   ├── evaluate_judge.py
│   └── run_lomo.py
├── src/
│   └── adele_judge/
└── tests/
```

## Requirements

Core data preparation and evaluation utilities run on ordinary Python environments.

Training is intended for **Linux + CUDA**. The default model and training stack are optimized for QLoRA fine-tuning of a 7B/8B/9B-class instruct model.

Default base model:

```yaml
model:
  model_name_or_path: Qwen/Qwen3-8B
  trust_remote_code: true
  attn_implementation: flash_attention_2
  thinking_mode:
    enabled: false
    apply_if_supported: true
```

For tokenizers whose chat template supports a thinking flag such as `enable_thinking`,
the default judge configuration uses non-thinking mode so the model is trained and
scored as a direct `"1"` through `"5"` continuation. Unsupported tokenizers skip this
option rather than receiving an unknown chat-template argument.

## Installation

This repository is managed with [`uv`](https://docs.astral.sh/uv/). The lockfile records the resolved dependency set, and the CLI examples below assume commands are run through `uv run`.

Core data preparation and reporting install:

```bash
uv sync
```

Developer/test install:

```bash
uv sync --extra dev
```

Training install for a Linux + CUDA machine:

```bash
uv sync --extra train
```

Training with FlashAttention 2 support:

```bash
uv sync --extra train --extra fa2
```

Training with DeepSpeed support:

```bash
uv sync --extra train --extra deepspeed
```

The `train` extra includes CUDA training/runtime helpers used by Unsloth, TRL, PEFT, and bitsandbytes:

```text
accelerate, bitsandbytes, ninja, packaging, peft, psutil, setuptools, torch, trl, unsloth, wheel
```

The `fa2` extra installs `flash-attn`. The `deepspeed` extra installs DeepSpeed. Both are intentionally separate because they are Linux/CUDA/PyTorch-build sensitive and can make CPU, macOS, or mismatched CUDA environments brittle.

If FlashAttention needs to be installed manually in an already-created environment, use:

```bash
uv pip install setuptools wheel packaging psutil ninja
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

Verify the CUDA training stack:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

Optional, but recommended for faster and less rate-limited Hugging Face downloads:

```bash
export HF_TOKEN=...
```

If you prefer `pip`, install equivalent extras with:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"
python -m pip install -e ".[train,fa2]"
python -m pip install -e ".[train,deepspeed]"
```

## Dependency Extras

The extras are:

- `dev`: test and lint tools.
- `train`: CUDA training stack without FlashAttention.
- `fa2`: FlashAttention 2, opt-in for compatible Linux/CUDA environments.
- `deepspeed`: DeepSpeed, opt-in for compatible Linux/CUDA environments.

`flash-attn` has undeclared build requirements in some environments, so [pyproject.toml](pyproject.toml) includes `tool.uv.extra-build-dependencies` for uv builds.

## Quick Start

Prepare filtered train/validation/test splits:

```bash
uv run adele-judge prepare --config configs/adele_judge_qwen3_8b.yaml
```

Inspect which tokens receive supervised loss:

```bash
uv run adele-judge debug-tokenization \
  --config configs/adele_judge_qwen3_8b.yaml \
  --num-examples 3
```

Train the judge:

```bash
uv run adele-judge train --config configs/adele_judge_qwen3_8b.yaml
```

Train a finalist adapter on all prepared rows:

```bash
uv run adele-judge train --config configs/adele_judge_qwen3_8b.yaml --finalist
```

Stage and upload a trained judge to the Hugging Face Hub:

```bash
uv run adele-judge push-to-hub \
  --config runs/qwen3_8b_adele_judge/inference_config.yaml \
  --repo-id USER_OR_ORG/MODEL_NAME
```

Run restricted-continuation prediction:

```bash
uv run adele-judge predict \
  --config runs/qwen3_8b_adele_judge/inference_config.yaml \
  --split validation
```

Evaluate saved predictions:

```bash
uv run adele-judge evaluate \
  --config runs/qwen3_8b_adele_judge/inference_config.yaml \
  --split validation
```

Run leave-one-model-out preparation or training:

```bash
uv run adele-judge lomo --config configs/adele_judge_qwen3_8b.yaml
```

To force a clean dataset rebuild before training:

```bash
uv run adele-judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --force-prepare
```

## Configuration

The main config is [configs/adele_judge_qwen3_8b.yaml](configs/adele_judge_qwen3_8b.yaml). Most behavior is controlled there: data paths, column names, filters, split policy, model, prompt, training settings, and inference settings.

CLI overrides use repeated `--override key.path=value` arguments:

```bash
uv run adele-judge prepare \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override data.filters.max_response_tokens=2048 \
  --override training.max_seq_length=8192
```

Hub packaging can be configured in YAML or via CLI flags:

```yaml
hub:
  repo_id: USER_OR_ORG/MODEL_NAME
  private: false
  commit_message: Upload ADeLe distilled judge
  local_checkpoint_dir: runs/qwen3_8b_adele_judge
  output_staging_dir: hub_staging/qwen3_8b_adele_judge
  create_pr: false
  max_shard_size: 5GB
```

Use `--no-push` to build the Hub staging directory without uploading.

The default column mapping is:

```yaml
data:
  columns:
    question: question
    reference_answer: ground_truth
    response: response
    judge_1_score: score_gpt4o
    judge_2_score: score_sonnet
    model_id: model_id
    benchmark: benchmark
    task: task
    example_id: instance_id
```

## Data Flow

1. Load `data/processed/response_scores.parquet`.
2. Validate configured columns and judge scores.
3. Compute judge disagreement.
4. Compute `target_score = floor(mean(judge_1_score, judge_2_score))`.
5. Derive `target_binary` for metrics only.
6. Compute response token length with the selected base-model tokenizer.
7. Apply disagreement and response-only length filters.
8. Format examples with the same chat template used for training and inference.
9. Enforce `max_seq_length` with `on_sequence_overflow: skip` or `error`.
10. Create model-held-out splits.
11. Write a preparation fingerprint so stale prepared splits are rebuilt when relevant settings change.

`data.filters.max_response_tokens` and `training.max_seq_length` are intentionally separate gates. The response cap applies only to the model response before prompt formatting. The sequence cap applies later to the full chat-formatted training example, including the system prompt, question, reference answer, response, chat-template tokens, and score target. Passing the response cap does not guarantee that an example fits in `max_seq_length`.

## Training Behavior

The default config trains with restricted 5-way score cross-entropy over the score token ids. The baseline `causal_lm` objective is still available; in that mode labels are masked so standard causal language-model loss is applied only to the assistant score continuation:

```text
system + user prompt -> assistant score digit
```

Prompt tokens, padding, and EOS tokens are labeled `-100`. The target is exactly one score string from `"1"` to `"5"` and no explanation, JSON, or rationale.

A discriminative alternative is selected in the same canonical config:

```yaml
training:
  objective: sequence_classification
  loss:
    type: ce_5way
    lambda_binary: 0.5
    class_weights: null
  train_sampling_strategy: random
  packing: false
```

This path loads `AutoModelForSequenceClassification` with five labels, maps scores `1..5` to labels `0..4`, and trains a classification head on the same chat-formatted prompt before the assistant score. The optional `ce_5way_plus_binary` loss adds a stable binary loss over score groups `{1, 2}` and `{3, 4, 5}` without changing the downstream threshold rule.

`training.objective` is the regime switch. Config normalization keeps `inference.method` compatible with that objective, so `sequence_classification` uses classifier logits and the causal regimes use restricted continuation scoring.

Packing is configurable:

```yaml
training:
  max_seq_length: 4096
  packing: false
  train_sampling_strategy: random
  length_column_name: length
```

`train_sampling_strategy: random` is the canonical default for both training regimes. Length bucketing remains available by setting `train_sampling_strategy: group_by_length`, but it is an explicit experiment rather than the default.

Use `scripts/debug_tokenization.py` before long runs to verify label masking on real examples.

Training writes `inference_config.yaml` with `model.adapter_path` pointing at the saved adapter. Prediction also auto-loads `runs/<run>/adapter` when it exists, which avoids accidentally evaluating the base model after a training run.

The default training config uses a validation monitor subset during training, stratified by `model_id` and `target_score`. Run full validation/test prediction and evaluation after training with the generated `inference_config.yaml`.

Finalist training is enabled with `--finalist`. It trains on the union of the prepared train, validation, and test splits, disables Trainer evaluation, and skips validation trainer metrics. Use a distinct `project.run_name` or output directory when you want to preserve artifacts from earlier validation runs.

## Distributed Training

Distributed training is config-driven through the top-level `distributed` section. The implementation uses the existing `transformers.Trainer` path, so dataloader sharding, metric gathering, and checkpoint coordination are handled by Trainer/Accelerate.

Single-GPU behavior is unchanged. When `distributed.enabled=true`, the training loader bypasses Unsloth and uses standard Transformers+PEFT. Training does not use `device_map="auto"` in distributed mode; each process owns its launcher-assigned local GPU.

DDP with QLoRA is the recommended multi-GPU path:

```bash
uv run torchrun --nproc_per_node=8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override distributed.strategy=ddp
```

The same run can be launched with Accelerate:

```bash
uv run accelerate launch --multi_gpu --num_processes 8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override distributed.strategy=ddp
```

For FSDP, disable 4-bit loading and configure the transformer layer class. For the default Qwen3 model, use `Qwen3DecoderLayer`:

```bash
uv run accelerate launch --multi_gpu --num_processes 8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override distributed.strategy=fsdp \
  --override training.load_in_4bit=false \
  --override distributed.fsdp.transformer_layer_cls_to_wrap=Qwen3DecoderLayer
```

DeepSpeed is configured inline in the YAML under `distributed.deepspeed`, so no separate JSON file is required. Install the extra, disable 4-bit loading, and select the strategy:

```bash
uv run accelerate launch --multi_gpu --num_processes 8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override distributed.strategy=deepspeed \
  --override training.load_in_4bit=false \
  --override distributed.deepspeed.zero_stage=2
```

Minimal SLURM template:

```bash
#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

uv run accelerate launch --multi_gpu --num_processes 8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override distributed.strategy=ddp
```

Limitations:

- Unsloth is kept for single-GPU causal training only.
- QLoRA/4-bit training is supported for DDP, not FSDP or DeepSpeed.
- DeepSpeed is optional and requires installing the `deepspeed` extra; its Trainer config is generated from `distributed.deepspeed` in the run YAML.
- Hub merge and upload remain a separate single-process `push-to-hub` step.

Training logs `per_device_train_batch_size`, `gradient_accumulation_steps`, `world_size`, and `effective_global_batch_size`. Resume Trainer checkpoints with:

```bash
uv run torchrun --nproc_per_node=8 -m adele_judge train \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override distributed.enabled=true \
  --override training.resume_from_checkpoint=runs/qwen3_8b_adele_judge/checkpoints/checkpoint-3000
```

## Inference

Default prediction does not use free-form generation. For each example, the pipeline scores the explicit continuations:

```text
"1", "2", "3", "4", "5"
```

When all score continuations are single-token, inference uses one batched forward pass per prompt batch and gathers the five score-token log probabilities from the prompt-final logits. If a continuation is multi-token, the code falls back to exact continuation scoring. The highest-probability continuation becomes `pred_score`, and `pred_binary` is derived with the fixed threshold.

## Hugging Face Hub Packaging

`adele-judge push-to-hub` prepares a Hub model repository from a trained local run. The command requires the run adapter at `runs/<run>/adapter`, prefers the run tokenizer at `runs/<run>/tokenizer`, merges the LoRA adapter into the base model at the repository root, and also copies the original adapter to `adapter/`.

The uploaded repo includes:

```text
config.json / model shards / tokenizer files
generation_config.json
README.md
adele_judge_pipeline.py
adele_judge_config.json
adele_judge_metadata.json
adapter/
training_config.yaml
```

The recommended Hub-side inference path is restricted continuation scoring:

```python
from transformers import pipeline

judge = pipeline(
    "adele-judge",
    model="USER_OR_ORG/MODEL_NAME",
    trust_remote_code=True,
    device_map="auto",
)
result = judge(
    {"question": "...", "reference_answer": "...", "model_response": "..."}
)

results = judge([
    {"question": "...", "reference_answer": "...", "model_response": "..."},
    {"question": "...", "ground_truth": "...", "model_response": "..."},
], batch_size=8)
```

The generated `generation_config.json` uses safe one-token defaults for debugging. `generate()` is not the recommended prediction method.

## Metrics

The evaluator reports:

- Ordinal accuracy
- Ordinal MAE
- Within-1 accuracy
- 5-way confusion matrix
- Binary accuracy
- Binary macro F1
- Per-class precision, recall, and F1
- False positive and false negative rates for the `CORRECT` class
- Per-model, per-benchmark, per-task, per-target-score, and response-length-bucket metrics
- Per-score precision, recall, F1, support, and ordinal macro-F1
- Confidence, score-margin, entropy, and simple calibration summaries when probabilities are present
- Majority binary and ordinal baselines from the training split

## Run Artifacts

Preparation and training write to `runs/{run_name}/`:

```text
config.yaml
prepared/train.parquet
prepared/validation.parquet
prepared/test.parquet
prepared/prepared_fingerprint.json
tokenized/
adapter/
tokenizer/
train_metrics.json
validation_trainer_metrics.json
training.log
trainer_log_history.json
length_statistics.json
dataset_filtering_report.json
token_supervision_debug.txt
score_tokenization_report.json
prepared_fingerprint.json
resolved_training_args.json
run_metadata.json
inference_config.yaml
```

`validation_trainer_metrics.json` is written only for standard training runs with validation monitoring enabled.

Prediction and evaluation add split-specific artifacts such as:

```text
predictions_validation.parquet
predictions_test.parquet
validation_metrics.json
test_metrics.json
majority_baseline_validation.json
majority_baseline_test.json
majority_ordinal_baseline_validation.json
majority_ordinal_baseline_test.json
confusion_matrix_ordinal_validation.csv
confusion_matrix_binary_validation.csv
validation_per_model_metrics.csv
validation_per_benchmark_metrics.csv
validation_per_task_metrics.csv
validation_per_target_score_metrics.csv
validation_per_response_length_bucket_metrics.csv
```

Some reports are also saved with non-split-specific convenience filenames, for example `per_model_metrics.csv` and `confusion_matrix_ordinal.csv`, and will reflect the most recently evaluated split.

`dataset_filtering_report.json` includes both response-cap removals and full-sequence overflow counts. It also records the effective prompt budget, computed as `training.max_seq_length - data.filters.max_response_tokens`, to make length pressure easy to diagnose.

## Validation

Run the lightweight test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

The tests cover target construction, split leakage checks, prompt formatting, supervised label masking, packing behavior, and metrics.
They also cover thinking-mode chat-template handling, leave-one-model-out split behavior, score-token contracts, fast batched inference, and configurable attention implementation.

## Notes For Operators

- The processed parquet already contains a `response_tokens` column, but this pipeline recomputes response lengths with the selected tokenizer.
- The default response cap is `3072`, while the default full sequence length is `4096`.
- Validation models, and any configured test models, must not appear in the training split. `test_models` may be empty for validation-only runs.
- In leave-one-model-out mode, the held-out model is reserved for test; validation is sampled from training-side models for monitoring.
- In finalist training mode, all prepared splits are merged for training and no held-out validation metrics are produced.
- Long examples are filtered or rejected; they are not silently truncated.
- Adapter loading for inference is controlled by `model.adapter_path`, with auto-resolution to the run adapter when present.
- Zero-shot scoring can be run by omitting `model.adapter_path` and using the same restricted continuation scorer.
- Unsloth chooses its own attention backend during training. `model.attn_implementation` is passed only to the Transformers fallback and inference loaders.
- FlashAttention 2 is optional. If it is missing or incompatible, use the `train` extra without `fa2` and let the stack fall back to the available attention backend.
