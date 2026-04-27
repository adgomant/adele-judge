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
- QLoRA training support through Unsloth/TRL/Transformers/PEFT-compatible tooling
- Restricted continuation log-probability inference over `"1"` through `"5"`
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
  model_name_or_path: Qwen/Qwen3-8B-Instruct
```

## Installation

Install the core package:

```bash
python -m pip install -e .
```

Install training dependencies on a CUDA machine:

```bash
python -m pip install -e ".[train]"
```

Install developer/test dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Prepare filtered train/validation/test splits:

```bash
adele-judge prepare --config configs/adele_judge_qwen3_8b.yaml
```

Inspect which tokens receive supervised loss:

```bash
adele-judge debug-tokenization \
  --config configs/adele_judge_qwen3_8b.yaml \
  --num-examples 3
```

Train the judge:

```bash
adele-judge train --config configs/adele_judge_qwen3_8b.yaml
```

Run restricted-continuation prediction:

```bash
adele-judge predict \
  --config configs/adele_judge_qwen3_8b.yaml \
  --split validation
```

Evaluate saved predictions:

```bash
adele-judge evaluate \
  --config configs/adele_judge_qwen3_8b.yaml \
  --split validation
```

Run leave-one-model-out preparation or training:

```bash
adele-judge lomo --config configs/adele_judge_qwen3_8b.yaml
```

## Configuration

The main config is [configs/adele_judge_qwen3_8b.yaml](configs/adele_judge_qwen3_8b.yaml). Most behavior is controlled there: data paths, column names, filters, split policy, model, prompt, training settings, and inference settings.

CLI overrides use repeated `--override key.path=value` arguments:

```bash
adele-judge prepare \
  --config configs/adele_judge_qwen3_8b.yaml \
  --override data.filters.max_response_tokens=2048 \
  --override training.max_seq_length=8192
```

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

`data.filters.max_response_tokens` and `training.max_seq_length` are intentionally separate gates. The response cap applies only to the model response before prompt formatting. The sequence cap applies later to the full chat-formatted training example, including the system prompt, benchmark/task metadata, question, reference answer, response, chat-template tokens, and score target. Passing the response cap does not guarantee that an example fits in `max_seq_length`.

## Training Behavior

The model is trained with standard causal language-model cross-entropy loss, but labels are masked so loss is applied only to the assistant score continuation:

```text
system + user prompt -> assistant score digit
```

Prompt tokens, padding, and EOS tokens are labeled `-100`. The target is exactly one score string from `"1"` to `"5"` and no explanation, JSON, or rationale.

Packing is configurable:

```yaml
training:
  max_seq_length: 8192
  packing: false
  group_by_length: true
```

Use `scripts/debug_tokenization.py` before long runs to verify label masking on real examples.

## Inference

Default prediction does not use free-form generation. For each example, the pipeline scores the explicit continuations:

```text
"1", "2", "3", "4", "5"
```

Each continuation is scored by log probability, including the robust case where a score string tokenizes into more than one token. The highest-probability continuation becomes `pred_score`, and `pred_binary` is derived with the fixed threshold.

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
- Majority binary baseline from the training split

## Run Artifacts

Each run writes to `runs/{run_name}/`:

```text
config.yaml
prepared/train.parquet
prepared/validation.parquet
prepared/test.parquet
adapter/
tokenizer/
train_metrics.json
validation_metrics.json
test_metrics.json
predictions_validation.parquet
predictions_test.parquet
per_model_metrics.csv
per_benchmark_metrics.csv
per_task_metrics.csv
confusion_matrix_ordinal.csv
confusion_matrix_binary.csv
length_statistics.json
dataset_filtering_report.json
token_supervision_debug.txt
```

Some reports are also saved with split-specific filenames, for example `validation_per_model_metrics.csv`.

`dataset_filtering_report.json` includes both response-cap removals and full-sequence overflow counts. It also records the effective prompt budget, computed as `training.max_seq_length - data.filters.max_response_tokens`, to make length pressure easy to diagnose.

## Validation

Run the lightweight test suite:

```bash
python -m pytest
```

Run linting:

```bash
ruff check .
```

The tests cover target construction, split leakage checks, prompt formatting, supervised label masking, packing behavior, and metrics.

## Notes For Operators

- The processed parquet already contains a `response_tokens` column, but this pipeline recomputes response lengths with the selected tokenizer.
- The default response cap remains `4096`, while the default full sequence length is `8192`.
- Validation and test models must not appear in the training split.
- Long examples are filtered or rejected; they are not silently truncated.
- Adapter loading for inference is controlled by `model.adapter_path`.
- Zero-shot scoring can be run by omitting `model.adapter_path` and using the same restricted continuation scorer.
