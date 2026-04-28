from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import track

from .config import copy_config, save_config
from .formatting import tokenizer_thinking_template_kwargs
from .modeling import load_model_for_training
from .pipeline import load_or_prepare_splits
from .tokenization import tokenize_supervised_example, validate_score_tokenization
from .utils import (
    ensure_dir,
    git_commit,
    package_versions,
    project_output_dir,
    set_seed,
    stable_json_hash,
    write_json,
)


@dataclass
class CausalDataCollator:
    tokenizer: Any
    include_score_metadata: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.include_score_metadata:
            batch["prompt_lengths"] = torch.tensor(
                [feature["prompt_length"] for feature in features],
                dtype=torch.long,
            )
            batch["target_scores"] = torch.tensor(
                [int(feature["target_score"]) for feature in features],
                dtype=torch.long,
            )
        return batch


def tokenize_training_dataframe(df: Any, tokenizer: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    system_prompt = config["prompt"]["system_prompt"]
    max_seq_length = int(config["training"]["max_seq_length"])
    overflow = config["data"]["filters"].get("on_sequence_overflow", "skip")
    rows = []
    skipped = 0
    for _, row in track(
        df.iterrows(),
        total=len(df),
        description="Tokenizing training rows",
    ):
        tokenized = tokenize_supervised_example(
            row.to_dict(),
            tokenizer,
            system_prompt,
            max_seq_length,
            overflow,
        )
        if tokenized is None:
            skipped += 1
            continue
        rows.append(
            {
                "input_ids": tokenized.input_ids,
                "attention_mask": tokenized.attention_mask,
                "labels": tokenized.labels,
                "length": tokenized.sequence_length,
                "prompt_length": tokenized.prompt_length,
                "target_score": int(row["target_score"]),
            }
        )
    if skipped:
        print(f"Skipped {skipped} overflowing examples during tokenization")
    if bool(config["training"].get("packing", False)):
        rows = pack_tokenized_rows(rows, int(config["training"]["max_seq_length"]))
    return rows


def pack_tokenized_rows(rows: list[dict[str, Any]], max_seq_length: int) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current = {"input_ids": [], "attention_mask": [], "labels": []}

    def flush() -> None:
        if current["input_ids"]:
            packed.append(
                {
                    "input_ids": current["input_ids"].copy(),
                    "attention_mask": current["attention_mask"].copy(),
                    "labels": current["labels"].copy(),
                    "length": len(current["input_ids"]),
                }
            )
            current["input_ids"].clear()
            current["attention_mask"].clear()
            current["labels"].clear()

    for row in rows:
        row_len = len(row["input_ids"])
        if row_len > max_seq_length:
            continue
        if current["input_ids"] and len(current["input_ids"]) + row_len > max_seq_length:
            flush()
        current["input_ids"].extend(row["input_ids"])
        current["attention_mask"].extend(row["attention_mask"])
        current["labels"].extend(row["labels"])
    flush()
    return packed


def make_restricted_score_trainer(
    base_trainer_class: Any,
    score_token_ids: list[int],
    class_weights: list[float] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    class _RestrictedScoreTrainer(base_trainer_class):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            prompt_lengths = inputs.pop("prompt_lengths")
            target_scores = inputs.pop("target_scores")
            inputs.pop("labels", None)
            outputs = model(**inputs)
            logits = outputs.logits
            positions = prompt_lengths.to(logits.device) - 1
            if torch.any(positions < 0):
                raise ValueError("All prompts must contain at least one token")
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            score_ids = torch.tensor(score_token_ids, dtype=torch.long, device=logits.device)
            score_logits = logits[batch_indices, positions][:, score_ids]
            labels = target_scores.to(logits.device) - 1
            weight = None
            if class_weights is not None:
                weight = torch.tensor(class_weights, dtype=score_logits.dtype, device=logits.device)
            loss = F.cross_entropy(score_logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    return _RestrictedScoreTrainer


def select_eval_subset(df: Any, config: dict[str, Any]) -> Any:
    subset_size = config["training"].get("eval_subset_size")
    if subset_size is None:
        return df
    subset_size = int(subset_size)
    if subset_size <= 0 or len(df) <= subset_size:
        return df
    seed = int(config["training"].get("eval_subset_seed", config["training"].get("seed", 42)))
    return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)


def tokenized_cache_fingerprint(
    split_name: str,
    config: dict[str, Any],
    df: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    components = {
        "schema_version": 1,
        "split_name": split_name,
        "num_examples": int(len(df)),
        "model_name_or_path": config["model"]["model_name_or_path"],
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "thinking_mode": config["model"].get("thinking_mode"),
        "thinking_mode_support_detected": getattr(
            tokenizer, "_adele_thinking_support_detected", None
        ),
        "chat_template_kwargs": tokenizer_thinking_template_kwargs(tokenizer),
        "prompt": config["prompt"],
        "max_seq_length": config["training"]["max_seq_length"],
        "overflow": config["data"]["filters"].get("on_sequence_overflow", "skip"),
        "packing": bool(config["training"].get("packing", False)),
        "objective": config["training"].get("objective", "causal_lm"),
        "allowed_scores": config["inference"].get("allowed_scores"),
        "data_digest": stable_json_hash(
            {
                "model_counts": df["model_id"].value_counts().to_dict()
                if "model_id" in df.columns
                else {},
                "target_counts": df["target_score"].value_counts().to_dict()
                if "target_score" in df.columns
                else {},
                "example_ids": df["example_id"].head(100).tolist()
                if "example_id" in df.columns
                else [],
            }
        ),
    }
    return {"fingerprint": stable_json_hash(components), "components": components}


def load_or_tokenize_dataset(
    split_name: str,
    df: Any,
    tokenizer: Any,
    config: dict[str, Any],
    output_dir: Path,
) -> Any:
    from datasets import Dataset, load_from_disk

    cache_enabled = bool(config["training"].get("cache_tokenized_datasets", True))
    fingerprint = tokenized_cache_fingerprint(split_name, config, df, tokenizer)
    cache_dir = output_dir / "tokenized" / f"{split_name}_{fingerprint['fingerprint'][:16]}"
    fingerprint_path = cache_dir / "fingerprint.json"
    if cache_enabled and cache_dir.exists() and fingerprint_path.exists():
        try:
            stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == fingerprint.get("fingerprint"):
                return load_from_disk(str(cache_dir))
        except Exception:
            pass

    rows = tokenize_training_dataframe(df, tokenizer, config)
    dataset = Dataset.from_list(rows)
    if cache_enabled:
        ensure_dir(cache_dir)
        dataset.save_to_disk(str(cache_dir))
        write_json(fingerprint_path, fingerprint)
    return dataset


def score_class_weights(train_df: Any, config: dict[str, Any]) -> list[float] | None:
    explicit = config["training"].get("score_class_weights")
    if explicit is not None:
        return [float(weight) for weight in explicit]
    if config["training"].get("class_weighting") != "inverse_frequency":
        return None
    counts = train_df["target_score"].astype(int).value_counts().to_dict()
    total = sum(int(counts.get(score, 0)) for score in range(1, 6))
    weights = []
    for score in range(1, 6):
        count = int(counts.get(score, 0))
        weights.append(float(total / (5 * count)) if count else 0.0)
    return weights


def train_judge(config: dict[str, Any], force_prepare: bool = False) -> dict[str, Any]:
    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    set_seed(seed)
    output_dir = ensure_dir(project_output_dir(config))
    model, tokenizer = load_model_for_training(config)
    from transformers import Trainer, TrainingArguments

    splits = load_or_prepare_splits(config, tokenizer, force_prepare=force_prepare)
    eval_split = select_eval_subset(splits["validation"], config)
    allowed_scores = [
        str(score) for score in config["inference"].get("allowed_scores", ["1", "2", "3", "4", "5"])
    ]
    objective = config["training"].get("objective", "causal_lm")
    score_report = validate_score_tokenization(
        tokenizer,
        allowed_scores,
        require_single_token=objective == "restricted_score_ce",
    )
    write_json(output_dir / "score_tokenization_report.json", score_report)

    train_dataset = load_or_tokenize_dataset("train", splits["train"], tokenizer, config, output_dir)
    eval_dataset = load_or_tokenize_dataset("validation_monitor", eval_split, tokenizer, config, output_dir)
    training = config["training"]

    training_args_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        "weight_decay": float(training["weight_decay"]),
        "optim": training["optim"],
        "logging_steps": int(training["logging_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "train_sampling_strategy": training.get("train_sampling_strategy", "group_by_length"),
        "length_column_name": training.get("length_column_name", "length"),
        "remove_unused_columns": False,
        "report_to": [],
        "seed": seed,
        "bf16": str(training.get("dtype", "")).lower() in {"bf16", "bfloat16"},
        "fp16": str(training.get("dtype", "")).lower() in {"fp16", "float16"},
    }
    if "warmup_steps" in training:
        training_args_kwargs["warmup_steps"] = int(training["warmup_steps"])
    elif "warmup_ratio" in training:
        training_args_kwargs["warmup_ratio"] = float(training["warmup_ratio"])

    args = TrainingArguments(**training_args_kwargs)
    trainer_class = Trainer
    score_token_ids: list[int] | None = None
    class_weights = score_class_weights(splits["train"], config)
    if objective == "restricted_score_ce":
        score_token_ids = [
            next(item for item in score_report if item["score"] == str(score))["token_ids"][0]
            for score in range(1, 6)
        ]
        trainer_class = make_restricted_score_trainer(Trainer, score_token_ids, class_weights)

    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalDataCollator(
            tokenizer,
            include_score_metadata=objective == "restricted_score_ce",
        ),
    )
    result = trainer.train()
    metrics = result.metrics
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))
    inference_config = copy_config(config)
    inference_config["model"]["adapter_path"] = str(adapter_dir)
    save_config(inference_config, output_dir / "inference_config.yaml")
    write_json(output_dir / "train_metrics.json", _jsonable(metrics))
    write_json(output_dir / "validation_trainer_metrics.json", _jsonable(eval_metrics))
    write_json(output_dir / "resolved_training_args.json", args.to_dict())
    write_json(
        output_dir / "run_metadata.json",
        {
            "git_commit": git_commit(),
            "package_versions": package_versions(),
            "training_objective": objective,
            "training_backend": getattr(model, "adele_training_backend", None),
            "score_token_ids": score_token_ids,
            "score_class_weights": class_weights,
            "validation_monitor_examples": int(len(eval_split)),
            "validation_full_examples": int(len(splits["validation"])),
        },
    )
    return {"train_metrics": metrics, "validation_trainer_metrics": eval_metrics}


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out
