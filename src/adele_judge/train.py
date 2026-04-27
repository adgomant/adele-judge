from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .modeling import load_model_for_training
from .pipeline import load_or_prepare_splits
from .tokenization import tokenize_supervised_example
from .utils import ensure_dir, project_output_dir, set_seed, write_json


@dataclass
class CausalDataCollator:
    tokenizer: Any

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
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_training_dataframe(df: Any, tokenizer: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    system_prompt = config["prompt"]["system_prompt"]
    max_seq_length = int(config["training"]["max_seq_length"])
    overflow = config["data"]["filters"].get("on_sequence_overflow", "skip")
    rows = []
    skipped = 0
    for _, row in df.iterrows():
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


def train_judge(config: dict[str, Any], force_prepare: bool = False) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    set_seed(seed)
    output_dir = ensure_dir(project_output_dir(config))
    model, tokenizer = load_model_for_training(config)
    splits = load_or_prepare_splits(config, tokenizer, force_prepare=force_prepare)

    train_rows = tokenize_training_dataframe(splits["train"], tokenizer, config)
    eval_rows = tokenize_training_dataframe(splits["validation"], tokenizer, config)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    training = config["training"]

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        weight_decay=float(training["weight_decay"]),
        optim=training["optim"],
        logging_steps=int(training["logging_steps"]),
        eval_steps=int(training["eval_steps"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        eval_strategy="steps",
        save_strategy="steps",
        group_by_length=bool(training.get("group_by_length", True)),
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
        bf16=str(training.get("dtype", "")).lower() in {"bf16", "bfloat16"},
        fp16=str(training.get("dtype", "")).lower() in {"fp16", "float16"},
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalDataCollator(tokenizer),
    )
    result = trainer.train()
    metrics = result.metrics
    eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))
    write_json(output_dir / "train_metrics.json", _jsonable(metrics))
    write_json(output_dir / "validation_trainer_metrics.json", _jsonable(eval_metrics))
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
