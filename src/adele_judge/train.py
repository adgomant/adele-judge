from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import copy_config, save_config
from .formatting import tokenizer_thinking_template_kwargs
from .metrics import all_metrics, binary_from_score
from .modeling import load_model_for_training
from .pipeline import load_or_prepare_splits
from .tokenization import (
    preprocessing_num_workers,
    tokenized_training_rows,
    validate_score_tokenization,
)
from .utils import (
    ensure_dir,
    git_commit,
    is_main_process,
    package_versions,
    project_output_dir,
    set_seed,
    setup_distributed_device,
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


@dataclass
class SequenceClassificationDataCollator:
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
            labels.append(int(feature["labels"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_training_dataframe(df: Any, tokenizer: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    system_prompt = config["prompt"]["system_prompt"]
    max_seq_length = int(config["training"]["max_seq_length"])
    overflow = config["data"]["filters"].get("on_sequence_overflow", "skip")
    objective = config["training"].get("objective", "causal_lm")
    tokenized_rows = tokenized_training_rows(
        df.to_dict("records"),
        tokenizer,
        system_prompt=system_prompt,
        max_seq_length=max_seq_length,
        overflow=overflow,
        objective=objective,
        num_workers=preprocessing_num_workers(config),
    )
    rows = [row for row in tokenized_rows if row is not None]
    skipped = len(tokenized_rows) - len(rows)
    if skipped and is_main_process():
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


def make_sequence_classification_trainer(
    base_trainer_class: Any,
    loss_config: dict[str, Any],
    class_weights: list[float] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    loss_type = loss_config.get("type", "ce_5way")
    lambda_binary = float(loss_config.get("lambda_binary", 0.5))

    class _SequenceClassificationTrainer(base_trainer_class):
        _last_component_log_step = -1

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **_: Any,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels = labels.to(logits.device)
            weight = None
            if class_weights is not None:
                weight = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)
            loss_5way = F.cross_entropy(logits, labels, weight=weight)
            loss = loss_5way
            loss_binary = None
            if loss_type == "ce_5way_plus_binary":
                logit_incorrect = torch.logsumexp(logits[:, :2], dim=-1)
                logit_correct = torch.logsumexp(logits[:, 2:], dim=-1)
                binary_logits = torch.stack([logit_incorrect, logit_correct], dim=-1)
                binary_labels = (labels >= 2).to(dtype=torch.long)
                loss_binary = F.cross_entropy(binary_logits, binary_labels)
                loss = loss_5way + lambda_binary * loss_binary

            self._maybe_log_loss_components(model, loss, loss_5way, loss_binary)
            return (loss, outputs) if return_outputs else loss

        def _maybe_log_loss_components(
            self,
            model: Any,
            loss: Any,
            loss_5way: Any,
            loss_binary: Any | None,
        ) -> None:
            is_world_process_zero = getattr(self, "is_world_process_zero", None)
            if callable(is_world_process_zero) and not is_world_process_zero():
                return
            if not getattr(model, "training", False):
                return
            step = int(getattr(self.state, "global_step", 0))
            logging_steps = max(1, int(getattr(self.args, "logging_steps", 1)))
            if step == self._last_component_log_step or step % logging_steps != 0:
                return
            payload = {
                "loss_5way": float(loss_5way.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
            }
            if loss_binary is not None:
                payload["loss_binary"] = float(loss_binary.detach().cpu())
            self.log(payload)
            self._last_component_log_step = step

    return _SequenceClassificationTrainer


def make_score_logits_preprocessor(score_token_ids: list[int]) -> Any:
    def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
        import torch

        if isinstance(logits, tuple):
            logits = logits[0]
        score_ids = torch.tensor(score_token_ids, dtype=torch.long, device=logits.device)
        if logits.ndim == 3:
            supervised = labels != -100
            target_positions = supervised.to(dtype=torch.long).argmax(dim=1)
            logit_positions = torch.clamp(target_positions - 1, min=0)
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            return logits[batch_indices, logit_positions][:, score_ids]
        if logits.ndim == 2 and logits.shape[-1] != len(score_token_ids):
            return logits[:, score_ids]
        return logits

    return preprocess_logits_for_metrics


def make_score_compute_metrics(score_token_ids: list[int], threshold: int = 3) -> Any:
    token_id_to_score = {int(token_id): score for score, token_id in enumerate(score_token_ids, start=1)}

    def compute_metrics(eval_prediction: Any) -> dict[str, Any]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]

        score_logits = np.asarray(predictions)
        label_ids = np.asarray(labels)
        if len(score_logits) == 0:
            return {}

        if label_ids.ndim == 2:
            supervised = label_ids != -100
            valid = supervised.any(axis=1)
            target_positions = supervised.argmax(axis=1)
            target_token_ids = label_ids[np.arange(len(label_ids)), target_positions]
        else:
            valid = np.ones(len(label_ids), dtype=bool)
            target_token_ids = label_ids

        target_scores = np.array(
            [_score_from_metric_label(token_id, token_id_to_score) for token_id in target_token_ids],
            dtype=float,
        )
        valid = valid & ~np.isnan(target_scores)
        if not np.any(valid):
            return {}

        return metrics_from_score_logits(
            score_logits[valid],
            target_scores[valid].astype(int),
            threshold,
        )

    return compute_metrics


def make_classification_compute_metrics(threshold: int = 3) -> Any:
    def compute_metrics(eval_prediction: Any) -> dict[str, Any]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        score_logits = np.asarray(predictions)
        label_ids = np.asarray(labels)
        if len(score_logits) == 0:
            return {}
        valid = (label_ids >= 0) & (label_ids < 5)
        if not np.any(valid):
            return {}
        return metrics_from_score_logits(
            score_logits[valid],
            label_ids[valid].astype(int) + 1,
            threshold,
        )

    return compute_metrics


def metrics_from_score_logits(
    score_logits: np.ndarray,
    target_scores: np.ndarray,
    threshold: int,
) -> dict[str, Any]:
    pred_scores = np.argmax(score_logits, axis=-1).astype(int) + 1
    probs = _softmax_np(score_logits)
    sorted_logits = np.sort(score_logits, axis=1)
    score_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    score_entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    metrics_df = pd.DataFrame(
        {
            "target_score": target_scores.astype(int),
            "pred_score": pred_scores.astype(int),
            "target_binary": [binary_from_score(score, threshold) for score in target_scores],
            "pred_binary": [binary_from_score(score, threshold) for score in pred_scores],
            "score_margin": score_margin,
            "score_entropy": score_entropy,
        }
    )
    for index, score in enumerate(range(1, 6)):
        metrics_df[f"prob_{score}"] = probs[:, index]
    return all_metrics(metrics_df, threshold)


def _score_from_metric_label(label: Any, token_id_to_score: dict[int, int]) -> float:
    value = int(label)
    if value in token_id_to_score:
        return float(token_id_to_score[value])
    if 1 <= value <= 5:
        return float(value)
    if 0 <= value < 5:
        return float(value + 1)
    return float("nan")


def _softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def select_eval_subset(df: Any, config: dict[str, Any]) -> Any:
    subset_size = config["training"].get("eval_subset_size")
    if subset_size is None:
        return df
    subset_size = int(subset_size)
    if subset_size <= 0 or len(df) <= subset_size:
        return df
    seed = int(config["training"].get("eval_subset_seed", config["training"].get("seed", 42)))
    if config["training"].get("eval_subset_strategy", "stratified") == "stratified":
        return stratified_eval_subset(
            df,
            subset_size,
            config["training"].get("eval_subset_stratify_columns", ["model_id", "target_score"]),
            seed,
        )
    return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)


def stratified_eval_subset(
    df: Any,
    subset_size: int,
    columns: list[str],
    seed: int,
) -> Any:
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

    grouped = df.groupby(available_columns, dropna=False, sort=True)
    group_sizes = grouped.size()
    ideal = group_sizes / len(df) * subset_size
    allocations = ideal.apply(np.floor).astype(int)
    non_empty = group_sizes > 0
    allocations[non_empty & (allocations == 0)] = 1
    allocations = allocations.clip(upper=group_sizes)

    while int(allocations.sum()) > subset_size:
        removable = allocations[allocations > 1]
        if removable.empty:
            break
        key = (removable - ideal.loc[removable.index]).idxmax()
        allocations.loc[key] -= 1

    while int(allocations.sum()) < subset_size:
        remaining = group_sizes - allocations
        expandable = remaining[remaining > 0]
        if expandable.empty:
            break
        key = (ideal.loc[expandable.index] - allocations.loc[expandable.index]).idxmax()
        allocations.loc[key] += 1

    chunks = []
    for key, group in grouped:
        n = int(allocations.loc[key])
        if n > 0:
            chunks.append(group.sample(n=n, random_state=seed))
    if not chunks:
        return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
    return pd.concat(chunks).sample(frac=1.0, random_state=seed).reset_index(drop=True)


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
    loss_weights = (config["training"].get("loss") or {}).get("class_weights")
    if isinstance(loss_weights, list):
        return [float(weight) for weight in loss_weights]
    if isinstance(loss_weights, str) and loss_weights in {"balanced", "inverse_frequency"}:
        return inverse_frequency_score_weights(train_df)
    explicit = config["training"].get("score_class_weights")
    if explicit is not None:
        return [float(weight) for weight in explicit]
    if config["training"].get("class_weighting") != "inverse_frequency":
        return None
    return inverse_frequency_score_weights(train_df)


def inverse_frequency_score_weights(train_df: Any) -> list[float]:
    counts = train_df["target_score"].astype(int).value_counts().to_dict()
    total = sum(int(counts.get(score, 0)) for score in range(1, 6))
    weights = []
    for score in range(1, 6):
        count = int(counts.get(score, 0))
        weights.append(float(total / (5 * count)) if count else 0.0)
    return weights


FINALIST_SOURCE_SPLITS = ("train", "validation", "test")


def source_split_counts(splits: dict[str, Any]) -> dict[str, int]:
    return {name: int(len(splits[name])) for name in FINALIST_SOURCE_SPLITS if name in splits}


def finalist_training_dataframe(splits: dict[str, Any]) -> Any:
    chunks = [
        splits[name]
        for name in FINALIST_SOURCE_SPLITS
        if name in splits and not splits[name].empty
    ]
    if not chunks:
        raise ValueError("Finalist mode found no examples across train, validation, and test splits")
    return pd.concat(chunks, ignore_index=True)


def distributed_training_args_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    distributed = config.get("distributed", {})
    if not bool(distributed.get("enabled", False)):
        return {}

    strategy = str(distributed.get("strategy", "ddp"))
    kwargs: dict[str, Any] = {}
    if bool(distributed.get("gradient_checkpointing", False)):
        kwargs["gradient_checkpointing"] = True

    if strategy == "ddp":
        kwargs["ddp_backend"] = distributed.get("backend", "nccl")
        kwargs["ddp_find_unused_parameters"] = bool(
            distributed.get("find_unused_parameters", False)
        )
    elif strategy == "fsdp":
        fsdp = distributed.get("fsdp", {})
        fsdp_options = [str(fsdp.get("sharding_strategy", "full_shard"))]
        transformer_layer = fsdp.get("transformer_layer_cls_to_wrap")
        if transformer_layer:
            fsdp_options.append("auto_wrap")
        fsdp_config: dict[str, Any] = {
            "activation_checkpointing": bool(fsdp.get("activation_checkpointing", True)),
            "use_orig_params": bool(fsdp.get("use_orig_params", True)),
        }
        if transformer_layer:
            fsdp_config["transformer_layer_cls_to_wrap"] = transformer_layer
        kwargs["fsdp"] = " ".join(fsdp_options)
        kwargs["fsdp_config"] = fsdp_config
    elif strategy == "deepspeed":
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "DeepSpeed training was requested, but the deepspeed package is not installed. "
                "Install with the deepspeed extra or use distributed.strategy='ddp'/'fsdp'."
            )
        kwargs["deepspeed"] = deepspeed_config_dict(config)
    return kwargs


def deepspeed_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    deepspeed = config.get("distributed", {}).get("deepspeed", {})
    zero_stage = int(deepspeed.get("zero_stage", 2))
    zero_optimization: dict[str, Any] = {"stage": zero_stage}

    optimizer_device = deepspeed.get("offload_optimizer_device", "none")
    if optimizer_device != "none":
        zero_optimization["offload_optimizer"] = {"device": optimizer_device}

    param_device = deepspeed.get("offload_param_device", "none")
    if param_device != "none":
        zero_optimization["offload_param"] = {"device": param_device}

    if zero_stage == 3:
        zero_optimization["stage3_gather_16bit_weights_on_model_save"] = bool(
            deepspeed.get("stage3_gather_16bit_weights_on_model_save", True)
        )

    ds_config: dict[str, Any] = {
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
        "zero_optimization": zero_optimization,
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_clipping": deepspeed.get("gradient_clipping", "auto"),
    }
    return _recursive_merge(ds_config, deepspeed.get("config_overrides", {}))


def _recursive_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def training_args_kwargs(
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    *,
    evaluation_enabled: bool,
) -> dict[str, Any]:
    training = config["training"]
    kwargs = {
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
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "eval_strategy": "steps" if evaluation_enabled else "no",
        "save_strategy": "steps",
        "train_sampling_strategy": training.get("train_sampling_strategy", "random"),
        "length_column_name": training.get("length_column_name", "length"),
        "remove_unused_columns": False,
        "report_to": [],
        "seed": seed,
        "bf16": str(training.get("dtype", "")).lower() in {"bf16", "bfloat16"},
        "fp16": str(training.get("dtype", "")).lower() in {"fp16", "float16"},
    }
    if "max_grad_norm" in training:
        kwargs["max_grad_norm"] = float(training["max_grad_norm"])
    if evaluation_enabled:
        kwargs["eval_steps"] = int(training["eval_steps"])
    if "warmup_steps" in training:
        kwargs["warmup_steps"] = int(training["warmup_steps"])
    elif "warmup_ratio" in training:
        kwargs["warmup_ratio"] = float(training["warmup_ratio"])
    if bool(config.get("distributed", {}).get("enabled", False)):
        kwargs["disable_tqdm"] = not is_main_process()
        kwargs["log_on_each_node"] = False
    kwargs.update(distributed_training_args_kwargs(config))
    return kwargs


def effective_global_batch_size(config: dict[str, Any], world_size: int) -> int:
    training = config["training"]
    return (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * int(world_size)
    )


def distributed_runtime_metadata(config: dict[str, Any], args: Any) -> dict[str, Any]:
    world_size = int(getattr(args, "world_size", 1))
    distributed = config.get("distributed", {})
    training = config["training"]
    return {
        "enabled": bool(distributed.get("enabled", False)),
        "strategy": distributed.get("strategy", "ddp"),
        "backend": distributed.get("backend", "nccl"),
        "mixed_precision": distributed.get("mixed_precision"),
        "gradient_checkpointing": bool(distributed.get("gradient_checkpointing", False)),
        "find_unused_parameters": distributed.get("find_unused_parameters"),
        "fsdp": distributed.get("fsdp"),
        "deepspeed": distributed.get("deepspeed"),
        "world_size": world_size,
        "launcher": detected_launcher(),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "effective_global_batch_size": effective_global_batch_size(config, world_size),
    }


def detected_launcher() -> str:
    if os.environ.get("ACCELERATE_USE_FSDP") or os.environ.get("ACCELERATE_USE_DEEPSPEED"):
        return "accelerate"
    if os.environ.get("ACCELERATE_MIXED_PRECISION"):
        return "accelerate"
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        return "torchrun"
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if os.environ.get("LOCAL_RANK") is not None:
        return "distributed"
    return "single_process"


def train_judge(
    config: dict[str, Any],
    force_prepare: bool = False,
    *,
    finalist: bool = False,
) -> dict[str, Any]:
    setup_distributed_device()
    seed = int(config["training"].get("seed", config["project"].get("seed", 42)))
    set_seed(seed)
    output_dir = ensure_dir(project_output_dir(config))
    from transformers import Trainer, TrainingArguments

    evaluation_enabled = not finalist
    args = TrainingArguments(
        **training_args_kwargs(config, output_dir, seed, evaluation_enabled=evaluation_enabled)
    )
    model, tokenizer = load_model_for_training(config)
    batch_metadata = distributed_runtime_metadata(config, args)
    if args.should_save:
        print(
            "Effective global batch size: "
            f"{batch_metadata['effective_global_batch_size']} "
            f"({batch_metadata['per_device_train_batch_size']} per device "
            f"x {batch_metadata['gradient_accumulation_steps']} grad accum "
            f"x {batch_metadata['world_size']} world size)"
        )

    with args.main_process_first(local=False, desc="prepare dataset"):
        splits = load_or_prepare_splits(
            config,
            tokenizer,
            force_prepare=force_prepare and args.should_save,
        )
    split_counts = source_split_counts(splits)
    training_df = finalist_training_dataframe(splits) if finalist else splits["train"]
    eval_split = None if finalist else select_eval_subset(splits["validation"], config)
    allowed_scores = [
        str(score) for score in config["inference"].get("allowed_scores", ["1", "2", "3", "4", "5"])
    ]
    objective = config["training"].get("objective", "causal_lm")
    score_report = validate_score_tokenization(
        tokenizer,
        allowed_scores,
        require_single_token=objective == "restricted_score_ce",
    )
    if args.should_save:
        write_json(output_dir / "score_tokenization_report.json", score_report)

    train_split_name = "train_finalist" if finalist else "train"
    with args.main_process_first(local=False, desc="tokenize train dataset"):
        train_dataset = load_or_tokenize_dataset(
            train_split_name,
            training_df,
            tokenizer,
            config,
            output_dir,
        )
    eval_dataset = None
    if evaluation_enabled and eval_split is not None:
        with args.main_process_first(local=False, desc="tokenize validation monitor"):
            eval_dataset = load_or_tokenize_dataset(
                "validation_monitor",
                eval_split,
                tokenizer,
                config,
                output_dir,
            )
    trainer_class = Trainer
    score_token_ids: list[int] | None = None
    class_weights = score_class_weights(training_df, config)
    compute_metrics = None
    preprocess_logits_for_metrics = None
    metric_score_token_ids = [
        item["token_ids"][0]
        for item in sorted(score_report, key=lambda row: int(row["score"]))
        if item["num_tokens"] == 1
    ]
    if objective == "sequence_classification":
        compute_metrics = make_classification_compute_metrics(
            threshold=int(config["inference"].get("binary_threshold", 3)),
        )
    elif len(metric_score_token_ids) == 5:
        compute_metrics = make_score_compute_metrics(
            metric_score_token_ids,
            threshold=int(config["inference"].get("binary_threshold", 3)),
        )
        preprocess_logits_for_metrics = make_score_logits_preprocessor(metric_score_token_ids)
    if objective == "restricted_score_ce":
        score_token_ids = [
            next(item for item in score_report if item["score"] == str(score))["token_ids"][0]
            for score in range(1, 6)
        ]
        trainer_class = make_restricted_score_trainer(Trainer, score_token_ids, class_weights)
    elif objective == "sequence_classification":
        trainer_class = make_sequence_classification_trainer(
            Trainer,
            config["training"].get("loss") or {},
            class_weights,
        )

    if objective == "sequence_classification":
        data_collator = SequenceClassificationDataCollator(tokenizer)
    else:
        data_collator = CausalDataCollator(
            tokenizer,
            include_score_metadata=objective == "restricted_score_ce",
        )

    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    resume_from_checkpoint = config["training"].get("resume_from_checkpoint")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    metrics = result.metrics
    eval_metrics = trainer.evaluate() if evaluation_enabled else {}

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    trainer.save_model(str(adapter_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(tokenizer_dir))
        inference_config = copy_config(config)
        inference_config["model"]["adapter_path"] = str(adapter_dir)
        save_config(inference_config, output_dir / "inference_config.yaml")
        write_json(output_dir / "train_metrics.json", _jsonable(metrics))
        if evaluation_enabled:
            write_json(output_dir / "validation_trainer_metrics.json", _jsonable(eval_metrics))
        write_json(output_dir / "trainer_log_history.json", trainer.state.log_history)
        write_json(output_dir / "resolved_training_args.json", args.to_dict())
        write_json(
            output_dir / "run_metadata.json",
            {
                "git_commit": git_commit(),
                "package_versions": package_versions(),
                "training_objective": objective,
                "training_loss": config["training"].get("loss"),
                "training_backend": getattr(model, "adele_training_backend", None),
                "training_mode": "finalist" if finalist else "standard",
                "score_token_ids": score_token_ids,
                "score_class_weights": class_weights,
                "source_split_counts": split_counts,
                "training_examples": int(len(training_df)),
                "evaluation_enabled": evaluation_enabled,
                "validation_monitor_examples": int(len(eval_split)) if eval_split is not None else 0,
                "validation_full_examples": int(len(splits["validation"])),
                "distributed": batch_metadata,
                "world_size": batch_metadata["world_size"],
                "effective_global_batch_size": batch_metadata["effective_global_batch_size"],
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
