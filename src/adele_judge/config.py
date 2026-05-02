from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or []:
        apply_override(config, override)
    normalize_config(config)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override}")
    key, raw_value = override.split("=", 1)
    value = yaml.safe_load(raw_value)
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def column_name(config: dict[str, Any], logical_name: str) -> str | None:
    return config["data"]["columns"].get(logical_name)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data")
    if isinstance(data, dict):
        data.setdefault("preprocessing_num_workers", "auto")
        data.setdefault("tokenizers_parallelism", True)

    model = config.setdefault("model", {})
    if "model_name_or_path" not in model and "name_or_path" in model:
        model["model_name_or_path"] = model["name_or_path"]
    model.setdefault("attn_implementation", None)
    thinking_mode = model.setdefault("thinking_mode", {})
    thinking_mode.setdefault("enabled", False)
    thinking_mode.setdefault("apply_if_supported", True)

    training = config.setdefault("training", {})
    if "train_sampling_strategy" not in training and training.get("group_by_length") is True:
        training["train_sampling_strategy"] = "group_by_length"
    training.setdefault("train_sampling_strategy", "random")
    if "warmup_steps" not in training and "warmup_ratio" not in training:
        training["warmup_ratio"] = 0.0
    training.setdefault("objective", "causal_lm")
    training.setdefault("class_weighting", None)
    training.setdefault("score_class_weights", None)
    loss = training.get("loss")
    if loss is None:
        loss = {}
        training["loss"] = loss
    if isinstance(loss, dict):
        loss.setdefault("type", "ce_5way")
        loss.setdefault("lambda_binary", 0.5)
        loss.setdefault("class_weights", None)
    training.setdefault("cache_tokenized_datasets", True)
    training.setdefault("eval_subset_size", None)
    training.setdefault("eval_subset_seed", training.get("seed", config.get("project", {}).get("seed", 42)))
    training.setdefault("eval_subset_strategy", "stratified")
    training.setdefault("eval_subset_stratify_columns", ["model_id", "target_score"])
    training.setdefault("max_grad_norm", 1.0)
    training.setdefault("resume_from_checkpoint", None)

    distributed = config.setdefault("distributed", {})
    distributed.setdefault("enabled", False)
    distributed.setdefault("strategy", "ddp")
    distributed.setdefault("backend", "nccl")
    distributed.setdefault("mixed_precision", None)
    distributed.setdefault("gradient_checkpointing", False)
    distributed.setdefault("find_unused_parameters", False)
    fsdp = distributed.setdefault("fsdp", {})
    fsdp.setdefault("sharding_strategy", "full_shard")
    fsdp.setdefault("transformer_layer_cls_to_wrap", None)
    fsdp.setdefault("activation_checkpointing", True)
    fsdp.setdefault("use_orig_params", True)
    deepspeed = distributed.setdefault("deepspeed", {})
    deepspeed.setdefault("zero_stage", 2)
    deepspeed.setdefault("offload_optimizer_device", "none")
    deepspeed.setdefault("offload_param_device", "none")
    deepspeed.setdefault("stage3_gather_16bit_weights_on_model_save", True)
    deepspeed.setdefault("gradient_clipping", "auto")
    deepspeed.setdefault("config_overrides", {})

    inference = config.setdefault("inference", {})
    inference.setdefault("allowed_scores", ["1", "2", "3", "4", "5"])
    inference.setdefault("binary_threshold", 3)
    if training.get("objective") == "sequence_classification":
        inference["method"] = "sequence_classification_logits"
    elif inference.get("method") == "sequence_classification_logits":
        inference["method"] = "restricted_continuation_logprobs_fast"
    else:
        inference.setdefault("method", "restricted_continuation_logprobs_fast")
    inference.setdefault("batch_size", 1)
    inference.setdefault("require_adapter", False)
    inference.setdefault("allow_base_model", True)

    split = config.setdefault("split", {})
    split.setdefault("lomo_validation_fraction", 0.05)
    split.setdefault("lomo_validation_max_examples", None)
    split.setdefault("lomo_validation_seed", config.get("project", {}).get("seed", 42))

    hub = config.setdefault("hub", {})
    hub.setdefault("repo_id", None)
    hub.setdefault("private", False)
    hub.setdefault("commit_message", "Upload ADeLe distilled judge")
    hub.setdefault("local_checkpoint_dir", None)
    hub.setdefault("output_staging_dir", None)
    hub.setdefault("create_pr", False)
    hub.setdefault("max_shard_size", "5GB")
    configure_cpu_environment(config)
    return config


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_num_workers(value: Any, *, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"auto", "all", "-1"}:
            return available_cpu_count()
        value = int(normalized)
    workers = int(value)
    if workers < 1:
        return available_cpu_count()
    return workers


def configure_cpu_environment(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return

    parallelism = data.get("tokenizers_parallelism", True)
    if parallelism is not None:
        os.environ.setdefault(
            "TOKENIZERS_PARALLELISM",
            "true" if bool(parallelism) else "false",
        )

    workers = resolve_num_workers(data.get("preprocessing_num_workers", 1))
    if workers > 0:
        os.environ.setdefault("RAYON_NUM_THREADS", str(workers))


def validate_config(config: dict[str, Any]) -> None:
    required_top_level = ["project", "data", "model", "prompt", "split", "training", "inference"]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    if "model_name_or_path" not in config["model"]:
        raise ValueError("model.model_name_or_path is required")
    attn_implementation = config["model"].get("attn_implementation")
    supported_attention = {None, "eager", "sdpa", "flash_attention_2", "flash_attention_3"}
    if attn_implementation not in supported_attention:
        raise ValueError(
            f"model.attn_implementation must be one of {sorted(item for item in supported_attention if item)}"
        )
    thinking_mode = config["model"].get("thinking_mode", {})
    if not isinstance(thinking_mode, dict):
        raise ValueError("model.thinking_mode must be a mapping")
    if thinking_mode.get("enabled") is not None and not isinstance(
        thinking_mode.get("enabled"), bool
    ):
        raise ValueError("model.thinking_mode.enabled must be true, false, or null")
    if not isinstance(thinking_mode.get("apply_if_supported", True), bool):
        raise ValueError("model.thinking_mode.apply_if_supported must be true or false")

    objective = config["training"].get("objective", "causal_lm")
    if objective not in {"causal_lm", "restricted_score_ce", "sequence_classification"}:
        raise ValueError(
            "training.objective must be 'causal_lm', 'restricted_score_ce', "
            "or 'sequence_classification'"
        )
    if objective in {"restricted_score_ce", "sequence_classification"} and bool(
        config["training"].get("packing", False)
    ):
        raise ValueError(
            "training.packing must be false when using restricted_score_ce "
            "or sequence_classification"
        )
    loss = config["training"].get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("training.loss must be a mapping")
    loss_type = loss.get("type", "ce_5way")
    if loss_type not in {"ce_5way", "ce_5way_plus_binary"}:
        raise ValueError("training.loss.type must be 'ce_5way' or 'ce_5way_plus_binary'")
    lambda_binary = float(loss.get("lambda_binary", 0.5))
    if lambda_binary < 0.0:
        raise ValueError("training.loss.lambda_binary must be non-negative")
    loss_weights = loss.get("class_weights")
    if isinstance(loss_weights, list) and len(loss_weights) != 5:
        raise ValueError("training.loss.class_weights must contain five weights")
    if isinstance(loss_weights, str) and loss_weights not in {"balanced", "inverse_frequency"}:
        raise ValueError(
            "training.loss.class_weights must be null, a five-weight list, "
            "'balanced', or 'inverse_frequency'"
        )
    class_weighting = config["training"].get("class_weighting")
    if class_weighting not in {None, "inverse_frequency"}:
        raise ValueError("training.class_weighting must be null or 'inverse_frequency'")
    score_weights = config["training"].get("score_class_weights")
    if score_weights is not None and len(score_weights) != 5:
        raise ValueError("training.score_class_weights must contain five weights")
    eval_subset_strategy = config["training"].get("eval_subset_strategy")
    if eval_subset_strategy not in {"random", "stratified"}:
        raise ValueError("training.eval_subset_strategy must be 'random' or 'stratified'")
    stratify_columns = config["training"].get("eval_subset_stratify_columns")
    if not isinstance(stratify_columns, list) or not stratify_columns:
        raise ValueError("training.eval_subset_stratify_columns must be a non-empty list")
    train_sampling_strategy = config["training"].get("train_sampling_strategy")
    if train_sampling_strategy not in {"random", "sequential", "group_by_length"}:
        raise ValueError(
            "training.train_sampling_strategy must be 'random', 'sequential', or 'group_by_length'"
        )
    if "warmup_steps" in config["training"] and int(config["training"]["warmup_steps"]) < 0:
        raise ValueError("training.warmup_steps must be >= 0")
    if "warmup_ratio" in config["training"]:
        warmup_ratio = float(config["training"]["warmup_ratio"])
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError("training.warmup_ratio must be between 0 and 1")
    preprocessing_workers = config.get("data", {}).get("preprocessing_num_workers", 1)
    try:
        resolve_num_workers(preprocessing_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data.preprocessing_num_workers must be a positive integer, 0, -1, 'auto', or 'all'"
        ) from exc
    validate_distributed_config(config)

    method = config["inference"].get("method", "restricted_continuation_logprobs_fast")
    supported_methods = {
        "restricted_continuation_logprobs",
        "restricted_continuation_logprobs_fast",
        "sequence_classification_logits",
    }
    if method not in supported_methods:
        raise ValueError(f"inference.method must be one of {sorted(supported_methods)}")

    batch_size = int(config["inference"].get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("inference.batch_size must be >= 1")

    allowed_scores = [str(score) for score in config["inference"].get("allowed_scores", [])]
    if set(allowed_scores) != {str(score) for score in range(1, 6)}:
        raise ValueError("inference.allowed_scores must contain exactly '1', '2', '3', '4', and '5'")

    validation_fraction = float(config["split"].get("lomo_validation_fraction", 0.05))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("split.lomo_validation_fraction must be between 0 and 1")


def validate_distributed_config(config: dict[str, Any]) -> None:
    distributed = config.get("distributed", {})
    if not isinstance(distributed, dict):
        raise ValueError("distributed must be a mapping")

    strategy = distributed.get("strategy", "ddp")
    if strategy not in {"ddp", "fsdp", "deepspeed"}:
        raise ValueError("distributed.strategy must be 'ddp', 'fsdp', or 'deepspeed'")

    backend = distributed.get("backend", "nccl")
    if backend not in {"nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"}:
        raise ValueError("distributed.backend must be a supported torch distributed backend")

    mixed_precision = _normalized_precision(distributed.get("mixed_precision"))
    training_precision = _normalized_precision(config["training"].get("dtype"))
    if mixed_precision not in {None, "no", "bf16", "fp16"}:
        raise ValueError("distributed.mixed_precision must be null, 'no', 'bf16', or 'fp16'")
    if mixed_precision not in {None, "no"} and training_precision != mixed_precision:
        raise ValueError("distributed.mixed_precision must match training.dtype")

    fsdp = distributed.get("fsdp", {})
    if not isinstance(fsdp, dict):
        raise ValueError("distributed.fsdp must be a mapping")
    if fsdp.get("sharding_strategy", "full_shard") not in {
        "full_shard",
        "shard_grad_op",
        "hybrid_shard",
        "hybrid_shard_zero2",
        "no_shard",
    }:
        raise ValueError("distributed.fsdp.sharding_strategy is not supported")
    transformer_layer = fsdp.get("transformer_layer_cls_to_wrap")
    if transformer_layer is not None and not isinstance(transformer_layer, (str, list)):
        raise ValueError("distributed.fsdp.transformer_layer_cls_to_wrap must be null, a string, or a list")
    if isinstance(transformer_layer, list) and not all(isinstance(item, str) for item in transformer_layer):
        raise ValueError("distributed.fsdp.transformer_layer_cls_to_wrap list entries must be strings")

    deepspeed = distributed.get("deepspeed", {})
    if not isinstance(deepspeed, dict):
        raise ValueError("distributed.deepspeed must be a mapping")
    zero_stage = int(deepspeed.get("zero_stage", 2))
    if zero_stage not in {1, 2, 3}:
        raise ValueError("distributed.deepspeed.zero_stage must be 1, 2, or 3")
    for key in ["offload_optimizer_device", "offload_param_device"]:
        if deepspeed.get(key, "none") not in {"none", "cpu", "nvme"}:
            raise ValueError(f"distributed.deepspeed.{key} must be 'none', 'cpu', or 'nvme'")
    if not isinstance(deepspeed.get("config_overrides", {}), dict):
        raise ValueError("distributed.deepspeed.config_overrides must be a mapping")

    if not bool(distributed.get("enabled", False)):
        return

    if (
        strategy == "fsdp"
        and bool(distributed.get("gradient_checkpointing", False))
        and bool(fsdp.get("activation_checkpointing", True))
    ):
        raise ValueError(
            "Use either distributed.gradient_checkpointing or "
            "distributed.fsdp.activation_checkpointing, not both."
        )
    if strategy in {"fsdp", "deepspeed"} and bool(config["training"].get("load_in_4bit", True)):
        raise ValueError(
            "QLoRA/4-bit training is only supported for single-GPU or DDP in this project. "
            "Set training.load_in_4bit=false for FSDP or DeepSpeed."
        )


def _normalized_precision(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16"}:
        return "fp16"
    if normalized in {"fp32", "float32", "no", "none"}:
        return "no"
    return normalized
