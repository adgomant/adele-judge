from __future__ import annotations

import copy
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
    if "warmup_steps" not in training and "warmup_ratio" not in training:
        training["warmup_steps"] = 0
    training.setdefault("objective", "causal_lm")
    training.setdefault("class_weighting", None)
    training.setdefault("score_class_weights", None)
    training.setdefault("cache_tokenized_datasets", True)
    training.setdefault("eval_subset_size", None)
    training.setdefault("eval_subset_seed", training.get("seed", config.get("project", {}).get("seed", 42)))

    inference = config.setdefault("inference", {})
    inference.setdefault("allowed_scores", ["1", "2", "3", "4", "5"])
    inference.setdefault("binary_threshold", 3)
    inference.setdefault("method", "restricted_continuation_logprobs_fast")
    inference.setdefault("batch_size", 1)
    inference.setdefault("require_adapter", False)
    inference.setdefault("allow_base_model", True)

    split = config.setdefault("split", {})
    split.setdefault("lomo_validation_fraction", 0.05)
    split.setdefault("lomo_validation_max_examples", None)
    split.setdefault("lomo_validation_seed", config.get("project", {}).get("seed", 42))
    return config


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
    if objective not in {"causal_lm", "restricted_score_ce"}:
        raise ValueError("training.objective must be 'causal_lm' or 'restricted_score_ce'")
    if objective == "restricted_score_ce" and bool(config["training"].get("packing", False)):
        raise ValueError("training.packing must be false when using restricted_score_ce")
    class_weighting = config["training"].get("class_weighting")
    if class_weighting not in {None, "inverse_frequency"}:
        raise ValueError("training.class_weighting must be null or 'inverse_frequency'")
    score_weights = config["training"].get("score_class_weights")
    if score_weights is not None and len(score_weights) != 5:
        raise ValueError("training.score_class_weights must contain five weights")

    method = config["inference"].get("method", "restricted_continuation_logprobs_fast")
    supported_methods = {
        "restricted_continuation_logprobs",
        "restricted_continuation_logprobs_fast",
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
