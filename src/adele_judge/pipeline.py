from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import save_config
from .data import (
    add_response_token_lengths,
    add_sequence_lengths_and_filter,
    apply_configured_filters,
    length_statistics,
    load_and_construct_targets,
)
from .formatting import configure_tokenizer_thinking_mode, tokenizer_thinking_template_kwargs
from .modeling import load_tokenizer
from .splits import create_splits, split_report
from .tokenization import validate_score_tokenization
from .utils import (
    ensure_dir,
    file_sha256,
    is_main_process,
    package_versions,
    prepared_dir,
    project_output_dir,
    read_json,
    stable_json_hash,
    write_json,
)


def prepare_dataset(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, pd.DataFrame]:
    output_dir = ensure_dir(project_output_dir(config))
    save_config(config, output_dir / "config.yaml")
    tokenizer = tokenizer or load_tokenizer(config)
    configure_tokenizer_thinking_mode(tokenizer, config)
    score_report = validate_score_tokenization(
        tokenizer,
        [str(score) for score in config["inference"].get("allowed_scores", ["1", "2", "3", "4", "5"])],
        require_single_token=config["training"].get("objective") == "restricted_score_ce",
    )
    write_json(output_dir / "score_tokenization_report.json", score_report)
    fingerprint = preparation_fingerprint(config, tokenizer)

    df = load_and_construct_targets(config)
    df = add_response_token_lengths(
        df,
        tokenizer,
        batch_size=int(config["data"].get("token_length_batch_size", 512)),
    )
    filtered, filter_report = apply_configured_filters(df, config)
    filtered, sequence_report = add_sequence_lengths_and_filter(filtered, tokenizer, config)
    splits = create_splits(filtered, config)

    pdir = ensure_dir(prepared_dir(config))
    for name, split in splits.items():
        split.to_parquet(pdir / f"{name}.parquet", index=False)

    write_json(output_dir / "dataset_filtering_report.json", {**filter_report, **sequence_report})
    write_json(output_dir / "length_statistics.json", length_statistics(filtered))
    write_json(output_dir / "split_report.json", split_report(splits))
    write_json(pdir / "prepared_fingerprint.json", fingerprint)
    write_json(output_dir / "prepared_fingerprint.json", fingerprint)
    return splits


def load_prepared_split(config: dict[str, Any], split_name: str) -> pd.DataFrame:
    path = prepared_dir(config) / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared split not found: {path}. Run scripts/prepare_dataset.py first."
        )
    return pd.read_parquet(path)


def load_or_prepare_splits(
    config: dict[str, Any],
    tokenizer: Any | None = None,
    force_prepare: bool = False,
) -> dict[str, pd.DataFrame]:
    pdir = prepared_dir(config)
    expected = [pdir / "train.parquet", pdir / "validation.parquet", pdir / "test.parquet"]
    tokenizer = tokenizer or load_tokenizer(config)
    configure_tokenizer_thinking_mode(tokenizer, config)
    if not force_prepare and all(path.exists() for path in expected):
        current = preparation_fingerprint(config, tokenizer, include_environment=False)
        stored_path = pdir / "prepared_fingerprint.json"
        if stored_path.exists():
            stored = read_json(stored_path)
            if stored.get("fingerprint") == current.get("fingerprint"):
                return {
                    name: pd.read_parquet(pdir / f"{name}.parquet")
                    for name in ["train", "validation", "test"]
                }
        elif is_main_process():
            print(f"Prepared split fingerprint missing at {stored_path}; rebuilding splits.")
    return prepare_dataset(config, tokenizer)


def preparation_fingerprint(
    config: dict[str, Any],
    tokenizer: Any,
    *,
    include_environment: bool = True,
) -> dict[str, Any]:
    data_path = Path(config["data"]["path"]).expanduser().resolve()
    chat_template = getattr(tokenizer, "chat_template", None)
    components = {
        "schema_version": 1,
        "model_name_or_path": config["model"]["model_name_or_path"],
        "model_revision": config["model"].get("revision"),
        "attn_implementation": config["model"].get("attn_implementation"),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "chat_template_hash": stable_json_hash(chat_template) if chat_template else None,
        "thinking_mode": config["model"].get("thinking_mode"),
        "thinking_mode_support_detected": getattr(
            tokenizer, "_adele_thinking_support_detected", None
        ),
        "chat_template_kwargs": tokenizer_thinking_template_kwargs(tokenizer),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "columns": config["data"]["columns"],
        "filters": config["data"]["filters"],
        "split": config["split"],
        "prompt": config["prompt"],
        "training_objective": config["training"].get("objective", "causal_lm"),
        "training_max_seq_length": config["training"]["max_seq_length"],
        "inference_allowed_scores": config["inference"].get("allowed_scores"),
        "binary_threshold": config["inference"].get("binary_threshold"),
    }
    fingerprint = {
        "fingerprint": stable_json_hash(components),
        "components": components,
    }
    if include_environment:
        fingerprint["package_versions"] = package_versions()
    return fingerprint


def resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
