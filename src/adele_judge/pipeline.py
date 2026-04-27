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
from .modeling import load_tokenizer
from .splits import create_splits, split_report
from .utils import ensure_dir, prepared_dir, project_output_dir, write_json


def prepare_dataset(config: dict[str, Any], tokenizer: Any | None = None) -> dict[str, pd.DataFrame]:
    output_dir = ensure_dir(project_output_dir(config))
    save_config(config, output_dir / "config.yaml")
    tokenizer = tokenizer or load_tokenizer(config)

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
    if not force_prepare and all(path.exists() for path in expected):
        return {name: pd.read_parquet(pdir / f"{name}.parquet") for name in ["train", "validation", "test"]}
    return prepare_dataset(config, tokenizer)


def resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
