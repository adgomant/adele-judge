from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import (
    BINARY_LABELS,
    ORDINAL_LABELS,
    add_length_bucket,
    all_metrics,
    confusion_matrix_df,
    grouped_metrics,
)
from .utils import ensure_dir, write_json


def prediction_dir(output_dir: str | Path, split_name: str) -> Path:
    return Path(output_dir) / "predictions" / split_name


def prediction_path(output_dir: str | Path, split_name: str) -> Path:
    return prediction_dir(output_dir, split_name) / "predictions.parquet"


def legacy_prediction_path(output_dir: str | Path, split_name: str) -> Path:
    return Path(output_dir) / f"predictions_{split_name}.parquet"


def resolve_prediction_path(
    output_dir: str | Path,
    split_name: str,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    scoped_path = prediction_path(output_dir, split_name)
    if scoped_path.exists():
        return scoped_path
    legacy_path = legacy_prediction_path(output_dir, split_name)
    if legacy_path.exists():
        return legacy_path
    return scoped_path


def evaluation_dir(output_dir: str | Path, split_name: str) -> Path:
    return Path(output_dir) / "evaluation" / split_name


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    split_name: str,
) -> Path:
    path = prediction_path(output_dir, split_name)
    ensure_dir(path.parent)
    predictions.to_parquet(path, index=False)
    return path


def save_evaluation_reports(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    split_name: str,
    threshold: int = 3,
    length_buckets: list[int] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(evaluation_dir(output_dir, split_name))
    metrics = all_metrics(predictions, threshold)
    write_json(output_dir / "metrics.json", metrics)

    ordinal_cm = confusion_matrix_df(
        predictions["target_score"].astype(int).tolist(),
        predictions["pred_score"].astype(int).tolist(),
        ORDINAL_LABELS,
    )
    ordinal_cm.to_csv(output_dir / "confusion_matrix_ordinal.csv")

    true_binary = predictions["target_binary"].tolist()
    pred_binary = predictions["pred_binary"].tolist()
    binary_cm = confusion_matrix_df(true_binary, pred_binary, BINARY_LABELS)
    binary_cm.to_csv(output_dir / "confusion_matrix_binary.csv")

    enriched = predictions
    if length_buckets:
        enriched = add_length_bucket(predictions, length_buckets)

    for group_col, filename in [
        ("model_id", "per_model_metrics.csv"),
        ("benchmark", "per_benchmark_metrics.csv"),
        ("task", "per_task_metrics.csv"),
        ("target_score", "per_target_score_metrics.csv"),
        ("response_length_bucket", "per_response_length_bucket_metrics.csv"),
    ]:
        table = grouped_metrics(enriched, group_col, threshold)
        if not table.empty:
            table.to_csv(output_dir / filename, index=False)
    return metrics
