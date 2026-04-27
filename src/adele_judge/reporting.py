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


def save_prediction_reports(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    split_name: str,
    threshold: int = 3,
    length_buckets: list[int] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    predictions.to_parquet(output_dir / f"predictions_{split_name}.parquet", index=False)
    metrics = all_metrics(predictions, threshold)
    write_json(output_dir / f"{split_name}_metrics.json", metrics)

    ordinal_cm = confusion_matrix_df(
        predictions["target_score"].astype(int).tolist(),
        predictions["pred_score"].astype(int).tolist(),
        ORDINAL_LABELS,
    )
    ordinal_cm.to_csv(output_dir / f"confusion_matrix_ordinal_{split_name}.csv")
    ordinal_cm.to_csv(output_dir / "confusion_matrix_ordinal.csv")

    true_binary = predictions["target_binary"].tolist()
    pred_binary = predictions["pred_binary"].tolist()
    binary_cm = confusion_matrix_df(true_binary, pred_binary, BINARY_LABELS)
    binary_cm.to_csv(output_dir / f"confusion_matrix_binary_{split_name}.csv")
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
            table.to_csv(output_dir / f"{split_name}_{filename}", index=False)
            table.to_csv(output_dir / filename, index=False)
    return metrics
