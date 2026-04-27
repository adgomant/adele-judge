from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BINARY_LABELS = ["INCORRECT", "CORRECT"]
ORDINAL_LABELS = [1, 2, 3, 4, 5]


def binary_from_score(score: int, threshold: int = 3) -> str:
    return "CORRECT" if int(score) >= threshold else "INCORRECT"


def confusion_matrix_df(y_true: list[Any], y_pred: list[Any], labels: list[Any]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for true, pred in zip(y_true, y_pred, strict=False):
        if true in matrix.index and pred in matrix.columns:
            matrix.loc[true, pred] += 1
    matrix.index.name = "target"
    matrix.columns.name = "prediction"
    return matrix


def ordinal_metrics(df: pd.DataFrame) -> dict[str, Any]:
    true = df["target_score"].astype(int).to_numpy()
    pred = df["pred_score"].astype(int).to_numpy()
    if len(true) == 0:
        return {"ordinal_accuracy": None, "ordinal_mae": None, "within_1_accuracy": None}
    return {
        "ordinal_accuracy": float(np.mean(true == pred)),
        "ordinal_mae": float(np.mean(np.abs(true - pred))),
        "within_1_accuracy": float(np.mean(np.abs(true - pred) <= 1)),
    }


def binary_metrics(df: pd.DataFrame, threshold: int = 3) -> dict[str, Any]:
    true = df.get("target_binary")
    if true is None:
        true = df["target_score"].map(lambda x: binary_from_score(x, threshold))
    pred = df.get("pred_binary")
    if pred is None:
        pred = df["pred_score"].map(lambda x: binary_from_score(x, threshold))
    true_list = true.tolist()
    pred_list = pred.tolist()
    if not true_list:
        return {}
    cm = confusion_matrix_df(true_list, pred_list, BINARY_LABELS)
    metrics: dict[str, Any] = {
        "binary_accuracy": float(np.mean(np.array(true_list) == np.array(pred_list))),
    }
    f1s = []
    for label in BINARY_LABELS:
        tp = int(cm.loc[label, label])
        fp = int(cm[label].sum() - tp)
        fn = int(cm.loc[label].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        lower = label.lower()
        metrics[f"precision_{lower}"] = precision
        metrics[f"recall_{lower}"] = recall
        metrics[f"f1_{lower}"] = f1
        f1s.append(f1)

    fp_correct = int(cm.loc["INCORRECT", "CORRECT"])
    tn_correct = int(cm.loc["INCORRECT", "INCORRECT"])
    fn_correct = int(cm.loc["CORRECT", "INCORRECT"])
    tp_correct = int(cm.loc["CORRECT", "CORRECT"])
    metrics["binary_macro_f1"] = float(np.mean(f1s))
    metrics["false_positive_rate_correct"] = _safe_div(fp_correct, fp_correct + tn_correct)
    metrics["false_negative_rate_correct"] = _safe_div(fn_correct, fn_correct + tp_correct)
    return metrics


def all_metrics(df: pd.DataFrame, threshold: int = 3) -> dict[str, Any]:
    out = {}
    out.update(ordinal_metrics(df))
    out.update(binary_metrics(df, threshold))
    out["num_examples"] = int(len(df))
    return out


def grouped_metrics(
    df: pd.DataFrame,
    group_col: str,
    threshold: int = 3,
) -> pd.DataFrame:
    rows = []
    if group_col not in df.columns:
        return pd.DataFrame()
    for value, group in df.groupby(group_col, dropna=False):
        row = {"group": group_col, "value": value}
        row.update(all_metrics(group, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def add_length_bucket(
    df: pd.DataFrame,
    buckets: list[int],
    source_col: str = "response_token_length",
) -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        return out
    labels = [f"{buckets[i]}-{buckets[i + 1]}" for i in range(len(buckets) - 1)]
    out["response_length_bucket"] = pd.cut(
        out[source_col],
        bins=buckets,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)
    return out


def majority_binary_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, threshold: int = 3) -> dict[str, Any]:
    train_labels = train_df["target_score"].map(lambda x: binary_from_score(x, threshold))
    majority = train_labels.value_counts().idxmax()
    baseline = eval_df.copy()
    baseline["pred_binary"] = majority
    baseline["pred_score"] = 3 if majority == "CORRECT" else 1
    metrics = binary_metrics(baseline, threshold)
    metrics["majority_class"] = majority
    metrics["num_train_examples"] = int(len(train_df))
    metrics["num_eval_examples"] = int(len(eval_df))
    return metrics


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0
