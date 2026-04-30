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
    metrics: dict[str, Any] = {
        "ordinal_accuracy": float(np.mean(true == pred)),
        "ordinal_mae": float(np.mean(np.abs(true - pred))),
        "within_1_accuracy": float(np.mean(np.abs(true - pred) <= 1)),
    }
    cm = confusion_matrix_df(true.tolist(), pred.tolist(), ORDINAL_LABELS)
    f1s = []
    for label in ORDINAL_LABELS:
        tp = int(cm.loc[label, label])
        fp = int(cm[label].sum() - tp)
        fn = int(cm.loc[label].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[f"precision_score_{label}"] = precision
        metrics[f"recall_score_{label}"] = recall
        metrics[f"f1_score_{label}"] = f1
        metrics[f"support_score_{label}"] = int(cm.loc[label].sum())
        f1s.append(f1)
    metrics["ordinal_macro_f1"] = float(np.mean(f1s))
    return metrics


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
    out.update(calibration_metrics(df))
    out["num_examples"] = int(len(df))
    for column, metric_name in [
        ("pred_score", "pred_score_counts"),
        ("target_score", "target_score_counts"),
        ("pred_binary", "pred_binary_counts"),
    ]:
        if column in df.columns:
            counts = df[column].value_counts(dropna=False).sort_index()
            out[metric_name] = {str(key): int(value) for key, value in counts.items()}
    if "target_binary" in df.columns:
        counts = df["target_binary"].value_counts()
        for label in BINARY_LABELS:
            out[f"support_{label.lower()}"] = int(counts.get(label, 0))
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


def majority_ordinal_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    threshold: int = 3,
) -> dict[str, Any]:
    majority = int(train_df["target_score"].astype(int).value_counts().idxmax())
    baseline = eval_df.copy()
    baseline["pred_score"] = majority
    baseline["pred_binary"] = baseline["pred_score"].map(lambda score: binary_from_score(score, threshold))
    metrics = ordinal_metrics(baseline)
    metrics["majority_score"] = majority
    metrics["num_train_examples"] = int(len(train_df))
    metrics["num_eval_examples"] = int(len(eval_df))
    return metrics


def calibration_metrics(df: pd.DataFrame, num_bins: int = 10) -> dict[str, Any]:
    prob_cols = [f"prob_{score}" for score in ORDINAL_LABELS if f"prob_{score}" in df.columns]
    if not prob_cols:
        return {}
    probs = df[prob_cols].astype(float).to_numpy()
    confidence = probs.max(axis=1)
    correct = (df["target_score"].astype(int).to_numpy() == df["pred_score"].astype(int).to_numpy())
    metrics: dict[str, Any] = {
        "confidence_mean": float(np.mean(confidence)) if len(confidence) else None,
        "confidence_p50": float(np.quantile(confidence, 0.50)) if len(confidence) else None,
        "confidence_p90": float(np.quantile(confidence, 0.90)) if len(confidence) else None,
        "expected_calibration_error_10bin": _expected_calibration_error(
            confidence,
            correct.astype(float),
            num_bins,
        ),
    }
    if "score_margin" in df.columns:
        metrics["score_margin_mean"] = float(pd.to_numeric(df["score_margin"]).mean())
    if "score_entropy" in df.columns:
        metrics["score_entropy_mean"] = float(pd.to_numeric(df["score_entropy"]).mean())
    return metrics


def _expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    num_bins: int,
) -> float:
    if len(confidence) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for index in range(num_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == num_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask])))
    return ece


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0
