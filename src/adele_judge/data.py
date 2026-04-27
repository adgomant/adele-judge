from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import track

from .config import column_name
from .formatting import clean_value
from .tokenization import batch_response_token_lengths, tokenize_supervised_example


REQUIRED_LOGICAL_COLUMNS = [
    "question",
    "reference_answer",
    "response",
    "judge_1_score",
    "judge_2_score",
    "model_id",
]


def load_source_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def validate_configured_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    missing = []
    for logical in REQUIRED_LOGICAL_COLUMNS:
        actual = column_name(config, logical)
        if not actual or actual not in df.columns:
            missing.append((logical, actual))
    if missing:
        raise ValueError(f"Missing required configured columns: {missing}")


def canonicalize_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validate_configured_columns(df, config)
    out = pd.DataFrame(index=df.index)
    for logical, actual in config["data"]["columns"].items():
        if actual and actual in df.columns:
            out[logical] = df[actual]
    for optional in ["benchmark", "task", "example_id", "source"]:
        if optional not in out.columns:
            out[optional] = None
    return out


def validate_scores(df: pd.DataFrame) -> None:
    for col in ["judge_1_score", "judge_2_score"]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = numeric.isna() | (numeric < 1) | (numeric > 5)
        if invalid.any():
            raise ValueError(f"{col} has {int(invalid.sum())} values outside 1..5 or missing")


def construct_targets(df: pd.DataFrame, binary_threshold: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["judge_1_score"] = pd.to_numeric(out["judge_1_score"], errors="raise")
    out["judge_2_score"] = pd.to_numeric(out["judge_2_score"], errors="raise")
    out["disagreement"] = (out["judge_1_score"] - out["judge_2_score"]).abs()
    out["mean_score"] = (out["judge_1_score"] + out["judge_2_score"]) / 2.0
    out["target_score"] = np.floor(out["mean_score"]).astype(int)
    invalid = ~out["target_score"].isin([1, 2, 3, 4, 5])
    if invalid.any():
        raise ValueError(f"target_score has {int(invalid.sum())} values outside 1..5")
    out["target_binary"] = np.where(out["target_score"] >= binary_threshold, "CORRECT", "INCORRECT")
    return out


def load_and_construct_targets(config: dict[str, Any]) -> pd.DataFrame:
    raw = load_source_dataframe(config)
    canonical = canonicalize_columns(raw, config)
    validate_scores(canonical)
    return construct_targets(
        canonical,
        binary_threshold=int(config["inference"].get("binary_threshold", 3)),
    )


def add_response_token_lengths(
    df: pd.DataFrame,
    tokenizer: Any,
    batch_size: int = 512,
) -> pd.DataFrame:
    out = df.copy()
    texts = [clean_value(x, fallback="") for x in out["response"].tolist()]
    out["response_token_length"] = batch_response_token_lengths(texts, tokenizer, batch_size)
    return out


def apply_configured_filters(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filters = config["data"]["filters"]
    total = len(df)
    max_disagreement = filters.get("max_disagreement")
    after_disagreement = df
    if max_disagreement is not None:
        after_disagreement = after_disagreement[
            after_disagreement["disagreement"] <= float(max_disagreement)
        ].copy()

    max_response_tokens = filters.get("max_response_tokens")
    after_length = after_disagreement
    if max_response_tokens is not None:
        after_length = after_length[
            after_length["response_token_length"] <= int(max_response_tokens)
        ].copy()

    report = {
        "raw_examples": total,
        "after_disagreement_filter": len(after_disagreement),
        "after_response_length_filter": len(after_length),
        "removed_by_disagreement": total - len(after_disagreement),
        "removed_by_response_length": len(after_disagreement) - len(after_length),
        "max_disagreement": max_disagreement,
        "max_response_tokens": max_response_tokens,
    }
    return after_length.reset_index(drop=True), report


def add_sequence_lengths_and_filter(
    df: pd.DataFrame,
    tokenizer: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_seq_length = int(config["training"]["max_seq_length"])
    max_response_tokens = config["data"]["filters"].get("max_response_tokens")
    overflow_mode = config["data"]["filters"].get("on_sequence_overflow", "skip")
    system_prompt = config["prompt"]["system_prompt"]
    prompt_lengths: list[int | float] = []
    target_lengths: list[int | float] = []
    sequence_lengths: list[int | float] = []
    keep: list[bool] = []
    for _, row in track(
        df.iterrows(),
        total=len(df),
        description="Checking sequence lengths",
    ):
        example = row.to_dict()
        tokenized = tokenize_supervised_example(
            example,
            tokenizer,
            system_prompt,
            max_seq_length,
            overflow_mode,
        )
        if tokenized is None:
            prompt_lengths.append(np.nan)
            target_lengths.append(np.nan)
            sequence_lengths.append(np.nan)
            keep.append(False)
        else:
            prompt_lengths.append(tokenized.prompt_length)
            target_lengths.append(tokenized.target_length)
            sequence_lengths.append(tokenized.sequence_length)
            keep.append(True)

    out = df.copy()
    out["prompt_token_length"] = prompt_lengths
    out["target_token_length"] = target_lengths
    out["sequence_length"] = sequence_lengths
    kept = out.loc[keep].reset_index(drop=True)
    overflowed = out.loc[[not item for item in keep]].copy()
    effective_prompt_budget = (
        max_seq_length - int(max_response_tokens) if max_response_tokens is not None else None
    )
    report = {
        "max_seq_length": max_seq_length,
        "max_response_tokens": max_response_tokens,
        "effective_prompt_budget_tokens": effective_prompt_budget,
        "on_sequence_overflow": overflow_mode,
        "examples_before_sequence_filter": len(df),
        "sequence_overflow_count": int(len(df) - len(kept)),
        "examples_after_sequence_filter": len(kept),
        "sequence_overflow_reason": (
            "Full chat-formatted sequence exceeded max_seq_length after response filtering. "
            "This includes system prompt, benchmark/task metadata, question, reference answer, "
            "model response, chat-template tokens, and target score tokens."
        ),
        "length_filter_warnings": length_filter_warnings(max_seq_length, max_response_tokens),
    }
    if len(overflowed) and "response_token_length" in overflowed.columns:
        report["overflowed_response_token_length"] = _series_stats(
            overflowed["response_token_length"]
        )
    if len(kept):
        report["kept_sequence_length"] = _series_stats(kept["sequence_length"])
        report["kept_prompt_token_length"] = _series_stats(kept["prompt_token_length"])
    return kept, report


def length_statistics(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {"num_examples": int(len(df))}
    for col in [
        "response_token_length",
        "prompt_token_length",
        "target_token_length",
        "sequence_length",
    ]:
        if col in df.columns and len(df):
            stats[col] = _series_stats(df[col])
    return stats


def length_filter_warnings(
    max_seq_length: int,
    max_response_tokens: int | None,
    *,
    small_prompt_budget_threshold: int = 1024,
) -> list[str]:
    if max_response_tokens is None:
        return []
    warnings = []
    if max_response_tokens >= max_seq_length:
        warnings.append(
            "max_response_tokens is greater than or equal to max_seq_length; examples can pass "
            "the response cap while having no room for the formatted prompt."
        )
    effective_prompt_budget = max_seq_length - max_response_tokens
    if effective_prompt_budget < small_prompt_budget_threshold:
        warnings.append(
            f"Effective prompt budget is only {effective_prompt_budget} tokens after reserving "
            "the response cap; consider increasing training.max_seq_length."
        )
    return warnings


def _series_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "min": int(numeric.min()) if len(numeric) else None,
        "max": int(numeric.max()) if len(numeric) else None,
        "mean": float(numeric.mean()) if len(numeric) else None,
        "p50": float(numeric.quantile(0.50)) if len(numeric) else None,
        "p90": float(numeric.quantile(0.90)) if len(numeric) else None,
        "p95": float(numeric.quantile(0.95)) if len(numeric) else None,
        "p99": float(numeric.quantile(0.99)) if len(numeric) else None,
    }
