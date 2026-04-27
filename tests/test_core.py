from __future__ import annotations

import math

import pandas as pd

from adele_judge.data import (
    add_sequence_lengths_and_filter,
    apply_configured_filters,
    canonicalize_columns,
    construct_targets,
    length_filter_warnings,
    validate_scores,
)
from adele_judge.formatting import build_user_message
from adele_judge.metrics import all_metrics, majority_binary_baseline
from adele_judge.splits import fixed_by_model_split
from adele_judge.tokenization import supervised_token_debug_rows, tokenize_supervised_example
from adele_judge.train import pack_tokenized_rows


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    pad_token = "<eos>"
    eos_token = "<eos>"
    chat_template = None

    def __call__(self, text, add_special_tokens=False, truncation=False, padding=False):
        if isinstance(text, list):
            return {"input_ids": [[ord(ch) for ch in item] for item in text]}
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != 0)


def config():
    return {
        "data": {
            "columns": {
                "question": "question",
                "reference_answer": "ground_truth",
                "response": "response",
                "judge_1_score": "score_gpt4o",
                "judge_2_score": "score_sonnet",
                "model_id": "model_id",
                "benchmark": "benchmark",
                "task": "task",
                "example_id": "instance_id",
            },
            "filters": {
                "max_disagreement": 1,
                "max_response_tokens": 4096,
                "on_sequence_overflow": "skip",
            },
        },
        "split": {
            "mode": "fixed_by_model",
            "validation_models": ["m2"],
            "test_models": ["m3"],
            "train_models": "auto_except_val_test",
        },
        "inference": {"binary_threshold": 3},
        "prompt": {"system_prompt": "Return one score."},
        "training": {"max_seq_length": 8192},
    }


def raw_df():
    return pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "ground_truth": ["a1", "a2", "a3"],
            "response": ["r1", "r2", "r3"],
            "score_gpt4o": [5, 2, 3],
            "score_sonnet": [4, 2, 4],
            "model_id": ["m1", "m2", "m3"],
            "benchmark": ["b", "b", "c"],
            "task": ["t1", "t2", "t3"],
            "instance_id": ["i1", "i2", "i3"],
        }
    )


def test_target_construction_floor_mean_and_binary():
    df = canonicalize_columns(raw_df(), config())
    validate_scores(df)
    out = construct_targets(df)
    assert out["target_score"].tolist() == [4, 2, 3]
    assert out["target_binary"].tolist() == ["CORRECT", "INCORRECT", "CORRECT"]
    assert out["disagreement"].tolist() == [1, 0, 1]


def test_fixed_split_auto_train_models_has_no_leakage():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    splits = fixed_by_model_split(df, config())
    assert splits["train"]["model_id"].tolist() == ["m1"]
    assert splits["validation"]["model_id"].tolist() == ["m2"]
    assert splits["test"]["model_id"].tolist() == ["m3"]


def test_formatting_omits_hidden_training_metadata():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    message = build_user_message(df.iloc[0].to_dict())
    assert "score_gpt4o" not in message
    assert "model_id" not in message
    assert "target_score" not in message
    assert "Question:" in message
    assert "Reference answer:" in message


def test_supervised_labels_only_cover_score_digit():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    tok = FakeTokenizer()
    tokenized = tokenize_supervised_example(
        df.iloc[0].to_dict(),
        tok,
        "Return one score.",
        max_seq_length=10000,
    )
    assert tokenized is not None
    supervised = [label for label in tokenized.labels if label != -100]
    assert tok.decode(supervised) == "4"
    rows = supervised_token_debug_rows(
        df.iloc[0].to_dict(),
        tok,
        "Return one score.",
        max_seq_length=10000,
    )
    assert [row["token"] for row in rows if row["supervised"]] == ["4"]


def test_sequence_overflow_skip():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    tokenized = tokenize_supervised_example(
        df.iloc[0].to_dict(),
        FakeTokenizer(),
        "Return one score.",
        max_seq_length=2,
        on_sequence_overflow="skip",
    )
    assert tokenized is None


def test_response_filter_is_independent_from_sequence_filter():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    df["response_token_length"] = [4096, 4097, 10]
    filtered, report = apply_configured_filters(df, config())
    assert filtered["response_token_length"].tolist() == [4096, 10]
    assert report["removed_by_response_length"] == 1


def test_sequence_filter_adds_prompt_target_and_budget_report():
    cfg = config()
    cfg["training"]["max_seq_length"] = 70
    cfg["data"]["filters"]["max_response_tokens"] = 40
    df = construct_targets(canonicalize_columns(raw_df(), cfg)).head(1)
    df["response_token_length"] = [2]
    kept, report = add_sequence_lengths_and_filter(df, FakeTokenizer(), cfg)
    assert list(kept.columns).count("prompt_token_length") == 1
    assert list(kept.columns).count("target_token_length") == 1
    assert list(kept.columns).count("sequence_length") == 1
    assert report["effective_prompt_budget_tokens"] == 30
    assert report["sequence_overflow_count"] in {0, 1}


def test_sequence_overflow_uses_full_formatted_prompt_length():
    cfg = config()
    cfg["training"]["max_seq_length"] = 30
    cfg["data"]["filters"]["max_response_tokens"] = 4096
    df = construct_targets(canonicalize_columns(raw_df(), cfg)).head(1)
    df["response_token_length"] = [2]
    kept, report = add_sequence_lengths_and_filter(df, FakeTokenizer(), cfg)
    assert kept.empty
    assert report["sequence_overflow_count"] == 1
    assert report["effective_prompt_budget_tokens"] == -4066


def test_length_filter_warnings_for_tight_prompt_budget():
    warnings = length_filter_warnings(max_seq_length=4096, max_response_tokens=4096)
    assert len(warnings) == 2


def test_metrics_and_majority_baseline():
    pred = pd.DataFrame(
        {
            "target_score": [1, 2, 3, 5],
            "pred_score": [1, 3, 3, 4],
            "target_binary": ["INCORRECT", "INCORRECT", "CORRECT", "CORRECT"],
            "pred_binary": ["INCORRECT", "CORRECT", "CORRECT", "CORRECT"],
        }
    )
    metrics = all_metrics(pred)
    assert math.isclose(metrics["ordinal_accuracy"], 0.5)
    assert math.isclose(metrics["within_1_accuracy"], 1.0)
    assert math.isclose(metrics["binary_accuracy"], 0.75)
    baseline = majority_binary_baseline(pred, pred)
    assert baseline["majority_class"] in {"CORRECT", "INCORRECT"}


def test_packing_preserves_supervised_labels():
    rows = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    packed = pack_tokenized_rows(rows, max_seq_length=10)
    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert [x for x in packed[0]["labels"] if x != -100] == [3, 5]
