from __future__ import annotations

import math

import pandas as pd
from typer.testing import CliRunner

from adele_judge.cli import app
from adele_judge.data import (
    add_sequence_lengths_and_filter,
    apply_configured_filters,
    canonicalize_columns,
    construct_targets,
    length_filter_warnings,
    validate_scores,
)
from adele_judge.config import normalize_config, validate_config
from adele_judge.formatting import (
    apply_chat_template_safe,
    build_messages,
    build_user_message,
    configure_tokenizer_thinking_mode,
    format_prompt,
)
from adele_judge.metrics import all_metrics, majority_binary_baseline
from adele_judge.inference import score_allowed_continuations_batch
from adele_judge.metrics import majority_ordinal_baseline
from adele_judge.modeling import model_from_pretrained_kwargs
from adele_judge.splits import fixed_by_model_split, lomo_split
from adele_judge.tokenization import (
    supervised_token_debug_rows,
    tokenize_supervised_example,
    validate_score_tokenization,
)
from adele_judge.train import pack_tokenized_rows, select_eval_subset
from adele_judge.train import make_score_compute_metrics, make_score_logits_preprocessor


runner = CliRunner()


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    pad_token = "<eos>"
    eos_token = "<eos>"
    chat_template = None

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    ):
        if isinstance(text, list):
            input_ids = [[ord(ch) for ch in item] for item in text]
            attention_mask = [[1] * len(ids) for ids in input_ids]
            if padding:
                max_len = max(len(ids) for ids in input_ids)
                input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
                attention_mask = [
                    mask + [0] * (max_len - len(mask))
                    for mask in attention_mask
                ]
            if return_tensors == "pt":
                import torch

                return {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                }
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != 0)

    def convert_ids_to_tokens(self, ids):
        return [self.decode([token_id]) for token_id in ids]


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        if kwargs:
            raise TypeError(f"Unsupported chat template kwargs: {kwargs}")
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\n"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text


class FakeThinkingChatTokenizer(FakeTokenizer):
    name_or_path = "Qwen/Qwen3-8B"
    chat_template = (
        "{% if enable_thinking is defined %}thinking={{ enable_thinking }}{% endif %}"
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        text = f"thinking={enable_thinking}\n"
        text += "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\n"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text


class AlwaysFiveModel:
    def __init__(self):
        import torch

        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.param

    def __call__(self, input_ids, attention_mask=None):
        import torch
        from types import SimpleNamespace

        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        logits[:, :, ord("5")] = 10.0
        return SimpleNamespace(logits=logits)


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


def test_lomo_split_reserves_held_out_for_test_only():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    splits = lomo_split(df, "m3", validation_fraction=0.5, seed=7)
    assert set(splits["test"]["model_id"]) == {"m3"}
    assert "m3" not in set(splits["train"]["model_id"])
    assert "m3" not in set(splits["validation"]["model_id"])
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == len(df)


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
    assert tokenized.target_length == 1
    supervised = [label for label in tokenized.labels if label != -100]
    assert tok.decode(supervised) == "4"
    rows = supervised_token_debug_rows(
        df.iloc[0].to_dict(),
        tok,
        "Return one score.",
        max_seq_length=10000,
    )
    assert [row["token"] for row in rows if row["supervised"]] == ["4"]


def test_qwen3_style_config_applies_non_thinking_when_supported():
    cfg = config()
    cfg["model"] = {"model_name_or_path": "Qwen/Qwen3-8B"}
    normalize_config(cfg)
    tokenizer = configure_tokenizer_thinking_mode(FakeThinkingChatTokenizer(), cfg, log=False)
    messages = build_messages(raw_df().iloc[0].to_dict(), "Return one score.")

    rendered = apply_chat_template_safe(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.calls[-1]["enable_thinking"] is False
    assert rendered.startswith("thinking=False")


def test_non_thinking_tokenizer_does_not_receive_unsupported_kwargs():
    cfg = config()
    cfg["model"] = {"model_name_or_path": "Example/Plain-Instruct"}
    normalize_config(cfg)
    tokenizer = configure_tokenizer_thinking_mode(FakeChatTokenizer(), cfg, log=False)
    messages = build_messages(raw_df().iloc[0].to_dict(), "Return one score.")

    apply_chat_template_safe(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.calls[-1]["kwargs"] == {}


def test_preprocessing_and_inference_prompt_formatting_are_consistent():
    cfg = config()
    cfg["model"] = {"model_name_or_path": "Qwen/Qwen3-8B"}
    normalize_config(cfg)
    tokenizer = configure_tokenizer_thinking_mode(FakeThinkingChatTokenizer(), cfg, log=False)
    df = construct_targets(canonicalize_columns(raw_df(), cfg))
    example = df.iloc[0].to_dict()

    prompt = format_prompt(example, tokenizer, cfg["prompt"]["system_prompt"])
    tokenized = tokenize_supervised_example(
        example,
        tokenizer,
        cfg["prompt"]["system_prompt"],
        max_seq_length=10000,
    )

    assert tokenized is not None
    assert tokenized.prompt_length == len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert tokenizer.calls[-2]["enable_thinking"] is False
    assert tokenizer.calls[-1]["enable_thinking"] is False


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
    ordinal_baseline = majority_ordinal_baseline(pred, pred)
    assert ordinal_baseline["majority_score"] in {1, 2, 3, 5}


def test_packing_preserves_supervised_labels():
    rows = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    packed = pack_tokenized_rows(rows, max_seq_length=10)
    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert [x for x in packed[0]["labels"] if x != -100] == [3, 5]


def test_eval_subset_can_be_stratified_by_model_and_score():
    rows = []
    for model_id in ["m1", "m2"]:
        for score in [1, 2, 3, 4, 5]:
            for index in range(10):
                rows.append({"model_id": model_id, "target_score": score, "row": index})
    df = pd.DataFrame(rows)
    cfg = config()
    cfg["project"] = {"seed": 42}
    normalize_config(cfg)
    cfg["training"]["eval_subset_size"] = 20
    cfg["training"]["eval_subset_strategy"] = "stratified"
    cfg["training"]["eval_subset_stratify_columns"] = ["model_id", "target_score"]
    subset = select_eval_subset(df, cfg)
    assert len(subset) == 20
    assert (subset.groupby(["model_id", "target_score"]).size() == 2).all()


def test_score_tokenization_contract_and_fast_inference():
    tokenizer = FakeTokenizer()
    report = validate_score_tokenization(tokenizer, ["1", "2", "3", "4", "5"])
    assert [item["num_tokens"] for item in report] == [1, 1, 1, 1, 1]
    scored = score_allowed_continuations_batch(
        AlwaysFiveModel(),
        tokenizer,
        ["prompt A", "prompt B"],
        ["1", "2", "3", "4", "5"],
    )
    assert [row["pred_score"] for row in scored] == [5, 5]
    assert {row["scoring_method"] for row in scored} == {"single_forward_single_token"}


def test_eval_score_metrics_use_reduced_score_logits():
    import torch
    from types import SimpleNamespace

    score_token_ids = [ord(str(score)) for score in range(1, 6)]
    logits = torch.zeros((2, 5, 128))
    labels = torch.full((2, 5), -100)
    labels[0, 3] = ord("1")
    labels[1, 4] = ord("5")
    logits[0, 2, ord("1")] = 10.0
    logits[1, 3, ord("4")] = 10.0

    preprocess = make_score_logits_preprocessor(score_token_ids)
    reduced = preprocess(logits, labels)
    assert reduced.shape == (2, 5)

    compute_metrics = make_score_compute_metrics(score_token_ids, threshold=3)
    metrics = compute_metrics(
        SimpleNamespace(
            predictions=reduced.numpy(),
            label_ids=labels.numpy(),
        )
    )
    assert math.isclose(metrics["ordinal_accuracy"], 0.5)
    assert math.isclose(metrics["within_1_accuracy"], 1.0)
    assert math.isclose(metrics["binary_accuracy"], 1.0)
    assert "expected_calibration_error_10bin" in metrics


def test_configurable_attention_implementation_for_transformers_loaders():
    cfg = config()
    cfg["project"] = {"seed": 42}
    cfg["model"] = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "trust_remote_code": True,
        "revision": None,
        "attn_implementation": "flash_attention_2",
    }
    normalize_config(cfg)
    validate_config(cfg)
    kwargs = model_from_pretrained_kwargs(cfg)
    assert kwargs["attn_implementation"] == "flash_attention_2"
    assert kwargs["trust_remote_code"] is True
    assert "revision" not in kwargs


def test_cli_exposes_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["prepare", "train", "predict", "evaluate", "debug-tokenization", "lomo"]:
        assert command in result.stdout
