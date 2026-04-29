from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
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
from adele_judge.hub import (
    add_custom_pipeline_metadata,
    collect_hub_metadata,
    render_model_card,
    resolve_checkpoint_paths,
    resolve_hub_options,
    write_generation_config,
)
from adele_judge.hub_pipeline import ADeLeJudgePipeline
from adele_judge.metrics import all_metrics, majority_binary_baseline
from adele_judge.inference import predict_dataframe, score_allowed_continuations_batch
from adele_judge.metrics import majority_ordinal_baseline
from adele_judge.modeling import model_from_pretrained_kwargs
from adele_judge.splits import fixed_by_model_split, lomo_split
from adele_judge.tokenization import (
    supervised_token_debug_rows,
    tokenize_supervised_example,
    validate_score_tokenization,
)
from adele_judge.train import (
    finalist_training_dataframe,
    make_score_compute_metrics,
    make_score_logits_preprocessor,
    pack_tokenized_rows,
    select_eval_subset,
    training_args_kwargs,
)
from adele_judge.utils import read_json, tee_output, write_json


runner = CliRunner()


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    pad_token = "<eos>"
    eos_token = "<eos>"
    chat_template = None
    padding_side = "right"

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
        input_ids = [ord(ch) for ch in text]
        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

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
        self.calls = 0
        self.config = SimpleNamespace(
            task_specific_params=None,
            _name_or_path=None,
            _commit_hash=None,
        )
        self.name_or_path = ""
        self.device = self.param.device

    def parameters(self):
        yield self.param

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def can_generate(self):
        return False

    def __call__(self, input_ids, attention_mask=None):
        import torch

        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        logits[:, :, ord("5")] = 10.0
        return SimpleNamespace(logits=logits)


class BatchIndexScoreModel:
    def __init__(self):
        import torch

        self.param = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.config = SimpleNamespace(
            task_specific_params=None,
            _name_or_path=None,
            _commit_hash=None,
        )
        self.name_or_path = ""
        self.device = self.param.device

    def parameters(self):
        yield self.param

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def can_generate(self):
        return False

    def __call__(self, input_ids, attention_mask=None):
        import torch

        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        for row_index in range(input_ids.shape[0]):
            score = min(row_index + 1, 5)
            logits[row_index, :, ord(str(score))] = 10.0
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


def hub_config(tmp_path):
    cfg = config()
    cfg["project"] = {
        "run_name": "test-run",
        "output_dir": str(tmp_path / "run"),
        "seed": 42,
    }
    cfg["model"] = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "revision": None,
        "attn_implementation": None,
        "trust_remote_code": True,
        "thinking_mode": {"enabled": False, "apply_if_supported": True},
    }
    cfg["training"].update(
        {
            "objective": "restricted_score_ce",
            "num_train_epochs": 1,
            "max_seq_length": 2048,
        }
    )
    cfg["inference"].update(
        {
            "allowed_scores": ["1", "2", "3", "4", "5"],
            "binary_threshold": 3,
            "method": "restricted_continuation_logprobs_fast",
        }
    )
    cfg["hub"] = {
        "repo_id": "user/test-run",
        "private": False,
        "commit_message": "Upload test run",
        "local_checkpoint_dir": str(tmp_path / "run"),
        "output_staging_dir": str(tmp_path / "staging"),
        "create_pr": False,
        "max_shard_size": "1GB",
    }
    normalize_config(cfg)
    return cfg


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


def test_fixed_split_allows_empty_test_models():
    cfg = config()
    cfg["split"]["test_models"] = []
    df = construct_targets(canonicalize_columns(raw_df(), cfg))
    splits = fixed_by_model_split(df, cfg)
    assert splits["test"].empty
    assert set(splits["train"]["model_id"]) == {"m1", "m3"}
    assert splits["validation"]["model_id"].tolist() == ["m2"]


def test_finalist_training_dataframe_includes_all_configured_splits():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    splits = fixed_by_model_split(df, config())
    combined = finalist_training_dataframe(splits)
    assert combined["model_id"].tolist() == ["m1", "m2", "m3"]


def test_finalist_training_dataframe_tolerates_empty_validation_and_test():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    empty = df.iloc[0:0]
    combined = finalist_training_dataframe(
        {"train": df.iloc[[0]], "validation": empty, "test": empty}
    )
    assert combined["model_id"].tolist() == ["m1"]


def test_finalist_training_dataframe_rejects_empty_training_pool():
    df = construct_targets(canonicalize_columns(raw_df(), config()))
    empty = df.iloc[0:0]
    with pytest.raises(ValueError, match="Finalist mode found no examples"):
        finalist_training_dataframe({"train": empty, "validation": empty, "test": empty})


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
    assert "### QUESTION" in message
    assert "### REFERENCE ANSWER" in message


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


def test_training_args_disable_eval_for_finalist_mode(tmp_path):
    cfg = config()
    cfg["training"].update(
        {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 3,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.0,
            "optim": "adamw_torch",
            "logging_steps": 5,
            "eval_steps": 6,
            "save_steps": 7,
            "save_total_limit": 8,
            "dtype": "float32",
            "warmup_steps": 0,
        }
    )
    standard = training_args_kwargs(cfg, tmp_path, 42, evaluation_enabled=True)
    finalist = training_args_kwargs(cfg, tmp_path, 42, evaluation_enabled=False)

    assert standard["eval_strategy"] == "steps"
    assert standard["eval_steps"] == 6
    assert finalist["eval_strategy"] == "no"
    assert "eval_steps" not in finalist


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


def test_empty_prediction_dataframe_keeps_report_columns():
    cfg = config()
    cfg["inference"]["allowed_scores"] = ["1", "2", "3", "4", "5"]
    predictions = predict_dataframe(
        pd.DataFrame(columns=["target_score", "target_binary"]),
        AlwaysFiveModel(),
        FakeTokenizer(),
        cfg,
    )
    assert predictions.empty
    assert {"target_score", "pred_score", "prob_5", "logprob_5"} <= set(predictions.columns)


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


def test_hub_options_resolve_from_config_and_cli_overrides(tmp_path):
    cfg = hub_config(tmp_path)
    options = resolve_hub_options(
        cfg,
        repo_id="org/override",
        private=True,
        commit_message="Custom upload",
        staging_dir=tmp_path / "custom-staging",
        create_pr=True,
        no_push=True,
    )
    assert options.repo_id == "org/override"
    assert options.private is True
    assert options.commit_message == "Custom upload"
    assert options.staging_dir == tmp_path / "custom-staging"
    assert options.create_pr is True
    assert options.no_push is True


def test_hub_options_require_repo_id(tmp_path):
    cfg = hub_config(tmp_path)
    cfg["hub"]["repo_id"] = None
    with pytest.raises(ValueError, match="repo_id"):
        resolve_hub_options(cfg)


def test_hub_checkpoint_requires_adapter(tmp_path):
    cfg = hub_config(tmp_path)
    options = resolve_hub_options(cfg)
    with pytest.raises(FileNotFoundError, match="Run directory"):
        resolve_checkpoint_paths(options)

    options.run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter"):
        resolve_checkpoint_paths(options)

    (options.run_dir / "adapter").mkdir()
    paths = resolve_checkpoint_paths(options)
    assert paths.adapter_dir == options.run_dir / "adapter"
    assert paths.tokenizer_dir is None


def test_hub_generation_config_uses_safe_debug_defaults(tmp_path):
    path = tmp_path / "generation_config.json"
    write_generation_config(path)
    data = read_json(path)
    assert data["max_new_tokens"] == 1
    assert data["do_sample"] is False


def test_hub_custom_pipeline_metadata_is_written(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"model_type": "qwen3"})
    add_custom_pipeline_metadata(path)
    data = read_json(path)
    assert data["custom_pipelines"]["adele-judge"]["impl"] == (
        "adele_judge_pipeline.ADeLeJudgePipeline"
    )
    assert data["custom_pipelines"]["adele-judge"]["pt"] == ["AutoModelForCausalLM"]


def test_hub_metadata_collects_available_run_artifacts(tmp_path):
    cfg = hub_config(tmp_path)
    options = resolve_hub_options(cfg)
    options.run_dir.mkdir()
    write_json(options.run_dir / "train_metrics.json", {"train_loss": 0.12})
    write_json(options.run_dir / "split_report.json", {"train": {"examples": 10}})
    metadata = collect_hub_metadata(cfg, options.run_dir, options)
    assert metadata["repo_id"] == "user/test-run"
    assert metadata["base_model"] == "Qwen/Qwen3-8B"
    assert metadata["artifacts"]["train_metrics.json"]["train_loss"] == 0.12
    assert metadata["artifacts"]["split_report.json"]["train"]["examples"] == 10


def test_hub_model_card_documents_restricted_scoring(tmp_path):
    cfg = hub_config(tmp_path)
    options = resolve_hub_options(cfg)
    metadata = collect_hub_metadata(cfg, tmp_path / "missing-run", options)
    card = render_model_card(cfg, metadata, options.repo_id)
    assert "ADeLe Distilled Judge" in card
    assert "restricted continuations" in card
    assert "generate()" in card
    assert "ADeLe-specific judge" in card


def adele_pipeline(model=None, tokenizer=None, *, adele_config=None):
    return ADeLeJudgePipeline(
        model=model or AlwaysFiveModel(),
        tokenizer=tokenizer or FakeTokenizer(),
        device=-1,
        adele_config=adele_config
        or {
            "prompt": {"system_prompt": "Return one score."},
            "inference": {"allowed_scores": ["1", "2", "3", "4", "5"], "binary_threshold": 3},
            "model": {"thinking_mode": {}},
        },
    )


def test_transformers_pipeline_loads_custom_pipeline_locally():
    from transformers import PretrainedConfig
    from transformers import pipeline

    config = PretrainedConfig()
    config.custom_pipelines = {
        "adele-judge": {
            "impl": "adele_judge_pipeline.ADeLeJudgePipeline",
            "pt": ["AutoModelForCausalLM"],
            "tf": [],
            "type": "text",
        }
    }
    pipe = pipeline(
        "adele-judge",
        model=AlwaysFiveModel(),
        config=config,
        tokenizer=FakeTokenizer(),
        pipeline_class=ADeLeJudgePipeline,
        device=-1,
        adele_config={
            "prompt": {"system_prompt": "Return one score."},
            "inference": {"allowed_scores": ["1", "2", "3", "4", "5"], "binary_threshold": 3},
            "model": {"thinking_mode": {}},
        },
    )
    assert isinstance(pipe, ADeLeJudgePipeline)


def test_hub_pipeline_scores_single_dict_with_restricted_continuations():
    model = AlwaysFiveModel()
    pipe = adele_pipeline(model=model)
    result = pipe({"question": "q", "reference_answer": "a", "model_response": "r"})
    assert result["score"] == 5
    assert result["label"] == "CORRECT"
    assert set(result) == {
        "score",
        "label",
        "probs",
        "logprobs",
        "confidence",
        "margin",
        "entropy",
    }
    assert set(result["probs"]) == {"1", "2", "3", "4", "5"}
    assert set(result["logprobs"]) == {"1", "2", "3", "4", "5"}
    assert result["confidence"] == max(result["probs"].values())
    assert model.calls == 1


def test_hub_pipeline_prompt_ignores_benchmark_and_task_fields():
    tokenizer = FakeChatTokenizer()
    pipe = adele_pipeline(tokenizer=tokenizer)
    pipe(
        {
            "question": "q",
            "reference_answer": "a",
            "model_response": "r",
            "benchmark": "must not appear",
            "task": "must not appear either",
        }
    )
    rendered_messages = "\n".join(
        message["content"] for message in tokenizer.calls[-1]["messages"]
    )
    assert "must not appear" not in rendered_messages
    assert "must not appear either" not in rendered_messages


def test_hub_pipeline_scores_list_with_standard_batching():
    model = BatchIndexScoreModel()
    pipe = adele_pipeline(model=model)
    results = pipe(
        [
            {"question": "q1", "reference_answer": "a1", "model_response": "r1"},
            {"question": "q2", "ground_truth": "a2", "model_response": "r2"},
            {"question": "q3", "reference_answer": "a3", "model_response": "r3"},
        ],
        batch_size=3,
    )
    assert [result["score"] for result in results] == [1, 2, 3]
    assert [result["label"] for result in results] == ["INCORRECT", "INCORRECT", "CORRECT"]
    assert model.calls == 1


def test_hub_pipeline_validates_required_fields():
    pipe = adele_pipeline()
    with pytest.raises(ValueError, match="question"):
        pipe({"reference_answer": "a", "model_response": "r"})
    with pytest.raises(ValueError, match="model_response"):
        pipe({"question": "q", "reference_answer": "a"})
    with pytest.raises(ValueError, match="reference_answer or ground_truth"):
        pipe({"question": "q", "model_response": "r"})


def test_hub_pipeline_requires_single_token_scores():
    class MultiTokenScoreTokenizer(FakeTokenizer):
        def __call__(self, text, *args, **kwargs):
            if text in {"1", "2", "3", "4", "5"}:
                return {"input_ids": [ord(text), ord(text)], "attention_mask": [1, 1]}
            return super().__call__(text, *args, **kwargs)

    with pytest.raises(ValueError, match="single tokens"):
        adele_pipeline(tokenizer=MultiTokenScoreTokenizer())


def test_hub_pipeline_applies_thinking_only_when_supported():
    supported_tokenizer = FakeThinkingChatTokenizer()
    supported_pipe = adele_pipeline(
        tokenizer=supported_tokenizer,
        adele_config={
            "prompt": {"system_prompt": "Return one score."},
            "inference": {"allowed_scores": ["1", "2", "3", "4", "5"], "binary_threshold": 3},
            "model": {"thinking_mode": {"enabled": False, "apply_if_supported": True}},
        },
    )
    supported_pipe({"question": "q", "reference_answer": "a", "model_response": "r"})
    assert supported_tokenizer.calls[-1]["enable_thinking"] is False

    unsupported_tokenizer = FakeChatTokenizer()
    unsupported_pipe = adele_pipeline(
        tokenizer=unsupported_tokenizer,
        adele_config={
            "prompt": {"system_prompt": "Return one score."},
            "inference": {"allowed_scores": ["1", "2", "3", "4", "5"], "binary_threshold": 3},
            "model": {"thinking_mode": {"enabled": False, "apply_if_supported": True}},
        },
    )
    unsupported_pipe({"question": "q", "reference_answer": "a", "model_response": "r"})
    assert unsupported_tokenizer.calls[-1]["kwargs"] == {}


def test_hub_pipeline_does_not_generate_by_default():
    class GenerateFailsModel(AlwaysFiveModel):
        def generate(self, *args, **kwargs):
            raise AssertionError("generate should not be called")

    pipe = adele_pipeline(model=GenerateFailsModel())
    result = pipe({"question": "q", "reference_answer": "a", "model_response": "r"})
    assert result["score"] == 5


def test_tee_output_mirrors_stdout_and_stderr_and_restores_streams(tmp_path, capsys):
    log_path = tmp_path / "training.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with tee_output(log_path):
        assert sys.stdout is not original_stdout
        assert sys.stderr is not original_stderr
        print("stdout line")
        print("stderr line", file=sys.stderr)

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    captured = capsys.readouterr()
    assert "stdout line" in captured.out
    assert "stderr line" in captured.err
    log_text = log_path.read_text(encoding="utf-8")
    assert "stdout line" in log_text
    assert "stderr line" in log_text


def test_cli_exposes_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "prepare",
        "train",
        "predict",
        "evaluate",
        "push-to-hub",
        "debug-tokenization",
        "lomo",
    ]:
        assert command in result.stdout


def test_train_cli_exposes_finalist_flag():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--finalist" in result.stdout


def test_push_to_hub_cli_exposes_packaging_flags():
    result = runner.invoke(app, ["push-to-hub", "--help"])
    assert result.exit_code == 0
    for flag in ["--repo-id", "--private", "--commit-message", "--staging-dir", "--no-push"]:
        assert flag in result.stdout
