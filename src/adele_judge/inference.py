from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import track

from .formatting import configure_tokenizer_thinking_mode, format_prompt
from .metrics import binary_from_score
from .modeling import load_model_for_inference
from .tokenization import encode_text, validate_score_tokenization


def continuation_logprob(model: Any, tokenizer: Any, prompt: str, continuation: str) -> float:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False, truncation=False)["input_ids"]
    if not continuation_ids:
        return float("-inf")
    input_ids = torch.tensor([prompt_ids + continuation_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    total = 0.0
    start = len(prompt_ids)
    for offset, token_id in enumerate(continuation_ids):
        position = start + offset - 1
        if position < 0:
            return float("-inf")
        log_probs = F.log_softmax(logits[0, position], dim=-1)
        total += float(log_probs[token_id].detach().cpu())
    return total


def _model_input_device(model: Any) -> Any:
    return next(model.parameters()).device


def _normalize_logprobs(
    allowed_scores: list[str],
    logprobs: dict[str, float],
    *,
    prompt_token_length: int | None = None,
    scoring_method: str,
    candidate_token_ids: dict[str, list[int]],
) -> dict[str, Any]:
    values = np.array([logprobs[score] for score in allowed_scores], dtype=np.float64)
    shifted = values - np.max(values)
    probs_arr = np.exp(shifted) / np.exp(shifted).sum()
    probs = {score: float(prob) for score, prob in zip(allowed_scores, probs_arr, strict=True)}
    pred_score = max(allowed_scores, key=lambda score: logprobs[score])
    sorted_values = np.sort(values)
    score_margin = float(sorted_values[-1] - sorted_values[-2]) if len(sorted_values) > 1 else 0.0
    entropy = float(-np.sum(probs_arr * np.log(np.clip(probs_arr, 1e-12, 1.0))))
    return {
        "pred_score": int(pred_score),
        "logprobs": logprobs,
        "probs": probs,
        "score_margin": score_margin,
        "score_entropy": entropy,
        "prompt_token_length_inference": prompt_token_length,
        "scoring_method": scoring_method,
        "candidate_token_ids": candidate_token_ids,
    }


def _empty_prediction_frame(
    allowed_scores: list[str],
    *,
    score_logit_prefix: str = "logprob",
) -> pd.DataFrame:
    columns = [
        "pred_score",
        "pred_binary",
        "target_score",
        "target_binary",
        "model_id",
        "benchmark",
        "task",
        "example_id",
        "response_token_length",
        "sequence_length",
        "prompt_token_length_inference",
        "score_margin",
        "score_entropy",
        "scoring_method",
        "candidate_token_ids",
    ]
    for score in allowed_scores:
        columns.extend([f"{score_logit_prefix}_{score}", f"prob_{score}"])
    return pd.DataFrame(columns=columns)


def score_allowed_continuations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    allowed_scores: list[str],
) -> dict[str, Any]:
    candidate_token_ids = {
        score: encode_text(tokenizer, score)
        for score in allowed_scores
    }
    logprobs = {
        score: continuation_logprob(model, tokenizer, prompt, score)
        for score in allowed_scores
    }
    prompt_length = len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
    return _normalize_logprobs(
        allowed_scores,
        logprobs,
        prompt_token_length=prompt_length,
        scoring_method="multi_forward_continuation",
        candidate_token_ids=candidate_token_ids,
    )


def score_allowed_continuations_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    allowed_scores: list[str],
    *,
    prefer_fast_single_token: bool = True,
) -> list[dict[str, Any]]:
    candidate_token_ids = {
        score: encode_text(tokenizer, score)
        for score in allowed_scores
    }
    if prefer_fast_single_token and all(len(ids) == 1 for ids in candidate_token_ids.values()):
        return _score_single_token_batch(model, tokenizer, prompts, allowed_scores, candidate_token_ids)
    return [
        score_allowed_continuations(model, tokenizer, prompt, allowed_scores)
        for prompt in prompts
    ]


def _score_single_token_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    allowed_scores: list[str],
    candidate_token_ids: dict[str, list[int]],
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if not prompts:
        return []
    device = _model_input_device(model)
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(prompt_lengths <= 0):
        raise ValueError("All prompts must contain at least one token")

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    token_positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    positions = (attention_mask * token_positions).max(dim=1).values.to(dtype=torch.long)
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    final_logits = logits[batch_indices, positions]
    full_log_probs = F.log_softmax(final_logits, dim=-1)
    score_ids = torch.tensor(
        [candidate_token_ids[score][0] for score in allowed_scores],
        dtype=torch.long,
        device=device,
    )
    score_logprobs = full_log_probs[:, score_ids].detach().cpu().numpy()
    prompt_lengths_np = prompt_lengths.detach().cpu().numpy()
    outputs = []
    for row_index, row_logprobs in enumerate(score_logprobs):
        logprobs = {
            score: float(row_logprobs[index])
            for index, score in enumerate(allowed_scores)
        }
        outputs.append(
            _normalize_logprobs(
                allowed_scores,
                logprobs,
                prompt_token_length=int(prompt_lengths_np[row_index]),
                scoring_method="single_forward_single_token",
                candidate_token_ids=candidate_token_ids,
            )
        )
    return outputs


def _normalize_classification_logits(
    allowed_scores: list[str],
    logits: np.ndarray,
    *,
    prompt_token_length: int | None = None,
) -> dict[str, Any]:
    values = logits.astype(np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    probs_arr = exp_values / exp_values.sum()
    probs = {score: float(prob) for score, prob in zip(allowed_scores, probs_arr, strict=True)}
    score_logits = {
        score: float(value)
        for score, value in zip(allowed_scores, values, strict=True)
    }
    pred_index = int(np.argmax(values))
    sorted_values = np.sort(values)
    score_margin = float(sorted_values[-1] - sorted_values[-2]) if len(sorted_values) > 1 else 0.0
    entropy = float(-np.sum(probs_arr * np.log(np.clip(probs_arr, 1e-12, 1.0))))
    return {
        "pred_score": int(allowed_scores[pred_index]),
        "logits": score_logits,
        "probs": probs,
        "score_margin": score_margin,
        "score_entropy": entropy,
        "prompt_token_length_inference": prompt_token_length,
        "scoring_method": "sequence_classification_logits",
        "candidate_token_ids": {},
    }


def score_sequence_classification_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    allowed_scores: list[str],
) -> list[dict[str, Any]]:
    import torch

    if not prompts:
        return []
    device = _model_input_device(model)
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lengths = attention_mask.sum(dim=1).detach().cpu().numpy()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_np = logits.detach().float().cpu().numpy()
    if logits_np.shape[-1] != len(allowed_scores):
        raise ValueError(
            f"Sequence classifier returned {logits_np.shape[-1]} labels; "
            f"expected {len(allowed_scores)}"
        )
    return [
        _normalize_classification_logits(
            allowed_scores,
            row_logits,
            prompt_token_length=int(prompt_lengths[row_index]),
        )
        for row_index, row_logits in enumerate(logits_np)
    ]


def predict_dataframe(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    allowed_scores = [str(score) for score in config["inference"].get("allowed_scores", ["1", "2", "3", "4", "5"])]
    threshold = int(config["inference"].get("binary_threshold", 3))
    system_prompt = config["prompt"]["system_prompt"]
    method = config["inference"].get("method", "restricted_continuation_logprobs_fast")
    prefer_fast = method == "restricted_continuation_logprobs_fast"
    batch_size = int(config["inference"].get("batch_size", 1))
    objective = config.get("training", {}).get("objective", "causal_lm")
    configure_tokenizer_thinking_mode(tokenizer, config)
    validate_score_tokenization(tokenizer, allowed_scores)
    if df.empty:
        return _empty_prediction_frame(
            allowed_scores,
            score_logit_prefix="logit" if objective == "sequence_classification" else "logprob",
        )
    rows = []
    starts = range(0, len(df), batch_size)
    for start in track(
        starts,
        total=(len(df) + batch_size - 1) // batch_size if len(df) else 0,
        description="Scoring continuations",
    ):
        batch_df = df.iloc[start : start + batch_size]
        examples = [row.to_dict() for _, row in batch_df.iterrows()]
        prompts = [format_prompt(example, tokenizer, system_prompt) for example in examples]
        if objective == "sequence_classification":
            scored_rows = score_sequence_classification_batch(
                model,
                tokenizer,
                prompts,
                allowed_scores,
            )
        else:
            scored_rows = score_allowed_continuations_batch(
                model,
                tokenizer,
                prompts,
                allowed_scores,
                prefer_fast_single_token=prefer_fast,
            )
        for example, scored in zip(examples, scored_rows, strict=True):
            pred_score = scored["pred_score"]
            out = {
                "pred_score": pred_score,
                "pred_binary": binary_from_score(pred_score, threshold),
                "target_score": int(example["target_score"]),
                "target_binary": example["target_binary"],
                "model_id": example.get("model_id"),
                "benchmark": example.get("benchmark"),
                "task": example.get("task"),
                "example_id": example.get("example_id"),
                "response_token_length": example.get("response_token_length"),
                "sequence_length": example.get("sequence_length"),
                "prompt_token_length_inference": scored["prompt_token_length_inference"],
                "score_margin": scored["score_margin"],
                "score_entropy": scored["score_entropy"],
                "scoring_method": scored["scoring_method"],
                "candidate_token_ids": json.dumps(scored["candidate_token_ids"], sort_keys=True),
            }
            for score in allowed_scores:
                if objective == "sequence_classification":
                    out[f"logit_{score}"] = scored["logits"][score]
                else:
                    out[f"logprob_{score}"] = scored["logprobs"][score]
                out[f"prob_{score}"] = scored["probs"][score]
            rows.append(out)
    return pd.DataFrame(rows)


def predict_with_config(config: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    model, tokenizer = load_model_for_inference(config)
    return predict_dataframe(df, model, tokenizer, config)
