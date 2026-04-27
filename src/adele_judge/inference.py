from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rich.progress import track

from .formatting import format_prompt
from .metrics import binary_from_score
from .modeling import load_model_for_inference


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


def score_allowed_continuations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    allowed_scores: list[str],
) -> dict[str, Any]:
    logprobs = {
        score: continuation_logprob(model, tokenizer, prompt, score)
        for score in allowed_scores
    }
    values = np.array([logprobs[score] for score in allowed_scores], dtype=np.float64)
    shifted = values - np.max(values)
    probs_arr = np.exp(shifted) / np.exp(shifted).sum()
    probs = {score: float(prob) for score, prob in zip(allowed_scores, probs_arr, strict=True)}
    pred_score = max(allowed_scores, key=lambda score: logprobs[score])
    return {"pred_score": int(pred_score), "logprobs": logprobs, "probs": probs}


def predict_dataframe(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    allowed_scores = [str(score) for score in config["inference"].get("allowed_scores", ["1", "2", "3", "4", "5"])]
    threshold = int(config["inference"].get("binary_threshold", 3))
    system_prompt = config["prompt"]["system_prompt"]
    rows = []
    for _, row in track(
        df.iterrows(),
        total=len(df),
        description="Scoring continuations",
    ):
        example = row.to_dict()
        prompt = format_prompt(example, tokenizer, system_prompt)
        scored = score_allowed_continuations(model, tokenizer, prompt, allowed_scores)
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
        }
        for score in allowed_scores:
            out[f"logprob_{score}"] = scored["logprobs"][score]
            out[f"prob_{score}"] = scored["probs"][score]
        rows.append(out)
    return pd.DataFrame(rows)


def predict_with_config(config: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    model, tokenizer = load_model_for_inference(config)
    return predict_dataframe(df, model, tokenizer, config)
