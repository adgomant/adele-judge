from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from transformers import Pipeline


THINKING_KWARG = "enable_thinking"
DEFAULT_SYSTEM_PROMPT = "Return only one score from 1 to 5. Do not explain."
DEFAULT_ALLOWED_SCORES = ["1", "2", "3", "4", "5"]
DEFAULT_BINARY_THRESHOLD = 3


def load_adele_judge_config(repo_id_or_path: str) -> dict[str, Any]:
    path = Path(repo_id_or_path) / "adele_judge_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id_or_path, "adele_judge_config.json")
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def load_adele_judge_config_or_default(model: Any, tokenizer: Any) -> dict[str, Any]:
    candidates = [
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "_name_or_path", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return load_adele_judge_config(str(candidate))
        except Exception:
            continue
    return {}


def adele_judge_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    prompt_config = config.get("prompt", {}) if isinstance(config.get("prompt"), dict) else {}
    inference_config = (
        config.get("inference", {}) if isinstance(config.get("inference"), dict) else {}
    )
    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    return {
        "system_prompt": prompt_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        "allowed_scores": [
            str(score)
            for score in inference_config.get("allowed_scores", DEFAULT_ALLOWED_SCORES)
        ],
        "binary_threshold": int(
            inference_config.get("binary_threshold", DEFAULT_BINARY_THRESHOLD)
        ),
        "thinking_mode": model_config.get("thinking_mode") or {},
    }


def clean_value(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value)
    if not text or text.lower() == "nan":
        return fallback
    return text


def validate_example(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        raise ValueError("ADeLe judge input must be a mapping")

    missing = []
    if inputs.get("question") is None:
        missing.append("question")
    if inputs.get("model_response") is None:
        missing.append("model_response")
    reference_answer = inputs.get("reference_answer")
    if reference_answer is None:
        reference_answer = inputs.get("ground_truth")
    if reference_answer is None:
        missing.append("reference_answer or ground_truth")
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    return {
        "question": inputs["question"],
        "reference_answer": reference_answer,
        "model_response": inputs["model_response"],
    }


def build_user_message(example: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"### QUESTION\n{clean_value(example.get('question'))}",
            f"### REFERENCE ANSWER\n{clean_value(example.get('reference_answer'))}",
            f"### MODEL RESPONSE\n{clean_value(example.get('model_response'), fallback='')}",
            "### SCORE\n",
        ]
    )


def build_messages(example: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": build_user_message(example)},
    ]


def chat_template_supports_thinking(tokenizer: Any) -> bool:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return False
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return False
    accepts_kwarg = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == THINKING_KWARG
        for name, parameter in signature.parameters.items()
    )
    if not accepts_kwarg:
        return False

    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and THINKING_KWARG in template:
        return True

    candidates = [
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "_name_or_path", None),
        getattr(tokenizer, "model_name", None),
    ]
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        candidates.extend([init_kwargs.get("name_or_path"), init_kwargs.get("tokenizer_file")])
    return any("qwen3" in str(candidate).lower() for candidate in candidates if candidate)


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    thinking_mode: dict[str, Any],
) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        template_kwargs = {}
        enabled = thinking_mode.get("enabled")
        if (
            enabled is not None
            and bool(thinking_mode.get("apply_if_supported", True))
            and chat_template_supports_thinking(tokenizer)
        ):
            template_kwargs[THINKING_KWARG] = bool(enabled)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

    rendered = [f"<|{message['role']}|>\n{message['content']}" for message in messages]
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def single_score_token_ids(tokenizer: Any, allowed_scores: list[str]) -> list[int]:
    token_ids = [encode_text(tokenizer, score) for score in allowed_scores]
    multi_token_scores = [
        score for score, ids in zip(allowed_scores, token_ids, strict=True) if len(ids) != 1
    ]
    if multi_token_scores:
        raise ValueError(
            "ADeLeJudgePipeline requires score continuations to be single tokens; "
            f"multi-token scores: {multi_token_scores}"
        )
    return [ids[0] for ids in token_ids]


class ADeLeJudgePipeline(Pipeline):
    """HF-native custom pipeline for restricted ADeLe judge scoring."""

    def __init__(
        self,
        *args: Any,
        adele_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.tokenizer is None:
            raise ValueError("ADeLeJudgePipeline requires a tokenizer")
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)

        settings = adele_judge_settings(
            adele_config
            if adele_config is not None
            else load_adele_judge_config_or_default(self.model, self.tokenizer)
        )
        self.system_prompt = settings["system_prompt"]
        self.allowed_scores = settings["allowed_scores"]
        self.binary_threshold = settings["binary_threshold"]
        self.thinking_mode = settings["thinking_mode"]
        self.score_token_ids = single_score_token_ids(self.tokenizer, self.allowed_scores)

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _sanitize_parameters(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return {}, {}, {}

    def preprocess(self, inputs: Any) -> dict[str, Any]:
        import torch

        example = validate_example(inputs)
        prompt = apply_chat_template_safe(
            self.tokenizer,
            build_messages(example, self.system_prompt),
            add_generation_prompt=True,
            thinking_mode=self.thinking_mode,
        )
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def _forward(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        token_positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        positions = (attention_mask * token_positions).max(dim=1).values.to(dtype=torch.long)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        final_logits = outputs.logits[batch_indices, positions]

        score_ids = torch.tensor(self.score_token_ids, dtype=torch.long, device=final_logits.device)
        score_logits = final_logits[:, score_ids]
        logprobs = F.log_softmax(score_logits, dim=-1)
        return {
            "score_indices": torch.argmax(logprobs, dim=-1),
            "probs": torch.exp(logprobs),
            "logprobs": logprobs,
        }

    def postprocess(self, model_outputs: dict[str, Any]) -> dict[str, Any]:
        import torch

        score_index = int(model_outputs["score_indices"].reshape(-1)[0])
        probs_tensor = model_outputs["probs"].reshape(-1, len(self.allowed_scores))[0]
        logprobs_tensor = model_outputs["logprobs"].reshape(-1, len(self.allowed_scores))[0]

        probs = {
            score: float(prob)
            for score, prob in zip(self.allowed_scores, probs_tensor.tolist(), strict=True)
        }
        logprobs = {
            score: float(logprob)
            for score, logprob in zip(self.allowed_scores, logprobs_tensor.tolist(), strict=True)
        }

        score = int(self.allowed_scores[score_index])
        sorted_logprobs = torch.sort(logprobs_tensor).values
        margin = (
            float(sorted_logprobs[-1] - sorted_logprobs[-2])
            if len(sorted_logprobs) > 1
            else 0.0
        )
        entropy = float(
            -(probs_tensor * torch.log(torch.clamp(probs_tensor, min=1e-12))).sum()
        )
        return {
            "score": score,
            "label": "CORRECT" if score >= self.binary_threshold else "INCORRECT",
            "probs": probs,
            "logprobs": logprobs,
            "confidence": max(probs.values()),
            "margin": margin,
            "entropy": entropy,
        }
