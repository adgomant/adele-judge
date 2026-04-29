from __future__ import annotations

import inspect
from typing import Any


THINKING_KWARG = "enable_thinking"


def clean_value(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value)
    if not text or text.lower() == "nan":
        return fallback
    return text


def build_user_message(example: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"### QUESTION\n{clean_value(example.get('question'))}",
            f"### REFERENCE ANSWER\n{clean_value(example.get('reference_answer'))}",
            f"### MODEL RESPONSE\n{clean_value(example.get('response'), fallback='')}",
            "### SCORE\n",
        ]
    )

def build_messages(
    example: dict[str, Any],
    system_prompt: str,
    target_score: int | str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": build_user_message(example)},
    ]
    if target_score is not None:
        messages.append({"role": "assistant", "content": str(target_score)})
    return messages


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    return apply_chat_template_safe(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def chat_template_supports_thinking(tokenizer: Any) -> bool:
    """Detect whether a tokenizer chat template understands enable_thinking."""

    accepts_kwarg = apply_chat_template_accepts_thinking_kwargs(tokenizer)
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and THINKING_KWARG in template and accepts_kwarg:
        return True

    name_candidates = [
        getattr(tokenizer, "name_or_path", None),
        getattr(tokenizer, "_name_or_path", None),
        getattr(tokenizer, "model_name", None),
    ]
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        name_candidates.extend(
            [
                init_kwargs.get("name_or_path"),
                init_kwargs.get("tokenizer_file"),
            ]
        )
    return accepts_kwarg and any(
        "qwen3" in str(candidate).lower() for candidate in name_candidates if candidate
    )


def apply_chat_template_accepts_thinking_kwargs(tokenizer: Any) -> bool:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return False
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == THINKING_KWARG
        for name, parameter in signature.parameters.items()
    )


def configure_tokenizer_thinking_mode(
    tokenizer: Any,
    config: dict[str, Any],
    *,
    log: bool = False,
) -> Any:
    """Attach model thinking-mode policy to a tokenizer for shared formatting paths."""

    thinking = config.get("model", {}).get("thinking_mode") or {}
    enabled = thinking.get("enabled")
    apply_if_supported = bool(thinking.get("apply_if_supported", True))
    supported = chat_template_supports_thinking(tokenizer)

    setattr(tokenizer, "_adele_thinking_enabled", enabled)
    setattr(tokenizer, "_adele_thinking_apply_if_supported", apply_if_supported)
    setattr(tokenizer, "_adele_thinking_support_detected", supported)

    signature = (supported, enabled, apply_if_supported)
    if log and getattr(tokenizer, "_adele_thinking_log_signature", None) != signature:
        print(f"Thinking mode support detected: {'yes' if supported else 'no'}")
        if supported and enabled is not None and apply_if_supported:
            print(f"Using {THINKING_KWARG}={bool(enabled)}")
        elif supported and enabled is None:
            print("No thinking-mode configuration requested for this tokenizer/model.")
        else:
            print("Skipping thinking-mode configuration for this tokenizer/model.")
        setattr(tokenizer, "_adele_thinking_log_signature", signature)
    return tokenizer


def tokenizer_thinking_template_kwargs(
    tokenizer: Any,
    thinking_enabled: bool | None = None,
) -> dict[str, bool]:
    enabled = (
        thinking_enabled
        if thinking_enabled is not None
        else getattr(tokenizer, "_adele_thinking_enabled", None)
    )
    if enabled is None:
        return {}
    if not bool(getattr(tokenizer, "_adele_thinking_apply_if_supported", True)):
        return {}
    supported = getattr(tokenizer, "_adele_thinking_support_detected", None)
    if supported is None:
        supported = chat_template_supports_thinking(tokenizer)
    if not supported:
        return {}
    return {THINKING_KWARG: bool(enabled)}


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    thinking_enabled: bool | None = None,
) -> str | list[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        kwargs = tokenizer_thinking_template_kwargs(tokenizer, thinking_enabled)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    rendered = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    text = "\n".join(rendered)
    if tokenize:
        return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return text


def format_prompt(example: dict[str, Any], tokenizer: Any, system_prompt: str) -> str:
    return apply_chat_template(
        tokenizer,
        build_messages(example, system_prompt),
        add_generation_prompt=True,
    )
