from __future__ import annotations

from typing import Any


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
            f"Benchmark:\n{clean_value(example.get('benchmark'))}",
            f"Task:\n{clean_value(example.get('task'))}",
            f"Question:\n{clean_value(example.get('question'))}",
            f"Reference answer:\n{clean_value(example.get('reference_answer'))}",
            f"Model response:\n{clean_value(example.get('response'), fallback='')}",
            "Score:",
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
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def format_prompt(example: dict[str, Any], tokenizer: Any, system_prompt: str) -> str:
    return apply_chat_template(
        tokenizer,
        build_messages(example, system_prompt),
        add_generation_prompt=True,
    )
