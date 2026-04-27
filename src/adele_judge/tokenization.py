from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .formatting import format_prompt


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_length: int
    target_length: int
    sequence_length: int


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def tokenize_supervised_example(
    example: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_seq_length: int,
    on_sequence_overflow: str = "skip",
) -> TokenizedExample | None:
    target = str(int(example["target_score"]))
    prompt_text = format_prompt(example, tokenizer, system_prompt)
    prompt_ids = encode_text(tokenizer, prompt_text)
    target_ids = encode_text(tokenizer, target)
    if not target_ids:
        raise ValueError(f"Target score {target!r} produced no tokens")

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(-100)

    if len(input_ids) > max_seq_length:
        if on_sequence_overflow == "skip":
            return None
        if on_sequence_overflow == "error":
            raise ValueError(
                f"Formatted sequence length {len(input_ids)} exceeds max_seq_length={max_seq_length}"
            )
        raise ValueError("on_sequence_overflow must be 'skip' or 'error'")

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        prompt_length=len(prompt_ids),
        target_length=len(target_ids),
        sequence_length=len(input_ids),
    )


def supervised_token_debug_rows(
    example: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_seq_length: int,
    on_sequence_overflow: str = "skip",
) -> list[dict[str, Any]]:
    tokenized = tokenize_supervised_example(
        example,
        tokenizer,
        system_prompt,
        max_seq_length,
        on_sequence_overflow,
    )
    if tokenized is None:
        return []
    rows = []
    for idx, token_id in enumerate(tokenized.input_ids):
        label = tokenized.labels[idx]
        rows.append(
            {
                "position": idx,
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "supervised": label != -100,
                "label_id": None if label == -100 else label,
                "label_text": "" if label == -100 else tokenizer.decode([label]),
            }
        )
    return rows


def batch_response_token_lengths(
    texts: list[str],
    tokenizer: Any,
    batch_size: int = 512,
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths
