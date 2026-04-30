from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from rich.progress import track

from .formatting import format_prompt


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_length: int
    target_length: int
    sequence_length: int


@dataclass
class TokenizedClassificationExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: int
    prompt_length: int
    sequence_length: int


def score_to_class_id(score: int | str) -> int:
    value = int(score)
    if value < 1 or value > 5:
        raise ValueError(f"Score must be in 1..5, got {score!r}")
    return value - 1


def class_id_to_score(class_id: int) -> int:
    value = int(class_id)
    if value < 0 or value >= 5:
        raise ValueError(f"Class id must be in 0..4, got {class_id!r}")
    return value + 1


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def score_tokenization_report(tokenizer: Any, allowed_scores: list[str]) -> list[dict[str, Any]]:
    rows = []
    for raw_score in allowed_scores:
        score = str(raw_score)
        token_ids = encode_text(tokenizer, score)
        rows.append(
            {
                "score": score,
                "token_ids": token_ids,
                "tokens": tokenizer.convert_ids_to_tokens(token_ids)
                if hasattr(tokenizer, "convert_ids_to_tokens")
                else [tokenizer.decode([token_id]) for token_id in token_ids],
                "num_tokens": len(token_ids),
            }
        )
    return rows


def validate_score_tokenization(
    tokenizer: Any,
    allowed_scores: list[str],
    *,
    require_single_token: bool = False,
) -> list[dict[str, Any]]:
    """Validate the bare score-string contract shared by training and inference."""

    report = score_tokenization_report(tokenizer, allowed_scores)
    seen_tokenizations: dict[tuple[int, ...], str] = {}
    expected_scores = {str(score) for score in range(1, 6)}
    configured_scores = {str(score) for score in allowed_scores}
    if configured_scores != expected_scores:
        raise ValueError(
            "allowed_scores must be exactly the bare strings '1', '2', '3', '4', and '5'"
        )
    for item in report:
        score = item["score"]
        if score.strip() != score or score != str(int(score)):
            raise ValueError(f"Score continuation must be a bare integer string, got {score!r}")
        token_ids = item["token_ids"]
        if not token_ids:
            raise ValueError(f"Score continuation {score!r} produced no tokens")
        if require_single_token and len(token_ids) != 1:
            raise ValueError(
                f"Score continuation {score!r} must be single-token for this objective/method; "
                f"got token ids {token_ids}"
            )
        key = tuple(token_ids)
        previous = seen_tokenizations.get(key)
        if previous is not None:
            raise ValueError(
                f"Score continuations {previous!r} and {score!r} share tokenization {token_ids}"
            )
        seen_tokenizations[key] = score
    return report


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


def tokenize_classification_example(
    example: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    max_seq_length: int,
    on_sequence_overflow: str = "skip",
) -> TokenizedClassificationExample | None:
    prompt_text = format_prompt(example, tokenizer, system_prompt)
    input_ids = encode_text(tokenizer, prompt_text)

    if len(input_ids) > max_seq_length:
        if on_sequence_overflow == "skip":
            return None
        if on_sequence_overflow == "error":
            raise ValueError(
                f"Formatted classifier input length {len(input_ids)} "
                f"exceeds max_seq_length={max_seq_length}"
            )
        raise ValueError("on_sequence_overflow must be 'skip' or 'error'")

    return TokenizedClassificationExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=score_to_class_id(example["target_score"]),
        prompt_length=len(input_ids),
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
    starts = range(0, len(texts), batch_size)
    total = ceil(len(texts) / batch_size) if texts else 0
    for start in track(starts, total=total, description="Measuring response tokens"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths
