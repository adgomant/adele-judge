from __future__ import annotations

from typing import Any

import pandas as pd


def create_splits(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    mode = config["split"]["mode"]
    if mode == "fixed_by_model":
        return fixed_by_model_split(df, config)
    if mode == "leave_one_model_out":
        held_out = config["split"].get("held_out_model")
        if not held_out:
            raise ValueError("split.held_out_model is required for a single leave-one-model-out fold")
        return lomo_split(df, held_out)
    raise ValueError(f"Unsupported split mode: {mode}")


def fixed_by_model_split(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    validation_models = set(config["split"].get("validation_models") or [])
    test_models = set(config["split"].get("test_models") or [])
    train_config = config["split"].get("train_models", "auto_except_val_test")
    all_models = set(df["model_id"].dropna().unique().tolist())

    if train_config == "auto_except_val_test":
        train_models = all_models - validation_models - test_models
    else:
        train_models = set(train_config or [])

    if not validation_models:
        raise ValueError("validation_models must not be empty")
    if not test_models:
        raise ValueError("test_models must not be empty")
    _validate_disjoint(train_models, validation_models, test_models)

    missing = (train_models | validation_models | test_models) - all_models
    if missing:
        raise ValueError(f"Configured split models not found in data: {sorted(missing)}")

    splits = {
        "train": df[df["model_id"].isin(train_models)].reset_index(drop=True),
        "validation": df[df["model_id"].isin(validation_models)].reset_index(drop=True),
        "test": df[df["model_id"].isin(test_models)].reset_index(drop=True),
    }
    validate_no_model_leakage(splits)
    return splits


def lomo_split(df: pd.DataFrame, held_out_model: str) -> dict[str, pd.DataFrame]:
    if held_out_model not in set(df["model_id"].dropna().unique()):
        raise ValueError(f"Held-out model {held_out_model!r} not found")
    train = df[df["model_id"] != held_out_model].reset_index(drop=True)
    test = df[df["model_id"] == held_out_model].reset_index(drop=True)
    return {"train": train, "validation": test.copy(), "test": test}


def enumerate_lomo_models(df: pd.DataFrame) -> list[str]:
    return sorted(df["model_id"].dropna().unique().tolist())


def _validate_disjoint(*sets: set[str]) -> None:
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"Split model sets overlap: {sorted(overlap)}")


def validate_no_model_leakage(splits: dict[str, pd.DataFrame]) -> None:
    model_sets = {
        name: set(split["model_id"].dropna().unique().tolist())
        for name, split in splits.items()
    }
    names = list(model_sets)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = model_sets[left_name] & model_sets[right_name]
            if overlap:
                raise ValueError(
                    f"Model leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )


def split_report(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: {
            "examples": int(len(split)),
            "models": sorted(split["model_id"].dropna().unique().tolist()),
            "num_models": int(split["model_id"].nunique()),
        }
        for name, split in splits.items()
    }
