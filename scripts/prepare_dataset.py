#!/usr/bin/env python
from __future__ import annotations

import argparse

from adele_judge.config import load_config
from adele_judge.pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    splits = prepare_dataset(config)
    for name, split in splits.items():
        print(f"{name}: {len(split):,} examples, {split['model_id'].nunique()} models")


if __name__ == "__main__":
    main()
