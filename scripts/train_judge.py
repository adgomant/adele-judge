#!/usr/bin/env python
from __future__ import annotations

import argparse

from adele_judge.config import load_config
from adele_judge.train import train_judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--finalist", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    train_judge(config, force_prepare=args.force_prepare, finalist=args.finalist)


if __name__ == "__main__":
    main()
