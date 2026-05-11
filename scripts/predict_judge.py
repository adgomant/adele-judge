#!/usr/bin/env python
from __future__ import annotations

import argparse

from adele_judge.config import load_config
from adele_judge.inference import predict_with_config
from adele_judge.pipeline import load_prepared_split
from adele_judge.reporting import save_predictions
from adele_judge.utils import project_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    df = load_prepared_split(config, args.split)
    predictions = predict_with_config(config, df)
    pred_path = save_predictions(predictions, project_output_dir(config), args.split)
    print(f"Wrote predictions for {args.split}: {pred_path}")


if __name__ == "__main__":
    main()
