#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adele_judge.config import load_config
from adele_judge.metrics import majority_binary_baseline
from adele_judge.pipeline import load_prepared_split
from adele_judge.reporting import save_prediction_reports
from adele_judge.utils import project_output_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    out_dir = project_output_dir(config)
    pred_path = Path(args.predictions) if args.predictions else out_dir / f"predictions_{args.split}.parquet"
    predictions = pd.read_parquet(pred_path)
    metrics = save_prediction_reports(
        predictions,
        out_dir,
        args.split,
        threshold=int(config["inference"]["binary_threshold"]),
        length_buckets=config.get("evaluation", {}).get("length_buckets"),
    )
    train_df = load_prepared_split(config, "train")
    eval_df = load_prepared_split(config, args.split)
    baseline = majority_binary_baseline(
        train_df,
        eval_df,
        threshold=int(config["inference"]["binary_threshold"]),
    )
    write_json(out_dir / f"majority_baseline_{args.split}.json", baseline)
    print(metrics)


if __name__ == "__main__":
    main()
