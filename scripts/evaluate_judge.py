#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adele_judge.config import load_config
from adele_judge.metrics import majority_binary_baseline, majority_ordinal_baseline
from adele_judge.pipeline import load_prepared_split
from adele_judge.reporting import evaluation_dir, resolve_prediction_path, save_evaluation_reports
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
    pred_path = resolve_prediction_path(out_dir, args.split, Path(args.predictions) if args.predictions else None)
    predictions = pd.read_parquet(pred_path)
    metrics = save_evaluation_reports(
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
    ordinal_baseline = majority_ordinal_baseline(
        train_df,
        eval_df,
        threshold=int(config["inference"]["binary_threshold"]),
    )
    out_eval_dir = evaluation_dir(out_dir, args.split)
    write_json(out_eval_dir / "majority_baseline.json", baseline)
    write_json(out_eval_dir / "majority_ordinal_baseline.json", ordinal_baseline)
    print(metrics)


if __name__ == "__main__":
    main()
