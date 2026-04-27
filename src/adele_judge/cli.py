from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, save_config
from .data import load_and_construct_targets
from .inference import predict_with_config
from .metrics import majority_binary_baseline
from .modeling import load_tokenizer
from .pipeline import load_or_prepare_splits, load_prepared_split, prepare_dataset
from .reporting import save_prediction_reports
from .splits import enumerate_lomo_models
from .tokenization import supervised_token_debug_rows
from .train import train_judge
from .utils import ensure_dir, project_output_dir, write_json


class SplitName(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


class EvalSplitName(str, Enum):
    validation = "validation"
    test = "test"


ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML run config.",
    ),
]
OverrideOption = Annotated[
    list[str] | None,
    typer.Option(
        "--override",
        "-o",
        help="Override a config value with key.path=value. Repeat as needed.",
    ),
]


console = Console()
app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Train and evaluate the ADeLe distilled judge.",
)


def _load_config(path: Path, overrides: list[str] | None) -> dict[str, Any]:
    config = load_config(path, overrides or [])
    run_name = config.get("project", {}).get("run_name", "unknown")
    console.print(f"[bold]Run:[/] {run_name}")
    console.print(f"[bold]Output:[/] {project_output_dir(config)}")
    return config


def _print_split_summary(splits: dict[str, pd.DataFrame]) -> None:
    table = Table(title="Prepared Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", justify="right")
    table.add_column("Models", justify="right")
    for name, split in splits.items():
        table.add_row(name, f"{len(split):,}", f"{split['model_id'].nunique():,}")
    console.print(table)


def _print_metrics(metrics: dict[str, Any], title: str) -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", overflow="fold")
    for key, value in metrics.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
        table.add_row(key, rendered)
    console.print(table)


@app.command()
def prepare(
    config: ConfigOption,
    override: OverrideOption = None,
) -> None:
    """Prepare filtered train/validation/test parquet splits."""
    run_config = _load_config(config, override)
    splits = prepare_dataset(run_config)
    _print_split_summary(splits)
    console.print("[green]Prepared dataset artifacts.[/]")


@app.command()
def train(
    config: ConfigOption,
    force_prepare: Annotated[
        bool,
        typer.Option(
            "--force-prepare",
            help="Rebuild prepared splits before training.",
        ),
    ] = False,
    override: OverrideOption = None,
) -> None:
    """Train the judge adapter."""
    run_config = _load_config(config, override)
    metrics = train_judge(run_config, force_prepare=force_prepare)
    _print_metrics(metrics["train_metrics"], "Training Metrics")
    _print_metrics(metrics["validation_trainer_metrics"], "Validation Trainer Metrics")
    console.print("[green]Training complete.[/]")


@app.command()
def predict(
    config: ConfigOption,
    split: Annotated[
        SplitName,
        typer.Option("--split", "-s", help="Prepared split to score."),
    ],
    override: OverrideOption = None,
) -> None:
    """Run restricted-continuation prediction for a prepared split."""
    run_config = _load_config(config, override)
    df = load_prepared_split(run_config, split.value)
    console.print(f"[bold]Scoring:[/] {len(df):,} examples from {split.value}")
    predictions = predict_with_config(run_config, df)
    metrics = save_prediction_reports(
        predictions,
        project_output_dir(run_config),
        split.value,
        threshold=int(run_config["inference"]["binary_threshold"]),
        length_buckets=run_config.get("evaluation", {}).get("length_buckets"),
    )
    _print_metrics(metrics, f"{split.value.title()} Metrics")
    console.print("[green]Wrote predictions and reports.[/]")


@app.command()
def evaluate(
    config: ConfigOption,
    split: Annotated[
        EvalSplitName,
        typer.Option("--split", "-s", help="Validation or test split to evaluate."),
    ],
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            "-p",
            dir_okay=False,
            readable=True,
            help="Optional predictions parquet path. Defaults to the run output directory.",
        ),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Evaluate saved predictions and write reports."""
    run_config = _load_config(config, override)
    out_dir = project_output_dir(run_config)
    pred_path = predictions or out_dir / f"predictions_{split.value}.parquet"
    console.print(f"[bold]Predictions:[/] {pred_path}")
    prediction_df = pd.read_parquet(pred_path)
    metrics = save_prediction_reports(
        prediction_df,
        out_dir,
        split.value,
        threshold=int(run_config["inference"]["binary_threshold"]),
        length_buckets=run_config.get("evaluation", {}).get("length_buckets"),
    )
    train_df = load_prepared_split(run_config, "train")
    eval_df = load_prepared_split(run_config, split.value)
    baseline = majority_binary_baseline(
        train_df,
        eval_df,
        threshold=int(run_config["inference"]["binary_threshold"]),
    )
    write_json(out_dir / f"majority_baseline_{split.value}.json", baseline)
    _print_metrics(metrics, f"{split.value.title()} Metrics")
    _print_metrics(baseline, "Majority Binary Baseline")
    console.print("[green]Evaluation reports updated.[/]")


@app.command("debug-tokenization")
def debug_tokenization(
    config: ConfigOption,
    split: Annotated[
        SplitName,
        typer.Option("--split", "-s", help="Prepared split to inspect."),
    ] = SplitName.train,
    num_examples: Annotated[
        int,
        typer.Option(
            "--num-examples",
            "-n",
            min=1,
            help="Number of examples to write.",
        ),
    ] = 3,
    override: OverrideOption = None,
) -> None:
    """Write token-level supervision debugging output."""
    run_config = _load_config(config, override)
    tokenizer = load_tokenizer(run_config)
    splits = load_or_prepare_splits(run_config, tokenizer)
    out_dir = ensure_dir(project_output_dir(run_config))
    out_path = out_dir / "token_supervision_debug.txt"
    chunks = []
    examples = splits[split.value].head(num_examples)
    for index, (_, row) in enumerate(examples.iterrows()):
        chunks.append(f"EXAMPLE {index}\n")
        rows = supervised_token_debug_rows(
            row.to_dict(),
            tokenizer,
            run_config["prompt"]["system_prompt"],
            int(run_config["training"]["max_seq_length"]),
            run_config["data"]["filters"].get("on_sequence_overflow", "skip"),
        )
        for item in rows:
            marker = "SUP" if item["supervised"] else "..."
            token_text = item["token"].replace("\n", "\\n")
            chunks.append(f"{item['position']:05d} {marker} {item['token_id']:>8} {token_text}\n")
        chunks.append("\n")
    out_path.write_text("".join(chunks), encoding="utf-8")
    console.print(f"[green]Wrote[/] {out_path}")


@app.command()
def lomo(
    config: ConfigOption,
    prepare_only: Annotated[
        bool,
        typer.Option(
            "--prepare-only",
            help="Prepare every leave-one-model-out fold without training.",
        ),
    ] = False,
    override: OverrideOption = None,
) -> None:
    """Run leave-one-model-out preparation or training folds."""
    base_config = _load_config(config, override)
    models = enumerate_lomo_models(load_and_construct_targets(base_config))
    root = ensure_dir(project_output_dir(base_config) / "lomo")
    console.print(f"[bold]LOMO folds:[/] {len(models):,}")
    for index, model_id in enumerate(models, start=1):
        console.rule(f"Fold {index}/{len(models)}: {model_id}")
        fold_config = deepcopy(base_config)
        safe_model = model_id.replace("/", "_").replace(" ", "_")
        fold_config["project"]["run_name"] = (
            f"{base_config['project']['run_name']}_lomo_{safe_model}"
        )
        fold_config["project"]["output_dir"] = str(root / safe_model)
        fold_config["split"]["mode"] = "leave_one_model_out"
        fold_config["split"]["held_out_model"] = model_id
        config_path = root / safe_model / "config.yaml"
        save_config(fold_config, config_path)
        if prepare_only:
            splits = prepare_dataset(fold_config)
            _print_split_summary(splits)
        else:
            train_judge(fold_config)
    console.print("[green]LOMO run complete.[/]")
