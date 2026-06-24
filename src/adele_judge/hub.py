from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import yaml

from .config import save_config
from .modeling import model_from_pretrained_kwargs, torch_dtype_from_name
from .utils import ensure_dir, git_commit, package_versions, read_json, write_json


DEFAULT_COMMIT_MESSAGE = "Upload ADeLe distilled judge"
DEFAULT_MAX_SHARD_SIZE = "5GB"
DEFAULT_IMPLEMENTATION_URL = "https://github.com/adgomant/adele-judge"
DEFAULT_ADELE_PROJECT_URL = "https://kinds-of-intelligence-cfi.github.io/ADELE/"
DEFAULT_ADELE_PAPER_TITLE = "General scales unlock AI evaluation with explanatory and predictive power"
DEFAULT_ADELE_PAPER_URL = "https://www.nature.com/articles/s41586-026-10303-2"
DEFAULT_ADELE_DATASET_ID = "CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
DEFAULT_ADELE_DATASET_URL = (
    "https://huggingface.co/datasets/CFI-Kinds-of-Intelligence/ADeLe_battery_v1dot0"
)
DEFAULT_ADELE_SOURCE_DATA_URL = (
    "https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/tree/main/"
    "ADeLe_battery_data/subject_specific_instance_level_data"
)
HUB_PIPELINE_FILENAME = "adele_judge_pipeline.py"
STAGING_MARKER = ".adele_judge_staging"
CUSTOM_PIPELINE_TASK = "adele-judge"
CUSTOM_PIPELINE_METADATA = {
    CUSTOM_PIPELINE_TASK: {
        "impl": "adele_judge_pipeline.ADeLeJudgePipeline",
        "pt": ["AutoModelForCausalLM"],
        "tf": [],
        "type": "text",
    }
}


@dataclass(frozen=True)
class HubOptions:
    repo_id: str
    run_dir: Path
    staging_dir: Path
    private: bool
    commit_message: str
    create_pr: bool
    max_shard_size: str
    no_push: bool = False


@dataclass(frozen=True)
class HubCheckpointPaths:
    run_dir: Path
    adapter_dir: Path
    tokenizer_dir: Path | None


@dataclass(frozen=True)
class HubPushResult:
    repo_id: str
    staging_dir: Path
    pushed: bool
    url: str | None = None


def resolve_hub_options(
    config: dict[str, Any],
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
    staging_dir: Path | None = None,
    create_pr: bool | None = None,
    no_push: bool = False,
) -> HubOptions:
    hub_config = config.get("hub", {}) or {}
    resolved_repo_id = repo_id or hub_config.get("repo_id")
    if not resolved_repo_id:
        raise ValueError("hub.repo_id or --repo-id is required")

    run_dir = Path(
        hub_config.get("local_checkpoint_dir")
        or config.get("project", {}).get("output_dir", "")
    ).expanduser()
    if not str(run_dir):
        raise ValueError("hub.local_checkpoint_dir or project.output_dir is required")

    resolved_staging_dir = staging_dir or hub_config.get("output_staging_dir")
    if resolved_staging_dir is None:
        run_name = config.get("project", {}).get("run_name") or resolved_repo_id.split("/")[-1]
        resolved_staging_dir = Path("hub_staging") / str(run_name)

    return HubOptions(
        repo_id=str(resolved_repo_id),
        run_dir=run_dir,
        staging_dir=Path(resolved_staging_dir).expanduser(),
        private=bool(hub_config.get("private", False) if private is None else private),
        commit_message=str(commit_message or hub_config.get("commit_message") or DEFAULT_COMMIT_MESSAGE),
        create_pr=bool(hub_config.get("create_pr", False) if create_pr is None else create_pr),
        max_shard_size=str(hub_config.get("max_shard_size") or DEFAULT_MAX_SHARD_SIZE),
        no_push=bool(no_push),
    )


def resolve_checkpoint_paths(options: HubOptions) -> HubCheckpointPaths:
    run_dir = options.run_dir
    adapter_dir = run_dir / "adapter"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Trained adapter directory does not exist: {adapter_dir}")
    tokenizer_dir = run_dir / "tokenizer"
    return HubCheckpointPaths(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        tokenizer_dir=tokenizer_dir if tokenizer_dir.exists() else None,
    )


def push_trained_judge_to_hub(
    config: dict[str, Any],
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
    staging_dir: Path | None = None,
    create_pr: bool | None = None,
    no_push: bool = False,
) -> HubPushResult:
    options = resolve_hub_options(
        config,
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        staging_dir=staging_dir,
        create_pr=create_pr,
        no_push=no_push,
    )
    stage_hub_repository(config, options)
    if options.no_push:
        return HubPushResult(options.repo_id, options.staging_dir, pushed=False)

    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=options.repo_id,
        private=options.private,
        repo_type="model",
        exist_ok=True,
    )
    upload_info = api.upload_folder(
        repo_id=options.repo_id,
        repo_type="model",
        folder_path=str(options.staging_dir),
        commit_message=options.commit_message,
        create_pr=options.create_pr,
    )
    return HubPushResult(
        options.repo_id,
        options.staging_dir,
        pushed=True,
        url=str(getattr(upload_info, "commit_url", None) or repo_url),
    )


def stage_hub_repository(config: dict[str, Any], options: HubOptions) -> Path:
    if config.get("training", {}).get("objective") == "sequence_classification":
        raise NotImplementedError(
            "Hugging Face Hub packaging for sequence_classification judges is not supported yet. "
            "The current Hub packager is specific to AutoModelForCausalLM restricted scoring."
        )
    paths = resolve_checkpoint_paths(options)
    reset_staging_dir(options.staging_dir)
    (options.staging_dir / STAGING_MARKER).write_text("ADeLe Hub staging directory\n", encoding="utf-8")
    save_merged_model(config, paths, options.staging_dir, max_shard_size=options.max_shard_size)
    add_custom_pipeline_metadata(options.staging_dir / "config.json")
    copy_adapter(paths.adapter_dir, options.staging_dir / "adapter")
    write_generation_config(options.staging_dir / "generation_config.json")
    write_hub_pipeline(options.staging_dir / HUB_PIPELINE_FILENAME)
    write_json(options.staging_dir / "adele_judge_config.json", hub_inference_config(config))
    metadata = collect_hub_metadata(config, paths.run_dir, options)
    write_json(options.staging_dir / "adele_judge_metadata.json", metadata)
    (options.staging_dir / "README.md").write_text(
        render_model_card(config, metadata, options.repo_id),
        encoding="utf-8",
    )
    save_config(config, options.staging_dir / "training_config.yaml")
    (options.staging_dir / STAGING_MARKER).unlink(missing_ok=True)
    return options.staging_dir


def reset_staging_dir(staging_dir: Path) -> None:
    if staging_dir.exists():
        marker = staging_dir / "adele_judge_metadata.json"
        staging_marker = staging_dir / STAGING_MARKER
        if any(staging_dir.iterdir()) and not marker.exists() and not staging_marker.exists():
            raise ValueError(
                f"Refusing to overwrite non-empty staging directory without ADeLe marker: {staging_dir}"
            )
        shutil.rmtree(staging_dir)
    ensure_dir(staging_dir)


def save_merged_model(
    config: dict[str, Any],
    paths: HubCheckpointPaths,
    staging_dir: Path,
    *,
    max_shard_size: str,
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(config.get("training", {}).get("dtype"))
    model_kwargs = model_from_pretrained_kwargs(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name_or_path"],
        torch_dtype=dtype or "auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    peft_model = PeftModel.from_pretrained(model, str(paths.adapter_dir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(
        str(staging_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    tokenizer_source = paths.tokenizer_dir or config["model"]["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(staging_dir))


def copy_adapter(adapter_dir: Path, destination: Path) -> None:
    shutil.copytree(adapter_dir, destination, dirs_exist_ok=True)


def write_generation_config(path: Path) -> None:
    write_json(
        path,
        {
            "max_new_tokens": 1,
            "do_sample": False,
            "num_beams": 1,
        },
    )


def write_hub_pipeline(path: Path) -> None:
    source = Path(__file__).with_name("hub_pipeline.py")
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def add_custom_pipeline_metadata(config_path: Path) -> None:
    model_config = read_json(config_path)
    model_config["custom_pipelines"] = CUSTOM_PIPELINE_METADATA
    write_json(config_path, model_config)


def hub_inference_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {
            "base_model": config["model"]["model_name_or_path"],
            "revision": config["model"].get("revision"),
            "thinking_mode": config["model"].get("thinking_mode") or {},
        },
        "prompt": {
            "system_prompt": config["prompt"]["system_prompt"],
        },
        "inference": {
            "allowed_scores": [
                str(score)
                for score in config.get("inference", {}).get("allowed_scores", ["1", "2", "3", "4", "5"])
            ],
            "binary_threshold": int(config.get("inference", {}).get("binary_threshold", 3)),
            "method": (
                "sequence_classification_logits"
                if config.get("training", {}).get("objective") == "sequence_classification"
                else "restricted_continuation_logprobs_fast"
            ),
        },
        "training": {
            "max_seq_length": config.get("training", {}).get("max_seq_length"),
            "objective": config.get("training", {}).get("objective"),
        },
    }


def collect_hub_metadata(
    config: dict[str, Any],
    run_dir: Path,
    options: HubOptions,
) -> dict[str, Any]:
    artifacts = {}
    for name in [
        "run_metadata.json",
        "dataset_filtering_report.json",
        "split_report.json",
        "length_statistics.json",
        "score_tokenization_report.json",
        "train_metrics.json",
        "validation_trainer_metrics.json",
    ]:
        path = run_dir / name
        if path.exists():
            artifacts[name] = read_json(path)

    for split in ["validation", "test"]:
        scoped_metrics_name = f"evaluation/{split}/metrics.json"
        scoped_metrics_path = run_dir / scoped_metrics_name
        if scoped_metrics_path.exists():
            artifacts[scoped_metrics_name] = read_json(scoped_metrics_path)
            continue
        legacy_metrics_name = f"{split}_metrics.json"
        legacy_metrics_path = run_dir / legacy_metrics_name
        if legacy_metrics_path.exists():
            artifacts[legacy_metrics_name] = read_json(legacy_metrics_path)

    for name in ["config.yaml", "inference_config.yaml"]:
        path = run_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                artifacts[name] = yaml.safe_load(handle) or {}

    return {
        "repo_id": options.repo_id,
        "base_model": config["model"]["model_name_or_path"],
        "training_objective": config.get("training", {}).get("objective"),
        "adapter_path": str(run_dir / "adapter"),
        "run_dir": str(run_dir),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "binary_threshold": int(config.get("inference", {}).get("binary_threshold", 3)),
        "allowed_scores": [
            str(score)
            for score in config.get("inference", {}).get("allowed_scores", ["1", "2", "3", "4", "5"])
        ],
        "max_seq_length": config.get("training", {}).get("max_seq_length"),
        "thinking_mode": config.get("model", {}).get("thinking_mode") or {},
        "artifacts": artifacts,
    }


VALIDATION_METRIC_KEYS = [
    ("binary_accuracy", "Binary accuracy"),
    ("binary_macro_f1", "Binary macro F1"),
    ("precision_correct", "Precision, CORRECT"),
    ("recall_correct", "Recall, CORRECT"),
    ("precision_incorrect", "Precision, INCORRECT"),
    ("recall_incorrect", "Recall, INCORRECT"),
    ("false_negative_rate_correct", "False negative rate, CORRECT"),
    ("false_positive_rate_correct", "False positive rate, CORRECT"),
    ("ordinal_accuracy", "Ordinal accuracy"),
    ("ordinal_macro_f1", "Ordinal macro F1"),
    ("confidence_mean", "Mean confidence"),
]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    prefixed = f"eval_{key}"
    if prefixed in metrics:
        return metrics[prefixed]
    return None


def validation_metrics_artifact(artifacts: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    for name in [
        "evaluation/validation/metrics.json",
        "validation_metrics.json",
        "validation_trainer_metrics.json",
    ]:
        value = artifacts.get(name)
        if isinstance(value, dict):
            return name, value
    return None, {}


def render_split_section(artifacts: dict[str, Any]) -> str:
    split_report = artifacts.get("split_report.json")
    if not isinstance(split_report, dict) or not split_report:
        return "No split report was found in the local packaging artifacts."

    rows = []
    detail_lines = []
    for split_name in ["train", "validation", "test"]:
        split = split_report.get(split_name)
        if not isinstance(split, dict):
            continue
        models = split.get("models") or []
        rows.append(
            [
                split_name,
                format_int(split.get("examples")),
                format_int(split.get("num_models", len(models))),
            ]
        )
        if models:
            detail_lines.append(f"- `{split_name}` models: " + ", ".join(f"`{model}`" for model in models))

    split_table = markdown_table(["Split", "Examples", "Models"], rows)
    if detail_lines:
        return split_table + "\n\n" + "\n".join(detail_lines)
    return split_table


def render_validation_section(artifacts: dict[str, Any]) -> str:
    artifact_name, metrics = validation_metrics_artifact(artifacts)
    if not metrics:
        return (
            "No validation metrics were found in the local run artifacts. "
            "If available, they are stored in `adele_judge_metadata.json`."
        )

    rows = []
    if "epoch" in metrics:
        rows.append(["Epoch", format_float(metrics["epoch"])])
    for key, label in VALIDATION_METRIC_KEYS:
        value = metric_value(metrics, key)
        if value is not None:
            rows.append([label, format_float(value)])

    note = f"Source artifact: `{artifact_name}`."
    return note + "\n\n" + markdown_table(["Metric", "Value"], rows)


def render_data_quality_section(config: dict[str, Any], artifacts: dict[str, Any]) -> str:
    data_config = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    columns = data_config.get("columns", {}) if isinstance(data_config.get("columns"), dict) else {}
    filters = data_config.get("filters", {}) if isinstance(data_config.get("filters"), dict) else {}
    judge_1 = columns.get("judge_1_score", "judge_1_score")
    judge_2 = columns.get("judge_2_score", "judge_2_score")
    threshold = int(config.get("inference", {}).get("binary_threshold", 3))
    max_disagreement = filters.get("max_disagreement")
    max_response_tokens = filters.get("max_response_tokens")
    max_seq_length = config.get("training", {}).get("max_seq_length")

    lines = [
        "Training labels are distilled from two proprietary judge scores used by the "
        "ADeLe evaluation pipeline to derive the official correctness signal. The "
        f"configured source columns are `{judge_1}` and `{judge_2}`.",
        "",
        f"- Ordinal target: `floor(mean({judge_1}, {judge_2}))`.",
        f"- Binary target: `CORRECT` when the ordinal target is >= `{threshold}`.",
    ]
    if max_disagreement is not None:
        lines.append(
            f"- Judge-agreement filter: keep examples with "
            f"`abs({judge_1} - {judge_2}) <= {max_disagreement}`."
        )
    if max_response_tokens is not None:
        lines.append(
            f"- Response-length filter: keep responses with at most `{max_response_tokens}` "
            "base-tokenizer tokens before prompt formatting."
        )
    if max_seq_length is not None:
        lines.append(
            f"- Sequence-length filter: keep full chat-formatted examples within "
            f"`max_seq_length={max_seq_length}`."
        )

    filtering_report = artifacts.get("dataset_filtering_report.json")
    if isinstance(filtering_report, dict) and filtering_report:
        rows = []
        for key, label in [
            ("raw_examples", "Raw examples"),
            ("removed_by_disagreement", "Removed by judge disagreement"),
            ("removed_by_response_length", "Removed by response length"),
            ("sequence_overflow_count", "Removed by full sequence overflow"),
            ("examples_after_sequence_filter", "Examples after filters"),
        ]:
            if key in filtering_report:
                rows.append([label, format_int(filtering_report[key])])
        if rows:
            lines.extend(["", markdown_table(["Filtering stage", "Examples"], rows)])

    length_statistics = artifacts.get("length_statistics.json")
    response_stats = (
        length_statistics.get("response_token_length")
        if isinstance(length_statistics, dict)
        else None
    )
    if isinstance(response_stats, dict):
        rows = []
        for key, label in [
            ("mean", "Mean"),
            ("p50", "P50"),
            ("p90", "P90"),
            ("p95", "P95"),
            ("p99", "P99"),
            ("max", "Max"),
        ]:
            if key in response_stats:
                rows.append([label, format_float(response_stats[key])])
        if rows:
            lines.extend(["", "Response-token length summary:", "", markdown_table(["Stat", "Tokens"], rows)])

    return "\n".join(lines)


def render_references_section(config: dict[str, Any]) -> str:
    hub_config = config.get("hub", {}) if isinstance(config.get("hub"), dict) else {}
    project_url = hub_config.get("project_url") or DEFAULT_ADELE_PROJECT_URL
    paper_title = hub_config.get("paper_title") or DEFAULT_ADELE_PAPER_TITLE
    paper_url = hub_config.get("paper_url") or DEFAULT_ADELE_PAPER_URL
    dataset_id = hub_config.get("dataset_id") or DEFAULT_ADELE_DATASET_ID
    dataset_url = hub_config.get("dataset_url") or DEFAULT_ADELE_DATASET_URL
    source_data_url = hub_config.get("source_data_url") or DEFAULT_ADELE_SOURCE_DATA_URL
    implementation_url = hub_config.get("implementation_url") or DEFAULT_IMPLEMENTATION_URL
    paper_reference = f"[{paper_title}]({paper_url})" if paper_url else paper_title
    return "\n".join(
        [
            f"- ADeLe project page: [ADeLe v1.0]({project_url}).",
            f"- ADeLe paper and official correctness definition: {paper_reference}.",
            f"- Official ADeLe dataset: [{dataset_id}]({dataset_url}).",
            f"- Official instance-level model-response data used for distillation: [{source_data_url}]({source_data_url}).",
            f"- Training and Hub packaging implementation: [{implementation_url}]({implementation_url}).",
        ]
    )


def render_model_card(
    config: dict[str, Any],
    metadata: dict[str, Any],
    repo_id: str,
) -> str:
    threshold = int(config.get("inference", {}).get("binary_threshold", 3))
    allowed_scores = ", ".join(
        str(score) for score in config.get("inference", {}).get("allowed_scores", ["1", "2", "3", "4", "5"])
    )
    base_model = config["model"]["model_name_or_path"]
    artifacts = metadata.get("artifacts", {})
    split_section = render_split_section(artifacts)
    validation_section = render_validation_section(artifacts)
    data_quality_section = render_data_quality_section(config, artifacts)
    references_section = render_references_section(config)
    return f"""---
library_name: transformers
tags:
- text-generation
- peft
- adele
- judge
base_model: {base_model}
---

# ADeLe Distilled Judge

This repository contains an ADeLe-suite-specific distilled judge. It scores a model response against a question and reference answer with an ordinal score from 1 to 5, then derives binary correctness with the ADeLe threshold.

The repository root contains a merged Transformers model for standard loading. The original LoRA adapter is also included under `adapter/` for provenance and reuse.

## Intended Use

Use this model to score ADeLe-style examples where a question, reference answer, and model response are available. It is intended for out-of-model evaluation within the ADeLe benchmark suite, not as a general-purpose evaluator.

## Input Format

The recommended helper accepts:

- `question`
- `reference_answer` or `ground_truth`
- `model_response`

## Score Rubric

Allowed scores: {allowed_scores}

- 1: surely incorrect
- 2: likely incorrect
- 3: minimally correct or sufficient
- 4: likely correct
- 5: surely correct

Binary label: scores greater than or equal to {threshold} are `CORRECT`; lower scores are `INCORRECT`.

## Training And Validation Data

{split_section}

## Data Quality And Label Construction

{data_quality_section}

## Validation Results

{validation_section}

## Recommended Inference

Do not use free-form generation as the primary prediction method. The recommended path scores the restricted continuations `"1"`, `"2"`, `"3"`, `"4"`, and `"5"`.

```python
from transformers import pipeline

judge = pipeline(
    "adele-judge",
    model="{repo_id}",
    trust_remote_code=True,
    device_map="auto",
)
result = judge(
    {{"question": "...", "reference_answer": "...", "model_response": "..."}}
)
print(result)

results = judge([
    {{"question": "...", "reference_answer": "...", "model_response": "..."}},
    {{"question": "...", "ground_truth": "...", "model_response": "..."}},
], batch_size=8)
```

The result has this shape:

```python
{{
    "score": 4,
    "label": "CORRECT",
    "probs": {{"1": 0.01, "2": 0.02, "3": 0.08, "4": 0.70, "5": 0.19}},
    "logprobs": {{"1": -5.0, "2": -4.2, "3": -2.9, "4": -0.8, "5": -2.1}},
    "confidence": 0.70,
    "margin": 1.3,
    "entropy": 0.82,
}}
```

## Standard Transformers Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
```

`generation_config.json` uses safe one-token defaults for debugging, but `generate()` is not the recommended scoring method.

## Metadata

Training, filtering, split, tokenization, and metric artifacts available at packaging time are stored in `adele_judge_metadata.json`.

The model is trained on distilled judge targets. These targets are useful for reproducing the ADeLe paper-style correctness signal at lower inference cost, but they should not be interpreted as independent human annotations.

## References

{references_section}

## Limitations

- ADeLe-specific judge; not a general-purpose evaluator.
- Distilled from proprietary judge labels and inherits their noise, calibration, and biases.
- Intended for scoring responses against a reference answer.
- It should not produce explanations; the expected output is a single score.
- Validation is out-of-model within the ADeLe suite, so transfer outside that suite should be measured before relying on it.
"""
