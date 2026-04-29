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
            "method": "restricted_continuation_logprobs_fast",
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
        "validation_metrics.json",
        "test_metrics.json",
    ]:
        path = run_dir / name
        if path.exists():
            artifacts[name] = read_json(path)

    for name in ["config.yaml", "inference_config.yaml"]:
        path = run_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                artifacts[name] = yaml.safe_load(handle) or {}

    return {
        "repo_id": options.repo_id,
        "base_model": config["model"]["model_name_or_path"],
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
    metrics = metadata.get("artifacts", {})
    metrics_note = "No evaluation metrics were found in the local run artifacts."
    if "validation_metrics.json" in metrics or "test_metrics.json" in metrics:
        metrics_note = "Validation/test metrics are included in `adele_judge_metadata.json`."
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

This repository contains an ADeLe-suite-specific distilled judge. It scores a model response against a question and reference answer with an ordinal score from 1 to 5.

The repository root contains a merged Transformers model for standard loading. The original LoRA adapter is also included under `adapter/` for provenance and reuse.

## Intended Use

Use this model to score ADeLe-style examples where a question, reference answer, and model response are available. It is not a general-purpose evaluator.

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

{metrics_note}

## Limitations

- ADeLe-specific judge; not a general-purpose evaluator.
- Distilled from judge labels and inherits their noise and biases.
- Intended for scoring responses against a reference answer.
- It should not produce explanations; the expected output is a single score.
"""
