from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from .formatting import configure_tokenizer_thinking_mode
from .utils import get_local_rank, setup_distributed_device


COMMON_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


SCORE_ID2LABEL = {index: str(index + 1) for index in range(5)}
SCORE_LABEL2ID = {label: index for index, label in SCORE_ID2LABEL.items()}


SUPPORTED_ATTENTION_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3"}


def torch_dtype_from_name(name: str | None) -> Any:
    import torch

    if name in (None, "auto"):
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    kwargs = {
        "trust_remote_code": bool(config["model"].get("trust_remote_code", True)),
    }
    if config["model"].get("revision"):
        kwargs["revision"] = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"], **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_tokenizer_thinking_mode(tokenizer, config)
    return tokenizer


def model_from_pretrained_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config["model"].get("trust_remote_code", True)),
    }
    if config["model"].get("revision"):
        kwargs["revision"] = config["model"]["revision"]
    attn_implementation = config["model"].get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def distributed_training_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("distributed", {}).get("enabled", False))


def distributed_training_strategy(config: dict[str, Any]) -> str:
    return str(config.get("distributed", {}).get("strategy", "ddp"))


def training_device_map(config: dict[str, Any]) -> str | dict[str, int] | None:
    """Return safe training placement for Transformers loaders."""

    if not distributed_training_enabled(config):
        return "auto"
    if distributed_training_strategy(config) != "ddp":
        if bool(config["training"].get("load_in_4bit", True)):
            raise ValueError(
                "QLoRA/4-bit training is only supported for single-GPU or DDP in this project. "
                "Set training.load_in_4bit=false for FSDP or DeepSpeed."
            )
        return None
    if bool(config["training"].get("load_in_4bit", True)):
        return {"": get_local_rank()}
    return None


def load_model_for_training(config: dict[str, Any]) -> tuple[Any, Any]:
    if distributed_training_enabled(config):
        setup_distributed_device()
    if config["training"].get("objective") == "sequence_classification":
        return load_sequence_classification_model_for_training(config)
    if distributed_training_enabled(config):
        return load_model_for_training_fallback(config)

    model_name = config["model"]["model_name_or_path"]
    training = config["training"]
    dtype = torch_dtype_from_name(training.get("dtype"))

    try:
        from unsloth import FastLanguageModel

        kwargs = {}
        if config["model"].get("revision"):
            kwargs["revision"] = config["model"]["revision"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=int(training["max_seq_length"]),
            dtype=dtype,
            load_in_4bit=bool(training.get("load_in_4bit", True)),
            **kwargs,
        )
        target_modules = training.get("target_modules", "auto")
        if target_modules == "auto":
            target_modules = COMMON_LORA_TARGET_MODULES
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(training["lora_r"]),
            target_modules=target_modules,
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(training.get("seed", config["project"].get("seed", 42))),
        )
        model.adele_training_backend = "unsloth"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        configure_tokenizer_thinking_mode(tokenizer, config)
        return model, tokenizer
    except Exception as exc:
        warn(
            f"Falling back to Transformers/PEFT training loader because Unsloth load failed: {exc}",
            stacklevel=2,
        )
        return load_model_for_training_fallback(config)


def load_model_for_training_fallback(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = load_tokenizer(config)
    training = config["training"]
    dtype = torch_dtype_from_name(training.get("dtype")) or torch.bfloat16
    quantization_config = None
    if bool(training.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    device_map = training_device_map(config)
    model_kwargs = model_from_pretrained_kwargs(config)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name_or_path"],
        torch_dtype=dtype,
        quantization_config=quantization_config,
        **model_kwargs,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    target_modules = training.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = COMMON_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        r=int(training["lora_r"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.adele_training_backend = "transformers_peft"
    return model, tokenizer


def load_sequence_classification_model_for_training(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    tokenizer = load_tokenizer(config)
    training = config["training"]
    dtype = torch_dtype_from_name(training.get("dtype")) or torch.bfloat16
    quantization_config = None
    if bool(training.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    device_map = training_device_map(config)
    model_kwargs = model_from_pretrained_kwargs(config)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["model_name_or_path"],
        num_labels=5,
        id2label=SCORE_ID2LABEL,
        label2id=SCORE_LABEL2ID,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    target_modules = training.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = COMMON_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        r=int(training["lora_r"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.adele_training_backend = "transformers_peft_sequence_classification"
    return model, tokenizer


def load_model_for_inference(config: dict[str, Any]) -> tuple[Any, Any]:
    if config.get("training", {}).get("objective") == "sequence_classification":
        return load_sequence_classification_model_for_inference(config)

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = load_tokenizer(config)
    dtype = torch_dtype_from_name(config.get("training", {}).get("dtype")) or torch.bfloat16
    quantization_config = None
    if bool(config.get("inference", {}).get("load_in_4bit", False)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name_or_path"],
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
        **model_from_pretrained_kwargs(config),
    )
    adapter_path = resolve_adapter_path(config)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def load_sequence_classification_model_for_inference(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    tokenizer = load_tokenizer(config)
    dtype = torch_dtype_from_name(config.get("training", {}).get("dtype")) or torch.bfloat16
    quantization_config = None
    if bool(config.get("inference", {}).get("load_in_4bit", False)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["model_name_or_path"],
        num_labels=5,
        id2label=SCORE_ID2LABEL,
        label2id=SCORE_LABEL2ID,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
        **model_from_pretrained_kwargs(config),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    adapter_path = resolve_adapter_path(config)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def resolve_adapter_path(config: dict[str, Any]) -> Path | None:
    configured = config["model"].get("adapter_path")
    if configured:
        path = Path(configured).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured adapter_path does not exist: {path}")
        return path

    default_adapter = Path(config["project"]["output_dir"]) / "adapter"
    if default_adapter.exists():
        warn(
            f"model.adapter_path is not set; auto-loading trained adapter at {default_adapter}",
            stacklevel=2,
        )
        return default_adapter

    inference = config.get("inference", {})
    if bool(inference.get("require_adapter", False)) or not bool(
        inference.get("allow_base_model", True)
    ):
        raise FileNotFoundError(
            "No adapter configured or found at the run output directory. "
            "Set model.adapter_path or disable inference.require_adapter."
        )
    warn(
        "No adapter configured or found; inference will run the base model.",
        stacklevel=2,
    )
    return None
