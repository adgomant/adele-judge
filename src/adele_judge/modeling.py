from __future__ import annotations

from typing import Any


COMMON_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_for_training(config: dict[str, Any]) -> tuple[Any, Any]:
    model_name = config["model"]["model_name_or_path"]
    training = config["training"]
    dtype = torch_dtype_from_name(training.get("dtype"))

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=int(training["max_seq_length"]),
            dtype=dtype,
            load_in_4bit=bool(training.get("load_in_4bit", True)),
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception:
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
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
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
    return get_peft_model(model, lora_config), tokenizer


def load_model_for_inference(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(config)
    dtype = torch_dtype_from_name(config.get("training", {}).get("dtype")) or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
        torch_dtype=dtype,
        device_map="auto",
    )
    adapter_path = config["model"].get("adapter_path")
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer
