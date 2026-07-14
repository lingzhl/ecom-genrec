#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.utils import load_yaml


def flash_attention_usable() -> tuple[bool, str | None]:
    if importlib.util.find_spec("flash_attn") is None:
        return False, "flash_attn is not installed"
    try:
        import torch
        from flash_attn import flash_attn_func

        if not torch.cuda.is_available():
            return False, "CUDA is not available"
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = torch.randn(1, 16, 2, 64, device=device, dtype=dtype)
        flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
        torch.cuda.synchronize(device)
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else repr(exc)
        return False, f"{type(exc).__name__}: {message}"
    return True, None


def select_attention_impl(requested: str) -> str:
    if requested != "auto":
        if requested == "flash_attention_2":
            ok, reason = flash_attention_usable()
            if not ok:
                raise SystemExit(
                    "flash_attention_2 was requested, but it cannot run in this environment: "
                    f"{reason}\nUse --attn-implementation auto or --attn-implementation sdpa."
                )
        return requested

    ok, reason = flash_attention_usable()
    if ok:
        return "flash_attention_2"
    print(f"FlashAttention 2 unavailable or incompatible ({reason}); using PyTorch SDPA.")
    return "sdpa"


def supported_kwargs(callable_obj, kwargs: dict) -> dict:
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def require_training_stack() -> None:
    missing = []
    for package in ["torch", "transformers", "datasets", "trl", "peft"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: pip install -r requirements.txt"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Attention backend. auto probes FlashAttention 2 and falls back to PyTorch SDPA.",
    )
    args = parser.parse_args()

    require_training_stack()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = load_yaml(args.config)
    train_cfg = cfg["training"]
    model_name = args.model or cfg["models"]["sft_base"]
    output_dir = args.output_dir or train_cfg["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation=select_attention_impl(args.attn_implementation),
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files={"train": args.train, "validation": args.eval})
    peft_config = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_kwargs = {
        "output_dir": output_dir,
        "dataset_text_field": "text",
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "num_train_epochs": float(train_cfg["num_train_epochs"]),
        "max_steps": args.max_steps if args.max_steps is not None else int(train_cfg["max_steps"]),
        "bf16": bool(train_cfg["bf16"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "save_steps": int(train_cfg["save_steps"]),
        "eval_steps": int(train_cfg["eval_steps"]),
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "deepspeed": args.deepspeed,
        "report_to": "none",
        "max_length": int(train_cfg["max_seq_length"]),
        "max_seq_length": int(train_cfg["max_seq_length"]),
    }
    sft_args = SFTConfig(**supported_kwargs(SFTConfig, sft_kwargs))
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }
    trainer = SFTTrainer(
        **supported_kwargs(SFTTrainer, trainer_kwargs)
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
