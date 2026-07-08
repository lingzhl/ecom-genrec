#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.utils import load_yaml


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
        attn_implementation="flash_attention_2",
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
    sft_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=int(train_cfg["max_seq_length"]),
        dataset_text_field="text",
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]),
        bf16=bool(train_cfg["bf16"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        evaluation_strategy="steps",
        save_strategy="steps",
        deepspeed=args.deepspeed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
