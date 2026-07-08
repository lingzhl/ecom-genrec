#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.instruction import extract_sids, load_sid_maps
from ecom_genrec.metrics import item_popularity
from ecom_genrec.utils import load_yaml, read_jsonl


def require_training_stack() -> None:
    missing = []
    for package in ["torch", "transformers", "datasets", "trl", "peft"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing GRPO dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: pip install -r requirements.txt"
        )


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def shared_prefix_length(a: str, b: str) -> int:
    a_parts = a.split("_")
    b_parts = b.split("_")
    length = 0
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        length += 1
    return length


def make_reward_func(sid_to_item: Dict[str, Dict[str, Any]], weights: Dict[str, float], popularity: Dict[str, int]):
    valid_sids = set(sid_to_item)
    total_popularity = max(1, sum(popularity.values()))

    def reward_func(completions, target_sid=None, target_category=None, **kwargs):
        rewards: List[float] = []
        target_sid = target_sid or kwargs.get("target_sid") or []
        target_category = target_category or kwargs.get("target_category") or []
        for idx, completion in enumerate(completions):
            text = normalize_completion(completion)
            preds = extract_sids(text)
            target = target_sid[idx] if isinstance(target_sid, list) else target_sid
            category = target_category[idx] if isinstance(target_category, list) else target_category
            reward = 0.0
            if target in preds:
                rank = preds.index(target) + 1
                reward += weights["hit"]
                reward += weights["ndcg"] * (1.0 / math.log2(rank + 1))
            elif preds:
                reward += weights.get("partial_match", 0.0) * max(shared_prefix_length(preds[0], target) - 1, 0) / 4.0
            if preds and preds[0] in valid_sids:
                reward += weights["valid_sid"]
            if preds:
                pred_cat = sid_to_item.get(preds[0], {}).get("category", "")
                if category and pred_cat == category:
                    reward += weights["category"]
                popularity_prob = max(1, popularity.get(preds[0], 0)) / total_popularity
                reward += weights.get("novelty", 0.0) * (-math.log2(popularity_prob) / 16.0)
                reward -= weights.get("popularity_bias", 0.0) * popularity_prob
            if len(set(preds[:5])) >= min(5, len(preds)):
                reward += weights["diversity"]
            if "推荐理由" in text and ("因为" in text or "因此" in text or "用户" in text):
                reward += weights["reasoning"]
            rewards.append(float(reward))
        return rewards

    return reward_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    require_training_stack()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    cfg = load_yaml(args.config)
    grpo_cfg = cfg["grpo"]
    train_cfg = cfg["training"]
    _item_to_sid, sid_to_item = load_sid_maps(args.sid_map)
    popularity = dict(item_popularity(read_jsonl(args.train)))
    output_dir = args.output_dir or grpo_cfg["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    dataset = load_dataset("json", data_files={"train": args.train})["train"]
    peft_config = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    reward_func = make_reward_func(sid_to_item, grpo_cfg["rewards"], popularity)
    args_grpo = GRPOConfig(
        output_dir=output_dir,
        max_prompt_length=int(grpo_cfg["max_prompt_length"]),
        max_completion_length=int(grpo_cfg["max_completion_length"]),
        num_generations=int(grpo_cfg["num_generations"]),
        per_device_train_batch_size=int(grpo_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(grpo_cfg["gradient_accumulation_steps"]),
        learning_rate=float(grpo_cfg["learning_rate"]),
        max_steps=int(grpo_cfg["max_steps"]),
        bf16=True,
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        deepspeed=args.deepspeed,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=args_grpo,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
