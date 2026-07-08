#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.instruction import completions_to_sid_lists, load_sid_maps
from ecom_genrec.metrics import evaluate_predictions
from ecom_genrec.utils import load_yaml, maybe_limit, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: pip install -r requirements.txt") from exc

    cfg = load_yaml(args.config)
    rows = maybe_limit(list(read_jsonl(args.eval)), args.max_samples or cfg["eval"].get("max_eval_samples"))
    _item_to_sid, sid_to_item = load_sid_maps(args.sid_map)
    fallback = sorted(sid_to_item)[: max(cfg["eval"]["k_values"])]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()

    completions = []
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    pred_lists = completions_to_sid_lists(completions, fallback=fallback, k=max(cfg["eval"]["k_values"]))
    result = evaluate_predictions(
        rows,
        pred_lists,
        k_values=cfg["eval"]["k_values"],
        catalog=set(sid_to_item),
        valid_sids=set(sid_to_item),
        sid_to_item=sid_to_item,
    )
    result["model"] = args.model
    result["samples"] = len(rows)
    write_json(args.out, {"metrics": result, "examples": completions[:5]})
    print(result)


if __name__ == "__main__":
    main()
