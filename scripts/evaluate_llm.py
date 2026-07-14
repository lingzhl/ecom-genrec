#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.instruction import extract_sids, load_sid_maps
from ecom_genrec.metrics import evaluate_predictions, item_popularity
from ecom_genrec.utils import load_yaml, maybe_limit, read_jsonl, write_json


def completions_to_sid_lists(
    completions: list[str],
    fallback: list[str],
    k: int,
    valid_sids: set[str],
    constrained_sid: bool,
) -> list[list[str]]:
    results = []
    for completion in completions:
        row = []
        for sid in extract_sids(completion):
            if constrained_sid and sid not in valid_sids:
                continue
            if sid not in row:
                row.append(sid)
        for sid in fallback:
            if len(row) >= k:
                break
            if sid not in row:
                row.append(sid)
        results.append(row[:k])
    return results


def generation_diagnostics(completions: list[str], valid_sids: set[str]) -> dict:
    total_sids = 0
    valid = 0
    empty = 0
    with_reason = 0
    for completion in completions:
        sids = extract_sids(completion)
        if not sids:
            empty += 1
        if "推荐理由" in completion:
            with_reason += 1
        total_sids += len(sids)
        valid += sum(1 for sid in sids if sid in valid_sids)
    total = max(1, len(completions))
    return {
        "RawGeneratedSIDAvg": round(total_sids / total, 6),
        "RawValidSIDRate": round(valid / max(1, total_sids), 6),
        "EmptySIDGenerationRate": round(empty / total, 6),
        "ReasonFieldRate": round(with_reason / total, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-reference", default=None)
    parser.add_argument(
        "--constrained-sid",
        action="store_true",
        help="Filter generated SIDs to the known catalog before fallback completion.",
    )
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
    popularity = dict(item_popularity(read_jsonl(args.train_reference))) if args.train_reference else {}

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

    valid_sids = set(sid_to_item)
    pred_lists = completions_to_sid_lists(
        completions,
        fallback=fallback,
        k=max(cfg["eval"]["k_values"]),
        valid_sids=valid_sids,
        constrained_sid=args.constrained_sid,
    )
    result = evaluate_predictions(
        rows,
        pred_lists,
        k_values=cfg["eval"]["k_values"],
        catalog=set(sid_to_item),
        valid_sids=valid_sids,
        sid_to_item=sid_to_item,
        popularity=popularity or None,
    )
    result.update(generation_diagnostics(completions, valid_sids))
    cold_threshold = cfg["eval"].get("cold_start_max_history", 5)
    cold_rows = [row for row in rows if row.get("history_len", 0) <= cold_threshold]
    cold_preds = [pred for row, pred in zip(rows, pred_lists) if row.get("history_len", 0) <= cold_threshold]
    cold_result = (
        evaluate_predictions(
            cold_rows,
            cold_preds,
            k_values=cfg["eval"]["k_values"],
            catalog=set(sid_to_item),
            valid_sids=set(sid_to_item),
            sid_to_item=sid_to_item,
            popularity=popularity or None,
        )
        if cold_rows
        else {"samples": 0}
    )
    result["model"] = args.model
    result["samples"] = len(rows)
    result["constrained_sid"] = args.constrained_sid
    write_json(args.out, {"metrics": result, "cold_start_metrics": cold_result, "examples": completions[:5]})
    print(result)


if __name__ == "__main__":
    main()
