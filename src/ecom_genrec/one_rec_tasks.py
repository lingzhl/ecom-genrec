from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .instruction import build_completion, build_prompt, load_sid_maps, render_history, simple_reason
from .utils import JsonDict, read_jsonl, truncate_words, write_json, write_jsonl


def sequence_task(row: JsonDict, item_to_sid: Dict[str, JsonDict], with_reasoning: bool) -> JsonDict | None:
    target_item = row["target_item"]
    if target_item not in item_to_sid:
        return None
    history_items = [x for x in row.get("history", []) if x in item_to_sid]
    if not history_items:
        return None
    target_sid = item_to_sid[target_item]["sid"]
    reason = simple_reason(history_items, target_item, item_to_sid) if with_reasoning else None
    prompt = build_prompt(history_items, item_to_sid, with_reasoning)
    completion = build_completion(target_sid, reason)
    return {
        "task_type": "sequence_recommendation",
        "user_id": row["user_id"],
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "history_item": history_items,
        "history_sid": [item_to_sid[x]["sid"] for x in history_items],
        "target_item": target_item,
        "target_sid": target_sid,
        "target_category": item_to_sid[target_item].get("category", "unknown"),
        "history_len": row.get("history_len", len(history_items)),
    }


def feature_alignment_tasks(items_path: str, sid_map_path: str, limit: int | None = None) -> List[JsonDict]:
    item_to_sid, _sid_to_item = load_sid_maps(sid_map_path)
    rows: List[JsonDict] = []
    for idx, item in enumerate(read_jsonl(items_path)):
        meta = item_to_sid.get(item["item_id"])
        if not meta:
            continue
        prompt = (
            "Instruction:\n根据商品特征预测对应的语义商品ID。\n\n"
            f'Input:\n商品标题：{truncate_words(meta.get("title", ""), 24)}\n'
            f'商品类目：{meta.get("category", "unknown")}\n'
            f'商品描述：{truncate_words(meta.get("description", ""), 48)}\n\n'
            "Output:\n"
        )
        completion = f'推荐商品：{meta["sid"]}'
        rows.append(
            {
                "task_type": "feature_alignment",
                "item_id": item["item_id"],
                "prompt": prompt,
                "completion": completion,
                "text": prompt + completion,
                "target_item": item["item_id"],
                "target_sid": meta["sid"],
                "target_category": meta.get("category", "unknown"),
                "history_len": 0,
            }
        )
        if limit is not None and idx + 1 >= limit:
            break
    return rows


def history_fusion_task(row: JsonDict, item_to_sid: Dict[str, JsonDict]) -> JsonDict | None:
    target_item = row["target_item"]
    if target_item not in item_to_sid:
        return None
    history_items = [x for x in row.get("history", []) if x in item_to_sid]
    if not history_items:
        return None
    target_sid = item_to_sid[target_item]["sid"]
    history_block = render_history(history_items, item_to_sid)
    latest_titles = ", ".join(truncate_words(item_to_sid[x].get("title", ""), 6) for x in history_items[-3:])
    prompt = (
        "Instruction:\n结合用户历史商品序列和近期商品特征，预测下一个商品语义ID，并给出简短推荐理由。\n\n"
        f"Input:\n用户历史商品：\n{history_block}\n\n"
        f"近期兴趣摘要：{latest_titles}\n\n"
        "Output:\n"
    )
    completion = build_completion(target_sid, simple_reason(history_items, target_item, item_to_sid))
    return {
        "task_type": "history_fusion",
        "user_id": row["user_id"],
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "history_item": history_items,
        "history_sid": [item_to_sid[x]["sid"] for x in history_items],
        "target_item": target_item,
        "target_sid": target_sid,
        "target_category": item_to_sid[target_item].get("category", "unknown"),
        "history_len": row.get("history_len", len(history_items)),
    }


def build_onerec_instruction_splits(
    processed_dir: str,
    sid_map_path: str,
    out_dir: str,
    with_reasoning: bool,
    limit: int | None = None,
) -> Dict[str, int]:
    out = Path(out_dir)
    item_to_sid, _sid_to_item = load_sid_maps(sid_map_path)
    items_path = str(Path(processed_dir) / "items.jsonl")
    counts: Dict[str, int] = {}
    for split in ["train", "valid", "test"]:
        seq_rows = list(read_jsonl(Path(processed_dir) / f"{split}.jsonl"))
        seq_data = [x for row in seq_rows if (x := sequence_task(row, item_to_sid, with_reasoning))]
        fusion_data = [x for row in seq_rows if (x := history_fusion_task(row, item_to_sid))]
        if limit is not None and split == "train":
            seq_data = seq_data[:limit]
            fusion_data = fusion_data[:limit]
        write_jsonl(out / f"sequence_{split}.jsonl", seq_data)
        write_jsonl(out / f"history_fusion_{split}.jsonl", fusion_data)
        counts[f"sequence_{split}"] = len(seq_data)
        counts[f"history_fusion_{split}"] = len(fusion_data)
    feature_data = feature_alignment_tasks(items_path, sid_map_path, limit=limit)
    split_idx = max(1, int(0.9 * len(feature_data))) if feature_data else 0
    write_jsonl(out / "feature_alignment_train.jsonl", feature_data[:split_idx])
    write_jsonl(out / "feature_alignment_valid.jsonl", feature_data[split_idx:])
    write_jsonl(out / "feature_alignment_test.jsonl", feature_data[split_idx:])
    counts["feature_alignment_train"] = len(feature_data[:split_idx])
    counts["feature_alignment_valid"] = len(feature_data[split_idx:])
    counts["feature_alignment_test"] = len(feature_data[split_idx:])
    for split in ["train", "valid", "test"]:
        merged: List[JsonDict] = []
        for prefix in ["sequence", "feature_alignment", "history_fusion"]:
            merged.extend(list(read_jsonl(out / f"{prefix}_{split}.jsonl")))
        write_jsonl(out / f"sft_{split}.jsonl", merged)
        counts[f"sft_{split}"] = len(merged)
    write_json(out / "counts.json", counts)
    return counts
