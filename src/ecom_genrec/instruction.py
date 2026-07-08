from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .utils import JsonDict, read_json, read_jsonl, truncate_words, write_jsonl


SID_RE = re.compile(r"SID(?:_[0-9]{3})+(?:_[0-9]{3})?")


def load_sid_maps(path: str) -> Tuple[Dict[str, JsonDict], Dict[str, JsonDict]]:
    item_to_sid = read_json(path)
    sid_to_item = {row["sid"]: row for row in item_to_sid.values()}
    return item_to_sid, sid_to_item


def render_history(history: List[str], item_to_sid: Dict[str, JsonDict]) -> str:
    lines = []
    for idx, item_id in enumerate(history, start=1):
        meta = item_to_sid.get(item_id, {})
        sid = meta.get("sid", item_id)
        title = truncate_words(meta.get("title", item_id), 12)
        category = meta.get("category", "unknown").split(" > ")[0]
        lines.append(f"{idx}. {sid} | {title} | {category}")
    return "\n".join(lines)


def simple_reason(history: List[str], target_sid: str, item_to_sid: Dict[str, JsonDict]) -> str:
    cats = [item_to_sid.get(item_id, {}).get("category", "unknown").split(" > ")[0] for item_id in history]
    target_cat = item_to_sid.get(target_sid, {}).get("category", "unknown").split(" > ")[0]
    focus = cats[-1] if cats else target_cat
    return (
        f"用户近期多次关注 {focus} 相关商品，历史兴趣与 {target_cat} 类目保持一致，"
        "因此推荐该商品作为下一步候选。"
    )


def build_prompt(history: List[str], item_to_sid: Dict[str, JsonDict], with_reasoning: bool) -> str:
    task = "根据用户历史购买/评论商品，预测用户下一个可能感兴趣的商品"
    if with_reasoning:
        task += "，并给出简短推荐理由"
    return (
        f"Instruction:\n{task}。\n\n"
        f"Input:\n用户历史商品：\n{render_history(history, item_to_sid)}\n\n"
        "Output:\n"
    )


def build_completion(target_sid: str, reason: str | None) -> str:
    if reason:
        return f"推荐商品：{target_sid}\n推荐理由：{reason}"
    return f"推荐商品：{target_sid}"


def sequence_to_instruction(row: JsonDict, item_to_sid: Dict[str, JsonDict], with_reasoning: bool) -> JsonDict | None:
    target_item = row["target_item"]
    if target_item not in item_to_sid:
        return None
    history_items = [x for x in row.get("history", []) if x in item_to_sid]
    if not history_items:
        return None
    target_sid = item_to_sid[target_item]["sid"]
    history_sid = [item_to_sid[x]["sid"] for x in history_items]
    reason = simple_reason(history_items, target_item, item_to_sid) if with_reasoning else None
    prompt = build_prompt(history_items, item_to_sid, with_reasoning)
    completion = build_completion(target_sid, reason)
    return {
        "user_id": row["user_id"],
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "history_item": history_items,
        "history_sid": history_sid,
        "target_item": target_item,
        "target_sid": target_sid,
        "target_category": item_to_sid[target_item].get("category", "unknown"),
        "history_len": row.get("history_len", len(history_items)),
    }


def build_instruction_split(
    sequence_path: str,
    sid_map_path: str,
    out_path: str,
    with_reasoning: bool,
    limit: int | None = None,
) -> int:
    item_to_sid, _sid_to_item = load_sid_maps(sid_map_path)
    rows: List[JsonDict] = []
    for row in read_jsonl(sequence_path):
        inst = sequence_to_instruction(row, item_to_sid, with_reasoning=with_reasoning)
        if inst is not None:
            rows.append(inst)
        if limit is not None and len(rows) >= limit:
            break
    write_jsonl(out_path, rows)
    return len(rows)


def extract_sids(text: str) -> List[str]:
    seen = set()
    sids = []
    for match in SID_RE.findall(text or ""):
        if match not in seen:
            seen.add(match)
            sids.append(match)
    return sids


def completions_to_sid_lists(completions: Iterable[str], fallback: List[str], k: int) -> List[List[str]]:
    results: List[List[str]] = []
    for completion in completions:
        row = extract_sids(completion)
        for sid in fallback:
            if len(row) >= k:
                break
            if sid not in row:
                row.append(sid)
        results.append(row[:k])
    return results
