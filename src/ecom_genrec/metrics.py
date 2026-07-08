from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .utils import JsonDict


def hit_at_k(preds: Sequence[str], target: str, k: int) -> float:
    return 1.0 if target in list(preds)[:k] else 0.0


def ndcg_at_k(preds: Sequence[str], target: str, k: int) -> float:
    for idx, item in enumerate(list(preds)[:k], start=1):
        if item == target:
            return 1.0 / math.log2(idx + 1)
    return 0.0


def mrr_at_k(preds: Sequence[str], target: str, k: int) -> float:
    for idx, item in enumerate(list(preds)[:k], start=1):
        if item == target:
            return 1.0 / idx
    return 0.0


def catalog_coverage(pred_lists: Iterable[Sequence[str]], catalog: Set[str], k: int) -> float:
    predicted: Set[str] = set()
    for preds in pred_lists:
        predicted.update(list(preds)[:k])
    return len(predicted & catalog) / max(1, len(catalog))


def valid_sid_rate(pred_lists: Iterable[Sequence[str]], valid_sids: Set[str], k: int) -> float:
    total = 0
    valid = 0
    for preds in pred_lists:
        for item in list(preds)[:k]:
            total += 1
            if item in valid_sids:
                valid += 1
    return valid / max(1, total)


def long_tail_ratio(pred_lists: Iterable[Sequence[str]], long_tail_items: Set[str], k: int) -> float:
    total = 0
    tail = 0
    for preds in pred_lists:
        for item in list(preds)[:k]:
            total += 1
            if item in long_tail_items:
                tail += 1
    return tail / max(1, total)


def category_consistency(
    pred_lists: Iterable[Sequence[str]],
    targets: Sequence[str],
    sid_to_item: Dict[str, JsonDict],
    k: int,
) -> float:
    total = 0
    same = 0
    for preds, target in zip(pred_lists, targets):
        target_cat = sid_to_item.get(target, {}).get("category", "unknown")
        for pred in list(preds)[:k]:
            total += 1
            if sid_to_item.get(pred, {}).get("category", "unknown") == target_cat:
                same += 1
    return same / max(1, total)


def evaluate_predictions(
    rows: Sequence[JsonDict],
    pred_lists: Sequence[Sequence[str]],
    k_values: Sequence[int],
    catalog: Optional[Set[str]] = None,
    valid_sids: Optional[Set[str]] = None,
    sid_to_item: Optional[Dict[str, JsonDict]] = None,
    long_tail_items: Optional[Set[str]] = None,
) -> JsonDict:
    targets = [r["target_sid"] if "target_sid" in r else r["target_item"] for r in rows]
    result: JsonDict = {"samples": len(rows)}
    for k in k_values:
        result[f"HR@{k}"] = sum(hit_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))
        result[f"NDCG@{k}"] = sum(ndcg_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))
        result[f"MRR@{k}"] = sum(mrr_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))
        if catalog is not None:
            result[f"Coverage@{k}"] = catalog_coverage(pred_lists, catalog, k)
        if valid_sids is not None:
            result[f"ValidSID@{k}"] = valid_sid_rate(pred_lists, valid_sids, k)
        if sid_to_item is not None:
            result[f"CategoryConsistency@{k}"] = category_consistency(pred_lists, targets, sid_to_item, k)
        if long_tail_items is not None:
            result[f"LongTailRatio@{k}"] = long_tail_ratio(pred_lists, long_tail_items, k)
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in result.items()}


def item_popularity(rows: Iterable[JsonDict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        if "target_sid" in row:
            counts[row["target_sid"]] += 1
        elif "target_item" in row:
            counts[row["target_item"]] += 1
        for item in row.get("history_sid", row.get("history", [])):
            counts[item] += 1
    return counts
