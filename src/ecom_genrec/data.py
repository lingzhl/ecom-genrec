from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from .utils import (
    JsonDict,
    first_present,
    normalize_category,
    read_jsonl,
    write_json,
    write_jsonl,
)


@dataclass
class DataStats:
    users: int
    items: int
    interactions: int
    train_samples: int
    valid_samples: int
    test_samples: int
    avg_history_len: float
    categories: int
    sparse_user_ratio: float
    long_tail_item_ratio: float

    def to_dict(self) -> JsonDict:
        return {
            "users": self.users,
            "items": self.items,
            "interactions": self.interactions,
            "train_samples": self.train_samples,
            "valid_samples": self.valid_samples,
            "test_samples": self.test_samples,
            "avg_history_len": round(self.avg_history_len, 4),
            "categories": self.categories,
            "sparse_user_ratio": round(self.sparse_user_ratio, 4),
            "long_tail_item_ratio": round(self.long_tail_item_ratio, 4),
        }


def normalize_review(row: JsonDict, category_hint: str = "") -> JsonDict:
    user_id = first_present(row, ["user_id", "reviewerID", "reviewer_id"])
    item_id = first_present(row, ["parent_asin", "asin", "item_id", "product_id"])
    timestamp = first_present(row, ["timestamp", "unixReviewTime", "time"], 0)
    rating = first_present(row, ["rating", "overall", "score"], None)
    return {
        "user_id": str(user_id) if user_id is not None else "",
        "item_id": str(item_id) if item_id is not None else "",
        "timestamp": int(timestamp or 0),
        "rating": float(rating) if rating is not None else None,
        "category": category_hint,
    }


def normalize_item(row: JsonDict, category_hint: str = "") -> JsonDict:
    item_id = first_present(row, ["parent_asin", "asin", "item_id", "product_id"])
    title = first_present(row, ["title", "name"], "")
    description = first_present(row, ["description", "details", "features"], "")
    if isinstance(description, list):
        description = " ".join(str(x) for x in description)
    if isinstance(description, dict):
        description = " ".join(f"{k}: {v}" for k, v in description.items())
    category = normalize_category(first_present(row, ["categories", "category", "main_category"], category_hint))
    price = first_present(row, ["price"], "")
    return {
        "item_id": str(item_id) if item_id is not None else "",
        "title": str(title or ""),
        "category": category or category_hint or "unknown",
        "description": str(description or ""),
        "price": str(price or ""),
    }


def load_reviews(paths: Sequence[str], categories: Sequence[str] | None = None) -> List[JsonDict]:
    rows: List[JsonDict] = []
    categories = categories or [""] * len(paths)
    for path, category in zip(paths, categories):
        for row in read_jsonl(path):
            item = normalize_review(row, category_hint=category)
            if item["user_id"] and item["item_id"]:
                rows.append(item)
    return rows


def load_items(paths: Sequence[str], categories: Sequence[str] | None = None) -> Dict[str, JsonDict]:
    items: Dict[str, JsonDict] = {}
    categories = categories or [""] * len(paths)
    for path, category in zip(paths, categories):
        for row in read_jsonl(path):
            item = normalize_item(row, category_hint=category)
            if item["item_id"] and item["item_id"] not in items:
                items[item["item_id"]] = item
    return items


def filter_interactions(rows: List[JsonDict], min_user: int, min_item: int) -> List[JsonDict]:
    filtered = rows
    while True:
        user_counts = Counter(r["user_id"] for r in filtered)
        item_counts = Counter(r["item_id"] for r in filtered)
        next_rows = [
            r
            for r in filtered
            if user_counts[r["user_id"]] >= min_user and item_counts[r["item_id"]] >= min_item
        ]
        if len(next_rows) == len(filtered):
            return next_rows
        filtered = next_rows


def split_by_user_time(rows: List[JsonDict], max_history: int) -> Tuple[List[JsonDict], List[JsonDict], List[JsonDict]]:
    by_user: Dict[str, List[JsonDict]] = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)

    train: List[JsonDict] = []
    valid: List[JsonDict] = []
    test: List[JsonDict] = []

    for user_id, seq in by_user.items():
        seq = sorted(seq, key=lambda x: (x.get("timestamp", 0), x["item_id"]))
        if len(seq) < 3:
            continue
        item_seq = [x["item_id"] for x in seq]
        for idx in range(1, len(seq) - 2):
            history = item_seq[:idx][-max_history:]
            train.append(make_sequence_row(user_id, history, item_seq[idx], seq[idx], len(seq)))
        valid_history = item_seq[: len(seq) - 2][-max_history:]
        test_history = item_seq[: len(seq) - 1][-max_history:]
        valid.append(make_sequence_row(user_id, valid_history, item_seq[-2], seq[-2], len(seq)))
        test.append(make_sequence_row(user_id, test_history, item_seq[-1], seq[-1], len(seq)))
    return train, valid, test


def make_sequence_row(
    user_id: str,
    history: List[str],
    target: str,
    target_row: JsonDict,
    full_history_len: int,
) -> JsonDict:
    return {
        "user_id": user_id,
        "history": history,
        "target_item": target,
        "target_category": target_row.get("category", "unknown"),
        "history_len": len(history),
        "full_history_len": full_history_len,
    }


def compute_stats(
    interactions: List[JsonDict],
    items: Dict[str, JsonDict],
    train: List[JsonDict],
    valid: List[JsonDict],
    test: List[JsonDict],
    cold_user_max_history: int,
    long_tail_quantile: float,
) -> DataStats:
    users = {r["user_id"] for r in interactions}
    item_ids = {r["item_id"] for r in interactions}
    item_counts = Counter(r["item_id"] for r in interactions)
    counts = sorted(item_counts.values())
    if counts:
        threshold_index = min(len(counts) - 1, max(0, int(len(counts) * long_tail_quantile)))
        long_tail_threshold = counts[threshold_index]
        long_tail_items = sum(1 for c in item_counts.values() if c <= long_tail_threshold)
        long_tail_ratio = long_tail_items / max(1, len(item_counts))
    else:
        long_tail_ratio = 0.0
    history_lengths = [r["history_len"] for r in test]
    sparse_ratio = sum(1 for x in history_lengths if x <= cold_user_max_history) / max(1, len(history_lengths))
    categories = {items.get(item_id, {}).get("category", "unknown") for item_id in item_ids}
    return DataStats(
        users=len(users),
        items=len(item_ids),
        interactions=len(interactions),
        train_samples=len(train),
        valid_samples=len(valid),
        test_samples=len(test),
        avg_history_len=mean(history_lengths) if history_lengths else 0.0,
        categories=len(categories),
        sparse_user_ratio=sparse_ratio,
        long_tail_item_ratio=long_tail_ratio,
    )


def write_processed(
    out_dir: str | Path,
    interactions: List[JsonDict],
    items: Dict[str, JsonDict],
    train: List[JsonDict],
    valid: List[JsonDict],
    test: List[JsonDict],
    stats: DataStats,
) -> None:
    out = Path(out_dir)
    write_jsonl(out / "interactions.jsonl", interactions)
    write_jsonl(out / "items.jsonl", items.values())
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)
    write_json(out / "stats.json", stats.to_dict())


def process_amazon(
    review_paths: Sequence[str],
    metadata_paths: Sequence[str],
    categories: Sequence[str],
    out_dir: str | Path,
    min_user: int,
    min_item: int,
    max_history: int,
    cold_user_max_history: int,
    long_tail_quantile: float,
) -> DataStats:
    interactions = load_reviews(review_paths, categories)
    items = load_items(metadata_paths, categories)
    for row in interactions:
        if row["item_id"] in items:
            row["category"] = items[row["item_id"]].get("category", row.get("category", "unknown"))
    filtered = filter_interactions(interactions, min_user=min_user, min_item=min_item)
    kept_items = {r["item_id"] for r in filtered}
    items = {k: v for k, v in items.items() if k in kept_items}
    train, valid, test = split_by_user_time(filtered, max_history=max_history)
    stats = compute_stats(filtered, items, train, valid, test, cold_user_max_history, long_tail_quantile)
    write_processed(out_dir, filtered, items, train, valid, test, stats)
    return stats
