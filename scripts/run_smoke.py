#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.baselines import evaluate_all_baselines
from ecom_genrec.data import compute_stats, filter_interactions, split_by_user_time, write_processed
from ecom_genrec.instruction import build_instruction_split
from ecom_genrec.semantic_id import build_sid_map
from ecom_genrec.utils import ensure_dir, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def synthetic_items():
    categories = ["Beauty", "Baby", "Sports", "Health", "Electronics", "Home"]
    rows = []
    for idx in range(48):
        cat = categories[idx % len(categories)]
        rows.append(
            {
                "item_id": f"ITEM_{idx:03d}",
                "title": f"{cat} product {idx:03d}",
                "category": cat,
                "description": f"A useful {cat.lower()} product with strong user interest signal {idx % 7}.",
                "price": str(9.99 + idx),
            }
        )
    return {row["item_id"]: row for row in rows}


def synthetic_interactions(items):
    item_ids = list(items)
    rows = []
    ts = 1
    for user_idx in range(60):
        preferred = user_idx % 6
        pool = [item for item in item_ids if int(item.split("_")[1]) % 6 == preferred]
        for step in range(8):
            item = pool[(user_idx + step) % len(pool)]
            rows.append(
                {
                    "user_id": f"USER_{user_idx:03d}",
                    "item_id": item,
                    "timestamp": ts,
                    "rating": 5.0 if step % 3 else 4.0,
                    "category": items[item]["category"],
                }
            )
            ts += 1
    return rows


def write_summary(stats, baselines, out_path: Path) -> None:
    lines = [
        "# Smoke Test Summary",
        "",
        "## Data Stats",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in stats.to_dict().items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Baseline Metrics", "", "| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |", "|---|---:|---:|---:|---:|"])
    for name, values in baselines.items():
        lines.append(
            f"| {name} | {values.get('HR@10', 0):.4f} | {values.get('NDCG@10', 0):.4f} | "
            f"{values.get('MRR@10', 0):.4f} | {values.get('Coverage@10', 0):.4f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    processed = ROOT / "data/processed/smoke"
    artifacts = ROOT / "artifacts/smoke"
    reports = ROOT / "reports/smoke"
    ensure_dir(processed)
    ensure_dir(artifacts)
    ensure_dir(reports)

    items = synthetic_items()
    interactions = synthetic_interactions(items)
    filtered = filter_interactions(interactions, min_user=5, min_item=1)
    train, valid, test = split_by_user_time(filtered, max_history=20)
    stats = compute_stats(filtered, items, train, valid, test, cold_user_max_history=5, long_tail_quantile=0.8)
    write_processed(processed, filtered, items, train, valid, test, stats)

    sid_map_path = artifacts / "sid_map.json"
    build_sid_map(
        item_path=str(processed / "items.jsonl"),
        out_path=str(sid_map_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        levels=3,
        clusters_per_level=8,
        sid_prefix="SID",
        batch_size=32,
    )

    inst_dir = processed / "instructions"
    ensure_dir(inst_dir)
    counts = {
        "train": build_instruction_split(str(processed / "train.jsonl"), str(sid_map_path), str(inst_dir / "sft_train.jsonl"), True),
        "valid": build_instruction_split(str(processed / "valid.jsonl"), str(sid_map_path), str(inst_dir / "sft_valid.jsonl"), True),
        "test": build_instruction_split(str(processed / "test.jsonl"), str(sid_map_path), str(inst_dir / "sft_test.jsonl"), True),
    }
    write_json(inst_dir / "counts.json", counts)

    baselines = evaluate_all_baselines(
        train_path=str(inst_dir / "sft_train.jsonl"),
        test_path=str(inst_dir / "sft_test.jsonl"),
        sid_map_path=str(sid_map_path),
        k_values=[5, 10, 20],
    )
    write_json(reports / "baselines.json", baselines)
    write_summary(stats, baselines, reports / "summary.md")
    print({"stats": stats.to_dict(), "counts": counts, "report": str(reports / "summary.md")})


if __name__ == "__main__":
    main()
