#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.baselines import load_sid_map, text_retrieval_predictions
from ecom_genrec.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Small retrieval demo before LLM training.")
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    rows = list(read_jsonl(args.instructions))
    item_to_sid, sid_to_item = load_sid_map(args.sid_map)
    row = rows[args.index]
    preds = text_retrieval_predictions(rows, [row], sid_to_item, k=10)[0]
    print(row["prompt"])
    print("Top-10 recommendations:")
    for rank, sid in enumerate(preds, start=1):
        item = sid_to_item.get(sid, {})
        print(f"{rank:02d}. {sid} | {item.get('title', '')} | {item.get('category', '')}")


if __name__ == "__main__":
    main()
