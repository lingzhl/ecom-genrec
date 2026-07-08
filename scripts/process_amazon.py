#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.data import process_amazon
from ecom_genrec.utils import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    categories = args.categories or cfg["data"]["categories"][: len(args.reviews)]
    stats = process_amazon(
        review_paths=args.reviews,
        metadata_paths=args.metadata,
        categories=categories,
        out_dir=args.out_dir,
        min_user=cfg["data"]["min_user_interactions"],
        min_item=cfg["data"]["min_item_interactions"],
        max_history=cfg["data"]["max_history"],
        cold_user_max_history=cfg["data"]["cold_user_max_history"],
        long_tail_quantile=cfg["data"]["long_tail_quantile"],
    )
    print(stats.to_dict())


if __name__ == "__main__":
    main()
