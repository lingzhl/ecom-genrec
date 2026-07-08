#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.instruction import build_instruction_split
from ecom_genrec.one_rec_tasks import build_onerec_instruction_splits
from ecom_genrec.utils import ensure_dir, load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-reasoning", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--task-mix", choices=["legacy", "onerec"], default="onerec")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = ensure_dir(args.out_dir)
    max_train = args.max_train_samples
    if max_train is None:
        max_train = cfg["data"].get("max_train_samples")

    if args.task_mix == "legacy":
        counts = {
            "train": build_instruction_split(
                str(Path(args.processed_dir) / "train.jsonl"),
                args.sid_map,
                str(out / "sft_train.jsonl"),
                with_reasoning=args.with_reasoning,
                limit=max_train,
            ),
            "valid": build_instruction_split(
                str(Path(args.processed_dir) / "valid.jsonl"),
                args.sid_map,
                str(out / "sft_valid.jsonl"),
                with_reasoning=args.with_reasoning,
            ),
            "test": build_instruction_split(
                str(Path(args.processed_dir) / "test.jsonl"),
                args.sid_map,
                str(out / "sft_test.jsonl"),
                with_reasoning=args.with_reasoning,
            ),
        }
        build_instruction_split(
            str(Path(args.processed_dir) / "train.jsonl"),
            args.sid_map,
            str(out / "grpo_train.jsonl"),
            with_reasoning=True,
            limit=max_train,
        )
    else:
        counts = build_onerec_instruction_splits(
            processed_dir=args.processed_dir,
            sid_map_path=args.sid_map,
            out_dir=str(out),
            with_reasoning=args.with_reasoning,
            limit=max_train,
        )
        build_instruction_split(
            str(Path(args.processed_dir) / "train.jsonl"),
            args.sid_map,
            str(out / "grpo_train.jsonl"),
            with_reasoning=True,
            limit=max_train,
        )
    write_json(out / "counts.json", counts)
    print(counts)


if __name__ == "__main__":
    main()
