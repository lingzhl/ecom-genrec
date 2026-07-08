#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.baselines import evaluate_all_baselines
from ecom_genrec.utils import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--sid-map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    inst_dir = Path(args.processed_dir) / "instructions"
    train_path = inst_dir / "sft_train.jsonl"
    test_path = inst_dir / "sft_test.jsonl"
    if not train_path.exists():
        train_path = Path(args.processed_dir) / "sft_train.jsonl"
        test_path = Path(args.processed_dir) / "sft_test.jsonl"
    result = evaluate_all_baselines(
        str(train_path),
        str(test_path),
        args.sid_map,
        k_values=cfg["eval"]["k_values"],
    )
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
