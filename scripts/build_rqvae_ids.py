#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_genrec.rqvae_sid import build_rqvae_style_sid_map
from ecom_genrec.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sid_cfg = cfg["semantic_id"]
    rq_cfg = sid_cfg["rqvae_style"]
    sid_map = build_rqvae_style_sid_map(
        item_path=args.items,
        out_path=args.out,
        embedding_model=sid_cfg["embedding_model"],
        levels=int(rq_cfg["levels"]),
        codebook_size=int(rq_cfg["codebook_size"]),
        sid_prefix=sid_cfg["sid_prefix"],
        batch_size=int(sid_cfg["batch_size"]),
    )
    print({"items": len(sid_map), "out": args.out, "method": "rqvae_style"})


if __name__ == "__main__":
    main()
