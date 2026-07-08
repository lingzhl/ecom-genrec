from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


JsonDict = Dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[JsonDict]:
    with open_text(path, "rt") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open_text(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open_text(path, "rt") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open_text(path, "wt") as f:
        json.dump(value, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def load_yaml(path: str | Path) -> JsonDict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def first_present(row: JsonDict, keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_category(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        flat: List[str] = []
        for item in value:
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
            else:
                flat.append(str(item))
        return " > ".join(x for x in flat if x) or "unknown"
    return str(value)


def truncate_words(text: str, max_words: int = 32) -> str:
    words = str(text or "").split()
    return " ".join(words[:max_words])


def batched(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def maybe_limit(rows: List[JsonDict], limit: Optional[int]) -> List[JsonDict]:
    if limit is None or limit < 0:
        return rows
    return rows[:limit]
