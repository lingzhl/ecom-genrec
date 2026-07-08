from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Sequence

from .utils import JsonDict, read_jsonl, truncate_words, write_json


def item_text(item: JsonDict) -> str:
    return " ".join(
        [
            str(item.get("title", "")),
            str(item.get("category", "")),
            truncate_words(str(item.get("description", "")), 48),
            str(item.get("price", "")),
        ]
    ).strip()


def hashed_embedding(text: str, dim: int = 128) -> List[float]:
    values = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[idx] += sign
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return [x / norm for x in values]


def encode_texts(texts: Sequence[str], model_name: str, batch_size: int) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return vectors.tolist()
    except Exception:
        return [hashed_embedding(text) for text in texts]


def _simple_cluster(vectors: List[List[float]], n_clusters: int) -> List[int]:
    if not vectors:
        return []
    try:
        from sklearn.cluster import MiniBatchKMeans

        k = min(n_clusters, len(vectors))
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=max(256, k * 8), n_init="auto")
        return model.fit_predict(vectors).tolist()
    except Exception:
        labels = []
        for vec in vectors:
            best_idx = max(range(len(vec)), key=lambda i: abs(vec[i]))
            labels.append(best_idx % max(1, min(n_clusters, len(vectors))))
        return labels


def hierarchical_labels(vectors: List[List[float]], levels: int, clusters_per_level: int) -> List[List[int]]:
    labels_by_item: List[List[int]] = [[] for _ in vectors]
    groups = {(): list(range(len(vectors)))}
    for _level in range(levels):
        next_groups: Dict[tuple, List[int]] = defaultdict(list)
        for prefix, idxs in groups.items():
            subset = [vectors[i] for i in idxs]
            labels = _simple_cluster(subset, clusters_per_level)
            for local_idx, label in enumerate(labels):
                item_idx = idxs[local_idx]
                labels_by_item[item_idx].append(int(label))
                next_groups[prefix + (int(label),)].append(item_idx)
        groups = next_groups
    return labels_by_item


def build_sid_map(
    item_path: str,
    out_path: str,
    embedding_model: str,
    levels: int,
    clusters_per_level: int,
    sid_prefix: str,
    batch_size: int,
) -> Dict[str, JsonDict]:
    items = list(read_jsonl(item_path))
    texts = [item_text(item) for item in items]
    vectors = encode_texts(texts, embedding_model, batch_size)
    labels = hierarchical_labels(vectors, levels=levels, clusters_per_level=clusters_per_level)

    collisions: Dict[str, int] = defaultdict(int)
    sid_map: Dict[str, JsonDict] = {}
    for item, item_labels in zip(items, labels):
        base_sid = sid_prefix + "_" + "_".join(f"{x:03d}" for x in item_labels)
        collisions[base_sid] += 1
        sid = base_sid if collisions[base_sid] == 1 else f"{base_sid}_{collisions[base_sid]:03d}"
        sid_map[item["item_id"]] = {
            "item_id": item["item_id"],
            "sid": sid,
            "title": item.get("title", ""),
            "category": item.get("category", "unknown"),
            "description": item.get("description", ""),
            "price": item.get("price", ""),
        }
    write_json(out_path, sid_map)
    return sid_map
