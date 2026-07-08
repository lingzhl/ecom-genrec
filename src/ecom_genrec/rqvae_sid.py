from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .semantic_id import _simple_cluster, encode_texts, item_text
from .utils import JsonDict, read_jsonl, write_json


def residual_quantized_labels(
    vectors: List[List[float]],
    levels: int,
    codebook_size: int,
) -> List[List[int]]:
    """Approximate RQ-VAE-style residual quantization with iterative residual KMeans."""
    if not vectors:
        return []
    residuals = [vec[:] for vec in vectors]
    labels_by_item: List[List[int]] = [[] for _ in vectors]
    for _level in range(levels):
        labels = _simple_cluster(residuals, codebook_size)
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            labels_by_item[idx].append(int(label))
            groups[int(label)].append(idx)
        centroids: Dict[int, List[float]] = {}
        for label, idxs in groups.items():
            dim = len(residuals[0])
            centroid = [0.0] * dim
            for item_idx in idxs:
                for dim_idx, value in enumerate(residuals[item_idx]):
                    centroid[dim_idx] += value
            centroids[label] = [value / len(idxs) for value in centroid]
        next_residuals: List[List[float]] = []
        for idx, label in enumerate(labels):
            centroid = centroids[int(label)]
            next_residuals.append([value - centroid[dim_idx] for dim_idx, value in enumerate(residuals[idx])])
        residuals = next_residuals
    return labels_by_item


def build_rqvae_style_sid_map(
    item_path: str,
    out_path: str,
    embedding_model: str,
    levels: int,
    codebook_size: int,
    sid_prefix: str,
    batch_size: int,
) -> Dict[str, JsonDict]:
    items = list(read_jsonl(item_path))
    texts = [item_text(item) for item in items]
    vectors = encode_texts(texts, embedding_model, batch_size)
    labels = residual_quantized_labels(vectors, levels=levels, codebook_size=codebook_size)
    collisions: Dict[str, int] = defaultdict(int)
    sid_map: Dict[str, JsonDict] = {}
    for item, item_labels in zip(items, labels):
        base_sid = sid_prefix + "_RQ_" + "_".join(f"{x:03d}" for x in item_labels)
        collisions[base_sid] += 1
        sid = base_sid if collisions[base_sid] == 1 else f"{base_sid}_{collisions[base_sid]:03d}"
        sid_map[item["item_id"]] = {
            "item_id": item["item_id"],
            "sid": sid,
            "title": item.get("title", ""),
            "category": item.get("category", "unknown"),
            "description": item.get("description", ""),
            "price": item.get("price", ""),
            "sid_source": "rqvae_style",
        }
    write_json(out_path, sid_map)
    return sid_map
