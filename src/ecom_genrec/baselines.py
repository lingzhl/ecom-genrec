from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .metrics import evaluate_predictions
from .utils import JsonDict, read_json, read_jsonl


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def load_sid_map(path: str) -> tuple[Dict[str, JsonDict], Dict[str, JsonDict]]:
    item_to_sid = read_json(path)
    sid_to_item = {row["sid"]: row for row in item_to_sid.values()}
    return item_to_sid, sid_to_item


def popular_predictions(train_rows: Sequence[JsonDict], test_rows: Sequence[JsonDict], k: int) -> List[List[str]]:
    counts = Counter()
    for row in train_rows:
        counts[row["target_sid"]] += 1
        counts.update(row.get("history_sid", []))
    popular = [sid for sid, _count in counts.most_common(k)]
    return [popular[:] for _ in test_rows]


def item_document(item: JsonDict) -> str:
    return " ".join(
        [
            str(item.get("sid", "")),
            str(item.get("title", "")),
            str(item.get("category", "")),
            str(item.get("description", "")),
        ]
    )


def text_retrieval_predictions(
    train_rows: Sequence[JsonDict],
    test_rows: Sequence[JsonDict],
    sid_to_item: Dict[str, JsonDict],
    k: int,
) -> List[List[str]]:
    candidate_sids = sorted({sid for row in train_rows for sid in [row["target_sid"], *row.get("history_sid", [])]})
    docs = {sid: tokenize(item_document(sid_to_item.get(sid, {}))) for sid in candidate_sids}
    df = Counter()
    for terms in docs.values():
        df.update(set(terms))
    total_docs = max(1, len(docs))
    idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    predictions: List[List[str]] = []
    for row in test_rows:
        query_terms = []
        for sid in row.get("history_sid", [])[-5:]:
            query_terms.extend(docs.get(sid, []))
        query = Counter(query_terms)
        scores = []
        history = set(row.get("history_sid", []))
        for sid, terms in docs.items():
            if sid in history:
                continue
            tf = Counter(terms)
            score = sum(query[t] * tf.get(t, 0) * idf.get(t, 1.0) for t in query)
            if score > 0:
                scores.append((score, sid))
        scores.sort(reverse=True)
        ranked = [sid for _score, sid in scores[:k]]
        if len(ranked) < k:
            ranked.extend([sid for sid in candidate_sids if sid not in ranked and sid not in history][: k - len(ranked)])
        predictions.append(ranked)
    return predictions


def embedding_retrieval_predictions(
    train_rows: Sequence[JsonDict],
    test_rows: Sequence[JsonDict],
    sid_to_item: Dict[str, JsonDict],
    k: int,
) -> List[List[str]]:
    try:
        from .semantic_id import encode_texts, item_text

        candidate_sids = sorted({sid for row in train_rows for sid in [row["target_sid"], *row.get("history_sid", [])]})
        texts = [item_text(sid_to_item.get(sid, {})) for sid in candidate_sids]
        vectors = encode_texts(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=128)

        def dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        sid_to_vec = dict(zip(candidate_sids, vectors))
        predictions = []
        for row in test_rows:
            history = [sid for sid in row.get("history_sid", []) if sid in sid_to_vec]
            if history:
                query = [sum(sid_to_vec[sid][i] for sid in history) / len(history) for i in range(len(vectors[0]))]
            else:
                query = [0.0] * len(vectors[0])
            history_set = set(history)
            ranked = sorted(
                ((dot(query, sid_to_vec[sid]), sid) for sid in candidate_sids if sid not in history_set),
                reverse=True,
            )
            predictions.append([sid for _score, sid in ranked[:k]])
        return predictions
    except Exception:
        return text_retrieval_predictions(train_rows, test_rows, sid_to_item, k)


def long_tail_items_from_rows(rows: Sequence[JsonDict], quantile: float = 0.8) -> set[str]:
    counts = Counter()
    for row in rows:
        counts[row["target_sid"]] += 1
        counts.update(row.get("history_sid", []))
    if not counts:
        return set()
    sorted_counts = sorted(counts.values())
    idx = min(len(sorted_counts) - 1, max(0, int(len(sorted_counts) * quantile)))
    threshold = sorted_counts[idx]
    return {sid for sid, count in counts.items() if count <= threshold}


def evaluate_all_baselines(
    train_path: str,
    test_path: str,
    sid_map_path: str,
    k_values: Sequence[int],
) -> JsonDict:
    train_rows = list(read_jsonl(train_path))
    test_rows = list(read_jsonl(test_path))
    _item_to_sid, sid_to_item = load_sid_map(sid_map_path)
    catalog = set(sid_to_item)
    long_tail = long_tail_items_from_rows(train_rows)
    max_k = max(k_values)
    methods = {
        "Popular": popular_predictions(train_rows, test_rows, max_k),
        "TextRetrieval": text_retrieval_predictions(train_rows, test_rows, sid_to_item, max_k),
        "EmbeddingRetrieval": embedding_retrieval_predictions(train_rows, test_rows, sid_to_item, max_k),
    }
    return {
        name: evaluate_predictions(
            test_rows,
            preds,
            k_values=k_values,
            catalog=catalog,
            valid_sids=catalog,
            sid_to_item=sid_to_item,
            long_tail_items=long_tail,
        )
        for name, preds in methods.items()
    }
