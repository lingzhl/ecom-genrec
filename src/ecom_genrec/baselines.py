from __future__ import annotations  # 允许使用较新的类型注解写法

import math  # 用于计算 IDF 里的 log
import re  # 用正则做简单英文/数字分词
from collections import Counter  # Counter 用于统计热门商品、词频、商品频次
from typing import Dict, List, Sequence  # 常用类型注解

from .metrics import evaluate_predictions  # 统一计算 HR/NDCG/MRR/Coverage/ValidSID 等指标
from .utils import JsonDict, read_json, read_jsonl  # JSON/JSONL 读取工具


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")  # 匹配英文和数字 token，用于文本检索 baseline


def tokenize(text: str) -> List[str]:  # 简单分词函数
    """把商品文本转成小写 token 列表，用于 Text Retrieval。"""
    return TOKEN_RE.findall((text or "").lower())  # 空文本兜底为 ""，再提取 token


def load_sid_map(path: str) -> tuple[Dict[str, JsonDict], Dict[str, JsonDict]]:  # 加载 SID 映射
    """读取 sid_map.json，同时返回 item_id->metadata 和 sid->metadata。"""
    item_to_sid = read_json(path)  # sid_map.json 的原始结构：item_id -> 商品信息 + SID
    sid_to_item = {row["sid"]: row for row in item_to_sid.values()}  # 反向索引：SID -> 商品信息
    return item_to_sid, sid_to_item  # 返回两个查表字典


def popular_predictions(train_rows: Sequence[JsonDict], test_rows: Sequence[JsonDict], k: int) -> List[List[str]]:
    """Popular baseline：不看用户个性化，永远推荐训练集中最热门的 Top-K 商品。"""
    counts = Counter()  # 统计每个 SID 在训练数据里出现了多少次
    for row in train_rows:  # 遍历训练样本
        counts[row["target_sid"]] += 1  # 目标商品出现一次，说明它是用户真实未来行为
        counts.update(row.get("history_sid", []))  # 历史商品也计入热度，表示它们在用户行为里常出现
    popular = [sid for sid, _count in counts.most_common(k)]  # 取出现次数最多的 Top-K SID
    return [popular[:] for _ in test_rows]  # 对每个测试用户都返回同一份热门商品列表


def item_document(item: JsonDict) -> str:  # 把一个商品组织成可检索文档
    """把商品 SID/title/category/description 拼成一段文本，用于文本相似度检索。"""
    return " ".join(  # 用空格拼接商品字段
        [
            str(item.get("sid", "")),  # 商品 SID，本身也可作为文本信号
            str(item.get("title", "")),  # 商品标题
            str(item.get("category", "")),  # 商品类目
            str(item.get("description", "")),  # 商品描述
        ]
    )


def text_retrieval_predictions(  # 文本检索 baseline
    train_rows: Sequence[JsonDict],  # 训练 instruction 样本
    test_rows: Sequence[JsonDict],  # 测试 instruction 样本
    sid_to_item: Dict[str, JsonDict],  # SID -> 商品 metadata
    k: int,  # 推荐 Top-K
) -> List[List[str]]:  # 返回每个测试样本的 Top-K SID
    """Text Retrieval baseline：用用户历史商品文本作为 query，检索文本最相似的商品。"""
    candidate_sids = sorted({sid for row in train_rows for sid in [row["target_sid"], *row.get("history_sid", [])]})  # 候选商品池
    docs = {sid: tokenize(item_document(sid_to_item.get(sid, {}))) for sid in candidate_sids}  # 每个候选商品的 token 文档
    df = Counter()  # document frequency：每个词出现在多少个商品文档里
    for terms in docs.values():  # 遍历每个商品文档的 token 列表
        df.update(set(terms))  # 一个词在同一商品里出现多次也只算一个文档
    total_docs = max(1, len(docs))  # 商品文档总数，至少为 1 避免除零
    idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}  # 计算平滑 IDF

    predictions: List[List[str]] = []  # 保存每个测试样本的推荐列表
    for row in test_rows:  # 遍历测试样本
        query_terms = []  # 用户 query 的 token 列表
        for sid in row.get("history_sid", [])[-5:]:  # 只取最近 5 个历史商品，模拟“近期兴趣”
            query_terms.extend(docs.get(sid, []))  # 把历史商品文本 token 加入 query
        query = Counter(query_terms)  # 统计 query 词频
        scores = []  # 保存候选商品的检索分数
        history = set(row.get("history_sid", []))  # 用户已经交互过的商品，推荐时排除
        for sid, terms in docs.items():  # 遍历每个候选商品
            if sid in history:  # 如果候选商品已经在用户历史里
                continue  # 不推荐用户已经交互过的商品
            tf = Counter(terms)  # 候选商品文档词频
            score = sum(query[t] * tf.get(t, 0) * idf.get(t, 1.0) for t in query)  # 简化 TF-IDF 相似度
            if score > 0:  # 只保留和用户历史有文本重叠的商品
                scores.append((score, sid))  # 保存分数和 SID
        scores.sort(reverse=True)  # 按分数从高到低排序
        ranked = [sid for _score, sid in scores[:k]]  # 取 Top-K SID
        if len(ranked) < k:  # 如果文本检索结果不足 K 个
            ranked.extend([sid for sid in candidate_sids if sid not in ranked and sid not in history][: k - len(ranked)])  # 用候选池补齐
        predictions.append(ranked)  # 保存当前测试用户的推荐列表
    return predictions  # 返回所有测试样本的推荐结果


def embedding_retrieval_predictions(  # 向量检索 baseline
    train_rows: Sequence[JsonDict],  # 训练 instruction 样本
    test_rows: Sequence[JsonDict],  # 测试 instruction 样本
    sid_to_item: Dict[str, JsonDict],  # SID -> 商品 metadata
    k: int,  # 推荐 Top-K
) -> List[List[str]]:  # 返回每个测试样本的 Top-K SID
    """Embedding Retrieval baseline：用商品 embedding 做相似召回，通常比纯文本重叠更强。"""
    try:  # 优先使用 semantic_id.py 里的真实 embedding/fallback embedding
        from .semantic_id import encode_texts, item_text  # 复用商品文本拼接和 embedding 编码逻辑

        candidate_sids = sorted({sid for row in train_rows for sid in [row["target_sid"], *row.get("history_sid", [])]})  # 候选商品池
        texts = [item_text(sid_to_item.get(sid, {})) for sid in candidate_sids]  # 把每个候选商品转成文本
        vectors = encode_texts(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=128)  # 商品文本 -> embedding

        def dot(a: List[float], b: List[float]) -> float:  # 两个向量的点积相似度
            return sum(x * y for x, y in zip(a, b))  # embedding 已归一化时，点积近似 cosine 相似度

        sid_to_vec = dict(zip(candidate_sids, vectors))  # SID -> embedding 向量
        predictions = []  # 保存每个测试样本的推荐列表
        for row in test_rows:  # 遍历测试样本
            history = [sid for sid in row.get("history_sid", []) if sid in sid_to_vec]  # 只保留有向量的历史 SID
            if history:  # 如果用户有可用历史
                query = [sum(sid_to_vec[sid][i] for sid in history) / len(history) for i in range(len(vectors[0]))]  # 历史商品向量平均作为用户兴趣向量
            else:  # 如果没有历史向量
                query = [0.0] * len(vectors[0])  # 用全 0 向量兜底
            history_set = set(history)  # 用户已交互商品集合，用于排除
            ranked = sorted(  # 对所有候选商品按向量相似度排序
                ((dot(query, sid_to_vec[sid]), sid) for sid in candidate_sids if sid not in history_set),  # 排除历史商品
                reverse=True,  # 分数从高到低
            )
            predictions.append([sid for _score, sid in ranked[:k]])  # 取 Top-K SID
        return predictions  # 返回 embedding 检索推荐结果
    except Exception:  # 如果 embedding 依赖、模型下载或编码出错
        return text_retrieval_predictions(train_rows, test_rows, sid_to_item, k)  # 回退到 Text Retrieval，保证评测链路可跑


def long_tail_items_from_rows(rows: Sequence[JsonDict], quantile: float = 0.8) -> set[str]:  # 计算长尾商品集合
    """按训练集出现频次定义长尾商品，用于评估推荐是否只推热门。"""
    counts = Counter()  # 统计 SID 频次
    for row in rows:  # 遍历训练样本
        counts[row["target_sid"]] += 1  # 目标 SID 计数
        counts.update(row.get("history_sid", []))  # 历史 SID 也计数
    if not counts:  # 如果没有商品
        return set()  # 返回空集合
    sorted_counts = sorted(counts.values())  # 频次从小到大排序
    idx = min(len(sorted_counts) - 1, max(0, int(len(sorted_counts) * quantile)))  # 找分位点下标
    threshold = sorted_counts[idx]  # 长尾阈值
    return {sid for sid, count in counts.items() if count <= threshold}  # 频次低于阈值的商品视为长尾


def evaluate_all_baselines(  # baseline 总入口
    train_path: str,  # 训练 instruction 文件路径
    test_path: str,  # 测试 instruction 文件路径
    sid_map_path: str,  # SID 映射文件路径
    k_values: Sequence[int],  # 要评估的 K，例如 [5, 10, 20]
) -> JsonDict:  # 返回每个 baseline 的指标字典
    """统一跑 Popular/Text/Embedding 三个 baseline，并用同一套指标评测。"""
    train_rows = list(read_jsonl(train_path))  # 读取训练 instruction 样本
    test_rows = list(read_jsonl(test_path))  # 读取测试 instruction 样本
    _item_to_sid, sid_to_item = load_sid_map(sid_map_path)  # 加载 SID -> 商品信息，item_to_sid 此处不用
    catalog = set(sid_to_item)  # 商品全集，也就是所有合法 SID
    long_tail = long_tail_items_from_rows(train_rows)  # 根据训练集频次定义长尾商品
    max_k = max(k_values)  # 三个 baseline 先生成最大的 K，再由指标函数切 @5/@10/@20
    methods = {  # 统一保存每个方法的预测结果
        "Popular": popular_predictions(train_rows, test_rows, max_k),  # 热门商品 baseline
        "TextRetrieval": text_retrieval_predictions(train_rows, test_rows, sid_to_item, max_k),  # 文本检索 baseline
        "EmbeddingRetrieval": embedding_retrieval_predictions(train_rows, test_rows, sid_to_item, max_k),  # 向量检索 baseline
    }
    return {  # 对每个 baseline 统一计算指标
        name: evaluate_predictions(  # 复用 metrics.py 的统一评测函数
            test_rows,  # 测试样本，里面有真实 target_sid
            preds,  # 当前 baseline 的 Top-K 推荐列表
            k_values=k_values,  # 评测 @5/@10/@20
            catalog=catalog,  # 商品全集，用于 Coverage
            valid_sids=catalog,  # 合法 SID 集合，用于 ValidSID
            sid_to_item=sid_to_item,  # 商品信息，用于 CategoryConsistency
            long_tail_items=long_tail,  # 长尾商品集合，用于 LongTailRatio
        )
        for name, preds in methods.items()  # 遍历 Popular/Text/Embedding 三个 baseline
    }
