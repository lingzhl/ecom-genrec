from __future__ import annotations  # 允许使用较新的类型注解写法

import math  # 用于 NDCG 里的 log2 计算
from collections import Counter  # 用于统计商品出现次数
from typing import Dict, Iterable, Optional, Sequence, Set  # 常用类型注解

from .utils import JsonDict  # 项目内 JSON 字典类型别名


def hit_at_k(preds: Sequence[str], target: str, k: int) -> float:  # 计算单个样本的 HR@K
    """HR@K：真实商品 target 是否出现在 Top-K 推荐列表里。"""
    return 1.0 if target in list(preds)[:k] else 0.0  # 命中返回 1，否则返回 0


def ndcg_at_k(preds: Sequence[str], target: str, k: int) -> float:  # 计算单个样本的 NDCG@K
    """NDCG@K：不仅看是否命中，还奖励更靠前的位置。"""
    for idx, item in enumerate(list(preds)[:k], start=1):  # 遍历 Top-K 推荐，idx 从 1 开始表示排名
        if item == target:  # 如果当前位置命中真实商品
            return 1.0 / math.log2(idx + 1)  # 排名越靠前分越高；第 1 名得 1/log2(2)=1
    return 0.0  # Top-K 里没命中则得 0


def mrr_at_k(preds: Sequence[str], target: str, k: int) -> float:  # 计算单个样本的 MRR@K
    """MRR@K：第一个命中位置的倒数，越早命中越好。"""
    for idx, item in enumerate(list(preds)[:k], start=1):  # 遍历 Top-K 推荐
        if item == target:  # 找到第一个命中的位置
            return 1.0 / idx  # 第 1 名命中得 1，第 2 名命中得 1/2，第 10 名命中得 1/10
    return 0.0  # Top-K 里没命中则得 0


def catalog_coverage(pred_lists: Iterable[Sequence[str]], catalog: Set[str], k: int) -> float:  # 计算 Coverage@K
    """Coverage@K：所有推荐结果覆盖了商品库里的多少比例。"""
    predicted: Set[str] = set()  # 保存所有样本 Top-K 推荐中出现过的商品
    for preds in pred_lists:  # 遍历每个用户/样本的推荐列表
        predicted.update(list(preds)[:k])  # 只统计 Top-K 推荐商品
    return len(predicted & catalog) / max(1, len(catalog))  # 覆盖商品数 / 商品库总数，max 防止除零


def valid_sid_rate(pred_lists: Iterable[Sequence[str]], valid_sids: Set[str], k: int) -> float:  # 计算 ValidSID@K
    """ValidSID@K：模型生成的 Top-K SID 中，有多少是真实存在于商品库的合法 SID。"""
    total = 0  # 统计 Top-K 推荐总数量
    valid = 0  # 统计合法 SID 数量
    for preds in pred_lists:  # 遍历每个样本的推荐列表
        for item in list(preds)[:k]:  # 只看 Top-K
            total += 1  # 推荐总数加 1
            if item in valid_sids:  # 如果这个 SID 在合法商品集合里
                valid += 1  # 合法数量加 1
    return valid / max(1, total)  # 合法 SID 比例，max 防止除零


def long_tail_ratio(pred_lists: Iterable[Sequence[str]], long_tail_items: Set[str], k: int) -> float:  # 计算长尾推荐比例
    """LongTailRatio@K：Top-K 推荐中有多少比例是长尾商品。"""
    total = 0  # 推荐总数
    tail = 0  # 长尾商品推荐数
    for preds in pred_lists:  # 遍历每个样本的推荐列表
        for item in list(preds)[:k]:  # 只看 Top-K
            total += 1  # 推荐总数加 1
            if item in long_tail_items:  # 如果推荐商品属于长尾集合
                tail += 1  # 长尾数量加 1
    return tail / max(1, total)  # 长尾推荐比例


def category_consistency(  # 计算推荐商品和真实目标商品的类目一致性
    pred_lists: Iterable[Sequence[str]],  # 每个样本的 Top-K 推荐列表
    targets: Sequence[str],  # 每个样本的真实目标 SID
    sid_to_item: Dict[str, JsonDict],  # SID -> 商品 metadata，里面有 category
    k: int,  # Top-K
) -> float:  # 返回类目一致比例
    """CategoryConsistency@K：推荐商品类目和真实目标商品类目一致的比例。"""
    total = 0  # 统计 Top-K 推荐总数量
    same = 0  # 统计类目一致数量
    for preds, target in zip(pred_lists, targets):  # 同时遍历推荐列表和真实目标
        target_cat = sid_to_item.get(target, {}).get("category", "unknown")  # 真实目标商品类目
        for pred in list(preds)[:k]:  # 遍历 Top-K 推荐
            total += 1  # 推荐总数加 1
            if sid_to_item.get(pred, {}).get("category", "unknown") == target_cat:  # 推荐商品类目是否等于目标类目
                same += 1  # 类目一致数量加 1
    return same / max(1, total)  # 类目一致比例


def evaluate_predictions(  # 统一评测入口
    rows: Sequence[JsonDict],  # 测试样本，每条里面有 target_sid 或 target_item
    pred_lists: Sequence[Sequence[str]],  # 每个测试样本对应的 Top-K 推荐列表
    k_values: Sequence[int],  # 要评测的 K 值，例如 [5, 10, 20]
    catalog: Optional[Set[str]] = None,  # 商品全集，用于 Coverage
    valid_sids: Optional[Set[str]] = None,  # 合法 SID 集合，用于 ValidSID
    sid_to_item: Optional[Dict[str, JsonDict]] = None,  # SID -> metadata，用于类目一致性
    long_tail_items: Optional[Set[str]] = None,  # 长尾商品集合，用于 LongTailRatio
) -> JsonDict:  # 返回指标字典
    """对一组预测结果统一计算 HR/NDCG/MRR/Coverage/ValidSID 等指标。"""
    targets = [r["target_sid"] if "target_sid" in r else r["target_item"] for r in rows]  # 真实答案列表，优先用 target_sid
    result: JsonDict = {"samples": len(rows)}  # 先记录评测样本数
    for k in k_values:  # 对每个 K 分别计算指标
        result[f"HR@{k}"] = sum(hit_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))  # 平均 HR@K
        result[f"NDCG@{k}"] = sum(ndcg_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))  # 平均 NDCG@K
        result[f"MRR@{k}"] = sum(mrr_at_k(p, t, k) for p, t in zip(pred_lists, targets)) / max(1, len(rows))  # 平均 MRR@K
        if catalog is not None:  # 如果提供了商品全集
            result[f"Coverage@{k}"] = catalog_coverage(pred_lists, catalog, k)  # 计算覆盖率
        if valid_sids is not None:  # 如果提供了合法 SID 集合
            result[f"ValidSID@{k}"] = valid_sid_rate(pred_lists, valid_sids, k)  # 计算合法 SID 比例
        if sid_to_item is not None:  # 如果提供了商品 metadata
            result[f"CategoryConsistency@{k}"] = category_consistency(pred_lists, targets, sid_to_item, k)  # 计算类目一致性
        if long_tail_items is not None:  # 如果提供了长尾商品集合
            result[f"LongTailRatio@{k}"] = long_tail_ratio(pred_lists, long_tail_items, k)  # 计算长尾推荐比例
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in result.items()}  # 浮点数保留 6 位，方便写报告


def item_popularity(rows: Iterable[JsonDict]) -> Counter:  # 统计样本里每个商品/SID 的出现次数
    """统计 target 和 history 中商品的出现频次，可用于分析热门/长尾。"""
    counts: Counter = Counter()  # 初始化计数器
    for row in rows:  # 遍历样本
        if "target_sid" in row:  # instruction 数据通常有 target_sid
            counts[row["target_sid"]] += 1  # 目标 SID 计数
        elif "target_item" in row:  # 原始 sequence 数据可能只有 target_item
            counts[row["target_item"]] += 1  # 目标 item_id 计数
        for item in row.get("history_sid", row.get("history", [])):  # 统计历史商品，优先用 history_sid，否则用 history
            counts[item] += 1  # 历史商品计数
    return counts  # 返回商品频次 Counter
