from __future__ import annotations  # 允许在类型注解里使用更现代的写法，例如 list[str] 风格

from collections import Counter, defaultdict  # Counter 用来计数；defaultdict 用来按 user_id 自动分组
from dataclasses import dataclass  # dataclass 用来快速定义只保存数据的统计结果类
from pathlib import Path  # Path 用来更安全地拼接和处理文件路径
from statistics import mean  # mean 用来计算平均历史长度
from typing import Dict, List, Sequence, Tuple  # 常用类型注解，让输入输出更清楚

from .utils import (  # 从本项目工具模块导入通用函数
    JsonDict,  # JsonDict 表示一条 JSON 数据，实际就是 Dict[str, Any]
    first_present,  # 从多个候选字段名里取第一个存在的字段
    normalize_category,  # 把 Amazon metadata 里的类目字段统一转成字符串
    read_jsonl,  # 逐行读取 jsonl/jsonl.gz 文件
    write_json,  # 写 JSON 文件
    write_jsonl,  # 写 JSONL 文件
)


@dataclass  # 自动生成 __init__，让 DataStats 只负责保存统计字段
class DataStats:  # 数据处理完成后的统计信息，后面会写入 stats.json
    users: int  # 用户数
    items: int  # 商品数
    interactions: int  # 交互数，也就是 review/行为记录数
    train_samples: int  # 训练样本数
    valid_samples: int  # 验证样本数
    test_samples: int  # 测试样本数
    avg_history_len: float  # 测试集中每个用户平均历史长度
    categories: int  # 商品类目数量
    sparse_user_ratio: float  # 稀疏用户比例，例如历史长度 <=5 的用户比例
    long_tail_item_ratio: float  # 长尾商品比例，用商品交互次数分位数估算

    def to_dict(self) -> JsonDict:  # 把 dataclass 转成普通 dict，方便写 JSON
        return {  # 返回可以序列化的统计字典
            "users": self.users,  # 用户数
            "items": self.items,  # 商品数
            "interactions": self.interactions,  # 总交互数
            "train_samples": self.train_samples,  # 训练样本数
            "valid_samples": self.valid_samples,  # 验证样本数
            "test_samples": self.test_samples,  # 测试样本数
            "avg_history_len": round(self.avg_history_len, 4),  # 平均历史长度，保留 4 位小数
            "categories": self.categories,  # 类目数量
            "sparse_user_ratio": round(self.sparse_user_ratio, 4),  # 稀疏用户比例，保留 4 位小数
            "long_tail_item_ratio": round(self.long_tail_item_ratio, 4),  # 长尾商品比例，保留 4 位小数
        }


def normalize_review(row: JsonDict, category_hint: str = "") -> JsonDict:  # 把 review 文件里一条用户行为标准化
    """把 Amazon review 行统一成 user_id/item_id/timestamp/rating/category。"""
    user_id = first_present(row, ["user_id", "reviewerID", "reviewer_id"])  # review 文件提供用户是谁
    item_id = first_present(row, ["parent_asin", "asin", "item_id", "product_id"])  # review 文件提供用户交互了哪个商品
    timestamp = first_present(row, ["timestamp", "unixReviewTime", "time"], 0)  # review 文件提供交互发生时间
    rating = first_present(row, ["rating", "overall", "score"], None)  # review 文件可能提供评分
    return {  # 返回项目内部统一格式
        "user_id": str(user_id) if user_id is not None else "",  # 统一转成字符串；缺失则为空
        "item_id": str(item_id) if item_id is not None else "",  # 统一转成字符串；缺失则为空
        "timestamp": int(timestamp or 0),  # 时间戳转成整数，后面按它排序
        "rating": float(rating) if rating is not None else None,  # 评分转成 float；没有评分就保留 None
        "category": category_hint,  # review 本身不一定有类目，先用外部传入的类目名兜底
    }


def normalize_item(row: JsonDict, category_hint: str = "") -> JsonDict:  # 把 metadata 文件里一条商品信息标准化
    """把 Amazon metadata 行统一成 item_id/title/category/description/price。"""
    item_id = first_present(row, ["parent_asin", "asin", "item_id", "product_id"])  # metadata 提供商品 ID
    title = first_present(row, ["title", "name"], "")  # metadata 提供商品标题
    description = first_present(row, ["description", "details", "features"], "")  # metadata 提供商品描述/详情/特征
    if isinstance(description, list):  # 有些 Amazon 字段是列表，例如 features
        description = " ".join(str(x) for x in description)  # 列表转成一段文本，方便 embedding 和 prompt 使用
    if isinstance(description, dict):  # 有些 Amazon 字段是字典，例如 details
        description = " ".join(f"{k}: {v}" for k, v in description.items())  # 字典转成 key: value 文本
    category = normalize_category(first_present(row, ["categories", "category", "main_category"], category_hint))  # 统一类目格式
    price = first_present(row, ["price"], "")  # 读取价格字段；没有就为空
    return {  # 返回项目内部统一商品格式
        "item_id": str(item_id) if item_id is not None else "",  # 商品 ID，后面要和 review 的 item_id 对齐
        "title": str(title or ""),  # 商品标题，是构造 Semantic ID 的重要文本
        "category": category or category_hint or "unknown",  # 商品类目，优先用 metadata，缺失则兜底
        "description": str(description or ""),  # 商品描述，是构造 Semantic ID 的重要文本
        "price": str(price or ""),  # 商品价格，作为附加商品信息
    }


def load_reviews(paths: Sequence[str], categories: Sequence[str] | None = None) -> List[JsonDict]:  # 读取一个或多个 review 文件
    """读取 review 文件；review 文件提供用户行为序列的原材料。"""
    rows: List[JsonDict] = []  # 保存所有标准化后的用户行为
    categories = categories or [""] * len(paths)  # 如果没传类目名，就给每个文件一个空类目兜底
    for path, category in zip(paths, categories):  # 一个 review 文件通常对应一个 Amazon 类目
        for row in read_jsonl(path):  # 逐行读取 review jsonl/jsonl.gz
            item = normalize_review(row, category_hint=category)  # 把原始字段名统一成内部字段名
            if item["user_id"] and item["item_id"]:  # 只有用户和商品都存在，才是一条有效交互
                rows.append(item)  # 保存有效用户行为
    return rows  # 返回所有用户行为，用于后续过滤和时间切分


def load_items(paths: Sequence[str], categories: Sequence[str] | None = None) -> Dict[str, JsonDict]:  # 读取 metadata 文件
    """读取 metadata 文件；metadata 文件提供商品标题、类目、描述等信息。"""
    items: Dict[str, JsonDict] = {}  # 保存 item_id -> 商品信息
    categories = categories or [""] * len(paths)  # 如果没传类目名，就给每个文件一个空类目兜底
    for path, category in zip(paths, categories):  # 一个 metadata 文件通常对应一个 Amazon 类目
        for row in read_jsonl(path):  # 逐行读取 metadata jsonl/jsonl.gz
            item = normalize_item(row, category_hint=category)  # 把原始商品字段统一成内部字段名
            if item["item_id"] and item["item_id"] not in items:  # 商品 ID 存在且未重复时才保存
                items[item["item_id"]] = item  # 建立 item_id -> metadata 的查表字典
    return items  # 返回商品字典，后续用于构造 Semantic ID 和 prompt


def filter_interactions(rows: List[JsonDict], min_user: int, min_item: int) -> List[JsonDict]:  # 过滤过稀疏的用户和商品
    """反复过滤交互太少的用户/商品，让序列推荐样本更稳定。"""
    filtered = rows  # 从全部用户行为开始过滤
    while True:  # 需要循环过滤，因为删掉一些商品后，某些用户交互数也可能变少
        user_counts = Counter(r["user_id"] for r in filtered)  # 统计每个用户有多少条行为
        item_counts = Counter(r["item_id"] for r in filtered)  # 统计每个商品被交互多少次
        next_rows = [  # 构造下一轮保留下来的行为
            r  # 当前行为记录
            for r in filtered  # 遍历当前保留的所有行为
            if user_counts[r["user_id"]] >= min_user and item_counts[r["item_id"]] >= min_item  # 用户和商品都要够频繁
        ]
        if len(next_rows) == len(filtered):  # 如果这一轮没有再删任何记录，说明过滤已经稳定
            return next_rows  # 返回最终过滤后的交互
        filtered = next_rows  # 否则继续下一轮过滤，直到稳定


def split_by_user_time(rows: List[JsonDict], max_history: int) -> Tuple[List[JsonDict], List[JsonDict], List[JsonDict]]:
    """按 user_id 分组、按 timestamp 排序，用过去行为预测未来商品。"""
    by_user: Dict[str, List[JsonDict]] = defaultdict(list)  # user_id -> 该用户的所有行为
    for row in rows:  # 遍历每条用户行为
        by_user[row["user_id"]].append(row)  # 按 user_id 分组，这是构造用户历史序列的第一步

    train: List[JsonDict] = []  # 训练集：用较早的历史预测较早的未来
    valid: List[JsonDict] = []  # 验证集：用倒数第 3 个及以前预测倒数第 2 个
    test: List[JsonDict] = []  # 测试集：用倒数第 2 个及以前预测最后 1 个

    for user_id, seq in by_user.items():  # 对每个用户单独处理，不能把不同用户的行为混在一起
        seq = sorted(seq, key=lambda x: (x.get("timestamp", 0), x["item_id"]))  # 按 timestamp 排序，保证历史在前、未来在后
        if len(seq) < 3:  # 少于 3 条行为无法同时构造 train/valid/test
            continue  # 跳过过短序列
        item_seq = [x["item_id"] for x in seq]  # 只取商品 ID 序列，例如 [A, B, C, D]
        for idx in range(1, len(seq) - 2):  # 用前面的多个切点生成训练样本，最后两条留给 valid/test
            history = item_seq[:idx][-max_history:]  # 历史只来自 idx 之前的商品，并截断到最大长度
            train.append(make_sequence_row(user_id, history, item_seq[idx], seq[idx], len(seq)))  # 训练目标是历史后的下一个商品
        valid_history = item_seq[: len(seq) - 2][-max_history:]  # 验证历史不能包含倒数第 2 个和最后 1 个
        test_history = item_seq[: len(seq) - 1][-max_history:]  # 测试历史可以包含倒数第 2 个，但不能包含最后 1 个
        valid.append(make_sequence_row(user_id, valid_history, item_seq[-2], seq[-2], len(seq)))  # 验证目标是倒数第 2 个商品
        test.append(make_sequence_row(user_id, test_history, item_seq[-1], seq[-1], len(seq)))  # 测试目标是最后 1 个商品
    return train, valid, test  # 返回按时间构造好的三份推荐样本


def make_sequence_row(  # 把一个“历史 -> 目标商品”的样本封装成统一格式
    user_id: str,  # 用户 ID
    history: List[str],  # 历史商品 ID 列表
    target: str,  # 要预测的下一个商品 ID
    target_row: JsonDict,  # 目标商品对应的原始行为记录，里面可能有类目等信息
    full_history_len: int,  # 该用户完整行为序列长度
) -> JsonDict:  # 返回一条推荐样本
    return {  # 统一样本格式
        "user_id": user_id,  # 当前样本属于哪个用户
        "history": history,  # 输入：用户过去交互过的商品
        "target_item": target,  # 输出/标签：用户未来交互的商品
        "target_category": target_row.get("category", "unknown"),  # 目标商品类目，用于统计或 reward
        "history_len": len(history),  # 当前样本可见的历史长度
        "full_history_len": full_history_len,  # 用户完整历史长度，用于分群分析
    }


def compute_stats(  # 计算处理后数据集的统计信息
    interactions: List[JsonDict],  # 过滤后的全部用户行为
    items: Dict[str, JsonDict],  # item_id -> 商品 metadata
    train: List[JsonDict],  # 训练样本
    valid: List[JsonDict],  # 验证样本
    test: List[JsonDict],  # 测试样本
    cold_user_max_history: int,  # 历史长度小于等于这个值的用户视为稀疏/冷启动用户
    long_tail_quantile: float,  # 用商品交互次数的分位点定义长尾商品
) -> DataStats:  # 返回 DataStats 对象
    users = {r["user_id"] for r in interactions}  # 去重统计用户数
    item_ids = {r["item_id"] for r in interactions}  # 去重统计商品数
    item_counts = Counter(r["item_id"] for r in interactions)  # 统计每个商品出现次数
    counts = sorted(item_counts.values())  # 商品交互次数从小到大排序
    if counts:  # 如果存在商品交互
        threshold_index = min(len(counts) - 1, max(0, int(len(counts) * long_tail_quantile)))  # 找分位点下标
        long_tail_threshold = counts[threshold_index]  # 分位点对应的交互次数阈值
        long_tail_items = sum(1 for c in item_counts.values() if c <= long_tail_threshold)  # 低于阈值的商品数
        long_tail_ratio = long_tail_items / max(1, len(item_counts))  # 长尾商品比例
    else:  # 如果没有商品交互
        long_tail_ratio = 0.0  # 长尾比例设为 0，避免除零
    history_lengths = [r["history_len"] for r in test]  # 取测试集中每个样本的历史长度
    sparse_ratio = sum(1 for x in history_lengths if x <= cold_user_max_history) / max(1, len(history_lengths))  # 稀疏用户比例
    categories = {items.get(item_id, {}).get("category", "unknown") for item_id in item_ids}  # 统计商品类目集合
    return DataStats(  # 汇总所有统计字段
        users=len(users),  # 用户数
        items=len(item_ids),  # 商品数
        interactions=len(interactions),  # 行为记录数
        train_samples=len(train),  # 训练样本数
        valid_samples=len(valid),  # 验证样本数
        test_samples=len(test),  # 测试样本数
        avg_history_len=mean(history_lengths) if history_lengths else 0.0,  # 测试样本平均历史长度
        categories=len(categories),  # 类目数量
        sparse_user_ratio=sparse_ratio,  # 稀疏用户比例
        long_tail_item_ratio=long_tail_ratio,  # 长尾商品比例
    )


def write_processed(  # 把处理后的数据写到磁盘，供后续 SID/SFT/baseline 使用
    out_dir: str | Path,  # 输出目录
    interactions: List[JsonDict],  # 过滤后的全部交互
    items: Dict[str, JsonDict],  # 商品 metadata
    train: List[JsonDict],  # 训练样本
    valid: List[JsonDict],  # 验证样本
    test: List[JsonDict],  # 测试样本
    stats: DataStats,  # 数据统计
) -> None:  # 只写文件，不返回值
    out = Path(out_dir)  # 把字符串路径转成 Path
    write_jsonl(out / "interactions.jsonl", interactions)  # 写过滤后的用户行为
    write_jsonl(out / "items.jsonl", items.values())  # 写商品 metadata
    write_jsonl(out / "train.jsonl", train)  # 写训练样本
    write_jsonl(out / "valid.jsonl", valid)  # 写验证样本
    write_jsonl(out / "test.jsonl", test)  # 写测试样本
    write_json(out / "stats.json", stats.to_dict())  # 写数据统计


def process_amazon(  # 真实 Amazon 数据处理入口
    review_paths: Sequence[str],  # review 文件路径列表，提供用户行为
    metadata_paths: Sequence[str],  # metadata 文件路径列表，提供商品信息
    categories: Sequence[str],  # 每个文件对应的类目名
    out_dir: str | Path,  # 输出目录
    min_user: int,  # 最少用户交互数过滤阈值
    min_item: int,  # 最少商品交互数过滤阈值
    max_history: int,  # 每条样本最多保留多少个历史商品
    cold_user_max_history: int,  # 稀疏用户历史长度阈值
    long_tail_quantile: float,  # 长尾商品分位点
) -> DataStats:  # 返回处理后的统计结果
    interactions = load_reviews(review_paths, categories)  # 从 review 文件读取用户行为
    items = load_items(metadata_paths, categories)  # 从 metadata 文件读取商品信息
    for row in interactions:  # 遍历每条用户行为
        if row["item_id"] in items:  # 如果该行为的商品在 metadata 里存在
            row["category"] = items[row["item_id"]].get("category", row.get("category", "unknown"))  # 用 metadata 类目补全行为类目
    filtered = filter_interactions(interactions, min_user=min_user, min_item=min_item)  # 过滤过稀疏的用户和商品
    kept_items = {r["item_id"] for r in filtered}  # 找到过滤后仍然出现的商品
    items = {k: v for k, v in items.items() if k in kept_items}  # 商品 metadata 也同步过滤，避免保留无用商品
    train, valid, test = split_by_user_time(filtered, max_history=max_history)  # 按用户时间序列构造推荐样本
    stats = compute_stats(filtered, items, train, valid, test, cold_user_max_history, long_tail_quantile)  # 计算统计信息
    write_processed(out_dir, filtered, items, train, valid, test, stats)  # 写出处理后的数据文件
    return stats  # 返回统计结果，命令行脚本会打印出来
