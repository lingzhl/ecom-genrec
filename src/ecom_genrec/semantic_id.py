from __future__ import annotations  # 允许更现代的类型注解写法

import hashlib  # 用于哈希 fallback embedding，保证没装模型依赖时也能跑 smoke test
import math  # 用于计算向量 L2 norm
from collections import defaultdict  # 用于统计 SID 冲突次数，以及按层级聚类分组
from typing import Dict, List, Sequence  # 常用类型注解

from .utils import JsonDict, read_jsonl, truncate_words, write_json  # 项目内 JSONL/JSON 工具和文本截断工具


def item_text(item: JsonDict) -> str:  # 把一条商品 metadata 拼成用于 embedding 的文本
    """把商品 title/category/description/price 拼成一段语义文本。"""
    return " ".join(  # 用空格连接各个字段，形成商品文本表示
        [
            str(item.get("title", "")),  # 商品标题，通常是最重要的语义信息
            str(item.get("category", "")),  # 商品类目，帮助 embedding 区分商品大类
            truncate_words(str(item.get("description", "")), 48),  # 商品描述，只保留前 48 个词，避免太长
            str(item.get("price", "")),  # 价格作为附加信息，帮助区分部分商品
        ]
    ).strip()  # 去掉首尾空格，得到最终商品文本


def hashed_embedding(text: str, dim: int = 128) -> List[float]:  # 无深度学习依赖时的哈希向量 fallback
    """用哈希技巧把文本转成固定维度向量，保证项目没装 embedding 模型时仍可跑通。"""
    values = [0.0] * dim  # 初始化一个 dim 维全 0 向量
    for token in text.lower().split():  # 把商品文本按空格切词，并统一小写
        digest = hashlib.md5(token.encode("utf-8")).digest()  # 对 token 做 MD5，得到稳定哈希
        idx = int.from_bytes(digest[:4], "little") % dim  # 用哈希前 4 字节决定 token 落到哪个维度
        sign = 1.0 if digest[4] % 2 == 0 else -1.0  # 用第 5 个字节决定加正数还是负数，减少碰撞偏差
        values[idx] += sign  # 把 token 信息累加到对应维度
    norm = math.sqrt(sum(x * x for x in values)) or 1.0  # 计算 L2 norm；如果全 0 则用 1 防止除零
    return [x / norm for x in values]  # 返回归一化向量，方便后续聚类


def encode_texts(texts: Sequence[str], model_name: str, batch_size: int) -> List[List[float]]:  # 商品文本 -> embedding
    """优先用 SentenceTransformer/BGE 生成 embedding；失败时回退到哈希向量。"""
    try:  # 尝试使用真正的语义 embedding 模型
        from sentence_transformers import SentenceTransformer  # 延迟导入，没装依赖时不会影响其他脚本

        model = SentenceTransformer(model_name)  # 加载配置里的 embedding 模型，例如 BAAI/bge-small-en-v1.5
        vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)  # 批量编码文本
        return vectors.tolist()  # numpy array 转成普通 Python list，方便后续处理和 JSON 兼容
    except Exception:  # 如果没装 sentence-transformers、模型下载失败、GPU 不可用等，就进入 fallback
        return [hashed_embedding(text) for text in texts]  # 用哈希向量替代真实 embedding，保证 smoke test 可复现


def _simple_cluster(vectors: List[List[float]], n_clusters: int) -> List[int]:  # 对一组向量做一层聚类
    """对当前组内商品做 KMeans；失败时用简单规则生成稳定标签。"""
    if not vectors:  # 如果没有向量
        return []  # 返回空标签列表
    try:  # 优先使用 sklearn 的 MiniBatchKMeans
        from sklearn.cluster import MiniBatchKMeans  # 延迟导入，没装 sklearn 时可 fallback

        k = min(n_clusters, len(vectors))  # 聚类数不能超过样本数
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=max(256, k * 8), n_init="auto")  # 小批量 KMeans
        return model.fit_predict(vectors).tolist()  # 返回每个商品所属 cluster id
    except Exception:  # 如果 sklearn 不可用或聚类失败，使用确定性 fallback
        labels = []  # 保存 fallback 标签
        for vec in vectors:  # 遍历每个向量
            best_idx = max(range(len(vec)), key=lambda i: abs(vec[i]))  # 找绝对值最大的维度，作为粗略语义信号
            labels.append(best_idx % max(1, min(n_clusters, len(vectors))))  # 映射到合法 cluster id 范围
        return labels  # 返回 fallback 聚类标签


def hierarchical_labels(vectors: List[List[float]], levels: int, clusters_per_level: int) -> List[List[int]]:  # 多层聚类
    """用层级 KMeans 生成多级标签，例如 [5, 2, 7] -> SID_005_002_007。"""
    labels_by_item: List[List[int]] = [[] for _ in vectors]  # 每个商品对应一个标签列表，长度最终等于 levels
    groups = {(): list(range(len(vectors)))}  # 第一层从一个根组开始，里面包含所有商品下标
    for _level in range(levels):  # 逐层聚类，例如 level 0、level 1、level 2
        next_groups: Dict[tuple, List[int]] = defaultdict(list)  # 保存下一层要继续细分的组
        for prefix, idxs in groups.items():  # 遍历当前层的每个组；prefix 是该组已有 SID 前缀
            subset = [vectors[i] for i in idxs]  # 取出这个组里的商品向量
            labels = _simple_cluster(subset, clusters_per_level)  # 在组内再做一层 KMeans 聚类
            for local_idx, label in enumerate(labels):  # 遍历组内每个商品得到的新标签
                item_idx = idxs[local_idx]  # local_idx 是组内下标，item_idx 是原始商品全局下标
                labels_by_item[item_idx].append(int(label))  # 把这一层标签追加到该商品的标签序列
                next_groups[prefix + (int(label),)].append(item_idx)  # 相同前缀+标签的商品进入下一层同一组
        groups = next_groups  # 当前层结束后，把下一层分组作为新的待细分分组
    return labels_by_item  # 返回每个商品的多级标签，例如 [[5,5,5], [1,3,7], ...]


def build_sid_map(  # 构建 item_id -> Semantic ID 的完整入口
    item_path: str,  # 输入商品 metadata 文件，例如 data/processed/.../items.jsonl
    out_path: str,  # 输出 SID 映射文件，例如 artifacts/.../sid_map.json
    embedding_model: str,  # embedding 模型名，例如 BAAI/bge-small-en-v1.5
    levels: int,  # SID 层数，例如 3 层
    clusters_per_level: int,  # 每层聚类数，例如 32
    sid_prefix: str,  # SID 字符串前缀，例如 SID
    batch_size: int,  # embedding 批处理大小
) -> Dict[str, JsonDict]:  # 返回 item_id -> 商品和 SID 信息
    """完整流程：items.jsonl -> 商品文本 -> embedding -> 层级聚类 -> SID -> sid_map.json。"""
    items = list(read_jsonl(item_path))  # 读取所有商品 metadata
    texts = [item_text(item) for item in items]  # 每个商品拼接 title/category/description/price
    vectors = encode_texts(texts, embedding_model, batch_size)  # 把商品文本编码成 embedding 向量
    labels = hierarchical_labels(vectors, levels=levels, clusters_per_level=clusters_per_level)  # 对 embedding 做层级聚类

    collisions: Dict[str, int] = defaultdict(int)  # 记录 base SID 出现次数，处理完全相同标签导致的冲突
    sid_map: Dict[str, JsonDict] = {}  # 保存最终 item_id -> SID 和商品信息
    for item, item_labels in zip(items, labels):  # 同时遍历商品和它的多层聚类标签
        base_sid = sid_prefix + "_" + "_".join(f"{x:03d}" for x in item_labels)  # 生成基础 SID，例如 SID_005_002_007
        collisions[base_sid] += 1  # 统计这个 base SID 第几次出现
        sid = base_sid if collisions[base_sid] == 1 else f"{base_sid}_{collisions[base_sid]:03d}"  # 如果冲突，加后缀保证唯一
        sid_map[item["item_id"]] = {  # 用原始 item_id 作为 key，方便后续 review 行为映射到 SID
            "item_id": item["item_id"],  # 原始商品 ID
            "sid": sid,  # 生成式推荐模型要输出的 Semantic ID
            "title": item.get("title", ""),  # 保留标题，方便 demo 和推荐解释
            "category": item.get("category", "unknown"),  # 保留类目，方便 reward 和 category consistency
            "description": item.get("description", ""),  # 保留描述，方便后续分析或 prompt
            "price": item.get("price", ""),  # 保留价格字段
        }
    write_json(out_path, sid_map)  # 把 SID 映射写到 artifacts/.../sid_map.json
    return sid_map  # 返回 SID 映射，供脚本或测试继续使用
