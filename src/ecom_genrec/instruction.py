from __future__ import annotations  # 允许使用较新的类型注解写法

import re  # 用正则表达式从模型输出文本里提取 SID
from typing import Dict, Iterable, List, Tuple  # 常用类型注解

from .utils import JsonDict, read_json, read_jsonl, truncate_words, write_jsonl  # 项目内 JSON/JSONL 和文本工具


SID_RE = re.compile(r"SID(?:_[0-9]{3})+(?:_[0-9]{3})?")  # 匹配 SID_001_002_003 或带冲突后缀的 SID


def load_sid_maps(path: str) -> Tuple[Dict[str, JsonDict], Dict[str, JsonDict]]:  # 加载 SID 映射文件
    """读取 sid_map.json，同时构造 item_id->metadata 和 sid->metadata 两种查表方式。"""
    item_to_sid = read_json(path)  # sid_map.json 原始格式是 item_id -> 商品信息和 SID
    sid_to_item = {row["sid"]: row for row in item_to_sid.values()}  # 反向索引：SID -> 商品信息
    return item_to_sid, sid_to_item  # 返回两个字典，方便不同场景查表


def render_history(history: List[str], item_to_sid: Dict[str, JsonDict]) -> str:  # 把历史 item_id 列表渲染成 prompt 文本
    """把用户历史商品 item_id 转成模型可读的多行历史：序号 + SID + 标题 + 类目。"""
    lines = []  # 保存每一行历史商品文本
    for idx, item_id in enumerate(history, start=1):  # 给历史商品编号，从 1 开始更符合 prompt 阅读习惯
        meta = item_to_sid.get(item_id, {})  # 根据 item_id 查商品 metadata 和 SID
        sid = meta.get("sid", item_id)  # 优先显示 Semantic ID；如果缺失则回退显示原始 item_id
        title = truncate_words(meta.get("title", item_id), 12)  # 商品标题最多保留 12 个词，避免 prompt 太长
        category = meta.get("category", "unknown").split(" > ")[0]  # 只取一级类目，降低 prompt 噪声
        lines.append(f"{idx}. {sid} | {title} | {category}")  # 一行历史商品：1. SID_xxx | title | category
    return "\n".join(lines)  # 多个历史商品用换行连接，形成 prompt 的用户历史部分


def simple_reason(history: List[str], target_sid: str, item_to_sid: Dict[str, JsonDict]) -> str:  # 生成一个简单推荐理由
    """用历史类目和目标类目生成模板化推荐理由；这里 target_sid 实际上传入的是 target_item。"""
    cats = [item_to_sid.get(item_id, {}).get("category", "unknown").split(" > ")[0] for item_id in history]  # 历史商品类目
    target_cat = item_to_sid.get(target_sid, {}).get("category", "unknown").split(" > ")[0]  # 目标商品类目
    focus = cats[-1] if cats else target_cat  # 优先用用户最近一次历史类目作为兴趣焦点
    return (  # 返回中文推荐理由，作为 SFT completion 的一部分
        f"用户近期多次关注 {focus} 相关商品，历史兴趣与 {target_cat} 类目保持一致，"  # 说明历史兴趣和目标类目一致
        "因此推荐该商品作为下一步候选。"  # 给出推荐结论
    )


def build_prompt(history: List[str], item_to_sid: Dict[str, JsonDict], with_reasoning: bool) -> str:  # 构造模型输入
    """构造 SFT prompt：告诉模型任务，并提供用户历史商品。"""
    task = "根据用户历史购买/评论商品，预测用户下一个可能感兴趣的商品"  # 基础任务描述
    if with_reasoning:  # 如果训练 reasoning 版本
        task += "，并给出简短推荐理由"  # 要求模型同时输出推荐理由
    return (  # 返回完整 prompt
        f"Instruction:\n{task}。\n\n"  # 指令区：告诉模型要做什么
        f"Input:\n用户历史商品：\n{render_history(history, item_to_sid)}\n\n"  # 输入区：历史 item_id 被渲染成 SID/title/category
        "Output:\n"  # 输出区开头：后面接 completion
    )


def build_completion(target_sid: str, reason: str | None) -> str:  # 构造模型要学习的答案
    """构造 SFT completion：推荐商品 SID，可选推荐理由。"""
    if reason:  # 如果需要 reasoning
        return f"推荐商品：{target_sid}\n推荐理由：{reason}"  # 输出推荐 SID + 推荐理由
    return f"推荐商品：{target_sid}"  # 只输出推荐 SID


def sequence_to_instruction(row: JsonDict, item_to_sid: Dict[str, JsonDict], with_reasoning: bool) -> JsonDict | None:
    """把一条序列推荐样本 history -> target_item 转成 LLM 训练样本 prompt/completion。"""
    target_item = row["target_item"]  # 原始目标商品 item_id，也就是用户未来真实交互的商品
    if target_item not in item_to_sid:  # 如果目标商品没有 SID 映射
        return None  # 丢弃该样本，否则模型无法学习合法输出
    history_items = [x for x in row.get("history", []) if x in item_to_sid]  # 只保留能映射到 SID 的历史商品
    if not history_items:  # 如果过滤后没有历史
        return None  # 丢弃该样本，因为没有输入历史就无法构造推荐任务
    target_sid = item_to_sid[target_item]["sid"]  # 把目标 item_id 转成模型要生成的目标 SID
    history_sid = [item_to_sid[x]["sid"] for x in history_items]  # 把历史 item_id 列表转成历史 SID 列表
    reason = simple_reason(history_items, target_item, item_to_sid) if with_reasoning else None  # 可选推荐理由
    prompt = build_prompt(history_items, item_to_sid, with_reasoning)  # 构造模型输入 prompt
    completion = build_completion(target_sid, reason)  # 构造模型输出 completion
    return {  # 返回一条完整训练样本
        "user_id": row["user_id"],  # 用户 ID，方便追踪样本来源
        "prompt": prompt,  # 模型输入：任务说明 + 用户历史
        "completion": completion,  # 模型答案：推荐 SID + 可选理由
        "text": prompt + completion,  # SFTTrainer 常用字段：把 prompt 和 completion 拼起来训练
        "history_item": history_items,  # 原始历史 item_id，方便调试和误差分析
        "history_sid": history_sid,  # 历史 SID，baseline 和评测也会用到
        "target_item": target_item,  # 原始目标 item_id
        "target_sid": target_sid,  # 目标 SID，也是真实 label
        "target_category": item_to_sid[target_item].get("category", "unknown"),  # 目标类目，用于 category reward/分析
        "history_len": row.get("history_len", len(history_items)),  # 历史长度，用于冷启动/稀疏用户分析
    }


def build_instruction_split(  # 把 train/valid/test 某个 split 批量转成 instruction jsonl
    sequence_path: str,  # 输入序列样本文件，例如 train.jsonl
    sid_map_path: str,  # 输入 SID 映射文件，例如 sid_map.json
    out_path: str,  # 输出 instruction 文件，例如 sft_train.jsonl
    with_reasoning: bool,  # 是否输出推荐理由
    limit: int | None = None,  # 可选样本上限，debug 训练时用
) -> int:  # 返回写出的样本数量
    item_to_sid, _sid_to_item = load_sid_maps(sid_map_path)  # 加载 item_id -> SID 映射；这里暂时不用 sid_to_item
    rows: List[JsonDict] = []  # 保存转换后的 instruction 样本
    for row in read_jsonl(sequence_path):  # 逐行读取序列推荐样本
        inst = sequence_to_instruction(row, item_to_sid, with_reasoning=with_reasoning)  # 单条样本转换成 prompt/completion
        if inst is not None:  # 如果样本有效
            rows.append(inst)  # 加入输出列表
        if limit is not None and len(rows) >= limit:  # 如果设置了 debug 样本上限且已达到
            break  # 停止继续读取
    write_jsonl(out_path, rows)  # 写出 sft_train/sft_valid/sft_test jsonl
    return len(rows)  # 返回样本数量，方便脚本打印和保存 counts.json


def extract_sids(text: str) -> List[str]:  # 从模型生成文本里抽取 SID
    """从任意生成文本中提取不重复 SID，保持出现顺序。"""
    seen = set()  # 记录已经见过的 SID，避免重复
    sids = []  # 保存按出现顺序抽取到的 SID
    for match in SID_RE.findall(text or ""):  # 用正则找出所有 SID
        if match not in seen:  # 如果这个 SID 还没出现过
            seen.add(match)  # 标记为已出现
            sids.append(match)  # 加入结果列表
    return sids  # 返回抽取到的 SID 列表


def completions_to_sid_lists(completions: Iterable[str], fallback: List[str], k: int) -> List[List[str]]:  # 生成文本 -> Top-K SID
    """把模型 completion 转成 Top-K 推荐 SID；不足 k 个时用 fallback 补齐。"""
    results: List[List[str]] = []  # 保存每条 completion 对应的 SID 列表
    for completion in completions:  # 遍历每条模型输出
        row = extract_sids(completion)  # 从输出文本中抽取 SID
        for sid in fallback:  # 如果模型生成的 SID 数量不足，就用 fallback 候选补齐
            if len(row) >= k:  # 已经达到 Top-K 长度
                break  # 停止补齐
            if sid not in row:  # 避免重复推荐同一个 SID
                row.append(sid)  # 补一个 fallback SID
        results.append(row[:k])  # 截断到 Top-K 并保存
    return results  # 返回所有样本的 Top-K SID 推荐列表
