#!/usr/bin/env python3  # 告诉系统用 python3 执行这个脚本
from __future__ import annotations  # 允许更现代的类型注解写法，兼容 Python 3.10

import sys  # 用来修改 Python 模块搜索路径
from pathlib import Path  # 用面向对象的方式处理文件路径

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 把项目 src/ 加入 import 路径

from ecom_genrec.baselines import evaluate_all_baselines  # 评测 Popular/Text/Embedding 三类 baseline
from ecom_genrec.data import compute_stats, filter_interactions, split_by_user_time, write_processed  # 数据处理闭环函数
from ecom_genrec.instruction import build_instruction_split  # 把推荐样本转成 SFT/GRPO 指令数据
from ecom_genrec.semantic_id import build_sid_map  # 把商品映射成 Semantic ID
from ecom_genrec.utils import ensure_dir, write_json  # 创建目录和写 JSON 文件的工具函数


ROOT = Path(__file__).resolve().parents[1]  # 项目根目录：/data1/zhl/ecom-genrec


def synthetic_items():  # 构造一批“假商品”，用于不依赖真实数据的 smoke test
    """生成合成商品元数据：item_id、title、category、description、price。"""
    categories = ["Beauty", "Baby", "Sports", "Health", "Electronics", "Home"]  # 模拟 6 个电商类目
    rows = []  # 用列表临时保存所有商品
    for idx in range(48):  # 生成 48 个商品，数量小，方便第一天快速跑通
        cat = categories[idx % len(categories)]  # 按编号循环分配类目，让商品分布在 6 个类目里
        rows.append(  # 添加一个商品字典
            {
                "item_id": f"ITEM_{idx:03d}",  # 商品原始 ID，例如 ITEM_000
                "title": f"{cat} product {idx:03d}",  # 商品标题，用类目名制造可学习的文本信号
                "category": cat,  # 商品类目，例如 Beauty
                "description": f"A useful {cat.lower()} product with strong user interest signal {idx % 7}.",  # 商品描述
                "price": str(9.99 + idx),  # 商品价格，这里只是模拟字段
            }
        )
    return {row["item_id"]: row for row in rows}  # 返回 item_id -> 商品信息 的字典，方便后面查表


def synthetic_interactions(items):  # 构造一批“假用户行为”，模拟用户按时间购买/评论商品
    """生成用户交互序列：每个用户偏好一个类目，并连续交互 8 个商品。"""
    item_ids = list(items)  # 取出所有商品 ID
    rows = []  # 用列表保存所有用户行为
    ts = 1  # 人造时间戳，从 1 开始递增，用于模拟时间顺序
    for user_idx in range(60):  # 生成 60 个用户
        preferred = user_idx % 6  # 每个用户绑定一个偏好类目编号
        pool = [item for item in item_ids if int(item.split("_")[1]) % 6 == preferred]  # 取该用户偏好的商品池
        for step in range(8):  # 每个用户产生 8 次行为，形成可切分的序列
            item = pool[(user_idx + step) % len(pool)]  # 从偏好商品池里按顺序取一个商品
            rows.append(  # 添加一条用户行为记录
                {
                    "user_id": f"USER_{user_idx:03d}",  # 用户 ID，例如 USER_000
                    "item_id": item,  # 当前交互的商品 ID
                    "timestamp": ts,  # 当前行为发生时间，用来做时间序列切分
                    "rating": 5.0 if step % 3 else 4.0,  # 模拟评分字段，不是本 smoke test 的核心
                    "category": items[item]["category"],  # 冗余保存商品类目，方便统计和评测
                }
            )
            ts += 1  # 时间戳递增，保证后面的行为发生在更晚时间
    return rows  # 返回所有用户行为，后面会按用户切 train/valid/test


def write_summary(stats, baselines, out_path: Path) -> None:  # 把统计和 baseline 指标写成 Markdown 报告
    """把 smoke test 的核心结果写入 reports/smoke/summary.md。"""
    lines = [  # Markdown 报告的开头部分
        "# Smoke Test Summary",  # 报告标题
        "",  # 空行
        "## Data Stats",  # 数据统计小节
        "",  # 空行
        "| Metric | Value |",  # Markdown 表头
        "|---|---:|",  # Markdown 表格对齐
    ]
    for key, value in stats.to_dict().items():  # 遍历数据统计指标
        lines.append(f"| {key} | {value} |")  # 把每个统计指标写成一行 Markdown 表格
    lines.extend(  # 添加 baseline 指标表头
        [
            "",  # 空行
            "## Baseline Metrics",  # baseline 结果小节
            "",  # 空行
            "| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |",  # baseline 表头
            "|---|---:|---:|---:|---:|",  # baseline 表格对齐
        ]
    )
    for name, values in baselines.items():  # 遍历每个 baseline 方法的指标
        lines.append(  # 添加一行 baseline 结果
            f"| {name} | {values.get('HR@10', 0):.4f} | {values.get('NDCG@10', 0):.4f} | "  # 命中率和排序指标
            f"{values.get('MRR@10', 0):.4f} | {values.get('Coverage@10', 0):.4f} |"  # MRR 和覆盖率
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保 reports/smoke/ 目录存在
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写入 Markdown 报告


def main() -> None:  # smoke test 主流程：把完整推荐闭环串起来
    """运行最小闭环：商品 -> 行为 -> 数据切分 -> SID -> 指令数据 -> baseline -> 报告。"""
    processed = ROOT / "data/processed/smoke"  # smoke test 的处理后数据目录
    artifacts = ROOT / "artifacts/smoke"  # smoke test 的中间产物目录，例如 sid_map.json
    reports = ROOT / "reports/smoke"  # smoke test 的实验报告目录
    ensure_dir(processed)  # 创建 data/processed/smoke
    ensure_dir(artifacts)  # 创建 artifacts/smoke
    ensure_dir(reports)  # 创建 reports/smoke

    # Step 1: 构造合成数据，模拟真实 Amazon 商品和用户行为。
    items = synthetic_items()  # 生成商品元数据：48 个商品，6 个类目
    interactions = synthetic_interactions(items)  # 生成用户行为序列：60 个用户，每人 8 条行为

    # Step 2: 过滤和时间切分，这是推荐系统评测的基础。
    filtered = filter_interactions(interactions, min_user=5, min_item=1)  # 过滤交互过少的用户/商品
    train, valid, test = split_by_user_time(filtered, max_history=20)  # 按用户时间序列切分训练/验证/测试
    stats = compute_stats(filtered, items, train, valid, test, cold_user_max_history=5, long_tail_quantile=0.8)  # 统计数据规模
    write_processed(processed, filtered, items, train, valid, test, stats)  # 写出 interactions/items/train/valid/test/stats

    # Step 3: 生成 Semantic ID，把商品变成模型可以稳定生成的结构化 ID。
    sid_map_path = artifacts / "sid_map.json"  # SID 映射文件输出路径
    build_sid_map(  # 生成 item_id -> SID 的映射
        item_path=str(processed / "items.jsonl"),  # 输入商品元数据
        out_path=str(sid_map_path),  # 输出 sid_map.json
        embedding_model="BAAI/bge-small-en-v1.5",  # 优先使用 BGE embedding；没装依赖时会 fallback
        levels=3,  # SID 有 3 个语义层级，例如 SID_005_005_005
        clusters_per_level=8,  # 每层最多 8 个聚类，smoke test 用小值保证速度
        sid_prefix="SID",  # SID 前缀
        batch_size=32,  # embedding 批处理大小
    )

    # Step 4: 构造 LLM 指令数据，把推荐任务变成 prompt/completion。
    inst_dir = processed / "instructions"  # 指令数据输出目录
    ensure_dir(inst_dir)  # 创建 data/processed/smoke/instructions
    counts = {  # 记录 train/valid/test 三个 split 的指令样本数量
        "train": build_instruction_split(  # 构造训练集 SFT 样本
            str(processed / "train.jsonl"),  # 输入训练序列数据
            str(sid_map_path),  # 输入 item_id -> SID 映射
            str(inst_dir / "sft_train.jsonl"),  # 输出训练指令数据
            True,  # True 表示输出推荐 SID + 推荐理由
        ),
        "valid": build_instruction_split(  # 构造验证集 SFT 样本
            str(processed / "valid.jsonl"),  # 输入验证序列数据
            str(sid_map_path),  # 输入 item_id -> SID 映射
            str(inst_dir / "sft_valid.jsonl"),  # 输出验证指令数据
            True,  # True 表示带 reasoning 推荐理由
        ),
        "test": build_instruction_split(  # 构造测试集 SFT 样本
            str(processed / "test.jsonl"),  # 输入测试序列数据
            str(sid_map_path),  # 输入 item_id -> SID 映射
            str(inst_dir / "sft_test.jsonl"),  # 输出测试指令数据
            True,  # True 表示带 reasoning 推荐理由
        ),
    }
    write_json(inst_dir / "counts.json", counts)  # 保存三份指令数据的样本数量

    # Step 5: 跑 baseline 和指标，验证推荐评测链路能工作。
    baselines = evaluate_all_baselines(  # 统一评测 Popular/Text Retrieval/Embedding Retrieval
        train_path=str(inst_dir / "sft_train.jsonl"),  # baseline 从训练数据里学习热门和候选商品
        test_path=str(inst_dir / "sft_test.jsonl"),  # 在测试集上评估推荐效果
        sid_map_path=str(sid_map_path),  # 用 SID 映射恢复商品类目等信息
        k_values=[5, 10, 20],  # 同时评测 @5、@10、@20
    )
    write_json(reports / "baselines.json", baselines)  # 保存完整 baseline 指标 JSON
    write_summary(stats, baselines, reports / "summary.md")  # 写出适合 GitHub 展示的 Markdown 报告
    print({"stats": stats.to_dict(), "counts": counts, "report": str(reports / "summary.md")})  # 在终端打印关键结果


if __name__ == "__main__":  # 只有直接执行 python3 scripts/run_smoke.py 时才进入这里
    main()  # 启动 smoke test 主流程
