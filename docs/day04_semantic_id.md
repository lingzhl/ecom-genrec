# Day 4：Semantic ID：如何把商品推荐变成大模型可控生成任务？

## 今日目标

把商品从原始 item_id 转成模型更容易学习和生成的 Semantic ID。

## 学习重点

- 为什么不直接生成商品标题。
- 商品文本 embedding。
- KMeans 聚类。
- 层级 Semantic ID。

## 执行命令

构建 SID：

```bash
python3 scripts/build_semantic_ids.py \
  --config configs/default.yaml \
  --items data/processed/all_beauty/items.jsonl \
  --out artifacts/all_beauty/sid_map.json
```

查看 SID：

```bash
head -40 artifacts/all_beauty/sid_map.json
```

重新构建 instruction：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out-dir data/processed/all_beauty/instructions \
  --with-reasoning
```

查看训练样本：

```bash
head -1 data/processed/all_beauty/instructions/sft_train.jsonl
```

## 实验记录

SID 覆盖率：

| Metric | Value |
|---|---:|
| item_count | TODO |
| sid_count | TODO |
| duplicate_sid_count | TODO |
| valid_mapping_rate | TODO |

示例：

```text
item_id: TODO
title: TODO
sid: TODO
```

## 今日技术理解

直接生成商品标题的问题：

- 模型可能生成不存在的商品。
- 标题自由文本很难精确评测。
- 商品库更新后，开放式生成更难约束。

Semantic ID 的优势：

- 输出空间可控。
- 可以映射回真实商品。
- 方便计算 HR@K、NDCG@K。
- 聚类编码比随机 item_id 更有语义结构。

## GitHub 产出

```bash
git add src scripts configs docs
git commit -m "Day 4: build semantic item ids for constrained generation"
```

## 今日完成标准

```text
[ ] sid_map.json 已生成
[ ] sft_train.jsonl 已生成
[ ] 所有训练样本 target 都能映射到合法 SID
[ ] 能解释 Semantic ID 的作用
[ ] 完成 Day 4 commit
```

## 今日问题记录

```text
TODO: 记录 embedding 模型下载、聚类耗时、SID 冲突等问题。
```
