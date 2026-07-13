# Day 4：Semantic ID：如何把商品推荐变成大模型可控生成任务？

## 今日目标

把商品从原始 item_id 转成模型更容易学习和生成的 Semantic ID。

## 学习重点

- 为什么不直接生成商品标题。
- 商品文本 embedding。
- KMeans 聚类。
- 层级 Semantic ID。
- RQ-VAE-style 残差量化 SID。

## 执行命令

先构建 KMeans SID：

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

如果要补 OneRec 风格对照，再构建 RQ-VAE-style SID：

```bash
python3 scripts/build_rqvae_ids.py \
  --config configs/default.yaml \
  --items data/processed/all_beauty/items.jsonl \
  --out artifacts/all_beauty/sid_map_rqvae.json
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
| item_count | 479 |
| sid_count | 479 |
| duplicate_sid_count | 0 |
| valid_mapping_rate | 100% |

KMeans vs RQ-VAE-style：

| Method | sid_count | valid_mapping_rate | codebook_usage | collapse_check |
|---|---:|---:|---:|---|
| KMeans SID | 479 | 100% | L1 32/32, L2 21/32, L3 16/32 | final duplicate 0；base code collision 416 |
| RQ-VAE-style SID | 479 | 100% | L1 53/64, L2 64/64, L3 63/64 | final duplicate 0；base code collision 104 |

示例：

```text
item_id: B08LYT4Q2X
title: Organic Sweet Almond Oil and Fractionated Coconut Oil Bundle for Hair and Skin, 100% Pure and Natural, Hexane-Free, Moisturizing, For Healthy Skin, Silky Hair, Multiuse Body Oil, 16 fl. Oz X 2
sid: SID_005_003_000
rqvae_sid: SID_RQ_005_062_062
```

Instruction 数据：

| Split | Count |
|---|---:|
| sequence_train | 2244 |
| sequence_valid | 357 |
| sequence_test | 357 |
| history_fusion_train | 2244 |
| history_fusion_valid | 357 |
| history_fusion_test | 357 |
| feature_alignment_train | 431 |
| feature_alignment_valid | 48 |
| feature_alignment_test | 48 |
| sft_train | 4919 |
| sft_valid | 762 |
| sft_test | 762 |

SFT target SID 合法性检查：

```text
sft_train: 4919/4919 valid, bad_targets=0
sft_valid: 762/762 valid, bad_targets=0
sft_test: 762/762 valid, bad_targets=0
```

Baseline 结果摘要：

| Method | HR@5 | NDCG@5 | HR@20 | NDCG@20 | ValidSID@20 |
|---|---:|---:|---:|---:|---:|
| Popular | 0.003937 | 0.001878 | 0.007874 | 0.002890 | 1.0 |
| TextRetrieval | 0.017060 | 0.008419 | 0.064304 | 0.020774 | 1.0 |
| EmbeddingRetrieval | 0.018373 | 0.011215 | 0.053806 | 0.021460 | 1.0 |
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

今天的实操顺序建议是：

```text
先做 KMeans SID 跑通
再补 RQ-VAE-style SID 做对照
```

## GitHub 产出

```bash
git add docs/day04_semantic_id.md
git add -f artifacts/all_beauty/sid_map.json artifacts/all_beauty/sid_map_rqvae.json
git add -f data/processed/all_beauty/instructions reports/all_beauty/baselines.json
git commit -m "Day 4: build semantic item ids for constrained generation"
```

## 今日完成标准

```text
[√] sid_map.json 已生成
[√] sid_map_rqvae.json 已生成或明确留到下一步
[√] sft_train.jsonl 已生成
[√] 所有训练样本 target 都能映射到合法 SID
[√] 能解释 Semantic ID 的作用
[√] 完成 Day 4 commit
```

## 今日问题记录

```text
使用 embedding_model=BAAI/bge-small-en-v1.5。
KMeans SID 和 RQ-VAE-style SID 最终都没有重复 SID。
KMeans base code collision=416，RQ-VAE-style base code collision=104；脚本通过追加后缀保证最终 SID 唯一。
artifacts/、data/processed/、reports/ 默认被 .gitignore 忽略，上传今天生成物时需要 git add -f。
```
