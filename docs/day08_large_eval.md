# Day 8：从单类目到多类目：生成式推荐的大规模评测怎么设计？

## 今日目标

从单类目扩到 3 类目，形成更像 JD 的大数据评测。

## 学习重点

- 多类目训练的意义。
- 单类目 vs 多类目泛化。
- 14B 和 32B 的尺度对比。
- 大模型实验如何控制变量。

## 准备数据

需要以下文件：

```text
data/raw/All_Beauty.jsonl.gz
data/raw/Baby_Products.jsonl.gz
data/raw/Sports_and_Outdoors.jsonl.gz
data/raw/meta_All_Beauty.jsonl.gz
data/raw/meta_Baby_Products.jsonl.gz
data/raw/meta_Sports_and_Outdoors.jsonl.gz
```

## 执行命令

处理多类目：

```bash
python3 scripts/process_amazon.py \
  --config configs/default.yaml \
  --reviews data/raw/All_Beauty.jsonl.gz data/raw/Baby_Products.jsonl.gz data/raw/Sports_and_Outdoors.jsonl.gz \
  --metadata data/raw/meta_All_Beauty.jsonl.gz data/raw/meta_Baby_Products.jsonl.gz data/raw/meta_Sports_and_Outdoors.jsonl.gz \
  --categories All_Beauty Baby_Products Sports_and_Outdoors \
  --out-dir data/processed/main
```

构建 SID：

```bash
python3 scripts/build_semantic_ids.py \
  --config configs/default.yaml \
  --items data/processed/main/items.jsonl \
  --out artifacts/main/sid_map.json
```

构建 instruction：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out-dir data/processed/main/instructions \
  --with-reasoning
```

跑 baseline：

```bash
python3 scripts/evaluate_baselines.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/baselines.json
```

## 实验记录

单类目 vs 多类目：

| Setting | Users | Items | Interactions | Train | Test |
|---|---:|---:|---:|---:|---:|
| All_Beauty | TODO | TODO | TODO | TODO | TODO |
| 3 Categories | TODO | TODO | TODO | TODO | TODO |

多类目 baseline：

| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Popular | TODO | TODO | TODO | TODO |
| Text Retrieval | TODO | TODO | TODO | TODO |
| Embedding Retrieval | TODO | TODO | TODO | TODO |

## 今日技术理解

多类目评测更接近真实电商场景，因为用户兴趣和商品语义不局限在单一类目。

32B 对比建议只做 Day 8 之后的 scale eval，不阻塞 14B 主线。

## GitHub 产出

```bash
git add configs scripts src docs reports
git commit -m "Day 8: scale to multi-category Amazon evaluation"
```

## 今日完成标准

```text
[ ] 3 类目数据链路跑通
[ ] reports/main/baselines.json 已生成
[ ] 记录单类目 vs 多类目规模差异
[ ] 明确 32B 只作为对比，不阻塞主线
[ ] 完成 Day 8 commit
```

## 今日问题记录

```text
TODO: 记录多类目数据过大、SID 构建耗时、baseline 变慢等问题。
```
