# Day 8：从单类目到多类目：生成式推荐的大规模评测怎么设计？

## 今日目标

加入冷启动专项后，再从单类目扩到 3 类目，形成更像 OneRec 复现的主实验评测。

## 学习重点

- 多类目训练的意义。
- 单类目 vs 多类目泛化。
- 冷启动用户为什么要单独统计。
- 为什么 7B/14B 要放到后续扩展阶段。

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

构建 KMeans SID：

```bash
python3 scripts/build_semantic_ids.py \
  --config configs/default.yaml \
  --items data/processed/main/items.jsonl \
  --out artifacts/main/sid_map.json
```

如果时间允许，再补 RQ-VAE-style SID：

```bash
python3 scripts/build_rqvae_ids.py \
  --config configs/default.yaml \
  --items data/processed/main/items.jsonl \
  --out artifacts/main/sid_map_rqvae.json
```

构建 instruction：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out-dir data/processed/main/instructions \
  --with-reasoning \
  --task-mix onerec
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

冷启动子集：

| Split | HR@10 | NDCG@10 | Sample Count |
|---|---:|---:|---:|
| cold_start_users | TODO | TODO | TODO |
| non_cold_start_users | TODO | TODO | TODO |

## 今日技术理解

多类目评测更接近真实电商场景，因为用户兴趣和商品语义不局限在单一类目。

当前 10 天主线仍然只对 1.5B 负责。7B/14B 是后续扩展，不在今天阻塞主实验。

## GitHub 产出

```bash
git add configs scripts src docs reports
git commit -m "Day 8: add cold-start and multi-category evaluation"
```

## 今日完成标准

```text
[ ] 3 类目数据链路跑通
[ ] reports/main/baselines.json 已生成
[ ] 有一张冷启动子集结果表
[ ] 记录单类目 vs 多类目规模差异
[ ] 明确 7B/14B 只作为后续扩展，不阻塞主线
[ ] 完成 Day 8 commit
```

## 今日问题记录

```text
TODO: 记录多类目数据过大、SID 构建耗时、baseline 变慢等问题。
```
