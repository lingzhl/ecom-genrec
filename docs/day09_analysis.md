# Day 9：实验结果复盘：SFT+GRPO 真的提升了推荐效果吗？

## 今日目标

把项目从“能跑”变成“有结果支撑”：主结果、消融、分群和失败案例。

## 学习重点

- 主实验表。
- 消融实验。
- 冷启动用户评测。
- 长尾商品评测。
- 失败案例分析。

## 执行命令

评测 SFT：

```bash
python3 scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec \
  --eval data/processed/main/instructions/sft_test.jsonl \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/qwen25_1p5b_sft_eval.json \
  --train-reference data/processed/main/instructions/sft_train.jsonl \
  --max-samples 1000
```

评测 GRPO：

```bash
python3 scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec-grpo \
  --eval data/processed/main/instructions/sft_test.jsonl \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/qwen25_1p5b_grpo_eval.json \
  --train-reference data/processed/main/instructions/sft_train.jsonl \
  --max-samples 1000
```

## 主结果表

| 方法 | HR@10 | NDCG@10 | Coverage@10 | Novelty@10 | Valid SID@10 |
|---|---:|---:|---:|---:|---:|
| Popular | TODO | TODO | TODO | TODO | - |
| Text Retrieval | TODO | TODO | TODO | TODO | - |
| Embedding Retrieval | TODO | TODO | TODO | TODO | - |
| Qwen2.5-1.5B SFT | TODO | TODO | TODO | TODO | TODO |
| Qwen2.5-1.5B SFT + GRPO | TODO | TODO | TODO | TODO | TODO |

## 消融实验

| 实验 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---|
| KMeans SID | TODO | TODO | 层级聚类 SID |
| RQ-VAE-style SID | TODO | TODO | 残差量化 SID |
| 无约束解码 | TODO | TODO | 可能出现非法 SID |
| 有约束解码 | TODO | TODO | 限制到合法 SID 空间 |
| SFT only | TODO | TODO | 监督微调 |
| SFT + GRPO | TODO | TODO | 后训练优化 |
| 无去偏奖励 | TODO | TODO | 不抑制热门偏置 |
| 有去偏奖励 | TODO | TODO | 加 popularity bias penalty |
| 无冷启动注入 | TODO | TODO | 普通 prompt |
| 有冷启动注入 | TODO | TODO | 注入冷启动提示 |

## 冷启动与分群评测

| 用户类型 | HR@10 | NDCG@10 | 样本数 |
|---|---:|---:|---:|
| 稀疏用户，历史 <=5 | TODO | TODO | TODO |
| 中等用户，历史 6-20 | TODO | TODO | TODO |
| 活跃用户，历史 >20 | TODO | TODO | TODO |
| 长尾商品 | TODO | TODO | TODO |
| 冷启动用户子集 | TODO | TODO | TODO |

## 案例分析

成功案例：

| Case | Why it works |
|---:|---|
| 1 | TODO |
| 2 | TODO |
| 3 | TODO |
| 4 | TODO |
| 5 | TODO |

失败案例：

| Case | Failure Type | Analysis |
|---:|---|---|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| 4 | TODO | TODO |
| 5 | TODO | TODO |

## 如果 GRPO 没提升

优先排查：

- reward 是否太稀疏。
- 生成格式是否不稳定。
- SFT checkpoint 是否太弱。
- GRPO 训练步数是否太少。
- evaluation 是否只评估了太小样本。

## GitHub 产出

```bash
git add docs reports README.md
git commit -m "Day 9: add full evaluation tables and error analysis"
```

## 今日完成标准

```text
[ ] 至少有一张完整主结果表
[ ] 至少有 5 条成功案例和 5 条失败案例
[ ] docs/RESULT_TABLES.md 已同步真实结果
[ ] 能说清楚 GRPO 是否有效
[ ] 完成 Day 9 commit
```
