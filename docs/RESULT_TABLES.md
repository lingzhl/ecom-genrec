# 实验结果记录模板

把真实训练和评测结果填到这里，再同步更新简历 bullet。不要提前编数字。

## 主结果表

| 方法 | HR@10 | NDCG@10 | MRR@10 | Coverage@10 | Valid SID@10 |
|---|---:|---:|---:|---:|---:|
| Popular | TBD | TBD | TBD | TBD | - |
| Text/BM25 Retrieval | TBD | TBD | TBD | TBD | - |
| Embedding Retrieval | TBD | TBD | TBD | TBD | - |
| Qwen2.5-14B SFT | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-14B SFT + GRPO | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-32B QLoRA / Eval | TBD | TBD | TBD | TBD | TBD |

## 消融实验

| 实验 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---|
| 无 SID | TBD | TBD | 直接生成商品标题 |
| 有 SID | TBD | TBD | 生成合法商品 ID |
| SFT only | TBD | TBD | 监督微调 |
| SFT + GRPO | TBD | TBD | 后训练优化 |
| 无 Reasoning | TBD | TBD | 只生成推荐 |
| 有 Reasoning | TBD | TBD | 推荐 + 理由 |
| 14B | TBD | TBD | 主训练模型 |
| 32B | TBD | TBD | 大模型尺度对比 |

## 分群评测

| 用户/商品类型 | HR@10 | NDCG@10 | 样本数 |
|---|---:|---:|---:|
| 稀疏用户，历史 <=5 | TBD | TBD | TBD |
| 中等用户，历史 6-20 | TBD | TBD | TBD |
| 活跃用户，历史 >20 | TBD | TBD | TBD |
| 长尾商品 | TBD | TBD | TBD |
| 热门商品 | TBD | TBD | TBD |

## 可写进简历的结果句式

```markdown
相比 Qwen2.5-14B SFT baseline，SFT+GRPO 在多类目 Amazon Reviews 2023 测试集上 HR@10 提升 X.X%，NDCG@10 提升 X.X%，Valid SID@10 提升至 X.X%，Catalog Coverage@10 达到 X.X%。
```
