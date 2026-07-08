# 实验结果记录模板

把真实训练和评测结果填到这里，再同步更新简历 bullet。不要提前编数字。

## 主结果表

| 方法 | HR@10 | NDCG@10 | Coverage@10 | Novelty@10 | Valid SID@10 |
|---|---:|---:|---:|---:|---:|
| Popular | TBD | TBD | TBD | TBD | - |
| Text/BM25 Retrieval | TBD | TBD | TBD | TBD | - |
| Embedding Retrieval | TBD | TBD | TBD | TBD | - |
| Qwen2.5-1.5B SFT | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-1.5B SFT + GRPO | TBD | TBD | TBD | TBD | TBD |

## 消融实验

| 实验 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---|
| 无 SID | TBD | TBD | 直接生成商品标题 |
| KMeans SID | TBD | TBD | 层级聚类 SID |
| RQ-VAE SID | TBD | TBD | 残差量化 SID |
| SFT only | TBD | TBD | 监督微调 |
| SFT + GRPO | TBD | TBD | 后训练优化 |
| 无 Reasoning | TBD | TBD | 只生成推荐 |
| 有 Reasoning | TBD | TBD | 推荐 + 理由 |
| 无约束解码 | TBD | TBD | 不限制非法 SID |
| 有约束解码 | TBD | TBD | 只接受合法 SID |
| 无去偏奖励 | TBD | TBD | 不抑制热门偏置 |
| 有去偏奖励 | TBD | TBD | 加入 popularity bias penalty |
| 无冷启动注入 | TBD | TBD | 只用 SID 历史 |
| 有冷启动注入 | TBD | TBD | 注入标题/类目信息 |

## 分群评测

| 用户/商品类型 | HR@10 | NDCG@10 | 样本数 |
|---|---:|---:|---:|
| 稀疏用户，历史 <=5 | TBD | TBD | TBD |
| 中等用户，历史 6-20 | TBD | TBD | TBD |
| 活跃用户，历史 >20 | TBD | TBD | TBD |
| 长尾商品 | TBD | TBD | TBD |
| 热门商品 | TBD | TBD | TBD |

## 冷启动子集

| 设置 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---|
| 基础 SFT | TBD | TBD | 不注入冷启动特征 |
| 冷启动 prompt 注入 | TBD | TBD | 注入标题/类目摘要 |
| 冷启动 + 去偏 GRPO | TBD | TBD | 去偏奖励 + 部分匹配奖励 |

## 可写进简历的结果句式

```markdown
相比 Qwen2.5-1.5B SFT baseline，SFT+GRPO 在多类目 Amazon Reviews 2023 测试集上 HR@10 提升 X.X%，NDCG@10 提升 X.X%，Valid SID@10 提升至 X.X%，Catalog Coverage@10 达到 X.X%，冷启动子集 HR@10 提升 X.X%。
```
