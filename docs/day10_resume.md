# Day 10：项目总结：我是如何用 Qwen2.5-1.5B 复现 OneRec 风格生成式推荐的？

## 今日目标

把项目包装成可投递、可面试、可放 GitHub 的完整项目。

## 学习重点

- README 怎么写得像真实项目。
- 简历 bullet 怎么写。
- 面试怎么讲技术取舍。
- 如何避免“只会调包”的感觉。

## README 最终结构

```markdown
# 面向电商场景的生成式商品推荐系统

## 项目背景
## 方法概览
## 数据集
## Semantic ID
## SFT
## GRPO
## 实验结果
## Demo
## 如何复现
## 项目亮点
## 局限性与下一步
```

## 最终简历 Bullet

```markdown
### 面向电商场景的生成式商品推荐系统 | Qwen2.5-1.5B, KMeans/RQ-VAE SID, SFT, GRPO

- 基于 Amazon Reviews 2023 构建 OneRec 风格生成式推荐数据集，完成用户行为时间序列切分、冷启动用户分群评测与多类目扩展。
- 设计 KMeans SID 与 RQ-VAE-style SID 双语义编码路径，将开放式商品生成转化为受约束商品 ID 生成，统计合法 SID 率与码本利用率。
- 基于 Qwen2.5-1.5B-Instruct 进行三任务 SFT，覆盖序列推荐、特征对齐、历史融合，并加入约束解码减少非法 SID。
- 使用 TRL GRPO 进行 7 类奖励后训练，覆盖 Hit、NDCG、Category、Valid SID、Diversity、Novelty、Reasoning，并补充去偏奖励与部分匹配奖励。
- 输出 HR@10、NDCG@10、Catalog Coverage、Novelty、Valid SID Rate 以及冷启动子集指标；后续扩展到 7B/14B 做规模对比。
```

真实数字句式：

```markdown
相比 Qwen2.5-1.5B SFT baseline，SFT+GRPO 在测试集上 HR@10 提升 X.X%，NDCG@10 提升 X.X%，Valid SID@10 提升至 X.X%，冷启动子集 HR@10 提升 X.X%。
```

## 面试 Q&A

### 1. 为什么使用生成式推荐？

TODO

### 2. 为什么不用模型直接生成商品标题？

TODO

### 3. Semantic ID 怎么构造？

TODO

### 4. SFT 数据怎么来？

TODO

### 5. GRPO 相比 SFT 解决什么问题？

TODO

### 6. reward 怎么设计？

TODO

### 7. HR@10 和 NDCG@10 怎么算？

TODO

### 8. 如果 GRPO 没提升怎么办？

TODO

### 9. 为什么先做 1.5B，再扩 7B/14B？

TODO

### 10. 这个项目和传统召回/排序推荐系统有什么区别？

TODO

## 最终检查

```text
[ ] README 有完整复现命令
[ ] docs 有 10 天技术博客
[ ] docs/RESULT_TABLES.md 有真实结果
[ ] reports 有 baseline/SFT/GRPO 结果文件
[ ] 简历 bullet 已替换真实数字
[ ] 能 3 分钟讲清楚项目
```

## GitHub 产出

```bash
git add README.md docs reports configs scripts src
git commit -m "Day 10: finalize project documentation and resume materials"
git log --oneline --decorate -10
```
