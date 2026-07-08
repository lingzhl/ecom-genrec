# Day 10：项目总结：我是如何用 Qwen2.5-14B 做电商生成式推荐的？

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
### 面向电商场景的生成式商品推荐系统 | Qwen2.5-14B/32B, SFT, GRPO, Semantic ID

- 基于 Amazon Reviews 2023 多类目数据构建大规模电商序列推荐数据集，完成用户行为时间序列切分、商品元数据融合、长尾/冷启动用户分群评测，形成覆盖 HR@K、NDCG@K、MRR、Catalog Coverage、Valid SID Rate 的完整评测体系。
- 设计层级商品语义 ID 表示方案，基于商品标题、类目和描述构建 embedding 聚类编码，将开放式商品文本生成转化为受约束商品 ID 生成，降低推荐幻觉并提升生成结果可控性。
- 基于 Qwen2.5-14B-Instruct 进行 LoRA SFT，构造“用户历史行为 -> 推荐商品 SID + 推荐理由”的高质量指令数据，探索电商推荐场景下的 Reasoning/CoT 推荐解释能力。
- 使用 TRL GRPO 进行模型后训练与偏好对齐，设计命中率、排序位置、类目一致性、合法 SID、多样性和推荐理由一致性等奖励函数，优化生成式推荐模型的命中率和稳定性。
- 对比 Popular、Embedding Retrieval、SFT、SFT+GRPO、Qwen2.5-32B 等多组实验，并完成多类目、大测试集、冷启动用户、长尾商品和消融实验分析，为推荐效果优化提供数据支撑。
```

真实数字句式：

```markdown
相比 Qwen2.5-14B SFT baseline，SFT+GRPO 在测试集上 HR@10 提升 X.X%，NDCG@10 提升 X.X%，Valid SID@10 提升至 X.X%。
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

### 9. 14B 和 32B 的对比意义是什么？

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
