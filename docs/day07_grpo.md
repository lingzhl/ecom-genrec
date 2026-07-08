# Day 7：用 GRPO 优化推荐模型：如何把 HR@10 写进 reward？

## 今日目标

在 SFT checkpoint 基础上跑通 GRPO 小规模后训练。

## 学习重点

- SFT 只能模仿数据。
- GRPO 用 reward 优化目标。
- 推荐任务 reward 设计。
- 合法 SID、命中率、类目一致性、多样性。

## 执行命令

启动 GRPO debug：

```bash
torchrun --nproc_per_node=2 scripts/train_grpo.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_reasoning/grpo_train.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --sft-checkpoint artifacts/checkpoints/qwen25-14b-genrec \
  --deepspeed configs/deepspeed_zero2.json
```

## Reward 设计

| Reward | Weight | 业务含义 |
|---|---:|---|
| Hit Reward | 1.0 | 推荐命中真实商品 |
| NDCG Reward | 0.6 | 命中越靠前越好 |
| Category Reward | 0.3 | 推荐类目和真实类目一致 |
| Valid SID Reward | 0.3 | 生成的 SID 必须存在 |
| Diversity Reward | 0.2 | 避免只推荐热门商品 |
| Reasoning Reward | 0.2 | 推荐理由包含用户兴趣逻辑 |

## 实验记录

Reward 日志：

```text
TODO: paste reward logs
```

SFT vs GRPO 样例对比：

| Case | SFT Output | GRPO Output | Difference |
|---:|---|---|---|
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO |

## 今日技术理解

GRPO 的作用不是让模型“知道更多商品”，而是让模型在已有 SFT 能力上更偏向业务目标：

```text
命中更高
排序更靠前
SID 更合法
类目更一致
推荐理由更稳定
```

## GitHub 产出

```bash
git add scripts src configs docs reports
git commit -m "Day 7: run GRPO post-training with recommendation rewards"
```

## 今日完成标准

```text
[ ] GRPO 能启动
[ ] reward 日志正常
[ ] 生成结果没有大面积崩坏
[ ] 记录至少 3 条 SFT vs GRPO 对比
[ ] 完成 Day 7 commit
```

## 今日问题记录

```text
TODO: 记录 reward 不升、输出变短、非法 SID、训练不稳定等问题。
```
