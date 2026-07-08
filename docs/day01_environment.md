# Day 1：从传统推荐到生成式推荐：为什么要让 LLM 生成商品 ID？

## 今日目标

跑通最小闭环，不碰真实大数据和 14B 训练。

你今天要理解：

- 生成式推荐和传统推荐的区别。
- Semantic ID 是什么。
- SFT 和 GRPO 分别解决什么问题。
- `HR@K / NDCG@K / MRR / Coverage` 的直觉。

## 执行步骤

进入项目：

```bash
cd /data1/zhl/ecom-genrec
```

创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

检查 GPU 和依赖：

```bash
nvidia-smi
python3 -c "import torch, transformers, trl, peft, deepspeed; print('ok')"
```

跑通 smoke test：

```bash
python3 scripts/run_smoke.py
cat reports/smoke/summary.md
```

## 实验记录

把 smoke test 的输出贴到这里：

```text
TODO: paste reports/smoke/summary.md
```

记录 GPU：

```text
TODO: paste nvidia-smi key lines
```

## 今日技术理解

### 传统推荐 vs 生成式推荐

传统推荐通常是：

```text
user_id + item features -> item ranking score
```

生成式推荐是：

```text
用户历史商品序列 -> LLM 生成下一个商品 SID
```

### 为什么先跑 smoke test

Smoke test 的作用是验证完整管线：

```text
数据 -> SID -> instruction -> baseline -> metrics -> report
```

它不追求效果，只验证项目不会在真实大数据阶段才发现基础链路错误。

### 指标解释

| 指标 | 含义 |
|---|---|
| HR@10 | Top-10 中是否命中真实商品 |
| NDCG@10 | 命中越靠前分越高 |
| MRR@10 | 第一个命中位置的倒数 |
| Coverage@10 | 推荐结果覆盖了多少商品 |

## GitHub 产出

```bash
git add README.md requirements.txt configs scripts src docs reports/smoke .gitignore
git commit -m "Day 1: setup environment and smoke test pipeline"
```

## 今日完成标准

```text
[ ] 环境能 import torch/transformers/trl/peft/deepspeed
[ ] smoke test 能跑通
[ ] reports/smoke/summary.md 存在
[ ] 能解释 HR@10 和 NDCG@10
[ ] 完成 Day 1 commit
```

## 今日问题记录

```text
TODO: 记录安装失败、CUDA、依赖冲突、下载慢等问题。
```
