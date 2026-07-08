# Day 6：电商推荐 Reasoning：推荐理由是解释，还是真推理？

## 今日目标

让模型输出 `推荐商品 SID + 推荐理由`，探索电商推荐 Reasoning。

## 学习重点

- 推荐 Reasoning 和 CoT 的区别。
- 推荐理由是否等于真实推理。
- 如何构造推荐解释数据。
- Reasoning Consistency 如何评估。

## 执行命令

构造 reasoning 数据：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out-dir data/processed/all_beauty/instructions_reasoning \
  --with-reasoning \
  --max-train-samples 10000
```

训练 reasoning 版 SFT：

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --eval data/processed/all_beauty/instructions_reasoning/sft_valid.jsonl \
  --deepspeed configs/deepspeed_zero2.json
```

## 实验记录

成功案例：

| Case | History Summary | Recommended SID | Reason Quality |
|---:|---|---|---|
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO |

失败案例：

| Case | Problem | Example |
|---:|---|---|
| 1 | TODO | TODO |
| 2 | TODO | TODO |

## 今日技术理解

推荐 Reasoning 不一定是真正的多步逻辑推理，更准确地说，它通常是：

```text
基于用户历史兴趣的可解释推荐生成
```

面试时要诚实区分：

- 解释生成：模型生成看起来合理的理由。
- 真实推理：理由能被行为、类目和命中结果验证。

## GitHub 产出

```bash
git add docs reports configs scripts src
git commit -m "Day 6: add recommendation reasoning SFT experiment"
```

## 今日完成标准

```text
[ ] reasoning 数据已生成
[ ] 模型能输出推荐商品和推荐理由
[ ] 至少记录 3 条成功案例和 2 条失败案例
[ ] 能解释推荐解释和真实推理的区别
[ ] 完成 Day 6 commit
```

## 今日问题记录

```text
TODO: 记录理由模板化、理由和商品不一致、输出格式不稳定等问题。
```
