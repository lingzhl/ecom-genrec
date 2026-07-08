# Day 5：第一次训练 Qwen2.5-1.5B：三任务 LoRA SFT 在推荐任务里学到了什么？

## 今日目标

用 1k-5k 样本 debug Qwen2.5-1.5B LoRA SFT，确认三任务训练链路可跑。

## 学习重点

- LoRA。
- bf16。
- DeepSpeed ZeRO-2。
- SFTTrainer。
- loss 曲线。
- 三任务 SFT。

## 执行命令

构建 debug 数据：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out-dir data/processed/all_beauty/instructions_debug \
  --with-reasoning \
  --task-mix onerec \
  --max-train-samples 5000
```

启动 SFT：

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_debug/sft_train.jsonl \
  --eval data/processed/all_beauty/instructions_debug/sft_valid.jsonl \
  --deepspeed configs/deepspeed_zero2.json
```

训练中观察：

```text
loss 是否下降
显存是否稳定
是否保存 checkpoint
tokens/s 或 step/s
```

## 实验记录

训练配置：

| Config | Value |
|---|---|
| model | Qwen2.5-1.5B-Instruct |
| LoRA r | TODO |
| LoRA alpha | TODO |
| learning rate | TODO |
| max sequence length | TODO |
| batch size | TODO |
| gradient accumulation | TODO |
| checkpoint | TODO |

三任务样本数：

| Task | Train Samples | Note |
|---|---:|---|
| sequence_recommendation | TODO | 历史序列预测下一个 SID |
| feature_alignment | TODO | 商品属性与 SID 对齐 |
| history_fusion | TODO | 多段历史融合生成推荐 |

Loss 记录：

```text
TODO: paste several training log lines
```

## 今日技术理解

### 为什么用 LoRA

LoRA 只训练少量适配参数，不需要更新全部 1.5B 参数，显存和训练成本更低，也更适合先复现完整方法。

### 为什么用 DeepSpeed ZeRO-2

ZeRO-2 会切分优化器状态和梯度，适合多卡训练大模型。

### 为什么要做三任务 SFT

OneRec 风格不是只做单一“历史 -> 下一个商品”任务，而是让模型同时学：

- 序列推荐
- 特征对齐
- 历史融合

这样后面做约束解码和 GRPO 时，模型基础能力会更完整。

## GitHub 产出

```bash
git add configs scripts src docs
git commit -m "Day 5: run Qwen2.5-1.5B multitask SFT debug training"
```

## 今日完成标准

```text
[ ] 1.5B SFT 能启动
[ ] loss 正常下降
[ ] checkpoint 已保存
[ ] 三任务数据来源已搞清楚
[ ] 记录显存占用
[ ] 完成 Day 5 commit
```

## 今日问题记录

```text
TODO: 记录 CUDA OOM、flash attention、TRL 版本、DeepSpeed 配置等问题。
```
