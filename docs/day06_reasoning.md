# Day 6：电商推荐 Reasoning：推荐理由是解释，还是真推理？

## 今日目标

在 Day 5 三任务 LoRA SFT 的基础上，完成三件事：

1. 生成一版更正式的 reasoning instruction 数据。
2. 对比无约束 SID 解析和有约束 SID 后处理。
3. 记录推荐理由质量，包括成功案例、失败案例和模板化问题。

今天不要急着做 GRPO。Day 6 的重点是让推荐输出从“能评测”变成“能解释、能分析、能做约束控制”。

## 核心理解

这里的 reasoning 不是严格意义上的多步 CoT。更准确的说法是：

```text
基于用户历史商品、类目和近期兴趣摘要生成推荐解释。
```

面试里要诚实区分：

```text
解释生成：模型生成看起来合理的推荐理由。
真实推理：理由能被历史行为、目标商品类目、命中结果或人工规则验证。
```

当前项目里的推荐理由主要来自 `src/ecom_genrec/instruction.py::simple_reason`，所以它偏模板化。Day 6 要记录这个限制，而不是把它包装成强推理能力。

## Step 0：确认 Day 5 产物

确认本地有 Day 5 checkpoint 和评测报告：

```bash
ls artifacts/checkpoints/qwen25-1p5b-onerec
ls reports/day05_inference_test.json
git status --short
```

预期：

```text
checkpoint 目录存在
reports/day05_inference_test.json 存在
git status 没有未提交的 Day5 遗留改动
```

Day 5 结果作为 Day 6 对照：

```text
HR@10 = 0.251969
NDCG@10 = 0.235313
MRR@10 = 0.230396
ValidSID@10 = 0.992913
Cold HR@10 = 0.395833
```

## Step 1：构造 reasoning 数据

生成 Day 6 专用 instruction 数据，输出到新目录，避免覆盖 Day 5 debug 数据：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out-dir data/processed/all_beauty/instructions_reasoning \
  --with-reasoning \
  --task-mix onerec \
  --max-train-samples 10000
```

检查样本数量：

```bash
python3 -m json.tool data/processed/all_beauty/instructions_reasoning/counts.json
wc -l data/processed/all_beauty/instructions_reasoning/sft_train.jsonl
wc -l data/processed/all_beauty/instructions_reasoning/sft_valid.jsonl
wc -l data/processed/all_beauty/instructions_reasoning/sft_test.jsonl
```

抽查一条样本，确认 completion 同时包含 SID 和理由：

```bash
python3 - <<'PY'
import json
path = "data/processed/all_beauty/instructions_reasoning/sft_train.jsonl"
with open(path, encoding="utf-8") as f:
    row = json.loads(next(f))
print(row["prompt"][:800])
print("\n--- completion ---")
print(row["completion"])
print("\n--- task_type ---")
print(row.get("task_type"))
PY
```

合格标准：

```text
completion 里有 推荐商品：SID_xxx
completion 里有 推荐理由：
task_type 至少覆盖 sequence_recommendation、feature_alignment、history_fusion
```

实际生成结果：

| Split / Task | Samples |
|---|---:|
| sequence_train | 2244 |
| history_fusion_train | 2244 |
| feature_alignment_train | 431 |
| sft_train | 4919 |
| sft_valid | 762 |
| sft_test | 762 |

说明：Day 6 reasoning 数据覆盖了 `sequence_recommendation`、`history_fusion`、`feature_alignment` 三类任务；`completion` 中包含 `推荐商品：SID_xxx` 和 `推荐理由：` 字段。

## Step 2：先用 Day 5 checkpoint 做小样本对比

这一步不重新训练，先验证 Day 6 评测链路。

无约束 SID 解析：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec \
  --eval data/processed/all_beauty/instructions_reasoning/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --out reports/day06_sft_unconstrained_200.json \
  --max-samples 200
```

有约束 SID 后处理：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec \
  --eval data/processed/all_beauty/instructions_reasoning/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --out reports/day06_sft_constrained_200.json \
  --max-samples 200 \
  --constrained-sid
```

重点看这些字段：

```text
HR@10
NDCG@10
ValidSID@10
RawGeneratedSIDAvg
RawValidSIDRate
EmptySIDGenerationRate
ReasonFieldRate
```

解释方式：

```text
RawValidSIDRate：模型原始生成的 SID 中有多少是真实合法 SID。
ValidSID@10：经过解析、去重、fallback 补齐后的 Top-K 合法率。
ReasonFieldRate：输出里是否稳定包含 推荐理由 字段。
EmptySIDGenerationRate：模型是否存在完全没生成 SID 的情况。
```

## Step 3：训练 Day 6 reasoning checkpoint

如果 Step 2 链路正常，再训练 reasoning 版 SFT。注意使用新 output dir，不覆盖 Day 5 checkpoint：

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --eval data/processed/all_beauty/instructions_reasoning/sft_valid.jsonl \
  --output-dir artifacts/checkpoints/qwen25-1p5b-onerec-reasoning \
  --deepspeed configs/deepspeed_zero2.json \
  --attn-implementation auto
```

如果想先快速 smoke test，可以加 `--max-steps 20`：

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --eval data/processed/all_beauty/instructions_reasoning/sft_valid.jsonl \
  --output-dir artifacts/checkpoints/qwen25-1p5b-onerec-reasoning-debug \
  --deepspeed configs/deepspeed_zero2.json \
  --attn-implementation auto \
  --max-steps 20
```

训练时记录：

```text
最终 global_step
train_loss
eval_loss
eval_mean_token_accuracy
是否 OOM
是否 fallback 到 SDPA
```

实际训练记录：

```text
reasoning checkpoint: artifacts/checkpoints/qwen25-1p5b-onerec-reasoning
最终 global_step: 154 / 154
最后一次训练日志: step=150, loss=0.131127, mean_token_accuracy=0.948401
最终验证: eval_loss=0.115737, eval_mean_token_accuracy=0.953685
是否 OOM: 否
attention: 训练时曾因 flash-attn wheel 不含 sm_120 kernel fallback 到 SDPA；会话末已用 CUDA 12.8 重新编译 flash-attn，当前 auto 可选择 flash_attention_2。
```

Debug 训练记录：

```text
debug checkpoint: artifacts/checkpoints/qwen25-1p5b-onerec-reasoning-debug
global_step: 20 / 20
eval_loss: 0.207635
eval_mean_token_accuracy: 0.929150
```

可选显存日志：

```bash
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > reports/day06_gpu_memory.csv
```

## Step 4：评测 reasoning checkpoint

先跑 5 条 smoke test：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec-reasoning \
  --eval data/processed/all_beauty/instructions_reasoning/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --out reports/day06_reasoning_smoke_5.json \
  --max-samples 5 \
  --constrained-sid
```

再跑完整测试集，无约束版本：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec-reasoning \
  --eval data/processed/all_beauty/instructions_reasoning/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --out reports/day06_reasoning_unconstrained.json
```

完整测试集，有约束版本：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec-reasoning \
  --eval data/processed/all_beauty/instructions_reasoning/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_reasoning/sft_train.jsonl \
  --out reports/day06_reasoning_constrained.json \
  --constrained-sid
```

实际产出文件：

```text
reports/day06_reasoning_smoke_5.json
reports/day06_reasoning_test_constrained.json
reports/day06_sft_unconstrained_200.json
reports/day06_sft_constrained_200.json
```

说明：本次 Day 6 完整测试集已完成 constrained 后处理版本，文件名为 `reports/day06_reasoning_test_constrained.json`；完整 unconstrained reasoning 版本本次未产出。

## Step 5：整理对比表

把结果填到这里：

| Result Set | Model | Decode | Samples | HR@10 | NDCG@10 | ValidSID@10 | RawValidSIDRate | ReasonFieldRate | Cold HR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Day5 full baseline | Day5 SFT | standard eval | 762 | 0.251969 | 0.235313 | 0.992913 | N/A | N/A | 0.395833 |
| Day6 200 | Day5 SFT | unconstrained | 200 | 0.050000 | 0.026470 | 1.000000 | 1.000000 | 1.000000 | 0.064935 |
| Day6 200 | Day5 SFT | constrained post-filter | 200 | 0.050000 | 0.026470 | 1.000000 | 1.000000 | 1.000000 | 0.064935 |
| Day6 full | Day6 Reasoning SFT | constrained post-filter | 762 | 0.250656 | 0.234000 | 1.000000 | 0.938320 | 0.952756 | 0.395833 |

对比结论：

```text
Day6 200 条无约束和有约束结果完全一致，因为原始生成 SID 已经全部合法，post-filter 没有可过滤对象。
Day6 full constrained 的 HR@10=0.250656，和 Day5 full baseline 的 0.251969 基本持平。
Day6 full constrained 的 NDCG@10=0.234000，和 Day5 full baseline 的 0.235313 基本持平。
Day6 full constrained 的 ValidSID@10=1.000000，高于 Day5 full baseline 的 0.992913。
Day6 full constrained 的 RawValidSIDRate=0.938320，说明仍有少量原始 SID 不合法，需要后处理或真正 constrained decoding。
Day6 full constrained 的 ReasonFieldRate=0.952756，说明大部分输出包含推荐理由字段，但格式稳定性还没到 100%。
Cold HR@10=0.395833，和 Day5 full baseline 持平；冷启动高分需要继续检查切分和 fallback 机制。
```

判断标准：

```text
如果 constrained 的 ValidSID@10 更高，但 HR@10 不下降太多，说明约束有价值。
如果 ReasonFieldRate 接近 1.0，说明输出格式稳定。
如果 RawValidSIDRate 已经很高，说明模型本身已经学会 SID 格式。
如果 HR@10 明显低于 Day5，说明 reasoning 训练可能牺牲了推荐准确性。
```

## Step 6：记录成功和失败案例

从报告里的 `examples` 先挑 5 条，再结合原始测试样本人工判断。

成功案例表：

| Case | History Summary | Recommended SID | Reason Quality |
|---:|---|---|---|
| 1 | All Beauty 历史兴趣 | SID_029_029_029_048 | SID 合法，包含推荐商品和推荐理由字段，解释与类目一致 |
| 2 | All Beauty 历史兴趣 | SID_023_003_005_004 | SID 合法，输出格式完整，说明模型能稳定生成推荐理由模板 |
| 3 | All Beauty 历史兴趣 | SID_012_012_012_012 | SID 合法，推荐理由引用了近期关注和历史兴趣 |

失败案例表：

| Case | Problem | Example |
|---:|---|---|
| 1 | 理由过于模板化 | 多数样例都使用“用户近期多次关注 All Beauty 相关商品，历史兴趣与 All Beauty 类目保持一致” |
| 2 | 字段格式未完全稳定 | Day6 full constrained 的 ReasonFieldRate=0.952756，约 4.7% 输出没有匹配到标准 `推荐理由` 字段 |
| 3 | 原始 SID 仍有非法生成 | Day6 full constrained 的 RawValidSIDRate=0.938320，说明 post-filter 前仍有约 6.2% 原始 SID 不合法 |
| 4 | 命中指标没有随 reasoning 明显提升 | Day6 full constrained HR@10=0.250656，略低于 Day5 full baseline HR@10=0.251969 |

写案例时优先记录这些问题：

```text
SID 是否合法
是否包含 推荐商品 和 推荐理由 两个字段
推荐理由有没有引用历史兴趣
推荐理由是否只是在复读类目
命中失败时是否仍然类目一致
是否过度推荐热门 SID
```

## Step 7：更新结果文档

更新两个文件：

```text
docs/day06_reasoning.md
docs/RESULT_TABLES.md
```

`docs/RESULT_TABLES.md` 里至少补上：

```text
Qwen2.5-1.5B SFT
Qwen2.5-1.5B SFT + reasoning
无约束解码
有约束解码
```

如果 Day 6 只完成了小样本 200 条对比，需要明确标注 `max_samples=200`，不要和完整测试集结果混在一起。

## GitHub 产出

```bash
git add docs/day06_reasoning.md docs/RESULT_TABLES.md scripts/evaluate_llm.py reports/day06*.json
git commit -m "Day 6: add reasoning evaluation and constrained SID analysis"
git push origin main
```

注意：checkpoint 目录默认被 `.gitignore` 忽略，不要直接推模型大文件。

## 今日问题记录

已记录：

```text
理由是否模板化：是，当前 simple_reason 风格明显，主要复述类目和近期兴趣。
理由和商品是否一致：大体一致，CategoryConsistency@10=0.885302。
输出格式是否稳定：较稳定，但 ReasonFieldRate=0.952756，还没到 1.0。
RawValidSIDRate 和 ValidSID@10 是否差距过大：有差距，RawValidSIDRate=0.938320，ValidSID@10=1.000000，说明 fallback / post-filter 对最终合法率有帮助。
约束后处理是否提高合法率但影响 HR/NDCG：200 条对比中无影响，因为原始 SID 全合法；完整 Day6 constrained 结果与 Day5 基本持平。
冷启动子集是否仍然异常高：是，Cold HR@10=0.395833，高于整体 HR@10=0.250656，需要继续排查冷启动定义、候选补齐和热门 fallback。
```

## 今日需要掌握的知识点

```text
1. 推荐解释不等于真实推理：当前 reasoning 更接近可读解释生成，不能包装成严格 CoT。
2. HR@K、NDCG@K、MRR@K 的含义：HR 看是否命中，NDCG 看命中位置折扣，MRR 看第一个命中位置。
3. ValidSID@K 和 RawValidSIDRate 的区别：前者是最终推荐列表合法率，后者是模型原始输出合法率。
4. constrained_sid 当前是后处理过滤，不是真正 constrained decoding；真正约束需要在生成阶段限制合法 SID 前缀。
5. ReasonFieldRate 衡量输出格式稳定性，不衡量理由质量。
6. LoRA SFT 的 checkpoint 只提交评测结果和文档，不直接提交大模型权重。
7. torchrun + DeepSpeed ZeRO-2 用于多卡训练；--max-steps 20 是 debug run，不代表完整训练。
8. FlashAttention 2 依赖 GPU 架构匹配；Blackwell sm_120 需要包含 sm_120 kernel 的 wheel 或从源码编译。
9. 完整评测和 200 条小样本不能混在同一结论里，小样本只用于链路验证。
10. 冷启动子集高分要谨慎解释，可能受数据切分、热门 fallback 或类目一致性影响。
```

## 今日完成标准

```text
[x] reasoning 数据已生成
[x] Day5 checkpoint 完成无约束 vs 有约束小样本对比
[x] reasoning checkpoint 已训练或完成 debug 训练
[x] reasoning checkpoint 能输出推荐商品和推荐理由
[x] 至少记录一组无约束 vs 有约束解码对比
[x] 至少记录 3 条成功案例和 2 条失败案例
[x] 能解释推荐解释和真实推理的区别
[x] 更新 RESULT_TABLES
[x] 完成 Day 6 commit
```
