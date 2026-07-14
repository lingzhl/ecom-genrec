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
| model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA r | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| learning rate | 1e-4 |
| max sequence length | 1024 |
| per-device batch size | 1 |
| GPU 数 | 2 |
| gradient accumulation | 16 |
| effective global batch size | 32 |
| bf16 | true |
| DeepSpeed | ZeRO-2 |
| attention backend | auto 探测后回退到 PyTorch SDPA |
| train epochs / steps | 1 epoch / 154 steps |
| checkpoint | `artifacts/checkpoints/qwen25-1p5b-onerec/checkpoint-154` |
| final adapter path | `artifacts/checkpoints/qwen25-1p5b-onerec` |

三任务样本数：

| Task | Train Samples | Note |
|---|---:|---|
| sequence_recommendation | 2244 | 历史序列预测下一个 SID |
| feature_alignment | 431 | 商品属性与 SID 对齐 |
| history_fusion | 2244 | 多段历史融合生成推荐 |
| total SFT train | 4919 | `sft_train.jsonl` |
| valid / test | 762 / 762 | `sft_valid.jsonl` / `sft_test.jsonl` |

Loss 记录：

```text
step=10   loss=0.7527  mean_token_accuracy=0.8118  lr=9.42e-5
step=20   loss=0.2294  mean_token_accuracy=0.9173  lr=8.77e-5
step=50   loss=0.1710  mean_token_accuracy=0.9342  lr=6.82e-5
step=100  loss=0.1309  mean_token_accuracy=0.9486  lr=3.57e-5
step=150  loss=0.1312  mean_token_accuracy=0.9480  lr=3.25e-6
step=154  eval_loss=0.1159  eval_mean_token_accuracy=0.9545

train_runtime=215.359s
train_samples_per_second=22.842
train_steps_per_second=0.715
train_loss=0.1893
```

显存记录：

```text
本次训练没有保存 nvidia-smi 采样日志，因此不能事后还原精确峰值。
训练结束后 GPU 空闲占用约 14 MiB / 97887 MiB，训练过程无 OOM。
下次训练可并行启动采样：

watch -n 1 nvidia-smi

或保存日志：

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > reports/day05_gpu_memory.csv
```

## 推理测试

先用 5 条测试样本做 smoke test，确认 LoRA checkpoint 可以加载，生成结果能解析成合法 SID：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec \
  --eval data/processed/all_beauty/instructions_debug/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_debug/sft_train.jsonl \
  --out reports/day05_inference_smoke_5.json \
  --max-samples 5
```

本次 smoke test 结果：

```text
samples=5
HR@5=0.0
HR@10=0.0
HR@20=0.0
ValidSID@5=1.0
ValidSID@10=1.0
ValidSID@20=1.0
```

5 条样本的 HR 不具备统计意义，主要用于验证推理链路。正式评估时去掉 `--max-samples 5`：

```bash
conda run -n ecom-genrec python scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec \
  --eval data/processed/all_beauty/instructions_debug/sft_test.jsonl \
  --sid-map artifacts/all_beauty/sid_map.json \
  --train-reference data/processed/all_beauty/instructions_debug/sft_train.jsonl \
  --out reports/day05_inference_test.json
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
git add docs/day05_sft.md scripts/train_sft.py reports/day05_inference_smoke_5.json reports/day05_inference_test.json
git commit -m "Day 5: run Qwen2.5-1.5B multitask SFT debug training"
```

## 今日完成标准

```text
[x] 1.5B SFT 能启动
[x] loss 正常下降
[x] checkpoint 已保存
[x] 三任务数据来源已搞清楚
[~] 记录显存占用：本次未保存峰值采样，下次需训练时同步记录
[x] 推理 smoke test 能跑通
[x] 完成 Day 5 commit
```

## 今日问题记录

### 问题：`flash-attn` 缺失或 CUDA kernel 不兼容导致训练失败

在 conda 环境 `ecom-genrec` 中启动 Qwen2.5 LoRA SFT 前，环境缺少 `flash-attn` 时，相关代码或依赖在导入 FlashAttention 时会失败。典型表现是：

```text
ModuleNotFoundError: No module named 'flash_attn'
```

或在安装 `flash-attn` 时没有匹配当前 Python / PyTorch / CUDA 组合的预编译 wheel，导致 pip 需要从源码编译。即使安装成功，如果编译出的 CUDA kernel 不支持当前 GPU 架构，也会在训练 forward 阶段失败。这次在 NVIDIA RTX PRO 6000 Blackwell 上复现到的关键报错是：

```text
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

排查思路：

```bash
conda activate ecom-genrec
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import flash_attn"
```

确认当前环境路径：

```text
/home/user/anaconda3/envs/ecom-genrec/lib/python3.10/site-packages
```

解决方法：

```bash
conda activate ecom-genrec
pip install flash-attn --no-build-isolation
```

这次安装没有可用的预编译 wheel，因此实际是从源码编译，耗时较久。安装完成后验证：

```bash
python - <<'PY'
import flash_attn
from flash_attn import flash_attn_func
print("flash_attn version:", flash_attn.__version__)
print("flash_attn_func import ok")
PY
```

验证结果：

```text
flash_attn version: 2.8.3.post1
flash_attn_func import ok
```

结论：`flash-attn` 已成功安装到 `ecom-genrec` 环境，后续如果再次遇到 FlashAttention 导入失败，优先检查是否进入了正确 conda 环境，以及 `torch` / `cuda` / `flash-attn` 版本是否匹配。

本机继续训练时不要强制 `flash_attention_2`。`scripts/train_sft.py` 已改为默认 `--attn-implementation auto`：先真实运行一个极小 FlashAttention kernel，能跑就用 `flash_attention_2`，不能跑就自动退回 PyTorch `sdpa`。也可以显式指定：

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/all_beauty/instructions_debug/sft_train.jsonl \
  --eval data/processed/all_beauty/instructions_debug/sft_valid.jsonl \
  --deepspeed configs/deepspeed_zero2.json \
  --attn-implementation sdpa
```
