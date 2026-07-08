# 面向电商场景的生成式商品推荐系统

This project implements a OneRec-style generative recommendation reproduction pipeline centered on Qwen2.5-1.5B:

- Amazon Reviews 2023 multi-category sequence recommendation data.
- Dual semantic ID paths: hierarchical KMeans SID and RQ-VAE-style residual quantization SID.
- Popular, BM25/text, and embedding retrieval baselines.
- Qwen2.5-1.5B multitask SFT and TRL GRPO post-training.
- Cold-start subset evaluation, novelty, coverage, valid SID rate, and ablation-ready reports.
- Post-10-day scale-up path for 7B/14B comparisons.

## Quick Smoke Test

The smoke test uses synthetic data and standard-library fallbacks, so it can run before installing the heavy training stack.

```bash
cd /data1/zhl/ecom-genrec
python3 scripts/run_smoke.py
```

Expected outputs:

- `data/processed/smoke/train.jsonl`
- `data/processed/smoke/valid.jsonl`
- `data/processed/smoke/test.jsonl`
- `artifacts/smoke/sid_map.json`
- `reports/smoke/baselines.json`
- `reports/smoke/summary.md`

## 10-Day Execution Plan

This repository is designed for a 10-day hands-on sprint. Each day has three required outputs: a technical blog, an experiment artifact, and a GitHub commit.

Start here:

- [10 天总执行计划](docs/10_DAY_EXECUTION_PLAN.md)
- [Day 1：环境搭建 + Smoke Test](docs/day01_environment.md)
- [Day 2：Amazon 数据处理](docs/day02_data.md)
- [Day 3：Baseline + 指标](docs/day03_baseline.md)
- [Day 4：Semantic ID 商品语义编码](docs/day04_semantic_id.md)
- [Day 5：Qwen2.5-1.5B SFT](docs/day05_sft.md)
- [Day 6：Reasoning 推荐解释](docs/day06_reasoning.md)
- [Day 7：GRPO 后训练](docs/day07_grpo.md)
- [Day 8：多类目大规模评测](docs/day08_large_eval.md)
- [Day 9：完整评测与错误分析](docs/day09_analysis.md)
- [Day 10：GitHub 包装 + 简历 + 面试稿](docs/day10_resume.md)

Daily checklist:

```text
[ ] 今天是否有技术博客 md？
[ ] 今天是否有实验结果或截图？
[ ] 今天是否更新 README 或 docs？
[ ] 今天是否 commit？
[ ] 今天是否记录失败问题？
[ ] 今天是否知道明天第一步做什么？
```

## Install Full Environment

```bash
cd /data1/zhl/ecom-genrec
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Data Preparation

For local Amazon Reviews 2023 files, put review and metadata JSONL/JSONL.GZ files under `data/raw`.

Example:

```bash
python3 scripts/process_amazon.py \
  --config configs/default.yaml \
  --reviews data/raw/All_Beauty.jsonl.gz data/raw/Baby_Products.jsonl.gz data/raw/Sports_and_Outdoors.jsonl.gz \
  --metadata data/raw/meta_All_Beauty.jsonl.gz data/raw/meta_Baby_Products.jsonl.gz data/raw/meta_Sports_and_Outdoors.jsonl.gz \
  --out-dir data/processed/main
```

## Semantic ID

```bash
python3 scripts/build_semantic_ids.py \
  --config configs/default.yaml \
  --items data/processed/main/items.jsonl \
  --out artifacts/main/sid_map.json
```

RQ-VAE-style residual quantization path:

```bash
python3 scripts/build_rqvae_ids.py \
  --config configs/default.yaml \
  --items data/processed/main/items.jsonl \
  --out artifacts/main/sid_map_rqvae.json
```

## Build SFT / GRPO Datasets

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out-dir data/processed/main/instructions \
  --with-reasoning \
  --task-mix onerec
```

## Baseline and Metrics

```bash
python3 scripts/evaluate_baselines.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/baselines.json
```

## Qwen2.5-1.5B SFT

```bash
accelerate launch --num_processes 2 --config_file configs/accelerate_zero2.yaml \
  scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/main/instructions/sft_train.jsonl \
  --eval data/processed/main/instructions/sft_valid.jsonl
```

If you do not use an accelerate config, pass DeepSpeed directly:

```bash
torchrun --nproc_per_node=2 scripts/train_sft.py \
  --config configs/default.yaml \
  --train data/processed/main/instructions/sft_train.jsonl \
  --eval data/processed/main/instructions/sft_valid.jsonl \
  --deepspeed configs/deepspeed_zero2.json
```

## GRPO Post-Training

```bash
torchrun --nproc_per_node=2 scripts/train_grpo.py \
  --config configs/default.yaml \
  --train data/processed/main/instructions/grpo_train.jsonl \
  --sid-map artifacts/main/sid_map.json \
  --sft-checkpoint artifacts/checkpoints/qwen25-1p5b-onerec \
  --deepspeed configs/deepspeed_zero2.json
```

## LLM Evaluation

```bash
python3 scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-1p5b-onerec-grpo \
  --eval data/processed/main/instructions/sft_test.jsonl \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/qwen25_1p5b_grpo_eval.json \
  --train-reference data/processed/main/instructions/sft_train.jsonl
```

## Resume Bullet Template

```markdown
### 面向电商场景的生成式商品推荐系统 | Qwen2.5-1.5B, KMeans/RQ-VAE SID, SFT, GRPO

- 基于 Amazon Reviews 2023 构建 OneRec 风格生成式推荐数据集，完成用户行为时间序列切分、冷启动用户分群评测与多类目扩展。
- 设计 KMeans SID 与 RQ-VAE-style SID 双语义编码路径，将开放式商品生成转化为受约束商品 ID 生成，统计合法 SID 率与码本利用率。
- 基于 Qwen2.5-1.5B-Instruct 进行三任务 SFT，覆盖序列推荐、特征对齐、历史融合，并加入约束解码减少非法 SID。
- 使用 TRL GRPO 进行 7 类奖励后训练，覆盖 Hit、NDCG、Category、Valid SID、Diversity、Novelty、Reasoning，并补充去偏奖励与部分匹配奖励。
- 输出 HR@10、NDCG@10、Catalog Coverage、Novelty、Valid SID Rate 以及冷启动子集指标，后续扩展到 7B/14B 做规模对比。
```
