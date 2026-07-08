# 面向电商场景的生成式商品推荐系统

This project implements a large-evaluation generative recommendation pipeline aligned with post-training / recommendation algorithm roles:

- Amazon Reviews 2023 multi-category sequence recommendation data.
- Semantic item IDs for constrained generation.
- Popular, BM25/text, and embedding retrieval baselines.
- Qwen2.5-14B LoRA SFT and TRL GRPO post-training.
- Qwen2.5-32B scale evaluation entrypoint.
- HR@K, NDCG@K, MRR@K, catalog coverage, valid SID rate, cold-user and long-tail reports.

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
- [Day 5：Qwen2.5-14B SFT](docs/day05_sft.md)
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

## Build SFT / GRPO Datasets

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out-dir data/processed/main/instructions \
  --with-reasoning
```

## Baseline and Metrics

```bash
python3 scripts/evaluate_baselines.py \
  --config configs/default.yaml \
  --processed-dir data/processed/main \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/baselines.json
```

## Qwen2.5-14B SFT

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
  --sft-checkpoint artifacts/checkpoints/qwen25-14b-genrec \
  --deepspeed configs/deepspeed_zero2.json
```

## LLM Evaluation

```bash
python3 scripts/evaluate_llm.py \
  --config configs/default.yaml \
  --model artifacts/checkpoints/qwen25-14b-genrec-grpo \
  --eval data/processed/main/instructions/sft_test.jsonl \
  --sid-map artifacts/main/sid_map.json \
  --out reports/main/qwen25_14b_grpo_eval.json
```

## Resume Bullet Template

```markdown
### 面向电商场景的生成式商品推荐系统 | Qwen2.5-14B/32B, SFT, GRPO, Semantic ID

- 基于 Amazon Reviews 2023 多类目数据构建大规模电商序列推荐数据集，完成用户行为时间序列切分、商品元数据融合、长尾/冷启动用户分群评测，形成覆盖 HR@K、NDCG@K、MRR、Catalog Coverage、Valid SID Rate 的完整评测体系。
- 设计层级商品语义 ID 表示方案，基于商品标题、类目和描述构建 embedding 聚类编码，将开放式商品文本生成转化为受约束商品 ID 生成，降低推荐幻觉并提升生成结果可控性。
- 基于 Qwen2.5-14B-Instruct 进行 LoRA SFT，构造“用户历史行为 -> 推荐商品 SID + 推荐理由”的高质量指令数据，探索电商推荐场景下的 Reasoning/CoT 推荐解释能力。
- 使用 TRL GRPO 进行模型后训练与偏好对齐，设计命中率、排序位置、类目一致性、合法 SID、多样性和推荐理由一致性等奖励函数，优化生成式推荐模型的命中率和稳定性。
- 对比 Popular、Embedding Retrieval、SFT、SFT+GRPO、Qwen2.5-32B 等多组实验，并完成多类目、大测试集、冷启动用户、长尾商品和消融实验分析，为推荐效果优化提供数据支撑。
```
