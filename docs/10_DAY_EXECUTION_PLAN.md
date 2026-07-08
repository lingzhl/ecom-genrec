# 10 天电商生成式推荐项目执行计划（1.5B 先复现版）

项目主题：面向电商场景的生成式商品推荐系统。

技术主线：

```text
Amazon Reviews 2023
-> 用户行为序列切分
-> KMeans SID / RQ-VAE-style SID
-> Baseline & Metrics
-> Qwen2.5-1.5B 三任务 SFT
-> 约束解码 / 合法 SID 控制
-> GRPO 后训练
-> 冷启动专项 + 多类目评测
-> GitHub + 简历 + 面试材料
```

## 每日固定产出

每天必须留下三类证据：

| 类型 | 产物 | 作用 |
|---|---|---|
| 技术博客 | `docs/dayXX_*.md` | 证明你理解了技术，不只是跑命令 |
| 实验产出 | `reports/`、截图、指标表 | 支撑简历中的真实数字 |
| GitHub 产出 | 每天至少 1 个 commit | 形成可展示的项目推进记录 |

## 推荐执行顺序

| Day | 主题 | 博客文件 | 核心实验 |
|---:|---|---|---|
| 1 | 环境 + Smoke Test | `docs/day01_environment.md` | 跑通最小闭环 |
| 2 | Amazon 数据处理 | `docs/day02_data.md` | 真实数据统计 |
| 3 | Baseline + 指标 | `docs/day03_baseline.md` | Popular/Text/Embedding |
| 4 | Semantic ID 双路径 | `docs/day04_semantic_id.md` | KMeans 与 RQ-VAE-style SID |
| 5 | Qwen2.5-1.5B 三任务 SFT | `docs/day05_sft.md` | 5k 样本 debug 训练 |
| 6 | 约束解码 + 推荐解释 | `docs/day06_reasoning.md` | 合法 SID 与解释生成 |
| 7 | GRPO | `docs/day07_grpo.md` | 7 类 reward 小规模跑通 |
| 8 | 冷启动 + 多类目扩展 | `docs/day08_large_eval.md` | 冷启动子集与 3 类目 baseline |
| 9 | 完整评测分析 | `docs/day09_analysis.md` | 主表/冷启动/消融 |
| 10 | 项目包装 | `docs/day10_resume.md` | README/简历/面试稿 |

## 每天结束 Checklist

```text
[ ] 今天的命令是否记录在 day 文档里？
[ ] 今天是否有实验结果、日志或截图？
[ ] 今天是否更新 README、docs 或 reports？
[ ] 今天是否写了遇到的问题和解决方案？
[ ] 今天是否完成一次 git commit？
[ ] 明天第一条命令是什么？
```

## Commit 规范

```text
Day 1: setup environment and smoke test pipeline
Day 2: process Amazon review sequences
Day 3: add recommendation baselines and metrics
Day 4: build semantic item ids for constrained generation
Day 5: run Qwen2.5-1.5B multitask SFT debug training
Day 6: add constrained decoding and reasoning outputs
Day 7: run GRPO post-training with recommendation rewards
Day 8: add cold-start and multi-category evaluation
Day 9: add full evaluation tables and error analysis
Day 10: finalize project documentation and resume materials
```

## 最终验收标准

- GitHub 仓库有完整 README、配置、脚本、docs 和实验表。
- `docs/` 中有 10 篇日更技术博客。
- `docs/RESULT_TABLES.md` 填入真实实验结果。
- 至少完成单类目 All_Beauty 的完整链路。
- 最理想完成 3 类目主实验和 Qwen2.5-1.5B SFT+GRPO 对比。
- 简历中的所有提升数字都能在 `reports/` 找到来源。
