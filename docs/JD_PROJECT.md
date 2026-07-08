# JD 对齐项目说明

## 项目定位

面向电商场景的生成式商品推荐系统，覆盖生成式推荐、推荐 Reasoning、SFT 指令微调、GRPO 后训练、表征学习和大规模评测。

## 与岗位职责的对应关系

| JD 要求 | 项目对应实现 |
|---|---|
| 生成式推荐模型研发 | 将用户历史行为序列转化为 LLM 生成下一个商品 SID 的任务 |
| 推荐 Reasoning 探索 | 构造“推荐商品 + 推荐理由”SFT 数据，并在 GRPO 中加入推荐理由奖励 |
| Post-training & Alignment | 使用 Qwen2.5-14B-Instruct 进行 LoRA SFT，再用 GRPO 做偏好优化 |
| 表征学习 | 使用商品文本 embedding 和层级聚类构建 Semantic ID |
| 大规模评测 | 多类目 Amazon Reviews 2023、HR/NDCG/MRR/Coverage/Valid SID、冷启动和长尾分群 |
| 前沿技术转化 | 把 LLM 后训练技术落到电商推荐业务目标：命中、排序、合法性、多样性 |

## 可交付材料

- 数据统计表：用户数、商品数、交互数、训练/验证/测试样本数、冷启动比例、长尾比例。
- 主结果表：Popular、Text/BM25、Embedding Retrieval、Qwen2.5-14B SFT、Qwen2.5-14B SFT+GRPO、Qwen2.5-32B Eval。
- 消融表：无 SID/有 SID、SFT/SFT+GRPO、无 Reasoning/有 Reasoning、14B/32B、单类目/多类目。
- Demo：输入用户历史商品，输出 Top-10 推荐商品和推荐理由。
- 简历 bullet：见 README 中模板，最终用真实实验数字替换占位。
