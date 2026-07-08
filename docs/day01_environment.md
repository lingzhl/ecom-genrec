# Day 1：从传统推荐到生成式推荐：为什么要让 LLM 生成商品 ID？

## 今日目标

跑通最小闭环，不碰真实大数据和 14B 训练。

你今天要理解：
这个项目最终要做的是：

```text
构建一个面向电商场景的生成式商品推荐系统。
```

更具体一点：

```text
给定用户历史浏览/购买/评论过的商品序列，让大模型生成用户下一个可能感兴趣的商品 Semantic ID，并给出推荐理由；再用 SFT 和 GRPO 优化推荐效果，最后用 HR@10、NDCG@10 等指标评测。
```

---

**输入是什么？**

核心输入是用户历史商品序列。

例如：

```text
用户历史商品：
1. 洁面乳
2. 爽肤水
3. 面霜
```

在项目里会被转成 Semantic ID：

```text
用户历史商品：
1. SID_012_003_021 | cleansing oil | beauty
2. SID_012_008_044 | facial toner | beauty
3. SID_019_002_031 | moisturizer | beauty
```

模型最终看到的是类似 prompt：

```text
根据用户历史购买/评论商品，预测用户下一个可能感兴趣的商品，并给出推荐理由。

用户历史商品：
1. SID_012_003_021 | cleansing oil | beauty
2. SID_012_008_044 | facial toner | beauty
3. SID_019_002_031 | moisturizer | beauty
```

---

**输出是什么？**

输出是推荐商品的 Semantic ID，加上推荐理由。

例如：

```text
推荐商品：SID_019_006_088
推荐理由：用户近期持续关注基础护肤和保湿相关商品，因此推荐同类目的保湿精华商品。
```

然后系统会把：

```text
SID_019_006_088
```

映射回真实商品：

```text
商品标题
商品类目
商品描述
商品 ID
```

---

**有哪些模块？**

这个项目可以分成 7 个模块。

**1. 数据处理模块**

对应代码：

```text
src/ecom_genrec/data.py
scripts/process_amazon.py
```

作用：

```text
读取 Amazon Reviews 2023
清洗用户行为和商品信息
按用户时间序列切分 train/valid/test
生成数据统计
```

---

**2. Semantic ID 模块**

对应代码：

```text
src/ecom_genrec/semantic_id.py
scripts/build_semantic_ids.py
```

作用：

```text
把商品 title/category/description/price 转成向量
再通过层级聚类生成 SID
```

例如：

```text
item_id: B00XXXXX
sid: SID_018_042_007
```

---

**3. 指令数据构造模块**

对应代码：

```text
src/ecom_genrec/instruction.py
scripts/build_instruction_data.py
```

作用：

```text
把推荐任务转成大模型 SFT 数据
```

也就是：

```text
用户历史商品 SID -> 下一个商品 SID + 推荐理由
```

---

**4. Baseline 模块**

对应代码：

```text
src/ecom_genrec/baselines.py
scripts/evaluate_baselines.py
```

作用：

```text
实现 Popular baseline
实现 Text Retrieval baseline
实现 Embedding Retrieval baseline
```

这些 baseline 用来证明大模型不是“自嗨”，必须和简单方法对比。

---

**5. 评测指标模块**

对应代码：

```text
src/ecom_genrec/metrics.py
```

作用：

```text
计算 HR@K
计算 NDCG@K
计算 MRR@K
计算 Coverage
计算 Valid SID Rate
计算 Long-tail Ratio
计算 Category Consistency
```

---

**6. SFT 训练模块**

对应代码：

```text
scripts/train_sft.py
```

作用：

```text
用 Qwen2.5-14B-Instruct 做 LoRA SFT
让模型学会根据用户历史生成推荐 SID
```

---

**7. GRPO 后训练模块**

对应代码：

```text
scripts/train_grpo.py
```

作用：

```text
在 SFT 模型基础上继续用 reward 优化
```

reward 包括：

```text
命中奖励
排序奖励
类目一致奖励
合法 SID 奖励
多样性奖励
推荐理由奖励
```

---

**一句话总结**

```text
输入：用户历史商品序列
处理：商品转 Semantic ID，构造 SFT 数据，训练 Qwen2.5-14B，再用 GRPO 对齐推荐目标
输出：推荐商品 SID + 推荐理由
评测：HR@10、NDCG@10、MRR、Coverage、Valid SID Rate
```

## 我今天看懂了哪些代码

### 我看懂了 scripts/run_smoke.py
输入：合成商品和合成用户行为
处理：切分数据、生成 SID、构造指令数据、跑 baseline
输出：`reports/smoke/summary.md`

### 我看懂了 src/ecom_genrec/data.py
输入：Amazon review 文件提供用户行为，metadata 文件提供商品信息
处理：标准化字段、过滤稀疏用户和商品、按 `user_id` 分组、按 `timestamp` 排序、用历史行为预测未来商品
输出：`train.jsonl`、`valid.jsonl`、`test.jsonl`、`stats.json`

### 我看懂了 src/ecom_genrec/semantic_id.py
输入：商品的 `title/category/description/price`
处理：商品文本 -> embedding -> 层级聚类 -> SID；当前先用 KMeans/哈希 fallback 跑通，不是 RQ-VAE
输出：`sid_map.json`

### 我看懂了 src/ecom_genrec/instruction.py
输入：序列样本里的 `history item_ids` 和 `target_item`，以及 `sid_map.json`
处理：把历史商品转成 `history_sid`，再构造 `prompt + completion` 形式的 SFT 样本
输出：`sft_train.jsonl`、`sft_valid.jsonl`、`sft_test.jsonl`

### 我看懂了 src/ecom_genrec/baselines.py
输入：训练/测试 instruction 数据和 SID 映射
处理：分别跑 Popular、Text Retrieval、Embedding Retrieval 三种 baseline，并生成 Top-K 推荐结果
输出：每个 baseline 的推荐列表，供后续指标评测使用

### 我看懂了 src/ecom_genrec/metrics.py
输入：真实目标商品和模型/baseline 的 Top-K 推荐列表
处理：计算 `HR@K`、`NDCG@K`、`MRR@K`、`Coverage`、`ValidSID` 等指标
输出：可写入 `baselines.json`、`summary.md` 和简历结果表的指标字典

## 我怎么理解这次 smoke test 结果

### baseline 表怎么读

这次 smoke test 的 baseline 结果是：

| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Popular | 0.0000 | 0.0000 | 0.0000 | 0.2083 |
| TextRetrieval | 1.0000 | 1.0000 | 1.0000 | 0.9375 |
| EmbeddingRetrieval | 1.0000 | 1.0000 | 1.0000 | 0.7292 |

我的理解：

- `Popular` 全为 0，说明在这份合成数据里，用户下一步行为不是靠“最热门商品”决定的。
- `TextRetrieval` 和 `EmbeddingRetrieval` 都达到 1.0，说明这份 smoke 数据里商品文本和用户兴趣模式非常规则，检索法可以轻松命中真实商品。
- `Coverage@10` 不一样，说明不同 baseline 即使命中率一样，覆盖的商品范围也会不同；Text Retrieval 在这份数据上覆盖更广。

Day 1 我不把这些数字当作最终效果，而是把它们当作“baseline 和指标评测链路已经跑通”的证据。

### 一条 SFT 样本怎么读

我看到的一条训练样本大致是：

```json
{
  "user_id": "USER_000",
  "history_item": ["ITEM_000"],
  "history_sid": ["SID_005_005_005"],
  "target_item": "ITEM_006",
  "target_sid": "SID_005_005_005_003"
}
```

以及它对应的训练文本：

```text
Instruction:
根据用户历史购买/评论商品，预测用户下一个可能感兴趣的商品，并给出简短推荐理由。

Input:
用户历史商品：
1. SID_005_005_005 | Beauty product 000 | Beauty

Output:
推荐商品：SID_005_005_005_003
推荐理由：用户近期多次关注 Beauty 相关商品，历史兴趣与 Beauty 类目保持一致，因此推荐该商品作为下一步候选。
```

我的理解：

- 原始数据先是 `history_item -> target_item`。
- 再通过 `sid_map.json` 映射成 `history_sid -> target_sid`。
- `instruction.py` 把它包装成 `prompt + completion`。
- `text = prompt + completion` 就是后面 SFT 真正要学习的训练样本。

这说明项目已经成功把推荐问题改写成了大模型可以学习的生成任务：输入历史商品，输出推荐商品 SID 和推荐理由。


## 生成式推荐和传统推荐的区别

传统推荐系统通常是“打分排序”：

```text
输入：user_id、item_id、用户特征、商品特征
输出：每个商品的分数
最后：按分数排序，取 Top-K 商品
```

例如：

```text
用户 A 对商品 1 的分数 = 0.91
用户 A 对商品 2 的分数 = 0.76
用户 A 对商品 3 的分数 = 0.33
```

生成式推荐是“直接生成推荐结果”：

```text
输入：用户历史商品序列
输出：模型生成下一个推荐商品 ID
```

例如：

```text
用户历史：
洁面乳、爽肤水、面霜

模型输出：
推荐商品：SID_019_006_088
推荐理由：用户近期关注基础护肤和保湿商品，因此推荐同类目的保湿精华。
```

区别：

| 对比点 | 传统推荐 | 生成式推荐 |
|---|---|---|
| 任务形式 | 商品打分排序 | 生成商品 ID |
| 输入 | 用户/商品特征 | 用户历史序列 + 文本 prompt |
| 输出 | 商品排序列表 | 商品 SID + 推荐理由 |
| 优点 | 高效、成熟、工业常用 | 可结合自然语言、解释、对话 |
| 难点 | 解释能力弱 | 生成合法商品、评测和稳定性更难 |

---

## Semantic ID 是什么

Semantic ID 是商品的“语义编号”。

普通商品 ID 可能是：

```text
B00ABCD123
```

这种 ID 对模型没有语义。模型不知道：

```text
B00ABCD123 和 B00XYZ789 是否相似
```

Semantic ID 类似：

```text
SID_018_042_007
```

它通常由商品文本信息构造：

```text
商品标题
商品类目
商品描述
商品价格
```

流程是：

```text
商品文本 -> embedding 向量 -> 聚类/量化 -> Semantic ID
```

Semantic ID 的好处：

```text
1. 让模型生成一个可控的商品编号
2. 避免模型编造不存在的商品标题
3. 方便把生成结果映射回真实商品
4. 方便计算 HR@10、NDCG@10 等推荐指标
5. 相似商品可能拥有相似的 ID 结构
```

一句话：

```text
Semantic ID 是把商品语义压缩成结构化 ID，让 LLM 可以更稳定地生成推荐结果。
```

---

## SFT 和 GRPO 分别解决什么问题

### SFT 解决什么问题

SFT 是监督微调。

它解决的问题是：

```text
让模型学会这个任务的输入输出格式。
```

在本项目里，SFT 让 Qwen 学会：

```text
输入：用户历史商品 SID
输出：下一个推荐商品 SID + 推荐理由
```

例如训练样本：

```text
Input:
用户历史商品：
1. SID_001_003_007
2. SID_001_004_011

Output:
推荐商品：SID_001_006_015
推荐理由：用户近期关注护肤类商品，因此推荐同类目的保湿商品。
```

SFT 的核心作用：

```text
让模型从“通用聊天模型”变成“懂推荐格式的模型”。
```

### GRPO 解决什么问题

GRPO 是强化学习后训练方法。

SFT 只是模仿训练数据，但它不一定真正优化推荐指标。

GRPO 解决的问题是：

```text
用 reward 直接优化推荐目标。
```

本项目里的 reward 包括：

```text
命中真实商品：加分
推荐排得靠前：加分
生成合法 SID：加分
推荐类目一致：加分
推荐结果多样：加分
推荐理由合理：加分
```

所以：

```text
SFT 让模型会做推荐
GRPO 让模型更符合推荐指标和业务目标
```

---

## HR@K / NDCG@K / MRR / Coverage 的直觉

假设真实答案是：

```text
真实下一个商品：A
```

模型推荐 Top-5：

```text
[B, C, A, D, E]
```

### HR@K

HR@K 看：

```text
真实商品有没有出现在 Top-K 里
```

如果真实商品 A 在 Top-5 里：

```text
HR@5 = 1
```

如果不在：

```text
HR@5 = 0
```

直觉：

```text
有没有推荐中。
```

### NDCG@K

NDCG@K 不只看有没有命中，还看排第几。

如果真实商品排第 1：

```text
分数最高
```

如果真实商品排第 10：

```text
也算命中，但分数较低
```

直觉：

```text
推荐得准不准，以及排得靠不靠前。
```

### MRR

MRR 看第一个命中的排名。

如果真实商品排第 1：

```text
MRR = 1
```

如果真实商品排第 2：

```text
MRR = 1/2
```

如果真实商品排第 5：

```text
MRR = 1/5
```

直觉：

```text
用户要滑多久才能看到正确推荐。
```

### Coverage

Coverage 看推荐覆盖了多少商品。

如果系统总是推荐同 10 个热门商品：

```text
Coverage 很低
```

如果系统能推荐更多不同商品：

```text
Coverage 更高
```

直觉：

```text
推荐系统是不是只会推热门商品，还是能覆盖更多长尾商品。
```

---

## 一句话总结

```text
生成式推荐是让 LLM 根据用户历史生成商品 SID；
Semantic ID 让商品生成更可控、更可评测；
SFT 让模型学会推荐任务格式；
GRPO 用 reward 优化命中率、排序、合法性和推荐理由；
HR@K 看有没有命中，NDCG@K 看命中位置，MRR 看第一个命中排名，Coverage 看推荐覆盖面。
```

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
造商品
-> 造用户行为
-> 切 train/valid/test
-> 生成 SID
-> 构造 SFT 数据
-> 跑 baseline
-> 生成报告

它不追求效果，只验证项目不会在真实大数据阶段才发现基础链路错误。

### 指标解释

| 指标 | 含义 |
|---|---|
| HR@10 | Top-10 中是否命中真实商品 |
| NDCG@10 | 命中越靠前分越高 |
| MRR@10 | 第一个命中位置的倒数 |
| Coverage@10 | 推荐结果覆盖了多少商品 |


## Day 1 学习结论

今天我已经搞清楚这个项目最小闭环的数据流：

合成商品和用户行为先进入 `scripts/run_smoke.py`，然后经过 `src/ecom_genrec/data.py` 做过滤和时间切分，生成 `train/valid/test` 序列样本；接着 `src/ecom_genrec/semantic_id.py` 把商品的 `title/category/description/price` 转成 Semantic ID；再由 `src/ecom_genrec/instruction.py` 把序列样本转成大模型训练需要的 `prompt + completion`；最后 `src/ecom_genrec/baselines.py` 和 `src/ecom_genrec/metrics.py` 负责 baseline 推荐和指标评测，输出 `reports/smoke/summary.md`。

我也看懂了 prompt 和 completion 是怎么来的：输入是用户历史商品，先从原始 `item_id` 映射成 `history_sid`，再渲染成 `SID | title | category` 形式的历史文本，构成 prompt；completion 则是目标商品对应的 `target_sid`，如果开启 reasoning，还会拼接推荐理由。

我现在理解为什么这个项目要用 SID 而不是直接生成商品标题：因为商品标题是开放文本，模型容易生成不存在的商品，也不方便精确评测；而 Semantic ID 是受约束的商品编号，既能映射回真实商品，又能稳定计算 `HR@K`、`NDCG@K`、`MRR` 和 `Coverage`。

我还知道 baseline 和指标分别在哪里算：`src/ecom_genrec/baselines.py` 里实现了 `Popular`、`Text Retrieval`、`Embedding Retrieval` 三种 baseline，`src/ecom_genrec/metrics.py` 里统一计算 `HR@K`、`NDCG@K`、`MRR`、`Coverage`、`ValidSID` 等指标。这意味着后面训练 Qwen2.5-14B 和 GRPO 时，我能明确知道应该和谁比、看哪些指标。

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
