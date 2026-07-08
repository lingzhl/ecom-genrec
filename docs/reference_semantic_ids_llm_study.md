# 精读笔记：eugeneyan/semantic-ids-llm

本笔记对应参考仓库：

```text
/data1/zhl/ecom-genrec/references/semantic-ids-llm
```

GitHub:

```text
https://github.com/eugeneyan/semantic-ids-llm
```

## 1. 这个项目一句话在做什么

它训练一个能同时理解自然语言和商品 Semantic ID 的 LLM 推荐模型。

传统推荐一般是：

```text
用户历史 -> 推荐系统召回/排序 -> 商品
```

这个项目想做成：

```text
用户历史或自然语言需求 -> LLM 直接生成商品 Semantic ID -> 映射回真实商品
```

README 里的核心说法是：让模型变成“bilingual”，既会英语，也会商品 ID。

## 2. 你第一天只需要看懂的核心概念

### 2.1 Semantic ID

普通商品 ID 类似：

```text
B0040JHNQG
```

这种 ID 对模型没有语义。模型不知道两个 ID 是否相似。

Semantic ID 类似：

```text
<|sid_start|><|sid_64|><|sid_313|><|sid_637|><|sid_768|><|sid_end|>
```

它是多级 token。相似商品可能共享前缀，因此更适合模型学习。

### 2.2 为什么不用 LLM 直接生成商品标题

直接生成标题有三个问题：

- 可能生成商品库不存在的商品。
- 标题是开放文本，很难精确评测 HR@10、NDCG@10。
- 商品标题变化、重复、很长，模型输出不稳定。

Semantic ID 的优势是：

- 输出空间可控。
- 可以映射回真实商品。
- 可以精确计算推荐指标。
- 可以把推荐和自然语言对话放到一个模型里。

## 3. 目录结构怎么读

重点文件：

| 文件 | 你要学什么 |
|---|---|
| `README.md` | 项目总思路，第一天重点读 |
| `notebooks/01-prep-items-and-sequences.ipynb` | 如何准备商品和用户序列 |
| `notebooks/06-get-semantic-ids-per-asin.ipynb` | 如何给商品生成 Semantic ID |
| `notebooks/08-prep-finetuning-data.ipynb` | 如何构造 LLM 微调数据，最值得精读 |
| `src/tokenize_items.py` | 商品文本如何 tokenize |
| `src/embed_items.py` | 商品文本如何生成 embedding |
| `src/train_rqvae.py` | 如何用 RQ-VAE 生成离散 Semantic ID |
| `src/finetune_qwen3_8b_vocab.py` | 如何给 LLM 增加 Semantic ID token |
| `src/finetune_qwen3_8b_full.py` | 如何做完整 SFT |

今天只需要重点读：

```text
README.md
notebooks/08-prep-finetuning-data.ipynb
src/tokenize_items.py
src/embed_items.py
```

今天先不要深挖：

```text
src/train_rqvae.py
src/finetune_qwen3_8b_full.py
```

这两个文件适合 Day 4 和 Day 5 再看。

## 4. 它的完整技术流程

这个项目的 pipeline 是：

```text
Amazon Reviews Video Games
-> 清洗商品标题、描述、元数据
-> 构造商品文本 item_context
-> 用 Qwen3-Embedding-0.6B 生成商品 embedding
-> 训练 RQ-VAE
-> 把 embedding 离散成 Semantic ID
-> 把 Semantic ID 加到 LLM tokenizer 词表
-> 构造 SFT conversation 数据
-> 微调 Qwen3-8B
-> 让模型能推荐、解释、对话
```

## 5. 这个项目和我们项目的对应关系

我们的项目：

```text
Amazon Reviews 2023 多类目
-> 商品文本 embedding
-> KMeans/Semantic ID
-> Qwen2.5-14B SFT
-> GRPO
-> 大规模评测
```

参考项目：

```text
Amazon Reviews 2023 Video Games
-> Qwen3 embedding
-> RQ-VAE Semantic ID
-> Qwen3-8B SFT
-> 对话式推荐
```

对照表：

| 参考项目 | 我们项目 |
|---|---|
| Video Games 单类目 | All_Beauty 起步，再扩多类目 |
| Qwen3-Embedding-0.6B | BGE / Sentence-BERT / hash fallback |
| RQ-VAE 生成 Semantic ID | 先用层级 KMeans，后续可升级 RQ-VAE |
| Qwen3-8B | Qwen2.5-14B |
| 词表扩展 `<|sid_x|>` | 先用文本格式 `SID_018_042_007` |
| SFT | SFT + GRPO |

## 6. notebook 08 最值得学习的点

`notebooks/08-prep-finetuning-data.ipynb` 不是只构造一种训练数据，而是构造多种任务。

### Type A：SemanticID -> Text

让模型看到 ID 后能说出商品标题、描述、类目。

例子：

```text
Input: Product <sid>... has title:
Output: 商品标题
```

意义：让模型理解 SID 对应什么商品。

### Type B：Text -> SemanticID

让模型看到商品标题或描述后能生成对应 SID。

例子：

```text
Input: The product "Mario Kart ..." has SemanticID:
Output: <sid>...
```

意义：让模型建立自然语言和 SID 的双向映射。

### Type C：User Sequence Prediction

这是和我们项目最像的部分。

例子：

```text
Input: Based on recent purchases: SID_A, SID_B, SID_C, next item:
Output: SID_D
```

意义：把推荐任务变成 LLM 生成任务。

### Type D：Category/Semantic Understanding

让模型理解 SID 前缀和类目之间的关系。

例子：

```text
Input: Products starting with prefix X are typically:
Output: 某类商品
```

意义：让 Semantic ID 不只是编号，而是带有语义结构。

### Type E：Multi-hop Reasoning

构造共购、相似、搭配等推理样本。

意义：支持推荐解释和对话式推荐。

## 7. 你今天可以借鉴到自己项目里的 3 件事

### 7.1 训练数据不只做“历史 -> 下一个商品”

我们后续可以增加：

```text
SID -> 商品标题
商品标题 -> SID
历史 SID -> 下一个 SID
历史 SID -> 推荐理由
SID 前缀 -> 商品类目
```

但第一版先做：

```text
历史 SID -> 下一个 SID + 推荐理由
```

### 7.2 Semantic ID 最好有层级结构

参考项目用 RQ-VAE 得到多级 token。

我们当前先用：

```text
Embedding -> KMeans level 1 -> KMeans level 2 -> KMeans level 3
```

简历可以写：

```text
设计层级商品 Semantic ID，将开放式商品标题生成转化为受约束商品 ID 生成。
```

### 7.3 推荐解释要和商品语义绑定

推荐理由不能瞎写，最好来自：

```text
历史商品类目
目标商品类目
标题关键词
用户最近行为
```

## 8. 小白今天不要被这些内容吓到

今天不要求你掌握：

- RQ-VAE 数学细节。
- Qwen3-8B full finetuning。
- Unsloth。
- W&B。
- SASRec 训练。
- Gemini 清洗数据。

今天只要求你能讲清楚：

```text
为什么要把商品变成 Semantic ID？
LLM 如何通过生成 SID 完成推荐？
```

## 9. 今天写进 Day 1 博客的内容

可以直接加到 `docs/day01_environment.md`：

```markdown
## 相关开源项目调研：semantic-ids-llm

今天阅读了 eugeneyan/semantic-ids-llm。这个项目的核心思想是让 LLM 同时理解自然语言和商品 Semantic ID，从而直接生成推荐商品。

我学到最重要的一点是：商品 ID 不能只是无意义的哈希编号，而应该通过商品文本 embedding 和离散化方法构造成有层级结构的 Semantic ID。这样相似商品可以共享前缀，模型更容易学习商品之间的语义关系。

这个项目构造了多种训练数据，包括 SemanticID -> Text、Text -> SemanticID、User Sequence Prediction、Category Understanding 和 Multi-hop Reasoning。其中和我的项目最相关的是 User Sequence Prediction，也就是根据用户历史商品 SID 预测下一个 SID。

我的项目第一版会先采用更简单的层级 KMeans Semantic ID，而不是直接复现 RQ-VAE。原因是我要先跑通数据、训练、评测闭环，再逐步升级 Semantic ID 方案。
```

## 10. 今日自测问题

你学完后要能回答：

```text
1. semantic-ids-llm 为什么说模型是 bilingual？
2. Semantic ID 和普通 item_id 有什么区别？
3. 为什么相似商品共享 SID 前缀有用？
4. Type C: User Sequence Prediction 和我们的项目有什么关系？
5. 为什么我们第一版不用 RQ-VAE，而是先用 KMeans？
```

