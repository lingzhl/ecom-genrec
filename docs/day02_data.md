# Day 2：Amazon Reviews 2023 数据处理：如何把用户评论转成推荐序列？

## 今日目标

用真实 Amazon 电商数据替代 smoke 数据，完成用户行为序列切分。

## 学习重点

- review 文件：用户行为。
- metadata 文件：商品信息。
- 时间序列切分：train/valid/test。
- 推荐任务为什么不能随机切分。

## 准备数据

把真实数据放到：

```text
data/raw/All_Beauty.jsonl.gz
data/raw/meta_All_Beauty.jsonl.gz
```

## 执行命令

```bash
cd /data1/zhl/ecom-genrec
source .venv/bin/activate

python3 scripts/process_amazon.py \
  --config configs/default.yaml \
  --reviews data/raw/All_Beauty.jsonl.gz \
  --metadata data/raw/meta_All_Beauty.jsonl.gz \
  --categories All_Beauty \
  --out-dir data/processed/all_beauty
```

查看结果：

```bash
cat data/processed/all_beauty/stats.json
head -3 data/processed/all_beauty/train.jsonl
head -3 data/processed/all_beauty/items.jsonl
```

## 实验记录

stats.json：

数据统计：

| Metric | Value |
|---|---:|
| users | 357 |
| items | 479 |
| interactions | 3315 |
| train_samples | 2244 |
| valid_samples | 357 |
| test_samples | 357 |
| avg_history_len | 7.6106 |
| sparse_user_ratio | 0.4706 |
| long_tail_item_ratio | 0.8643 |

review 文件提供用户行为，metadata 文件提供商品语义信息。推荐任务需要按 user_id 分组，因为模型要学习每个用户自己的兴趣序列；按 timestamp 排序是为了保证“用过去预测未来”。最后一次交互作为 test，倒数第二次作为 valid，避免随机切分带来的未来信息泄漏。stats.json 中 users、items、interactions、train/valid/test_samples 和 avg_history_len 是判断数据规模与任务难度的核心字段。


train.jsonl：
{"user_id": "AFSKPY37N3C43SOI5IEXEK5JSIYA", "history": ["B07J3GH1W1", "B07W397QG4", "B07KG1TWP5"], "target_item": "B08JTNQFZY", "target_category": "All Beauty", "history_len": 3, "full_history_len": 7}

items.jsonl：
{"item_id": "B081ZN3TD5", "title": "JPNK 4PCS Anti-Static Detangling Fine & Wide Tooth Shower Comb Set", "category": "All Beauty", "description": "Material: Plastic Brand: JPNK Hair Type: Wavy,Straight Unit Count: 1.00 Count Package Dimensions: 12 x 3.1 x 0.7 inches; 5.61 Ounces", "price": ""}

history_len 是这条样本给模型看的长度，full_history_len 是这个用户全部行为的长度

## 今日技术理解

### 为什么按时间切分

推荐系统要模拟真实线上场景：

```text
过去行为 -> 预测未来行为
```

如果随机切分，模型可能看到用户未来兴趣，导致评测虚高。

### 过滤规则

本项目默认：

```text
用户交互数 >= 5
商品交互数 >= 5
```

这样可以减少极端稀疏数据，让序列推荐任务更稳定。

## GitHub 产出

不要提交大数据文件。提交代码和文档：

```bash
git add docs README.md configs scripts src .gitignore
git commit -m "Day 2: process Amazon review sequences"
```

## 今日完成标准

```text
[√] train.jsonl / valid.jsonl / test.jsonl 已生成
[√] stats.json 已生成
[√] 能解释为什么不能随机划分
[√] 完成 Day 2 博客
[√] 完成 Day 2 commit
```

## 今日问题记录

```text
/items.jsonl商品信息为什么"category": "unknown", "description": "", "price": "没有

原因找到了，而且已经修好了。
原来不是数据坏了，而是代码取字段时有个小问题：
meta_All_Beauty.jsonl.gz 原始数据里长这样：
"main_category": "All Beauty",
"categories": [],
"description": [],
"details": {...},
"price": null
之前代码优先取 categories，但 categories 是空列表 []，所以没有继续往后取 main_category，最后就变成了：
"category": "unknown"
description 也是类似问题：原始 description 经常是空列表，但 details 里有商品信息。现在已经改成：空列表、空字典也当作缺失，继续往后找字段。
我已经修改了：
[utils.py (line 57)](/data1/zhl/ecom-genrec/src/ecom_genrec/utils.py:57)
并重新处理了数据。现在前几条商品已经正常了：
"category": "All Beauty"
"description": "Brand: Shiny Leaf Scent: Almond Oil ..."
当前处理后统计：
items: 479
All Beauty: 467
Premium Beauty: 12
empty_description: 11
empty_price: 392
所以现在结论是：
category 的问题是代码字段兜底问题，已修好。
description 大部分已经能从 details 补出来，只剩 11 个空。
price 为空是正常的，因为 Amazon 原始数据里很多商品 price: null，不是你处理错了。
```
