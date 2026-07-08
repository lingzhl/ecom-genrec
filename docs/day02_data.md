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

数据统计：

| Metric | Value |
|---|---:|
| users | TODO |
| items | TODO |
| interactions | TODO |
| train_samples | TODO |
| valid_samples | TODO |
| test_samples | TODO |
| avg_history_len | TODO |
| sparse_user_ratio | TODO |
| long_tail_item_ratio | TODO |

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
[ ] train.jsonl / valid.jsonl / test.jsonl 已生成
[ ] stats.json 已生成
[ ] 能解释为什么不能随机划分
[ ] 完成 Day 2 博客
[ ] 完成 Day 2 commit
```

## 今日问题记录

```text
TODO: 记录数据下载、字段不匹配、内存、过滤后样本过少等问题。
```
