# Day 3：推荐系统不能只看感觉：HR@10、NDCG@10 和 Coverage 怎么算？

## 今日目标

建立训练大模型前的 baseline 和评测指标。

## 学习重点

- Popular baseline。
- Text Retrieval / BM25 思想。
- Embedding Retrieval 思想。
- HR@K、NDCG@K、MRR、Coverage 的计算方式。

## 前置条件

需要先有：

```text
data/processed/all_beauty/
artifacts/all_beauty/sid_map.json
```

如果还没有 SID，先完成 Day 4，再回来跑本日 baseline。

## 执行命令

构建 instruction 数据：

```bash
python3 scripts/build_instruction_data.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out-dir data/processed/all_beauty/instructions \
  --with-reasoning
```

跑 baseline：

```bash
python3 scripts/evaluate_baselines.py \
  --config configs/default.yaml \
  --processed-dir data/processed/all_beauty \
  --sid-map artifacts/all_beauty/sid_map.json \
  --out reports/all_beauty/baselines.json
```

查看结果：

```bash
cat reports/all_beauty/baselines.json
```

## 实验记录

| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Popular | TODO | TODO | TODO | TODO |
| Text Retrieval | TODO | TODO | TODO | TODO |
| Embedding Retrieval | TODO | TODO | TODO | TODO |

## 今日技术理解

### Popular baseline

推荐训练集中最热门的商品。它简单但非常重要，因为很多推荐场景存在明显热门偏置。

### HR@10

```text
如果真实下一个商品出现在 Top-10 推荐里，HR@10 = 1，否则 = 0。
```

### NDCG@10

命中越靠前越好。第 1 位命中比第 10 位命中得分高。

### Coverage@10

衡量推荐是否只集中在少数热门商品。Coverage 越高，说明推荐覆盖的商品越多。

## GitHub 产出

```bash
git add src scripts configs docs reports
git commit -m "Day 3: add recommendation baselines and metrics"
```

## 今日完成标准

```text
[ ] 至少有 Popular 和 Text Retrieval 结果
[ ] baselines.json 已生成
[ ] 能解释 HR@10 和 NDCG@10 的区别
[ ] 完成 Day 3 博客
[ ] 完成 Day 3 commit
```

## 今日问题记录

```text
TODO: 记录 baseline 过高/过低、指标计算疑问、SID 缺失等问题。
```
