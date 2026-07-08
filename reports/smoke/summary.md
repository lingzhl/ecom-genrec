# Smoke Test Summary

## Data Stats

| Metric | Value |
|---|---:|
| users | 60 |
| items | 48 |
| interactions | 480 |
| train_samples | 300 |
| valid_samples | 60 |
| test_samples | 60 |
| avg_history_len | 7 |
| categories | 6 |
| sparse_user_ratio | 0.0 |
| long_tail_item_ratio | 1.0 |

## Baseline Metrics

| Method | HR@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Popular | 0.0000 | 0.0000 | 0.0000 | 0.2083 |
| TextRetrieval | 1.0000 | 1.0000 | 1.0000 | 0.9375 |
| EmbeddingRetrieval | 1.0000 | 1.0000 | 1.0000 | 0.7292 |
