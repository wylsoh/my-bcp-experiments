# BCP 测试结果汇总表

> 自动生成于: 2026-05-27 19:17
> 数据来源: [`master_metrics.csv`](master_metrics.csv)

| # | 实验 | 标签 | 模型 | 阶段 | GPU | 时间 | 训练 Dice | 训练 HD95 | RV Dice | RV HD95 | MYO Dice | MYO HD95 | LV Dice | LV HD95 | Avg Dice | Avg HD95 |
|---|------|------|------|------|-----|------|-----------|-----------|---------|---------|----------|----------|---------|---------|----------|----------|
| 1 | BCP_CMC_fusion_student | 7 | unet | self_train | 2 | 05-27 19:09 | 0.8833 | 2.17 | 0.8897 | 1.7507 | 0.8663 | 4.4885 | 0.9170 | 4.6389 | 0.8910 | 3.6261 |
| 2 | BCP_CMC_fusion_student | 7 | unet | self_train | 1 | 05-27 19:17 | 0.8833 | 2.17 | 0.8897 | 1.7507 | 0.8663 | 4.4885 | 0.9170 | 4.6389 | 0.8910 | 3.6261 |

## 用法

### 查看所有结果
```bash
cat test_results/master_metrics.csv | column -t -s','
```

### 用 Python 分析
```python
import pandas as pd
df = pd.read_csv('test_results/master_metrics.csv')
df.groupby('exp')[['avg_dice','avg_hd95']].mean()
```

### 追加新结果
```bash
./run_test.sh --exp <实验名> --labelnum <标签数>
# 自动追加到 master_metrics.csv
```
