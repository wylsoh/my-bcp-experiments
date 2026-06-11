# 当前最优结果：NA-CMC V2 (Soft Mask, σ=0.3)

## 模型概述

| 项目 | 值 |
|------|-----|
| **实验名称** | `BCP_CMC_NA_v2_s0.3_ps8` |
| **算法** | BCP + CMC v1 改良版 — NA-CMC V2 软权重掩码 |
| **数据集** | ACDC (7 labeled / 全部) |
| **训练框架** | 2D U-Net, SGD optimizer |
| **迭代次数** | 30,000 (self-train) |
| **CM Patch Size** | 8 |
| **Soft Sigma (σ)** | 0.3 |
| **Batch Size** | 12 (labeled_bs=6) |

## 核心创新

NA-CMC V2 将硬性 0/1 互补掩码替换为 **连续值软权重掩码**：

- 使用 **bilinear 插值**（替代 nearest）产生边缘渐变
- 使用 **avg_pool 平滑** 扩大过渡带宽度
- 边界处像素权重从 0/1 变为 0.1~0.9，同时接收两视图监督信号
- σ=0.3 对应 avg_pool 核大小 3px，过渡带宽约 1~2px

## 测试结果

| 指标 | 值 |
|------|------|
| **Dice** 🥇 | **0.9031** |
| **JC (Jaccard)** | 0.8277 |
| **HD95 (mm)** | 1.3215 |
| **ASD (mm)** | 0.4583 |

## 排行榜

| 排名 | 实验 | Dice |
|------|------|------|
| 🥇 | **NA_v2_s0.3 (soft_mask, σ=0.3)** | **0.9031** |
| 🥈 | ablation_cmc_only_ps8 (CMC-only, ps=8) | 0.9002 |
| 🥉 | asymmetric_ps8 (Asymmetric CMC, ps=8) | 0.9000 |
| 4 | NA_v3_bs12 (border_loss w=1.0/t=1.0) | 0.8988 |
| 5 | classbal_ps8 (Class-balanced CMC, ps=8) | 0.8961 |

## 训练命令

```bash
# Pre-train (从零训练)
python BCP_CMC_v1_NA_v2_soft_mask.py \
  --exp BCP_CMC_NA_v2_s0.3_ps8 \
  --cmc_patch_size 8 \
  --cmc_soft_sigma 0.3 \
  --cmc_mutual_weight 0.5 \
  --gpu 2 \
  --labelnum 7 \
  --pre_iterations 10000 \
  --max_iterations 30000 \
  --batch_size 12 \
  --labeled_bs 6

# Self-train (从已有 pretrain 继续)
python BCP_CMC_v1_NA_v2_soft_mask.py \
  --exp BCP_CMC_NA_v2_s0.3_ps8 \
  --cmc_patch_size 8 \
  --cmc_soft_sigma 0.3 \
  --cmc_mutual_weight 0.5 \
  --gpu 2 \
  --labelnum 7 \
  --pre_iterations 0 \
  --max_iterations 30000 \
  --batch_size 12 \
  --labeled_bs 6
```

## 测试命令

```bash
CUDA_VISIBLE_DEVICES=3 python test_ACDC.py \
  --exp BCP_CMC_NA_v2_s0.3_ps8 \
  --model unet \
  --labelnum 7
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `unet_best_model.pth` | 最终训练完成的 U-Net 权重 (7.4 MB) |
| `iter_20600_dice_0.893.pth` | 最佳中间检查点 (iter=20600, Dice=0.893) |
| `train.log` | 完整训练日志 (含参数配置) |
| `self_train_log.txt` | self-train 阶段日志 |
| `BCP_CMC_v1_NA_v2_soft_mask.py` | 训练用源代码 |
