#!/bin/bash
# ============================================================
# GPU 2 接力实验：NA_v2 (soft_mask) → NA_v3 (border_loss)
# 利用 GPU 2 空闲资源，bs=12 安全运行
# ============================================================
cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python
GPU=2

echo "============================================================"
echo "  GPU ${GPU} 接力实验启动"
echo "  开始时间: $(date)"
echo "============================================================"

# ========================
# [1/2] NA_v2 — Soft Mask
# ========================
echo ""
echo "=============================="
echo "  [1/2] NA_v2 (soft_mask, bs=12)"
echo "=============================="
echo ""

$PYTHON BCP_CMC_v1_NA_v2_soft_mask.py \
    --exp "BCP_CMC_NA_v2_bs12" \
    --cmc_soft_sigma 0.5 \
    --cmc_patch_size 8 \
    --gpu ${GPU} \
    --labelnum 7 \
    --pre_iterations 10000 \
    --max_iterations 30000 \
    --batch_size 12 \
    --labeled_bs 6 2>&1 | tee "./run_NA_v2_bs12.log"

echo ""
echo "  [1/2] NA_v2 完成! 结束时间: $(date)"
echo ""

# ========================
# [2/2] NA_v3 — Border Loss
# ========================
echo ""
echo "=============================="
echo "  [2/2] NA_v3 (border_loss, bs=12)"
echo "=============================="
echo ""

$PYTHON BCP_CMC_v1_NA_v3_border_loss.py \
    --exp "BCP_CMC_NA_v3_bs12" \
    --cmc_border_loss_weight 1.5 \
    --cmc_patch_size 8 \
    --gpu ${GPU} \
    --labelnum 7 \
    --pre_iterations 10000 \
    --max_iterations 30000 \
    --batch_size 12 \
    --labeled_bs 6 2>&1 | tee "./run_NA_v3_bs12.log"

echo ""
echo "=============================="
echo "  [2/2] NA_v3 完成!"
echo "  GPU ${GPU} 所有接力实验结束: $(date)"
echo "=============================="
