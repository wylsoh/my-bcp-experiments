#!/bin/bash
# ============================================================
# ACDC CMC 卷积核邻居策略实验
# 新策略：exclusive mask 扩展为 kernel_size×kernel_size 邻居区域
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python

EXP_NAME="BCP_CMC_v1_mutual_mw06_ps8_conv3"
WEIGHT=0.6
GPU=${1:-2}

echo ""
echo "============================================================"
echo "  开始实验: ${EXP_NAME}"
echo "  cmc_mutual_weight = ${WEIGHT}"
echo "  cmc_patch_size = 8"
echo "  cmc_kernel_size = 3 (卷积核邻居策略)"
echo "  GPU = ${GPU}"
echo "  开始时间: $(date)"
echo "============================================================"
echo ""

$PYTHON BCP_CMC_v1_mutual_convkernel.py \
    --exp "${EXP_NAME}" \
    --cmc_mutual_weight "${WEIGHT}" \
    --cmc_patch_size 8 \
    --cmc_kernel_size 3 \
    --gpu "${GPU}" \
    --labelnum 7 \
    --pre_iterations 10000 \
    --max_iterations 30000 \
    --batch_size 24 \
    --labeled_bs 12 2>&1 | tee "./run_${EXP_NAME}.log"

echo ""
echo "============================================================"
echo "  完成: ${EXP_NAME}"
echo "  完成时间: $(date)"
echo "============================================================"
