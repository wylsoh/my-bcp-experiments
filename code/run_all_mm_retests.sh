#!/bin/bash
# 按顺序在 GPU3 上运行全部 5 个 MM 重测试
# 每个测试完成后自动进入下一个

cd /home/hjj/ssq/my-bcp-experiments/code

GPU=3
PYTHON="/home/hjj/anaconda3/envs/yll/bin/python"
SCRIPT="test_MM.py"

echo "=========================================="
echo "启动全部 5 个 MM 重测试 @ $(date)"
echo "GPU: ${GPU}"
echo "=========================================="

# ============ 1. BCP 10% ============
echo ""
echo "[1/5] BCP 10% self_train test starting..."
$PYTHON $SCRIPT --gpu $GPU --exp MM_BCP --labelnum 10 --num_classes 8 --stage_name self_train
echo "[1/5] BCP 10% test completed @ $(date)"

# ============ 2. BCP 20% ============
echo ""
echo "[2/5] BCP 20% self_train test starting..."
$PYTHON $SCRIPT --gpu $GPU --exp MM_BCP --labelnum 20 --num_classes 8 --stage_name self_train
echo "[2/5] BCP 20% test completed @ $(date)"

# ============ 3. NA_v2 10% ============
echo ""
echo "[3/5] NA_v2 10% self_train test starting..."
$PYTHON $SCRIPT --gpu $GPU --exp MM_BCP_CMC_NA_v2_s0.3 --labelnum 10 --num_classes 8 --stage_name self_train
echo "[3/5] NA_v2 10% test completed @ $(date)"

# ============ 4. NA_v2 20% ============
echo ""
echo "[4/5] NA_v2 20% self_train test starting..."
$PYTHON $SCRIPT --gpu $GPU --exp MM_BCP_CMC_NA_v2_s0.3 --labelnum 20 --num_classes 8 --stage_name self_train
echo "[4/5] NA_v2 20% test completed @ $(date)"

# ============ 5. NA_v2 5_labeled ============
echo ""
echo "[5/5] NA_v2 5_labeled self_train test starting..."
$PYTHON $SCRIPT --gpu $GPU --exp MM_BCP_CMC_NA_v2_s0.3_5_labeled --labelnum 5 --num_classes 8 --stage_name self_train
echo "[5/5] NA_v2 5_labeled test completed @ $(date)"

echo ""
echo "=========================================="
echo "全部 5 个 MM 重测试已完成 @ $(date)"
echo "=========================================="
