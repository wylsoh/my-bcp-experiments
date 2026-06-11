#!/bin/bash
# ============================================================
# 已完成实验测试 + GPU 0 剩余接力实验
# 测试用 CUDA_VISIBLE_DEVICES=0 与 Task B 共享 GPU 0
# ============================================================
cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python

echo "============================================================"
echo "  测试已完成实验"
echo "  开始时间: $(date)"
echo "============================================================"

# =====================================
# 需要测试的已完成实验列表
# =====================================
test_exps=(
    "BCP_CMC_v1_mutual_mw10 0.5_old_ps16"
    "BCP_CMC_v1_mutual_mw01_ps8 0.1_ps8"
)

for entry in "${test_exps[@]}"; do
    exp_name=$(echo $entry | awk '{print $1}')
    label=$(echo $entry | awk '{print $2}')
    
    echo ""
    echo "--- 测试: ${exp_name} (${label}) ---"
    echo ""
    
    CUDA_VISIBLE_DEVICES=0 $PYTHON test_ACDC.py \
        --exp "${exp_name}" \
        --labelnum 7 \
        --stage_name "self_train" 2>&1
    
    echo ""
    echo "--- 完成: ${exp_name} ---"
    echo ""
done

echo ""
echo "============================================================"
echo "  所有测试完成! 完成时间: $(date)"
echo "============================================================"
