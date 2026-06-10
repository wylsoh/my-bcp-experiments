#!/bin/bash
# ============================================================
# ACDC 互教权重敏感性分析实验
# 串行执行 5 组实验，每组包含 pretrain (10K) + self-train (30K)
# 使用 GPU 3，总预估时间约 15~20 小时
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python

# 权重列表：exp_name weight
experiments=(
    "ACDC_BCP_CMC_v1_mutual_mw01 0.1"
    "ACDC_BCP_CMC_v1_mutual_mw03 0.3"
    "ACDC_BCP_CMC_v1_mutual_mw05 0.5"
    "ACDC_BCP_CMC_v1_mutual_mw07 0.7"
    "ACDC_BCP_CMC_v1_mutual_mw10 1.0"
)

total=${#experiments[@]}
current=1

for exp_entry in "${experiments[@]}"; do
    exp_name=$(echo $exp_entry | awk '{print $1}')
    weight=$(echo $exp_entry | awk '{print $2}')

    echo ""
    echo "============================================================"
    echo "  [${current}/${total}] 开始实验: ${exp_name}"
    echo "  cmc_mutual_weight = ${weight}"
    echo "  开始时间: $(date)"
    echo "============================================================"
    echo ""

    $PYTHON BCP_CMC_v1_mutual.py \
        --exp "${exp_name}" \
        --cmc_mutual_weight "${weight}" \
        --gpu 3 \
        --labelnum 7 \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 24 \
        --labeled_bs 12

    exp_status=$?
    if [ $exp_status -ne 0 ]; then
        echo ""
        echo "  [WARNING] 实验 ${exp_name} 退出码 ${exp_status}"
    fi

    echo ""
    echo "  [${current}/${total}] 完成: ${exp_name} (weight=${weight})"
    echo "  完成时间: $(date)"
    echo ""

    current=$((current + 1))
done

echo ""
echo "============================================================"
echo "  所有实验已完成!"
echo "  完成时间: $(date)"
echo "============================================================"
