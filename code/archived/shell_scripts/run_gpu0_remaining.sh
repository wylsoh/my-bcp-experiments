#!/bin/bash
# ============================================================
# GPU 0 接力：启动剩余的 5 个消融/变体实验
# 与 Task B (convkernel) 共享 GPU 0，bs=12 安全运行
# 串行执行，每个完成后自动启动下一个
# ============================================================
cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python

# 等待 Task B 训练完成（预计还有 ~4h）
# 但可以先用小 bs 测试
WAIT_TASK_B=true

echo "============================================================"
echo "  GPU 0 剩余实验接力启动"
echo "  开始时间: $(date)"
echo "============================================================"

# 实验队列：脚本名, exp_name, 额外参数
experiments=(
    "BCP_CMC_v1_ablation_cmc_only,BCP_CMC_v1_ablation_cmc_only_ps8,-"
    "BCP_CMC_v1_asymmetric,BCP_CMC_v1_asymmetric_ps8,-"
    "BCP_CMC_v1_attention,BCP_CMC_v1_attention_ps8,-"
    "BCP_CMC_v1_classbal,BCP_CMC_v1_classbal_ps8,-"
    "BCP_CMC_v1_freq,BCP_CMC_v1_freq_ps8,-"
)

total=${#experiments[@]}
current=1

for entry in "${experiments[@]}"; do
    script_name=$(echo $entry | cut -d',' -f1)
    exp_name=$(echo $entry | cut -d',' -f2)
    extra=$(echo $entry | cut -d',' -f3)

    echo ""
    echo "============================================================"
    echo "  [${current}/${total}] 启动: ${script_name}"
    echo "  exp: ${exp_name}"
    echo "  开始时间: $(date)"
    echo "============================================================"
    echo ""

    # 先用 bs=12, cmc_patch_size=8, mw=0.5 运行
    CUDA_VISIBLE_DEVICES=0 $PYTHON "${script_name}" \
        --exp "${exp_name}" \
        --cmc_patch_size 8 \
        --cmc_mutual_weight 0.5 \
        --gpu 0 \
        --labelnum 7 \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 12 \
        --labeled_bs 6 2>&1 | tee "./run_${exp_name}.log"

    exp_status=$?
    echo ""
    echo "  [${current}/${total}] 完成: ${exp_name} (exit=$exp_status)"
    echo "  完成时间: $(date)"
    echo ""

    current=$((current + 1))
done

echo ""
echo "============================================================"
echo "  所有剩余实验已完成! 完成时间: $(date)"
echo "============================================================"
