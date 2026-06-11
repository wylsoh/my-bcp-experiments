#!/bin/bash
# NA消融实验pipeline - 从旧pre_train启动self_train
# 全部使用 cmc_patch_size=8, batch_size=24, labeled_bs=12
# 策略: 按GPU可用性排队执行

BASE_DIR="/home/hjj/ssq/my-bcp-experiments/code"
cd "$BASE_DIR" || exit 1
PIPELINE_LOG="run_NA_ablation.log"

echo "============================================" | tee -a "$PIPELINE_LOG"
echo "NA Ablation Pipeline 启动: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================" | tee -a "$PIPELINE_LOG"

# ===== 实验队列 =====
# 格式: "SCRIPT EXP_NAME EXTRA_ARGS GPU"
# SCRIPT: 训练脚本名
# EXP_NAME: 实验名称（不包含exp前缀）
# EXTRA_ARGS: 消融参数
EXPERIMENTS=()

# NA_v1 (border_shared): border_width=1, 2
EXPERIMENTS+=("BCP_CMC_v1_NA_v1_border_shared.py BCP_CMC_NA_v1_bw1_ps8 --cmc_border_width 1")
EXPERIMENTS+=("BCP_CMC_v1_NA_v1_border_shared.py BCP_CMC_NA_v1_bw2_ps8 --cmc_border_width 2")

# NA_v2 (soft_mask): soft_sigma=0.3, 0.5, 0.8
EXPERIMENTS+=("BCP_CMC_v1_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.3_ps8 --cmc_soft_sigma 0.3")
EXPERIMENTS+=("BCP_CMC_v1_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.5_ps8 --cmc_soft_sigma 0.5")
EXPERIMENTS+=("BCP_CMC_v1_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.8_ps8 --cmc_soft_sigma 0.8")

# NA_v3 (border_loss): weight=1.0/1.5/3.0 × temp=1.0/2.0
EXPERIMENTS+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.0_t1.0_ps8 --cmc_border_loss_weight 1.0 --cmc_border_kl_temp 1.0")
EXPERIMENTS+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t1.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 1.0")
EXPERIMENTS+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w3.0_t1.0_ps8 --cmc_border_loss_weight 3.0 --cmc_border_kl_temp 1.0")
EXPERIMENTS+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t2.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 2.0")

# ===== GPU 分配函数 =====
find_free_gpu() {
    local min_free=4000  # 要求至少 4000MB
    for gpu in 0 1 2 3; do
        local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        if [ -n "$free_mem" ] && [ "$free_mem" -ge "$min_free" ]; then
            # 检查是否已有训练进程在跑
            local has_train=$(ps aux | grep "python.*BCP_CMC.*gpu $gpu" | grep -v grep | wc -l)
            if [ "$has_train" -eq 0 ]; then
                echo "$gpu"
                return 0
            fi
        fi
    done
    echo ""
    return 1
}

# ===== 主循环 =====
for exp_entry in "${EXPERIMENTS[@]}"; do
    read -r script exp_name extra_args <<< "$exp_entry"
    
    # 检查实验目录是否已存在且包含self_train的best_model（说明已完成）
    EXP_DIR="$BASE_DIR/model/BCP/ACDC_${exp_name}_7_labeled"
    if [ -f "$EXP_DIR/self_train/unet_best_model.pth" ]; then
        echo "[SKIP] $exp_name - 已存在 best_model" | tee -a "$PIPELINE_LOG"
        continue
    fi
    
    # 等待可用GPU
    while true; do
        gpu=$(find_free_gpu)
        if [ -n "$gpu" ]; then
            break
        fi
        echo "[$(date)] 等待可用GPU... ($exp_name)" | tee -a "$PIPELINE_LOG"
        sleep 60
    done
    
    # 构建命令行
    LOG_FILE="${exp_name}.log"
    CMD="nohup python $script \
        --exp $exp_name \
        --cmc_patch_size 8 \
        --cmc_mutual_weight 0.5 \
        --batch_size 24 --labeled_bs 12 \
        --gpu $gpu --labelnum 7 \
        --pre_iterations 0 --max_iterations 30000 \
        $extra_args \
        > $LOG_FILE 2>&1 &"
    
    echo "[$(date)] 启动: $exp_name on GPU $gpu" | tee -a "$PIPELINE_LOG"
    echo "  CMD: $CMD" | tee -a "$PIPELINE_LOG"
    
    eval "$CMD"
    
    # 等待几秒确认进程启动
    sleep 5
    PID=$!
    if ps -p $PID > /dev/null 2>&1; then
        echo "  PID: $PID" | tee -a "$PIPELINE_LOG"
    else
        # 尝试通过grep找进程
        NEW_PID=$(ps aux | grep "python.*$script.*$exp_name" | grep -v grep | awk '{print $2}')
        if [ -n "$NEW_PID" ]; then
            echo "  PID: $NEW_PID" | tee -a "$PIPELINE_LOG"
        fi
    fi
    
    # 等待一下再找下一个GPU
    sleep 10
done

echo "" | tee -a "$PIPELINE_LOG"
echo "============================================" | tee -a "$PIPELINE_LOG"
echo "所有实验已提交启动: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================" | tee -a "$PIPELINE_LOG"
