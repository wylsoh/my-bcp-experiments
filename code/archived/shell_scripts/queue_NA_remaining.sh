#!/bin/bash
# NA消融剩余实验排队启动
# 自动等待GPU空闲后启动
# 当前已占用:
#   GPU0: mw05_label7 self_train
#   GPU1: NA_v1_bw1
#   GPU2: NA_v2_s0.3
#   GPU3: NA_v3_w1.0_t1.0
# 剩余: NA_v1_bw2, NA_v2_s0.5, NA_v2_s0.8, NA_v3_w1.5_t1.0, NA_v3_w3.0_t1.0, NA_v3_w1.5_t2.0

PY=/home/hjj/anaconda3/envs/yll/bin/python
BASE_DIR="/home/hjj/ssq/my-bcp-experiments/code"
cd "$BASE_DIR" || exit
LOG="queue_NA_remaining.log"

echo "============================================" | tee -a "$LOG"
echo "NA Remaining Queue 启动: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

declare -a QUEUE

# NA_v1 bw=2 (可以用batch_size=24，因为和bw=1结构一样)
QUEUE+=("BCP_CMC_v1_NA_v1_border_shared.py NA_v1_bw2_ps8 --cmc_border_width 2 --batch_size 24 --labeled_bs 12")

# NA_v2 s=0.5, 0.8 (batch_size=12)
QUEUE+=("BCP_CMC_v1_NA_v2_soft_mask.py NA_v2_s0.5_ps8 --cmc_soft_sigma 0.5 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v2_soft_mask.py NA_v2_s0.8_ps8 --cmc_soft_sigma 0.8 --batch_size 12 --labeled_bs 6")

# NA_v3 w=1.5/t=1.0, w=3.0/t=1.0, w=1.5/t=2.0 (batch_size=12)
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py NA_v3_w1.5_t1.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py NA_v3_w3.0_t1.0_ps8 --cmc_border_loss_weight 3.0 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py NA_v3_w1.5_t2.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 2.0 --batch_size 12 --labeled_bs 6")

# ===== GPU查找函数 =====
find_free_gpu() {
    for gpu in 0 1 2 3; do
        local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
        if [ -n "$free_mem" ] && [ "$free_mem" -ge 4000 ]; then
            echo "$gpu"
            return 0
        fi
    done
    echo ""
    return 1
}

# ===== 主循环 =====
for entry in "${QUEUE[@]}"; do
    read -r script exp_name extra_args <<< "$entry"
    
    EXP_DIR="$BASE_DIR/model/BCP/ACDC_${exp_name}_7_labeled"
    if [ -f "$EXP_DIR/self_train/unet_best_model.pth" ]; then
        echo "[$(date)] SKIP: $exp_name - 已存在best_model" | tee -a "$LOG"
        continue
    fi
    
    # 等待可用GPU
    while true; do
        gpu=$(find_free_gpu)
        if [ -n "$gpu" ]; then
            break
        fi
        echo "[$(date)] 等待GPU空闲... ($exp_name)" | tee -a "$LOG"
        sleep 120  # 每2分钟检查一次
    done
    
    LOG_FILE="run_${exp_name}.log"
    CMD="nohup $PY $script \
        --exp $exp_name \
        --cmc_patch_size 8 --cmc_mutual_weight 0.5 \
        --gpu $gpu --labelnum 7 \
        --pre_iterations 0 --max_iterations 30000 \
        $extra_args \
        > $LOG_FILE 2>&1 &"
    
    echo "[$(date)] 启动: $exp_name on GPU $gpu" | tee -a "$LOG"
    echo "  $CMD" | tee -a "$LOG"
    eval "$CMD"
    
    sleep 10
    # 检查是否启动成功
    if ps aux | grep -q "python.*$exp_name" 2>/dev/null; then
        echo "  ✓ 启动成功" | tee -a "$LOG"
    else
        echo "  ✗ 启动可能失败，检查log" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "所有剩余实验已提交: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
