#!/bin/bash
# ============================================================
# NA_v2 最优结果扩展实验队列
#
# 实验列表：
#   1. ACDC label3 NA_v2_s0.3 (soft_mask, σ=0.3, ps=8)
#   2. LA label8 NA_v2_s0.3 (soft_mask, σ=0.3, ps=16)
#   3. LA label4 NA_v2_s0.3
#   4. Pancreas label6 (10%) NA_v2_s0.3
#   5. Pancreas label12 (20%) NA_v2_s0.3
#
# 注意：NA_v3 剩余实验由 queue_NA_v3_remaining.sh (PID 26094) 管理
# ============================================================
PY=/home/hjj/anaconda3/envs/yll/bin/python
BASE_DIR="/home/hjj/ssq/my-bcp-experiments/code"
cd "$BASE_DIR" || exit
LOG="queue_na_v2_extended.log"

find_free_gpu() {
    for gpu in 0 1 2 3; do
        local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
        [ -n "$free" ] && [ "$free" -ge 4000 ] && echo "$gpu" && return 0
    done
    echo "" && return 1
}

echo "============================================" | tee -a "$LOG"
echo "NA_v2 扩展实验 Queue 启动: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# ============================================================
# 1. ACDC label3 NA_v2_s0.3
# ============================================================
EXP1_NAME="BCP_CMC_NA_v2_s0.3_label3"
if [ -f "$BASE_DIR/model/BCP/ACDC_${EXP1_NAME}_3_labeled/self_train/unet_best_model.pth" ]; then
    echo "[$(date)] SKIP: $EXP1_NAME - 已完成" | tee -a "$LOG"
else
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (ACDC label3 NA_v2)" | tee -a "$LOG"
        sleep 120
    done
    CMD="nohup $PY BCP_CMC_v1_NA_v2_soft_mask.py \
        --exp $EXP1_NAME \
        --cmc_patch_size 8 \
        --cmc_soft_sigma 0.3 \
        --cmc_mutual_weight 0.5 \
        --gpu $gpu \
        --labelnum 3 \
        --pre_iterations 6000 \
        --max_iterations 30000 \
        --batch_size 12 \
        --labeled_bs 6 \
        > run_${EXP1_NAME}.log 2>&1 &"
    echo "[$(date)] 启动: $EXP1_NAME on GPU $gpu" | tee -a "$LOG"
    eval "$CMD"
    sleep 15
    ps aux | grep -q "python.*$EXP1_NAME" 2>/dev/null && echo "  ✓ OK" | tee -a "$LOG" || echo "  ✗ FAIL" | tee -a "$LOG"
fi

# ============================================================
# 2. LA label8 NA_v2_s0.3
# ============================================================
EXP2_NAME="BCP_CMC_NA_v2_s0.3"
if [ -f "$BASE_DIR/model/BCP/LA_${EXP2_NAME}_8_labeled/self_train/VNet_best_model.pth" ]; then
    echo "[$(date)] SKIP: LA label8 $EXP2_NAME - 已完成" | tee -a "$LOG"
else
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (LA label8 NA_v2)" | tee -a "$LOG"
        sleep 120
    done
    CMD="nohup $PY LA_BCP_CMC_NA_v2_soft_mask.py \
        --exp $EXP2_NAME \
        --cmc_patch_size 16 \
        --cmc_soft_sigma 0.3 \
        --cmc_mutual_weight 0.5 \
        --batch_size 8 \
        --labeled_bs 4 \
        --gpu $gpu \
        --labelnum 8 \
        --pre_max_iteration 2000 \
        --self_max_iteration 15000 \
        --max_samples 80 \
        > run_LA_${EXP2_NAME}_8.log 2>&1 &"
    echo "[$(date)] 启动: LA label8 $EXP2_NAME on GPU $gpu" | tee -a "$LOG"
    eval "$CMD"
    sleep 15
    ps aux | grep -q "python.*$EXP2_NAME" 2>/dev/null && echo "  ✓ OK" | tee -a "$LOG" || echo "  ✗ FAIL" | tee -a "$LOG"
fi

# ============================================================
# 3. LA label4 NA_v2_s0.3
# ============================================================
if [ -f "$BASE_DIR/model/BCP/LA_${EXP2_NAME}_4_labeled/self_train/VNet_best_model.pth" ]; then
    echo "[$(date)] SKIP: LA label4 $EXP2_NAME - 已完成" | tee -a "$LOG"
else
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (LA label4 NA_v2)" | tee -a "$LOG"
        sleep 120
    done
    CMD="nohup $PY LA_BCP_CMC_NA_v2_soft_mask.py \
        --exp $EXP2_NAME \
        --cmc_patch_size 16 \
        --cmc_soft_sigma 0.3 \
        --cmc_mutual_weight 0.5 \
        --batch_size 8 \
        --labeled_bs 4 \
        --gpu $gpu \
        --labelnum 4 \
        --pre_max_iteration 2000 \
        --self_max_iteration 15000 \
        --max_samples 80 \
        > run_LA_${EXP2_NAME}_4.log 2>&1 &"
    echo "[$(date)] 启动: LA label4 $EXP2_NAME on GPU $gpu" | tee -a "$LOG"
    eval "$CMD"
    sleep 15
    ps aux | grep -q "python.*$EXP2_NAME" 2>/dev/null && echo "  ✓ OK" | tee -a "$LOG" || echo "  ✗ FAIL" | tee -a "$LOG"
fi

# ============================================================
# 4. Pancreas label6 (10%) NA_v2_s0.3
# ============================================================
PANCREAS_DIR="/home/hjj/ssq/BCP_original/code/pancreas"
if [ -d "$PANCREAS_DIR/result/cutmix_bcp_cmc_na_v2/10percent/self_train" ] && \
   [ -f "$PANCREAS_DIR/result/cutmix_bcp_cmc_na_v2/10percent/self_train/best_ema10_self.pth" ]; then
    echo "[$(date)] SKIP: Pancreas 10% NA_v2 - 已完成" | tee -a "$LOG"
else
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (Pancreas 10% NA_v2)" | tee -a "$LOG"
        sleep 120
    done
    CMD="nohup $PY train_pancreas_bcp_cmc_na_v2.py \
        --gpu $gpu \
        --label_percent 10 \
        --cmc_patch_size 16 \
        --cmc_soft_sigma 0.3 \
        --cmc_mutual_weight 0.5 \
        --batch_size 2 \
        --lr 1e-3 \
        > $BASE_DIR/run_pancreas_na_v2_10pct.log 2>&1 &"
    echo "[$(date)] 启动: Pancreas 10% NA_v2 on GPU $gpu" | tee -a "$LOG"
    cd "$PANCREAS_DIR" && eval "$CMD" && cd "$BASE_DIR"
    sleep 15
    ps aux | grep -q "python.*train_pancreas_bcp_cmc_na_v2" 2>/dev/null && echo "  ✓ OK" | tee -a "$LOG" || echo "  ✗ FAIL" | tee -a "$LOG"
fi

# ============================================================
# 5. Pancreas label12 (20%) NA_v2_s0.3
# ============================================================
if [ -d "$PANCREAS_DIR/result/cutmix_bcp_cmc_na_v2/20percent/self_train" ] && \
   [ -f "$PANCREAS_DIR/result/cutmix_bcp_cmc_na_v2/20percent/self_train/best_ema20_self.pth" ]; then
    echo "[$(date)] SKIP: Pancreas 20% NA_v2 - 已完成" | tee -a "$LOG"
else
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (Pancreas 20% NA_v2)" | tee -a "$LOG"
        sleep 120
    done
    CMD="nohup $PY train_pancreas_bcp_cmc_na_v2.py \
        --gpu $gpu \
        --label_percent 20 \
        --cmc_patch_size 16 \
        --cmc_soft_sigma 0.3 \
        --cmc_mutual_weight 0.5 \
        --batch_size 2 \
        --lr 1e-3 \
        > $BASE_DIR/run_pancreas_na_v2_20pct.log 2>&1 &"
    echo "[$(date)] 启动: Pancreas 20% NA_v2 on GPU $gpu" | tee -a "$LOG"
    cd "$PANCREAS_DIR" && eval "$CMD" && cd "$BASE_DIR"
    sleep 15
    ps aux | grep -q "python.*train_pancreas_bcp_cmc_na_v2" 2>/dev/null && echo "  ✓ OK" | tee -a "$LOG" || echo "  ✗ FAIL" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "所有 NA_v2 扩展实验已提交: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "当前运行实验检查:" | tee -a "$LOG"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"