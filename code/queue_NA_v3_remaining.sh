#!/bin/bash
# NA_v3 剩余3个实验排队启动（修正前缀）
PY=/home/hjj/anaconda3/envs/yll/bin/python
cd /home/hjj/ssq/my-bcp-experiments/code || exit
LOG="queue_NA_v3_remaining.log"

find_free_gpu() {
    for gpu in 0 1 2 3; do
        local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
        [ -n "$free" ] && [ "$free" -ge 4000 ] && echo "$gpu" && return 0
    done
    echo "" && return 1
}

declare -a QUEUE
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t1.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w3.0_t1.0_ps8 --cmc_border_loss_weight 3.0 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t2.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 2.0 --batch_size 12 --labeled_bs 6")

echo "============================================" | tee -a "$LOG"
echo "NA_v3 Remaining Queue 启动: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

for entry in "${QUEUE[@]}"; do
    read -r script exp_name extra_args <<< "$entry"
    
    EXP_DIR="model/BCP/ACDC_${exp_name}_7_labeled"
    if [ -f "$EXP_DIR/self_train/unet_best_model.pth" ]; then
        echo "[$(date)] SKIP: $exp_name - 已完成训练" | tee -a "$LOG"
        continue
    fi
    if [ ! -f "$EXP_DIR/pre_train/unet_best_model.pth" ]; then
        echo "[$(date)] ERROR: $exp_name - 无预训练模型！" | tee -a "$LOG"
        continue
    fi
    
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... ($exp_name)" | tee -a "$LOG"
        sleep 120
    done
    
    LOG_FILE="run_${exp_name}.log"
    CMD="nohup $PY $script --exp $exp_name --cmc_patch_size 8 --cmc_mutual_weight 0.5 --gpu $gpu --labelnum 7 --pre_iterations 0 --max_iterations 30000 $extra_args > $LOG_FILE 2>&1 &"
    
    echo "[$(date)] 启动: $exp_name on GPU $gpu" | tee -a "$LOG"
    eval "$CMD"
    sleep 15
    if ps aux | grep -q "python.*$exp_name" 2>/dev/null; then
        echo "  ✓ 启动成功" | tee -a "$LOG"
    else
        echo "  ✗ 启动失败" | tee -a "$LOG"
        tail -3 "$LOG_FILE" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "所有NA_v3实验已提交: $(date)" | tee -a "$LOG"
