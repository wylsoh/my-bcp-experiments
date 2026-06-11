#!/bin/bash
# ============================================================
# GPU 自动监控队列：等 lxm 任务释放 GPU 后依次启动
#   1. ACDC label3 NA_v2_s0.3（batch_size=24，需 ≥23GB）
#   2. LA label8  NA_v2_s0.3（batch_size=8，需 ≥10GB）
#   3. LA label4  NA_v2_s0.3（batch_size=8，需 ≥10GB）
# ============================================================
PY=/home/hjj/anaconda3/envs/yll/bin/python
BASE=/home/hjj/ssq/my-bcp-experiments/code
cd "$BASE" || exit

LOG="$BASE/monitor_acdc_la_queue.log"
PID_FILE="$BASE/monitor_acdc_la_queue.pid"

echo "============================================" | tee -a "$LOG"
echo "ACDC+LA Queue Monitor 启动: $(date)" | tee -a "$LOG"
echo "PID: $$" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# 写入 PID
echo $$ > "$PID_FILE"

# ============================================================
# 辅助函数：查找满足最小空闲显存的 GPU
# ============================================================
find_free_gpu() {
    local min_mb=$1
    for gpu in 0 1 2 3; do
        local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
        [ -n "$free" ] && [ "$free" -ge $min_mb ] && echo "$gpu" && return 0
    done
    echo "" && return 1
}

# ============================================================
# 辅助函数：检查实验是否已完成
# ============================================================
check_acdc_done() {
    [ -f "$BASE/model/BCP/ACDC_BCP_CMC_NA_v2_s0.3_3_labeled/self_train/unet_best_model.pth" ]
}

check_la_done() {
    local labelnum=$1
    [ -f "$BASE/model/BCP/LA_BCP_CMC_NA_v2_s0.3_${labelnum}_labeled/self_train/VNet_best_model.pth" ]
}

# ============================================================
# 辅助函数：在指定 GPU 上启动实验
# ============================================================
run_on_gpu() {
    local gpu=$1
    local logfile=$2
    shift 2
    local cmd="$@"
    echo "[$(date)] 启动在 GPU $gpu: $cmd" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY $cmd > "$BASE/$logfile" 2>&1 &
    local pid=$!
    echo "[$(date)] PID=$pid, 日志: $logfile" | tee -a "$LOG"
    return $pid
}

# ============================================================
# 实验命令定义
# ============================================================
ACDC_CMD="BCP_CMC_v1_NA_v2_soft_mask.py \
    --exp BCP_CMC_NA_v2_s0.3_label3 \
    --cmc_patch_size 8 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 \
    --batch_size 24 --labeled_bs 12 --labelnum 3 \
    --pre_iterations 6000 --max_iterations 30000"

LA8_CMD="LA_BCP_CMC_NA_v2_soft_mask.py \
    --exp BCP_CMC_NA_v2_s0.3 \
    --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 \
    --batch_size 8 --labeled_bs 4 --labelnum 8 \
    --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80"

LA4_CMD="LA_BCP_CMC_NA_v2_soft_mask.py \
    --exp BCP_CMC_NA_v2_s0.3 \
    --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 \
    --batch_size 8 --labeled_bs 4 --labelnum 4 \
    --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80"

# ============================================================
# 主循环
# ============================================================
STEP=0  # 0=等待ACDC, 1=等待LA8, 2=等待LA4, 3=全部完成

while [ $STEP -le 2 ]; do
    echo "[$(date)] 当前进度: STEP=$STEP" | tee -a "$LOG"
    
    if [ $STEP -eq 0 ]; then
        # === ACDC label3 ===
        if check_acdc_done; then
            echo "[$(date)] ACDC label3 已完成，跳过" | tee -a "$LOG"
            STEP=1
            continue
        fi
        
        GPU=$(find_free_gpu 23000)
        if [ -n "$GPU" ]; then
            echo "[$(date)] GPU $GPU 空闲显存充足 (>=23000MB)，启动 ACDC label3" | tee -a "$LOG"
            run_on_gpu "$GPU" "run_acdc_label3_full.log" "$ACDC_CMD"
            echo "[$(date)] ACDC label3 已启动，等待完成..." | tee -a "$LOG"
            
            # 等 ACDC 完成后再继续
            while ! check_acdc_done; do
                sleep 120
            done
            echo "[$(date)] ACDC label3 完成！" | tee -a "$LOG"
            STEP=1
        else
            echo "[$(date)] 没有 GPU 满足 ACDC label3 需求 (>=23000MB)，60秒后重试" | tee -a "$LOG"
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
            sleep 60
        fi
        
    elif [ $STEP -eq 1 ]; then
        # === LA label8 ===
        if check_la_done 8; then
            echo "[$(date)] LA label8 已完成，跳过" | tee -a "$LOG"
            STEP=2
            continue
        fi
        
        GPU=$(find_free_gpu 10000)
        if [ -n "$GPU" ]; then
            echo "[$(date)] GPU $GPU 空闲显存充足 (>=10000MB)，启动 LA label8" | tee -a "$LOG"
            run_on_gpu "$GPU" "run_la_label8_full.log" "$LA8_CMD"
            echo "[$(date)] LA label8 已启动，等待完成..." | tee -a "$LOG"
            
            while ! check_la_done 8; do
                sleep 120
            done
            echo "[$(date)] LA label8 完成！" | tee -a "$LOG"
            STEP=2
        else
            echo "[$(date)] 没有 GPU 满足 LA label8 需求 (>=10000MB)，60秒后重试" | tee -a "$LOG"
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
            sleep 60
        fi
        
    elif [ $STEP -eq 2 ]; then
        # === LA label4 ===
        if check_la_done 4; then
            echo "[$(date)] LA label4 已完成，跳过" | tee -a "$LOG"
            STEP=3
            continue
        fi
        
        GPU=$(find_free_gpu 10000)
        if [ -n "$GPU" ]; then
            echo "[$(date)] GPU $GPU 空闲显存充足 (>=10000MB)，启动 LA label4" | tee -a "$LOG"
            run_on_gpu "$GPU" "run_la_label4_full.log" "$LA4_CMD"
            echo "[$(date)] LA label4 已启动，等待完成..." | tee -a "$LOG"
            
            while ! check_la_done 4; do
                sleep 120
            done
            echo "[$(date)] LA label4 完成！" | tee -a "$LOG"
            STEP=3
        else
            echo "[$(date)] 没有 GPU 满足 LA label4 需求 (>=10000MB)，60秒后重试" | tee -a "$LOG"
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
            sleep 60
        fi
    fi
done

echo "============================================" | tee -a "$LOG"
echo "全部实验已完成！$(date)" | tee -a "$LOG"
echo "  ACDC label3: ✅" | tee -a "$LOG"
echo "  LA label8:   ✅" | tee -a "$LOG"
echo "  LA label4:   ✅" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# 清理 PID 文件
rm -f "$PID_FILE"
