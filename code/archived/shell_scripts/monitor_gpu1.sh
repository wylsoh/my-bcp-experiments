#!/bin/bash
# GPU 1 监控脚本：等待 NA_v1_bw2_ps8 完成后检查队列是否成功启动
# 每 5 分钟检查一次

LOG="monitor_gpu1.log"
PY=/home/hjj/anaconda3/envs/yll/bin/python
SCRIPT_DIR="/home/hjj/ssq/my-bcp-experiments/code"

cd "$SCRIPT_DIR" || exit

echo "============================================" | tee -a "$LOG"
echo "GPU1 监控启动: $(date)" | tee -a "$LOG"
echo "目标: NA_v1_bw2_ps8 (iter 26117/30000)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

while true; do
    # 检查 GPU 1 显存使用量
    gpu1_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    gpu1_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    
    now=$(date)
    
    if [ -n "$gpu1_free" ] && [ "$gpu1_free" -ge 15000 ]; then
        echo "[$now] ✅ GPU 1 已空闲! (used=${gpu1_used}MB, free=${gpu1_free}MB)" | tee -a "$LOG"
        echo "队列应已自动启动 NA_v3_w1.5_t2.0_ps8" | tee -a "$LOG"
        
        # 检查队列日志
        sleep 10
        if ps aux | grep -q "python.*NA_v3_w1.5_t2.0_ps8" 2>/dev/null; then
            echo "  ✓ 实验已成功启动!" | tee -a "$LOG"
            tail -5 "$SCRIPT_DIR/run_BCP_CMC_NA_v3_w1.5_t2.0_ps8.log" 2>/dev/null | tee -a "$LOG"
        else
            echo "  ✗ 实验未启动! 检查队列状态:" | tee -a "$LOG"
            ps aux | grep queue_NA_v3 | grep -v grep | tee -a "$LOG"
            tail -5 "$SCRIPT_DIR/queue_NA_v3_remaining.log" 2>/dev/null | tee -a "$LOG"
        fi
        break
    fi
    
    # 打印进度
    last_line=$(tail -1 "$SCRIPT_DIR/run_BCP_CMC_NA_v1_bw2_ps8.log" 2>/dev/null)
    echo "[$now] GPU1: used=${gpu1_used}MB free=${gpu1_free}MB | $last_line" | tee -a "$LOG"
    
    sleep 300  # 5 分钟
done

echo "============================================" | tee -a "$LOG"
echo "监控结束: $(date)" | tee -a "$LOG"
