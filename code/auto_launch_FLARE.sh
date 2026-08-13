#!/bin/bash
# auto_launch_FLARE.sh — 等 NA_v2 训练完成后自动启动 FLARE 训练
#
# 用法:
#   ./auto_launch_FLARE.sh                    # 等 GPU1 上 NA_v2 10% 跑完，启动 FLARE 10%
#   ./auto_launch_FLARE.sh 2 42              # 等 GPU2 上 NA_v2 20% 跑完，启动 FLARE 20%
#
# 监控日志：
#   tail -f /home/hjj/ssq/my-bcp-experiments/code/model/BCP/nohup_flare_na_v2_10.log

LOG_PATH="/home/hjj/ssq/my-bcp-experiments/code/model/BCP/nohup_self_train_na_v2_10.log"
MAX_ITER=30000
SLEEP_INTERVAL=60

TARGET_GPU="${1:-1}"
LABELNUM="${2:-42}"
PYTHON="/home/hjj/anaconda3/envs/yll/bin/python"
SCRIPT="/home/hjj/ssq/my-bcp-experiments/code/FLARE_BCP_CMC_NA_v2_soft_mask.py"
FLARE_LOG="/home/hjj/ssq/my-bcp-experiments/code/model/BCP/nohup_flare_na_v2_10.log"
START_SCRIPT="/home/hjj/ssq/my-bcp-experiments/code/start_FLARE.sh"

echo "[$(date +'%H:%M:%S')] auto_launch_FLARE: waiting for NA_v2 to finish..."
echo "  Monitor log: $LOG_PATH"
echo "  Target GPU: $TARGET_GPU"
echo "  Labelnum: $LABELNUM"
echo "  Poll interval: ${SLEEP_INTERVAL}s"
echo ""

# 轮询等待训练完成
last_iter=-1
while true; do
    if [ ! -f "$LOG_PATH" ]; then
        echo "[$(date +'%H:%M:%S')] Log not found: $LOG_PATH, waiting..."
        sleep $SLEEP_INTERVAL
        continue
    fi

    # 解析最后一行中的迭代号
    last_line=$(tail -1 "$LOG_PATH" 2>/dev/null)
    current_iter=$(echo "$last_line" | grep -oP 'iter \K\d+' | head -1)

    if [ -n "$current_iter" ] && [ "$current_iter" -ge "$MAX_ITER" ]; then
        echo "[$(date +'%H:%M:%S')] NA_v2 completed at iter $current_iter!"
        break
    fi

    if [ -n "$current_iter" ]; then
        progress=$((current_iter * 100 / MAX_ITER))
        remaining=$((MAX_ITER - current_iter))
        echo "[$(date +'%H:%M:%S')] NA_v2: iter $current_iter/$MAX_ITER (${progress}%), ~${remaining} iters left"
    fi

    sleep $SLEEP_INTERVAL
done

# 等待 GPU 释放
echo "[$(date +'%H:%M:%S')] Waiting 120s for GPU memory release..."
sleep 120

# 启动 FLARE
echo "[$(date +'%H:%M:%S')] Launching FLARE on GPU$TARGET_GPU with labelnum=$LABELNUM..."
echo "  nohup $PYTHON $SCRIPT --gpu $TARGET_GPU --labelnum $LABELNUM > $FLARE_LOG 2>&1 &"
echo "  Log: $FLARE_LOG"

cd /home/hjj/ssq/my-bcp-experiments/code
nohup $PYTHON $SCRIPT --gpu $TARGET_GPU --labelnum $LABELNUM > $FLARE_LOG 2>&1 &

echo "[$(date +'%H:%M:%S')] FLARE training started on GPU$TARGET_GPU"
echo "  Monitor: tail -f $FLARE_LOG"
echo "  Also: $START_SCRIPT (for manual restart)"
