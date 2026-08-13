#!/bin/bash
# start_FLARE.sh — 重启 FLARE 训练（跳过 pre-train）
# GPU1 当前完全空闲 (24 GiB)，可以用完整 batch_size=8

PYTHON="/home/hjj/anaconda3/envs/yll/bin/python"
SCRIPT="/home/hjj/ssq/my-bcp-experiments/code/FLARE_BCP_CMC_NA_v2_soft_mask.py"
LOG="/home/hjj/ssq/my-bcp-experiments/code/model/BCP/nohup_flare_na_v2_10_v2.log"

cd /home/hjj/ssq/my-bcp-experiments/code

# 清理旧进程
pkill -f "FLARE_BCP_CMC_NA_v2_soft_mask" 2>/dev/null
sleep 3

# GPU1 完全空闲，使用默认 batch_size=8
nohup $PYTHON $SCRIPT --gpu 1 --labelnum 42 --batch_size 8 --labeled_bs 2 > $LOG 2>&1 &

echo "FLARE started (PID: $!, batch_size=8)"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
