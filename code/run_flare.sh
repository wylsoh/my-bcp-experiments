#!/bin/bash
# 先跑 labeled_num=42
echo "========================================"
echo "[$(date)] Starting labeled_num=42 on GPU 2"
echo "========================================"
conda run -n yll python flare_BCP_CMC.py \
    --labeled_num 42 \
    --gpu 2 \
    --exp BCP_CMC_v1_FLARE42 \
    --self_max_iteration 15000
FLARE42_EXIT=$?
echo "[$(date)] labeled_num=42 finished with exit code $FLARE42_EXIT"

# 再跑 labeled_num=21
echo "========================================"
echo "[$(date)] Starting labeled_num=21 on GPU 2"
echo "========================================"
conda run -n yll python flare_BCP_CMC.py \
    --labeled_num 21 \
    --gpu 2 \
    --exp BCP_CMC_v1_FLARE21 \
    --self_max_iteration 15000
FLARE21_EXIT=$?
echo "[$(date)] labeled_num=21 finished with exit code $FLARE21_EXIT"

echo "========================================"
echo "[$(date)] All FLARE training completed!"
echo "  labeled_num=42: exit $FLARE42_EXIT"
echo "  labeled_num=21: exit $FLARE21_EXIT"
echo "========================================"
