#!/bin/bash
# ============================================================
# LA 启动监控：等 ACDC label3 训练完成后自动启动 LA
#   1. LA label8  NA_v2_s0.3（batch_size=8）
#   2. LA label4  NA_v2_s0.3（batch_size=8）
# ============================================================
PY=/home/hjj/anaconda3/envs/yll/bin/python
BASE=/home/hjj/ssq/my-bcp-experiments/code
cd "$BASE" || exit

LOG="$BASE/run_la_after_acdc.log"
echo "============================================" | tee -a "$LOG"
echo "LA 启动监控: $(date)" | tee -a "$LOG"
echo "等待 ACDC label3 完成..." | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# 等待 ACDC label3 完成
while [ ! -f "$BASE/model/BCP/ACDC_BCP_CMC_NA_v2_s0.3_3_labeled/self_train/unet_best_model.pth" ]; do
    sleep 120
done
echo "[$(date)] ACDC label3 已完成！" | tee -a "$LOG"

# ============================================================
# LA label8
# ============================================================
echo "[$(date)] 启动 LA label8..." | tee -a "$LOG"
nohup $PY LA_BCP_CMC_NA_v2_soft_mask.py \
    --exp BCP_CMC_NA_v2_s0.3 \
    --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 \
    --batch_size 8 --labeled_bs 4 --labelnum 8 \
    --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80 \
    --gpu 2 \
    > "$BASE/run_la_label8_full.log" 2>&1 &
echo "[$(date)] LA label8 PID: $!" | tee -a "$LOG"

while [ ! -f "$BASE/model/BCP/LA_BCP_CMC_NA_v2_s0.3_8_labeled/self_train/VNet_best_model.pth" ]; do
    sleep 120
done
echo "[$(date)] LA label8 已完成！" | tee -a "$LOG"

# ============================================================
# LA label4
# ============================================================
echo "[$(date)] 启动 LA label4..." | tee -a "$LOG"
nohup $PY LA_BCP_CMC_NA_v2_soft_mask.py \
    --exp BCP_CMC_NA_v2_s0.3 \
    --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 \
    --batch_size 8 --labeled_bs 4 --labelnum 4 \
    --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80 \
    --gpu 2 \
    > "$BASE/run_la_label4_full.log" 2>&1 &
echo "[$(date)] LA label4 PID: $!" | tee -a "$LOG"

while [ ! -f "$BASE/model/BCP/LA_BCP_CMC_NA_v2_s0.3_4_labeled/self_train/VNet_best_model.pth" ]; do
    sleep 120
done
echo "[$(date)] LA label4 已完成！" | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"
echo "所有实验已完成！$(date)" | tee -a "$LOG"
echo "  ACDC label3: ✅" | tee -a "$LOG"
echo "  LA label8:   ✅" | tee -a "$LOG"
echo "  LA label4:   ✅" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
