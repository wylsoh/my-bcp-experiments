#!/bin/bash
# ============================================================
# Comprehensive Queue: NA_v2_soft_mask 最优结果扩展实验
# 
# 包含实验（按优先级排列）：
#   1-3: NA_v3_w1.5_t1.0 / w3.0_t1.0 / w1.5_t2.0 (来自现有队列)
#   4:   ACDC label3 NA_v2_s0.3
#   5-6: LA label8 / label4 NA_v2_s0.3
#   7-8: Pancreas label6 / label12 NA_v2_s0.3
# ============================================================
PY=/home/hjj/anaconda3/envs/yll/bin/python
cd /home/hjj/ssq/my-bcp-experiments/code || exit
LOG="queue_all_na_v2_extended.log"

find_free_gpu() {
    for gpu in 0 1 2 3; do
        local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
        [ -n "$free" ] && [ "$free" -ge 4000 ] && echo "$gpu" && return 0
    done
    echo "" && return 1
}

# ============================================================
# 队列定义
# ============================================================
declare -a QUEUE

# --- NA_v3 剩余实验（同原队列） ---
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t1.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w3.0_t1.0_ps8 --cmc_border_loss_weight 3.0 --cmc_border_kl_temp 1.0 --batch_size 12 --labeled_bs 6")
QUEUE+=("BCP_CMC_v1_NA_v3_border_loss.py BCP_CMC_NA_v3_w1.5_t2.0_ps8 --cmc_border_loss_weight 1.5 --cmc_border_kl_temp 2.0 --batch_size 12 --labeled_bs 6")

# --- ACDC label3 NA_v2_s0.3 ---
# 用已有 pretrain (ACDC_BCP_CMC_NA_v2_s0.3_ps8_7_labeled/pre_train) 无法直接复用，
# 因为 label3 的数据划分不同。需要从头预训练。
QUEUE+=("BCP_CMC_v1_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.3_label3 --cmc_patch_size 8 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --batch_size 12 --labeled_bs 6 --labelnum 3 --pre_iterations 6000 --max_iterations 30000")

# --- LA label8 NA_v2_s0.3 ---
# LA 需要 3D 版本脚本，位于 my-bcp-experiments/code/ 下
QUEUE+=("LA_BCP_CMC_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.3 --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --batch_size 8 --labeled_bs 4 --labelnum 8 --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80")

# --- LA label4 NA_v2_s0.3 ---
QUEUE+=("LA_BCP_CMC_NA_v2_soft_mask.py BCP_CMC_NA_v2_s0.3 --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --batch_size 8 --labeled_bs 4 --labelnum 4 --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80")

# --- Pancreas label6 (10%) NA_v2_s0.3 ---
# Pancreas 使用 epoch 制训练，在 BCP_original/code/pancreas/ 下运行
# label6 = 10percent (6 labeled out of 62 total)
QUEUE+=("pancreas/train_pancreas_bcp_cmc_na_v2.py pancreas_6 --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --label_percent 10 --batch_size 2 --lr 1e-3 --skip_pretrain")

# --- Pancreas label12 (20%) NA_v2_s0.3 ---
# label12 = 20percent (12 labeled out of 62 total)
QUEUE+=("pancreas/train_pancreas_bcp_cmc_na_v2.py pancreas_12 --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --label_percent 20 --batch_size 2 --lr 1e-3 --skip_pretrain")

echo "============================================" | tee -a "$LOG"
echo "NA_v2 扩展实验 Queue 启动: $(date)" | tee -a "$LOG"
echo "共计 ${#QUEUE[@]} 个实验" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

for idx in "${!QUEUE[@]}"; do
    entry="${QUEUE[$idx]}"
    read -r script exp_name extra_args <<< "$entry"
    
    # 检查是否已完成
    if [ "$script" = "BCP_CMC_v1_NA_v2_soft_mask.py" ] || [ "$script" = "BCP_CMC_v1_NA_v3_border_loss.py" ]; then
        # ACDC 实验目录
        EXP_DIR="model/BCP/ACDC_${exp_name}_7_labeled"
        [ "$script" = "BCP_CMC_v1_NA_v2_soft_mask.py" ] && [ ! -z "$(echo $extra_args | grep labelnum=3)" ] && EXP_DIR="model/BCP/ACDC_${exp_name/_label3/}_3_labeled"
        if [ -f "$EXP_DIR/self_train/unet_best_model.pth" ]; then
            echo "[$(date)] SKIP #$((idx+1)): $exp_name - 已完成" | tee -a "$LOG"
            continue
        fi
    elif [ "$script" = "LA_BCP_CMC_NA_v2_soft_mask.py" ]; then
        EXP_DIR="model/BCP/LA_${exp_name}_${extra_args#*labelnum=}"
        EXP_DIR="model/BCP/LA_${exp_name}_$(echo $extra_args | grep -oP 'labelnum=\K\d+')_labeled"
        if [ -d "$EXP_DIR/self_train" ] && [ -f "$EXP_DIR/self_train/VNet_best_model.pth" ]; then
            echo "[$(date)] SKIP #$((idx+1)): $exp_name - LA 已完成" | tee -a "$LOG"
            continue
        fi
    fi
    
    while true; do
        gpu=$(find_free_gpu)
        [ -n "$gpu" ] && break
        echo "[$(date)] 等待GPU空闲... (#$((idx+1)): $exp_name)" | tee -a "$LOG"
        sleep 120
    done
    
    LOG_FILE="run_${exp_name}.log"
    
    if [ "$script" = "pancreas/train_pancreas_bcp_cmc_na_v2.py" ]; then
        # Pancreas 实验在 BCP_original 目录下运行
        cd /home/hjj/ssq/BCP_original/code/pancreas
        # 解析 label_percent
        lp=$(echo $extra_args | grep -oP 'label_percent=\K\d+')
        CMD="nohup $PY train_pancreas_bcp_cmc_na_v2.py --gpu $gpu --label_percent $lp --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --batch_size 2 --lr 1e-3 > /home/hjj/ssq/my-bcp-experiments/code/${LOG_FILE} 2>&1 &"
        cd /home/hjj/ssq/my-bcp-experiments/code
    elif [ "$script" = "LA_BCP_CMC_NA_v2_soft_mask.py" ]; then
        # LA 实验
        labelnum=$(echo $extra_args | grep -oP 'labelnum=\K\d+')
        CMD="nohup $PY $script --exp $exp_name --cmc_patch_size 16 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --batch_size 8 --labeled_bs 4 --gpu $gpu --labelnum $labelnum --pre_max_iteration 2000 --self_max_iteration 15000 --max_samples 80 > $LOG_FILE 2>&1 &"
    else
        # ACDC 实验
        CMD="nohup $PY $script --exp $exp_name --cmc_patch_size 8 --cmc_soft_sigma 0.3 --cmc_mutual_weight 0.5 --gpu $gpu --labelnum 3 --pre_iterations 6000 --max_iterations 30000 --batch_size 12 --labeled_bs 6 $extra_args > $LOG_FILE 2>&1 &"
    fi
    
    echo "[$(date)] 启动 #$((idx+1)): $exp_name on GPU $gpu" | tee -a "$LOG"
    eval "$CMD"
    sleep 15
    if ps aux | grep -q "python.*$exp_name" 2>/dev/null; then
        echo "  ✓ 启动成功" | tee -a "$LOG"
    else
        echo "  ✗ 启动失败" | tee -a "$LOG"
        tail -3 "$LOG_FILE" 2>/dev/null | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "所有实验已提交: $(date)" | tee -a "$LOG"
