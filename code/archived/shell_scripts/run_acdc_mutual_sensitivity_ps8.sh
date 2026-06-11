#!/bin/bash
# ============================================================
# ACDC cmc_patch_size=8 敏感性分析实验 (智能GPU分配版)
# 每组：pretrain (10K) + self-train (30K)
# GPU策略: 如果有"我的进程"(yll+BCP_CMC)在GPU上,预留6000MB
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python
OUR_RESERVED=6000  # "我的进程"预留显存(MB)
MIN_MEM=5000       # 最小有效空闲显存要求

# 判断进程是否是"我们的进程"(yll环境的BCP_CMC训练)
is_our_process() {
    local pid=$1
    local cmdline
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmdline" | grep -q "/envs/yll/" && echo "$cmdline" | grep -qE "BCP_CMC"; then
        return 0
    fi
    return 1
}

# 检查指定GPU上是否有我们的进程
gpu_has_our_process() {
    local gpu_id=$1
    local pids
    pids=$(nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for pid in $pids; do
        pid=$(echo $pid | xargs)
        if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
            if is_our_process "$pid"; then
                return 0
            fi
        fi
    done
    return 1
}

# 计算有效空闲显存
get_effective_free() {
    local gpu_id=$1
    local mem_free
    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    if gpu_has_our_process "$gpu_id"; then
        local effective=$((mem_free - OUR_RESERVED))
        [ "$effective" -lt 0 ] && effective=0
        echo $effective
    else
        echo $mem_free
    fi
}

# 查找最佳空闲GPU
find_best_gpu() {
    local min_mem=${1:-$MIN_MEM}
    local best_gpu=-1
    local best_effective=0
    for gpu_id in 0 1 2 3; do
        local eff_free
        eff_free=$(get_effective_free "$gpu_id")
        if [ "$eff_free" -gt "$min_mem" ] && [ "$eff_free" -gt "$best_effective" ]; then
            best_gpu=$gpu_id
            best_effective=$eff_free
        fi
        local actual_free
        actual_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
        if gpu_has_our_process "$gpu_id"; then
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB (有我方进程)"
        else
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB"
        fi
    done
    echo $best_gpu
}

# 等待空闲GPU
wait_for_free_gpu() {
    local min_mem=${1:-$MIN_MEM}
    echo "⏳ 等待空闲GPU (需有效空闲 >${min_mem}MB, 有我方进程的预留${OUR_RESERVED}MB)..."
    while true; do
        local gpu
        gpu=$(find_best_gpu $min_mem)
        if [ "$gpu" -ge 0 ]; then
            echo "✅ 选择 GPU ${gpu}"
            return $gpu
        fi
        echo "⏳ 所有GPU忙碌, 15分钟后重试... ($(date))"
        sleep 900
    done
}

# 权重列表
experiments=(
    "BCP_CMC_v1_mutual_mw01_ps8 0.1"
    "BCP_CMC_v1_mutual_mw03_ps8 0.3"
    "BCP_CMC_v1_mutual_mw05_ps8 0.5"
    "BCP_CMC_v1_mutual_mw07_ps8 0.7"
    "BCP_CMC_v1_mutual_mw10_ps8 1.0"
)

total=${#experiments[@]}
current=1

echo "============================================================"
echo "  ACDC cmc_patch_size=8 敏感性分析实验 (智能GPU分配)"
echo "  开始时间: $(date)"
echo "============================================================"

for exp_entry in "${experiments[@]}"; do
    exp_name=$(echo $exp_entry | awk '{print $1}')
    weight=$(echo $exp_entry | awk '{print $2}')

    echo ""
    echo "============================================================"
    echo "  [${current}/${total}] 等待GPU: ${exp_name}"
    echo "  cmc_mutual_weight = ${weight}, cmc_patch_size = 8"
    echo "============================================================"

    # 等待空闲GPU
    GPU=$(wait_for_free_gpu)

    echo "  [${current}/${total}] 开始实验: ${exp_name} on GPU ${GPU}"
    echo "  开始时间: $(date)"
    echo ""

    $PYTHON BCP_CMC_v1_mutual.py \
        --exp "${exp_name}" \
        --cmc_mutual_weight "${weight}" \
        --cmc_patch_size 8 \
        --gpu "${GPU}" \
        --labelnum 7 \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 24 \
        --labeled_bs 12 2>&1 | tee "./run_${exp_name}.log"

    exp_status=$?
    if [ $exp_status -ne 0 ]; then
        echo ""
        echo "  [WARNING] 实验 ${exp_name} 退出码 ${exp_status}"
    fi

    echo ""
    echo "  [${current}/${total}] 完成: ${exp_name} (weight=${weight})"
    echo "  完成时间: $(date)"
    echo ""

    current=$((current + 1))
done

echo ""
echo "============================================================"
echo "  所有实验已完成! 完成时间: $(date)"
echo "============================================================"
