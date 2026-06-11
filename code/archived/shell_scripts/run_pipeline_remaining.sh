#!/bin/bash
# ============================================================
# ACDC 剩余实验智能调度流水线
# 自动找空闲 GPU 启动，有"我的进程"的 GPU 预留 6000MB
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python
PIPELINE_LOG="./pipeline_remaining.log"
OUR_RESERVED=6000
MIN_MEM=5000

echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "  剩余实验智能调度流水线" | tee -a "$PIPELINE_LOG"
echo "  开始时间: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"

# =================== 辅助函数 ===================

is_our_process() {
    local pid=$1
    local cmdline
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmdline" | grep -q "/envs/yll/" && echo "$cmdline" | grep -qE "BCP_CMC"; then
        return 0
    fi
    return 1
}

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
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB (有我方进程)" >> "$PIPELINE_LOG"
        else
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB" >> "$PIPELINE_LOG"
        fi
    done
    echo $best_gpu
}

wait_for_free_gpu() {
    local min_mem=${1:-$MIN_MEM}
    echo "⏳ 等待空闲GPU (有效空闲 >${min_mem}MB)..." | tee -a "$PIPELINE_LOG"
    while true; do
        local gpu
        gpu=$(find_best_gpu $min_mem)
        if [ "$gpu" -ge 0 ]; then
            echo "✅ 选择 GPU ${gpu}" | tee -a "$PIPELINE_LOG"
            return $gpu
        fi
        echo "⏳ 所有GPU忙碌, 15分钟后重试... ($(date))" | tee -a "$PIPELINE_LOG"
        sleep 900
    done
}

run_experiment_and_wait() {
    local script=$1
    local name=$2
    local extra_args=$3
    local gpu=$4
    local log_file="./run_${name}.log"

    echo "$(date) - 🚀 启动 ${name} on GPU ${gpu}" | tee -a "$PIPELINE_LOG"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$script" \
        --exp "${name}" \
        --cmc_patch_size 8 \
        --cmc_mutual_weight 0.5 \
        --gpu 0 \
        --labelnum 7 \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 12 \
        --labeled_bs 6 $extra_args 2>&1 | tee "$log_file"

    echo "$(date) - ✅ ${name} 完成" | tee -a "$PIPELINE_LOG"
}

# =================== 剩余实验队列 ===================

declare -a queue_scripts=(
    "BCP_CMC_v1_ablation_cmc_only.py"
    "BCP_CMC_v1_asymmetric.py"
    "BCP_CMC_v1_attention.py"
    "BCP_CMC_v1_classbal.py"
    "BCP_CMC_v1_freq.py"
)

declare -a queue_names=(
    "BCP_CMC_v1_ablation_cmc_only_ps8"
    "BCP_CMC_v1_asymmetric_ps8"
    "BCP_CMC_v1_attention_ps8"
    "BCP_CMC_v1_classbal_ps8"
    "BCP_CMC_v1_freq_ps8"
)

declare -a queue_args=(
    ""
    ""
    ""
    ""
    ""
)

total=${#queue_scripts[@]}

echo "" | tee -a "$PIPELINE_LOG"
echo "===== 剩余实验队列 (${total}个) =====" | tee -a "$PIPELINE_LOG"
for i in $(seq 0 $((total-1))); do
    echo "  [$((i+1))] ${queue_names[$i]}" | tee -a "$PIPELINE_LOG"
done
echo "===================================" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

# =================== 主循环 ===================

for i in $(seq 0 $((total-1))); do
    script="${queue_scripts[$i]}"
    name="${queue_names[$i]}"
    extra="${queue_args[$i]}"

    echo "" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  [$((i+1))/${total}] 等待GPU: ${name}" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"

    wait_for_free_gpu $MIN_MEM
    next_gpu=$?

    run_experiment_and_wait "$script" "$name" "$extra" "$next_gpu"
done

# =================== 完成 ===================
echo "" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "  🎉 所有剩余实验已完成!" | tee -a "$PIPELINE_LOG"
echo "  完成时间: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
