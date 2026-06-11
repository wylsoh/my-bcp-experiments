#!/bin/bash
# ============================================================
# ACDC CMC 完整自动化流水线 v3 (智能GPU分配)
# 功能：
#   1. 动态选择最空闲GPU启动实验
#   2. 如果GPU上有"我的进程"(yll环境下的BCP_CMC训练)，预留6000MB
#   3. 每组完成后自动运行 test_ACDC.py + 自动清理中间文件
#   4. 串行执行队列，避免超分
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python
TEST_SCRIPT="/home/hjj/ssq/my-bcp-experiments/code/test_ACDC.py"
MODEL_BASE="/home/hjj/ssq/my-bcp-experiments/code/model/BCP"
PIPELINE_LOG="./pipeline_overall.log"
MIN_MEM=5000  # 最低有效空闲显存要求(MB)
OUR_RESERVED=6000  # "我的进程"预留显存(MB)

echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "  ACDC CMC 完整自动化流水线 v3 (智能GPU分配)" | tee -a "$PIPELINE_LOG"
echo "  开始时间: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

# ============== 辅助函数 ==============

# 判断进程是否是"我们的进程"(yll环境的BCP_CMC训练)
is_our_process() {
    local pid=$1
    local cmdline
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    # 检查是否包含 yll 环境 + BCP_CMC
    if echo "$cmdline" | grep -q "/envs/yll/" && echo "$cmdline" | grep -qE "BCP_CMC"; then
        return 0  # 是我们的进程
    fi
    return 1  # 不是
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
                return 0  # 有我们的进程
            fi
        fi
    done
    return 1  # 没有
}

# 计算GPU的有效空闲显存
# 有效空闲 = 实际空闲 - 6000(如果有我们的进程在跑)
get_effective_free() {
    local gpu_id=$1
    local mem_free
    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    
    if gpu_has_our_process "$gpu_id"; then
        local effective=$((mem_free - OUR_RESERVED))
        # 有效空闲最低为0
        if [ "$effective" -lt 0 ]; then
            effective=0
        fi
        echo $effective
    else
        echo $mem_free
    fi
}

# 查找最佳空闲GPU
# 将结果写入 $1 指定的变量名；通过返回值(echo)仅输出GPU ID
# 注意: 所有调试信息用 >>$PIPELINE_LOG 而非 tee，避免被 $() 捕获
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
            echo "$(date) -   GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效空闲=${eff_free}MB (🟡 有我方进程)" >> "$PIPELINE_LOG"
        else
            echo "$(date) -   GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效空闲=${eff_free}MB" >> "$PIPELINE_LOG"
        fi
    done
    
    if [ "$best_gpu" -ge 0 ]; then
        actual_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $best_gpu 2>/dev/null)
        eff_free=$(get_effective_free "$best_gpu")
        echo "$(date) - ✅ 最佳选择: GPU ${best_gpu} (实际空闲=${actual_free}MB, 有效=${eff_free}MB)" >> "$PIPELINE_LOG"
    fi
    
    echo $best_gpu
}

wait_for_free_gpu() {
    local min_mem=${1:-$MIN_MEM}
    echo "$(date) - ⏳ 等待空闲GPU (需有效空闲 >${min_mem}MB) ..." | tee -a "$PIPELINE_LOG"
    echo "$(date) - ⚠️  注意: 有我的进程的GPU会自动预留${OUR_RESERVED}MB" | tee -a "$PIPELINE_LOG"
    while true; do
        local gpu
        gpu=$(find_best_gpu $min_mem)
        if [ "$gpu" -ge 0 ]; then
            return $gpu
        fi
        echo "$(date) - ⏳ 所有GPU忙碌, 15分钟后重试..." | tee -a "$PIPELINE_LOG"
        sleep 900  # 15分钟
    done
}

run_test() {
    local exp_name=$1
    local labelnum=$2
    local stage=$3
    local gpu=$4
    local test_log="./test_${exp_name}.log"

    echo "$(date) - 🚀 测试: ${exp_name} (GPU ${gpu})" | tee -a "$PIPELINE_LOG"
    cd /home/hjj/ssq/my-bcp-experiments/code
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $TEST_SCRIPT \
        --exp "$exp_name" \
        --labelnum $labelnum \
        --stage_name $stage 2>&1 | tee -a "$test_log"
    echo "$(date) - ✅ 测试完成: ${exp_name}" | tee -a "$PIPELINE_LOG"

    perf_file="${MODEL_BASE}/ACDC_${exp_name}_${labelnum}_labeled/performance.txt"
    if [ -f "$perf_file" ]; then
        echo "========== ${exp_name} 测试结果 ==========" | tee -a "$PIPELINE_LOG"
        cat "$perf_file" | tee -a "$PIPELINE_LOG"
        echo "==========================================" | tee -a "$PIPELINE_LOG"
    fi
}

cleanup_checkpoints() {
    local exp_name=$1
    local labelnum=$2
    local pre_dir="${MODEL_BASE}/ACDC_${exp_name}_${labelnum}_labeled/pre_train"
    local self_dir="${MODEL_BASE}/ACDC_${exp_name}_${labelnum}_labeled/self_train"

    echo "$(date) - 🧹 清理中间文件: ${exp_name}" | tee -a "$PIPELINE_LOG"

    if [ -d "$pre_dir" ]; then
        ls "$pre_dir"/iter_*.pth 2>/dev/null | grep -v "iter_10000" | xargs -I {} rm -f {}
        echo "  - pretrain 清理完成" | tee -a "$PIPELINE_LOG"
    fi

    if [ -d "$self_dir" ]; then
        latest=$(ls "$self_dir"/iter_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -1)
        second=$(ls "$self_dir"/iter_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -2 | head -1)
        for f in $(ls "$self_dir"/iter_*.pth 2>/dev/null); do
            if [ "$f" != "$latest" ] && [ "$f" != "$second" ]; then
                rm -f "$f"
            fi
        done
        echo "  - self-train 清理完成" | tee -a "$PIPELINE_LOG"
    fi
}

run_experiment_and_wait() {
    local script=$1
    local exp_name=$2
    local extra_args=$3
    local gpu=$4

    # 每次进入实验前确保在 code 目录
    cd /home/hjj/ssq/my-bcp-experiments/code

    echo "" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "  🏃 启动实验: ${exp_name}" | tee -a "$PIPELINE_LOG"
    echo "  脚本: ${script}" | tee -a "$PIPELINE_LOG"
    echo "  GPU: ${gpu}" | tee -a "$PIPELINE_LOG"
    echo "  开始时间: $(date)" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"

    # 后台启动训练
    $PYTHON "$script" \
        --exp "${exp_name}" \
        --gpu "${gpu}" \
        --labelnum 7 \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 24 \
        --labeled_bs 12 \
        ${extra_args} > "./run_${exp_name}.log" 2>&1 &
    
    local train_pid=$!
    echo "$(date) - PID: ${train_pid}" | tee -a "$PIPELINE_LOG"

    # 监控训练进度，直到完成
    while true; do
        sleep 1800  # 30分钟检查一次
        if [ ! -d /proc/$train_pid ]; then
            wait $train_pid 2>/dev/null
            local status=$?
            echo "$(date) - 实验 ${exp_name} 完成, 退出码: ${status}" | tee -a "$PIPELINE_LOG"
            break
        fi
        
        # 检查是否已完成(iter 30000 + mean_dice)
        has_30000=$(grep -c "iteration 30000" "./run_${exp_name}.log" 2>/dev/null)
        has_mean_dice=$(grep -c "mean_dice" "./run_${exp_name}.log" 2>/dev/null)
        if [ "$has_30000" -gt 0 ] && [ "$has_mean_dice" -gt 0 ]; then
            echo "$(date) - ✅ ${exp_name} 训练完成 (detected iter 30000)" | tee -a "$PIPELINE_LOG"
            sleep 120  # 等日志写完
            break
        fi
        
        # 定期输出进度
        last_iter=$(tail -5 "./run_${exp_name}.log" 2>/dev/null | grep -oP 'iteration \K[0-9]+' | tail -1)
        echo "$(date) - ${exp_name} progress: iter ${last_iter:-unknown}" | tee -a "$PIPELINE_LOG"
    done

    # 运行测试
    run_test "$exp_name" 7 "self_train" "$gpu"

    # 清理中间文件
    cleanup_checkpoints "$exp_name" 7
}

# ============== 主流水线 ==============

# 实验队列
declare -a queue_scripts=(
    "BCP_CMC_v1_mutual_convkernel.py"
    "BCP_CMC_v1_NA_v1_border_shared.py"
    "BCP_CMC_v1_NA_v2_soft_mask.py"
    "BCP_CMC_v1_NA_v3_border_loss.py"
)

declare -a queue_names=(
    "BCP_CMC_v1_mutual_mw06_ps8_conv3"
    "BCP_CMC_NA_v1_bw1"
    "BCP_CMC_NA_v2"
    "BCP_CMC_NA_v3"
)

declare -a queue_args=(
    "--cmc_mutual_weight 0.6 --cmc_patch_size 8 --cmc_kernel_size 3"
    "--cmc_border_width 1"
    "--cmc_soft_sigma 0.5"
    "--cmc_border_loss_weight 1.5 --cmc_border_kl_temp 1.0"
)

total=${#queue_scripts[@]}
completed=0

echo "" | tee -a "$PIPELINE_LOG"
echo "===== 智能GPU调度流水线 =====" | tee -a "$PIPELINE_LOG"
echo "  队列中 ${total} 个实验" | tee -a "$PIPELINE_LOG"
echo "  GPU策略: 如果有'我的进程'在GPU上,预留${OUR_RESERVED}MB" | tee -a "$PIPELINE_LOG"
echo "  有效空闲 > ${MIN_MEM}MB 才启动新实验" | tee -a "$PIPELINE_LOG"
echo "" | tee -a "$PIPELINE_LOG"

while [ $completed -lt $total ]; do
    echo "$(date) - 📋 进度: ${completed}/${total} 已完成" | tee -a "$PIPELINE_LOG"
    
    # 等待空闲GPU
    wait_for_free_gpu $MIN_MEM
    next_gpu=$?

    # 启动下一个实验
    script="${queue_scripts[$completed]}"
    name="${queue_names[$completed]}"
    args="${queue_args[$completed]}"

    echo "$(date) - 🎯 启动队列中第 $((completed+1)) 个实验: ${name}" | tee -a "$PIPELINE_LOG"
    
    # 后台运行实验（包含监控、测试、清理）
    run_experiment_and_wait "$script" "$name" "$args" "$next_gpu"
    
    completed=$((completed + 1))
done

# ============== 完成 ==============
echo "" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
echo "  🎉 所有实验已完成!" | tee -a "$PIPELINE_LOG"
echo "  完成时间: $(date)" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"

# 汇总所有测试结果
echo "" | tee -a "$PIPELINE_LOG"
echo "========== 最终测试结果汇总 ==========" | tee -a "$PIPELINE_LOG"

for name in "${queue_names[@]}"; do
    perf_file="${MODEL_BASE}/ACDC_${name}_7_labeled/performance.txt"
    if [ -f "$perf_file" ]; then
        echo "" | tee -a "$PIPELINE_LOG"
        echo "--- ${name} ---" | tee -a "$PIPELINE_LOG"
        cat "$perf_file" | tee -a "$PIPELINE_LOG"
    fi
done

echo "" | tee -a "$PIPELINE_LOG"
echo "============================================================" | tee -a "$PIPELINE_LOG"
