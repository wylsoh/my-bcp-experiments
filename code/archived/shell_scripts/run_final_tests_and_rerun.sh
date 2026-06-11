#!/bin/bash
# ============================================================
# 最终测试 + NA_v3 重跑脚本
# 1) 对所有已完成实验执行测试
# 2) 重跑 NA_v3 (border_loss, bug已修复)
# ============================================================

cd /home/hjj/ssq/my-bcp-experiments/code

PYTHON=/home/hjj/anaconda3/envs/yll/bin/python
TEST_SCRIPT="/home/hjj/ssq/BCP/code/test_ACDC.py"
TEST_LOG="./run_final_tests.log"
OUR_RESERVED=6000
MIN_MEM=5000

echo "============================================================" | tee -a "$TEST_LOG"
echo "  最终测试 + NA_v3 重跑" | tee -a "$TEST_LOG"
echo "  开始时间: $(date)" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"

# test_ACDC.py 使用路径: ./model/BCP/ACDC_{exp}_{labelnum}_labeled/{stage_name}/unet_best_model.pth
# 所以 exp = 目录名去掉 "ACDC_" 前缀和 "_7_labeled" 后缀
# 例如: ACDC_BCP_CMC_v1_mutual_mw01_ps8_7_labeled → exp: BCP_CMC_v1_mutual_mw01_ps8

# =================== GPU 辅助函数 ===================

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
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB (有我方进程)" >> "$TEST_LOG"
        else
            echo "  GPU ${gpu_id}: 实际空闲=${actual_free}MB, 有效=${eff_free}MB" >> "$TEST_LOG"
        fi
    done
    echo $best_gpu
}

wait_for_free_gpu() {
    local min_mem=${1:-$MIN_MEM}
    echo "⏳ 等待空闲GPU (有效空闲 >${min_mem}MB)..." | tee -a "$TEST_LOG"
    while true; do
        local gpu
        gpu=$(find_best_gpu $min_mem)
        if [ "$gpu" -ge 0 ]; then
            echo "✅ 选择 GPU ${gpu}" | tee -a "$TEST_LOG"
            return $gpu
        fi
        echo "⏳ 所有GPU忙碌, 5分钟后重试... ($(date))" | tee -a "$TEST_LOG"
        sleep 300
    done
}

# =================== 测试函数 ===================
# test_ACDC.py 自动拼接路径: ./model/BCP/ACDC_{exp}_{labelnum}_labeled/{stage}/unet_best_model.pth
# 注意: 已完成的实验目录名以 ACDC_ 开头，例如 ACDC_BCP_CMC_v1_mutual_mw05_7_labeled
#       所以 exp 参数应为 "BCP_CMC_v1_mutual_mw05" (去掉 ACDC_ 前缀)
#       但有些历史实验目录名是 ACDC_ACDC_BCP_... (test_ACDC.py 会再加 ACDC_)

run_test() {
    local exp=$1
    local labelnum=$2
    local gpu=$3

    echo "$(date) - 🧪 测试 ${exp} (labelnum=${labelnum}) on GPU ${gpu}" | tee -a "$TEST_LOG"

    # 必须在 my-bcp-experiments/code 下运行，因为 test_ACDC.py 使用相对路径 ./model/BCP/
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$TEST_SCRIPT" \
        --exp "${exp}" \
        --labelnum "${labelnum}" 2>&1 | tee -a "$TEST_LOG"

    echo "$(date) - ✅ ${exp} 测试完成" | tee -a "$TEST_LOG"
}

# =================== 需要测试的实验列表 ===================
# exp_name(=目录) | 用于 test_ACDC.py 的 --exp | labelnum
declare -a test_dirs=(
    "ACDC_BCP_CMC_v1_mutual_mw01_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_mutual_mw03_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_mutual_mw05_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_mutual_mw07_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_mutual_mw06_ps8_conv3_7_labeled"
    "ACDC_BCP_CMC_NA_v1_bw1_bs12_7_labeled"
    "ACDC_BCP_CMC_NA_v2_bs12_7_labeled"
    "ACDC_BCP_CMC_v1_ablation_cmc_only_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_asymmetric_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_attention_ps8_7_labeled"
    "ACDC_BCP_CMC_v1_classbal_ps8_7_labeled"
)

echo "" | tee -a "$TEST_LOG"
echo "===== 测试队列 (${#test_dirs[@]}个) =====" | tee -a "$TEST_LOG"
for d in "${test_dirs[@]}"; do
    echo "  - ${d}" | tee -a "$TEST_LOG"
done
echo "" | tee -a "$TEST_LOG"

# =================== 执行测试 ===================
# 验证模型存在
MODEL_BASE="./model/BCP"
for d in "${test_dirs[@]}"; do
    model_path="${MODEL_BASE}/${d}/self_train/unet_best_model.pth"
    if [ ! -f "$model_path" ]; then
        echo "⚠️ 模型不存在: ${model_path}" | tee -a "$TEST_LOG"
    fi
done

for d in "${test_dirs[@]}"; do
    echo "------------------------------------------------------------" | tee -a "$TEST_LOG"
    echo "  测试: ${d}" | tee -a "$TEST_LOG"
    echo "------------------------------------------------------------" | tee -a "$TEST_LOG"

    wait_for_free_gpu 2000
    test_gpu=$?

    # 从目录名提取 exp 名称
    # ACDC_BCP_CMC_v1_mutual_mw01_ps8_7_labeled → BCP_CMC_v1_mutual_mw01_ps8
    exp=$(echo "$d" | sed 's/^ACDC_//' | sed 's/_7_labeled$//')
    labelnum=$(echo "$d" | grep -oP '(\d+)(?=_labeled$)')

    run_test "$exp" "$labelnum" "$test_gpu"
done

echo "" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"
echo "  🧪 所有测试已完成!" | tee -a "$TEST_LOG"
echo "  时间: $(date)" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"

# =================== NA_v3 重跑 ===================

echo "" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"
echo "  🔄 重跑 NA_v3 (border_loss)" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"

wait_for_free_gpu 5000
na3_gpu=$?

echo "$(date) - 🚀 启动 NA_v3 on GPU ${na3_gpu}" | tee -a "$TEST_LOG"

CUDA_VISIBLE_DEVICES=$na3_gpu $PYTHON BCP_CMC_v1_NA_v3_border_loss.py \
    --exp BCP_CMC_NA_v3_bs12 \
    --cmc_border_loss_weight 1.5 \
    --cmc_patch_size 8 \
    --gpu 0 \
    --labelnum 7 \
    --pre_iterations 10000 \
    --max_iterations 30000 \
    --batch_size 12 \
    --labeled_bs 6 2>&1 | tee ./run_NA_v3_bs12.log

echo "$(date) - ✅ NA_v3 完成" | tee -a "$TEST_LOG"

# =================== 完成 ===================
echo "" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"
echo "  🎉 全部完成!" | tee -a "$TEST_LOG"
echo "  完成时间: $(date)" | tee -a "$TEST_LOG"
echo "============================================================" | tee -a "$TEST_LOG"
