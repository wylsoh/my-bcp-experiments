#!/bin/bash
# =============================================================================
# run_test.sh — 优化版测试运行与结果记录脚本 v2
#
# 修复问题:
#   1. test_ACDC.py 无 --gpu 参数 → 使用 CUDA_VISIBLE_DEVICES 环境变量
#   2. cd $CODE_DIR 在子 shell 中失效 → 改用 pushd/popd 确保工作目录正确
#   3. 路径依赖于从 code/ 目录运行 → 自动推导正确的工作目录
#   4. GPU 不可用时自动等待 → 集成等待机制
#   5. 输出解析脆弱 → 改进正则匹配
#
# 用法：
#   ./run_test.sh [--mem <MiB>] [--gpu <id>] --exp <exp_name> --labelnum <num> [其他参数...]
#
# 参数说明：
#   --exp <name>       实验名称（必需）
#   --labelnum <num>   标签数量（必需）
#   --model <name>     模型名称，默认 unet
#   --stage_name <s>   阶段名称，默认 self_train（可选 pre_train）
#   --gpu <id>         指定 GPU ID，不指定则自动选择空闲显存最多的 GPU
#   --mem <MiB>        最低空闲显存要求（配合自动选择使用），默认 4000
#   --root_path <path> 数据根路径，相对 code/ 目录，默认 ../data_split/ACDC
#   --num_classes <n>  分类数，默认 4
#   -b, --background   后台运行
#   -h, --help         显示帮助信息
#
# 示例：
#   ./run_test.sh --exp BCP_CMC_fusion_student --labelnum 7
#   ./run_test.sh --exp BCP_MAE --labelnum 7 --model unet --stage_name self_train
#   ./run_test.sh --exp BCP_MAE --labelnum 7 --gpu 0
#   ./run_test.sh --exp BCP_MAE --labelnum 7 --stage_name pre_train --model unet
# =============================================================================

set -o pipefail

# ---------- 配置 ----------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${PROJECT_DIR}/code"
CONDA_ENV="yll"
CONDA_BASE="${HOME}/anaconda3"
MIN_FREE_MEMORY=4000
GPUS=(0 1 2 3)
WAIT_INTERVAL=30       # 等待空闲 GPU 的轮询间隔（秒）

TEST_RESULTS_DIR="${PROJECT_DIR}/test_results"
MASTER_CSV="${TEST_RESULTS_DIR}/master_metrics.csv"

# ---------- 颜色定义 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------- 解析参数 ----------
BACKGROUND=false
GPU_SPECIFIED=false
GPU_ID=""
EXP=""
LABELNUM=""
MODEL="unet"
STAGE_NAME="self_train"
ROOT_PATH="../data_split/ACDC"
NUM_CLASSES=4
EXTRA_ARGS=()

show_help() {
    # 提取文件头部注释中的帮助信息（第1行到第1个空行之间的 # 行）
    sed -n '/^# /,/^$/p; /^$/q' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        --mem)
            MIN_FREE_MEMORY="$2"
            shift 2
            ;;
        --gpu)
            GPU_SPECIFIED=true
            GPU_ID="$2"
            shift 2
            ;;
        --exp)
            EXP="$2"
            shift 2
            ;;
        --labelnum)
            LABELNUM="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --stage_name)
            STAGE_NAME="$2"
            shift 2
            ;;
        --root_path)
            ROOT_PATH="$2"
            shift 2
            ;;
        --num_classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------- 参数校验 ----------
if [ -z "$EXP" ] || [ -z "$LABELNUM" ]; then
    echo -e "${RED}错误: --exp 和 --labelnum 是必需参数${NC}"
    echo ""
    echo "用法: ./run_test.sh --exp <exp_name> --labelnum <num> [其他参数]"
    echo "示例: ./run_test.sh --exp BCP_CMC_fusion_student --labelnum 7"
    exit 1
fi

# ---------- 日志 & 结果文件 ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$RUN_LOG_DIR"
mkdir -p "$TEST_RESULTS_DIR"

TEST_LOG="${RUN_LOG_DIR}/test_${EXP}_${LABELNUM}_${TIMESTAMP}.log"
RESULT_FILE="${TEST_RESULTS_DIR}/${EXP}_${LABELNUM}_${STAGE_NAME}_${TIMESTAMP}.txt"

# ---------- 工具函数 ----------

log() {
    local level="$1"
    shift
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
    echo "$msg" | tee -a "$TEST_LOG"
}

info()    { log "INFO"    "$@"; }
warn()    { log "WARN"    "$@"; }
error()   { log "ERROR"   "$@"; }
success() { log "SUCCESS" "$@"; }

echo_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           BCP 测试运行与结果记录脚本 v2                         ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  实验:        %-48s║\n" "${EXP}"
    printf "║  标签数:      %-48s║\n" "${LABELNUM}"
    printf "║  模型:        %-48s║\n" "${MODEL}"
    printf "║  阶段:        %-48s║\n" "${STAGE_NAME}"
    printf "║  时间:        %-48s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

# 激活 conda 环境
activate_conda() {
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        local conda_path
        conda_path=$(which conda 2>/dev/null)
        if [ -n "$conda_path" ]; then
            local conda_dir
            conda_dir="$(dirname "$(dirname "$conda_path")")"
            source "${conda_dir}/etc/profile.d/conda.sh" 2>/dev/null
        fi
    fi

    if ! conda activate ${CONDA_ENV} 2>/dev/null; then
        error "无法激活 conda 环境 '${CONDA_ENV}'"
        error "请检查 CONDA_BASE 路径或环境名"
        exit 1
    fi
    info "Conda 环境 '${CONDA_ENV}' 已激活"
}

# GPU 显存查询
get_gpu_memory_free() {
    local gpu_id="$1"
    local used total
    used=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    total=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
        echo $((total - used))
    else
        echo 0
    fi
}

get_gpu_memory_total() {
    local gpu_id="$1"
    nvidia-smi --id="$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# 获取所有 GPU 按空闲显存排序（只返回 >= MIN_FREE_MEMORY 的）
get_gpus_sorted_by_free_memory() {
    local gpu_list=()
    for gpu_id in "${GPUS[@]}"; do
        local free_mem
        free_mem=$(get_gpu_memory_free "$gpu_id")
        if [ "$free_mem" -ge "$MIN_FREE_MEMORY" ]; then
            gpu_list+=("$gpu_id:$free_mem")
        fi
    done

    # 冒泡排序降序
    for ((i = 0; i < ${#gpu_list[@]}; i++)); do
        for ((j = i + 1; j < ${#gpu_list[@]}; j++)); do
            local free_i="${gpu_list[i]#*:}"
            local free_j="${gpu_list[j]#*:}"
            if [ "$free_i" -lt "$free_j" ]; then
                local tmp="${gpu_list[i]}"
                gpu_list[i]="${gpu_list[j]}"
                gpu_list[j]="$tmp"
            fi
        done
    done
    echo "${gpu_list[@]}"
}

# 自动选择最优 GPU
select_best_gpu() {
    info "扫描 GPU 显存状态（最低需求: ${MIN_FREE_MEMORY} MiB）..."
    for gpu_id in "${GPUS[@]}"; do
        local free_mem
        free_mem=$(get_gpu_memory_free "$gpu_id")
        local total
        total=$(get_gpu_memory_total "$gpu_id")
        info "  GPU ${gpu_id}: 空闲 ${free_mem} MiB / 总 ${total} MiB"
    done

    local sorted_gpus
    sorted_gpus=($(get_gpus_sorted_by_free_memory))

    if [ ${#sorted_gpus[@]} -eq 0 ]; then
        error "没有 GPU 满足 ${MIN_FREE_MEMORY} MiB 空闲显存要求"
        # 兜底：使用空闲显存最多的 GPU
        local best_gpu=""
        local max_free=0
        for gpu_id in "${GPUS[@]}"; do
            local free_mem
            free_mem=$(get_gpu_memory_free "$gpu_id")
            if [ "$free_mem" -gt "$max_free" ]; then
                max_free="$free_mem"
                best_gpu="$gpu_id"
            fi
        done
        if [ -n "$best_gpu" ]; then
            warn "选择 GPU ${best_gpu}（空闲 ${max_free} MiB，低于推荐 ${MIN_FREE_MEMORY} MiB）"
            GPU_ID="$best_gpu"
            return 0
        fi
        error "无法找到任何可用的 GPU"
        return 1
    fi

    local best_entry="${sorted_gpus[0]}"
    GPU_ID="${best_entry%%:*}"
    local free_mem="${best_entry##*:}"
    info "选择 GPU ${GPU_ID}（空闲 ${free_mem} MiB，显存最充足）"
    return 0
}

# 等待可用 GPU
wait_for_gpu() {
    info "📡 所有 GPU 显存均不足 ${MIN_FREE_MEMORY} MiB，进入等待模式 ..."
    info "每 ${WAIT_INTERVAL} 秒检查一次 GPU 显存状态"

    while true; do
        info "检查 GPU 状态 ..."
        local sorted_gpus
        sorted_gpus=($(get_gpus_sorted_by_free_memory))
        if [ ${#sorted_gpus[@]} -gt 0 ]; then
            local best="${sorted_gpus[0]}"
            local best_id="${best%%:*}"
            local best_free="${best##*:}"
            info "✅ 发现 GPU ${best_id} 空闲显存 ${best_free} MiB"
            GPU_ID="$best_id"
            return 0
        fi
        info "⏳ 暂无 GPU 满足条件，${WAIT_INTERVAL} 秒后重试 ..."
        sleep "$WAIT_INTERVAL"
    done
}

# ====================================================================
# 核心改进：从 test_ACDC.py 的输出中解析性能指标
# test_ACDC.py 输出格式：
#   [array([dice, jc, hd95, asd]), array([...]), array([...])]
#   [avg_dice, avg_jc, avg_hd95, avg_asd]
# ====================================================================

parse_metrics_table() {
    local test_output="$1"
    local performance_file="$2"

    # 从 performance.txt 读取原始 metric line
    local metric_line=""
    if [ -f "$performance_file" ]; then
        metric_line=$(grep "metric is" "$performance_file")
    fi

    # 从 test output 提取平均值行
    local avg_line
    avg_line=$(echo "$test_output" | grep -oP '\[[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\]' | tail -1)

    echo ""
    echo "  ┌──────────┬──────────┬──────────┬──────────┬──────────┐"
    echo "  │  类别    │   Dice   │    JC    │   HD95   │   ASD    │"
    echo "  ├──────────┼──────────┼──────────┼──────────┼──────────┤"

    if [ -n "$metric_line" ]; then
        # 解析三个类别的指标
        for class_idx in 1 2 3; do
            local class_name=""
            case $class_idx in
                1) class_name="RV"  ;;
                2) class_name="MYO" ;;
                3) class_name="LV"  ;;
            esac

            # 提取第 class_idx 个 array(...) 的内容
            local arr
            arr=$(echo "$metric_line" | grep -oP 'array\(\[([0-9.,\s]+)\]\)' | sed -n "${class_idx}p")
            if [ -n "$arr" ]; then
                # 提取 4 个数值
                local vals=()
                while IFS= read -r v; do
                    vals+=("$v")
                done < <(echo "$arr" | grep -oP '[\d]+\.[\d]+')
                if [ ${#vals[@]} -ge 4 ]; then
                    printf "  │  %-6s │  %s  │  %s  │  %s  │  %s  │\n" \
                        "$class_name" \
                        "$(printf "%.4f" "${vals[0]}")" \
                        "$(printf "%.4f" "${vals[1]}")" \
                        "$(printf "%.4f" "${vals[2]}")" \
                        "$(printf "%.4f" "${vals[3]}")"
                fi
            fi
        done
    fi

    echo "  ├──────────┼──────────┼──────────┼──────────┼──────────┤"

    if [ -n "$avg_line" ]; then
        local avg_dice avg_jc avg_hd95 avg_asd
        avg_dice=$(echo "$avg_line" | awk '{print $1}')
        avg_jc=$(echo "$avg_line" | awk '{print $2}')
        avg_hd95=$(echo "$avg_line" | awk '{print $3}')
        avg_asd=$(echo "$avg_line" | awk '{print $4}')
        printf "  │  %-6s │  %.4f  │  %.4f  │  %.4f  │  %.4f  │\n" \
            "Avg" "$avg_dice" "$avg_jc" "$avg_hd95" "$avg_asd"
    fi

    echo "  └──────────┴──────────┴──────────┴──────────┴──────────┘"
    echo ""
}

# record_results — 记录结果到文件
record_results() {
    local exp_name="$1"
    local labelnum="$2"
    local model="$3"
    local stage="$4"
    local gpu_id="$5"
    local test_output="$6"
    local performance_file="$7"
    local train_final_dice="$8"
    local train_final_hd95="$9"

    # 清空并写入结果文件
    > "$RESULT_FILE"
    {
        echo "=================================================================================="
        echo "                         测试实验结果记录"
        echo "=================================================================================="
        echo ""
        echo "  实验名称:        ${exp_name}"
        echo "  模型:            ${model}"
        echo "  标签数量:        ${labelnum}"
        echo "  阶段:            ${stage}"
        echo "  使用的 GPU:      ${gpu_id}"
        echo "  测试时间:        $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  结果文件:        ${RESULT_FILE}"
        echo ""
        echo "----------------------------------------------------------------------------------"
        echo "  参数设置"
        echo "----------------------------------------------------------------------------------"
        echo "  --exp           ${exp_name}"
        echo "  --labelnum      ${labelnum}"
        echo "  --model         ${model}"
        echo "  --stage_name    ${stage}"
        echo "  --root_path     ${ROOT_PATH}"
        echo "  --num_classes   ${NUM_CLASSES}"
        for arg in "${EXTRA_ARGS[@]}"; do
            echo "  ${arg}"
        done
        echo ""

        # 训练过程的最终指标（如果存在）
        if [ -n "$train_final_dice" ]; then
            echo "----------------------------------------------------------------------------------"
            echo "  训练阶段最佳指标（来自 log.txt）"
            echo "----------------------------------------------------------------------------------"
            echo "  Best mean_dice : ${train_final_dice}"
            echo "  Best mean_hd95 : ${train_final_hd95}"
            echo ""
        fi

        echo "----------------------------------------------------------------------------------"
        echo "  测试指标结果"
        echo "----------------------------------------------------------------------------------"

        # 写入格式化的指标表格
        parse_metrics_table "$test_output" "$performance_file"

        echo "----------------------------------------------------------------------------------"
        echo "  测试脚本完整输出"
        echo "----------------------------------------------------------------------------------"
        echo "${test_output}"
        echo ""

        # 读取 performance.txt
        if [ -f "$performance_file" ]; then
            echo "----------------------------------------------------------------------------------"
            echo "  performance.txt 内容"
            echo "----------------------------------------------------------------------------------"
            cat "$performance_file"
            echo ""
        fi

        echo "=================================================================================="
        echo "  记录时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=================================================================================="
    } >> "$RESULT_FILE"

    success "测试结果已保存到: ${RESULT_FILE}"
}

# ====================================================================
# 提取数值化指标 — 从 performance_file 解析出 RV/MYO/LV/AVG 的 4 项指标
# 输出: 以空格分隔的 16 个数值 (rv_dice rv_jc rv_hd95 rv_asd myo_dice myo_jc myo_hd95 myo_asd
#                                   lv_dice lv_jc lv_hd95 lv_asd avg_dice avg_jc avg_hd95 avg_asd)
#       若解析失败对应位置输出空字符串
# ====================================================================
extract_metrics_values() {
    local performance_file="$1"

    # 默认 16 个空值
    local rv_dice="" rv_jc="" rv_hd95="" rv_asd=""
    local myo_dice="" myo_jc="" myo_hd95="" myo_asd=""
    local lv_dice="" lv_jc="" lv_hd95="" lv_asd=""
    local avg_dice="" avg_jc="" avg_hd95="" avg_asd=""

    if [ ! -f "$performance_file" ]; then
        echo "${rv_dice} ${rv_jc} ${rv_hd95} ${rv_asd} ${myo_dice} ${myo_jc} ${myo_hd95} ${myo_asd} ${lv_dice} ${lv_jc} ${lv_hd95} ${lv_asd} ${avg_dice} ${avg_jc} ${avg_hd95} ${avg_asd}"
        return
    fi

    local metric_line
    metric_line=$(grep "metric is" "$performance_file")

    local avg_line
    avg_line=$(grep "average metric is" "$performance_file")

    if [ -n "$metric_line" ]; then
        # 提取每个 array(...) 中的 4 个数值
        local idx=0
        for class_idx in 1 2 3; do
            local arr
            arr=$(echo "$metric_line" | grep -oP 'array\(\[([0-9.,\s]+)\]\)' | sed -n "${class_idx}p")
            if [ -n "$arr" ]; then
                local vals=()
                while IFS= read -r v; do
                    vals+=("$v")
                done < <(echo "$arr" | grep -oP '[\d]+\.[\d]+')

                if [ ${#vals[@]} -ge 4 ]; then
                    case $class_idx in
                        1) rv_dice="${vals[0]}"; rv_jc="${vals[1]}"; rv_hd95="${vals[2]}"; rv_asd="${vals[3]}" ;;
                        2) myo_dice="${vals[0]}"; myo_jc="${vals[1]}"; myo_hd95="${vals[2]}"; myo_asd="${vals[3]}" ;;
                        3) lv_dice="${vals[0]}"; lv_jc="${vals[1]}"; lv_hd95="${vals[2]}"; lv_asd="${vals[3]}" ;;
                    esac
                fi
            fi
            ((idx++))
        done
    fi

    if [ -n "$avg_line" ]; then
        # average metric is [0.89099157 0.80961849 3.62607378 1.0792405]
        avg_dice=$(echo "$avg_line" | grep -oP '\[([0-9.\s]+)\]' | tr -d '[]' | awk '{print $1}')
        avg_jc=$(echo "$avg_line"   | grep -oP '\[([0-9.\s]+)\]' | tr -d '[]' | awk '{print $2}')
        avg_hd95=$(echo "$avg_line" | grep -oP '\[([0-9.\s]+)\]' | tr -d '[]' | awk '{print $3}')
        avg_asd=$(echo "$avg_line"  | grep -oP '\[([0-9.\s]+)\]' | tr -d '[]' | awk '{print $4}')
    fi

    echo "${rv_dice} ${rv_jc} ${rv_hd95} ${rv_asd} ${myo_dice} ${myo_jc} ${myo_hd95} ${myo_asd} ${lv_dice} ${lv_jc} ${lv_hd95} ${lv_asd} ${avg_dice} ${avg_jc} ${avg_hd95} ${avg_asd}"
}

# ====================================================================
# 追加结果到主 CSV 汇总表 (master_metrics.csv)
# CSV 列:
#   exp,labelnum,model,stage,gpu_id,test_time,
#   train_best_dice,train_best_hd95,
#   RV_dice,RV_jc,RV_hd95,RV_asd,
#   MYO_dice,MYO_jc,MYO_hd95,MYO_asd,
#   LV_dice,LV_jc,LV_hd95,LV_asd,
#   avg_dice,avg_jc,avg_hd95,avg_asd
# ====================================================================
append_to_master_csv() {
    local exp_name="$1"
    local labelnum="$2"
    local model="$3"
    local stage="$4"
    local gpu_id="$5"
    local performance_file="$6"
    local train_best_dice="$7"
    local train_best_hd95="$8"

    local test_time
    test_time=$(date '+%Y-%m-%d %H:%M:%S')

    # 解析数值化指标
    local metrics_str
    metrics_str=$(extract_metrics_values "$performance_file")
    read -r rv_dice rv_jc rv_hd95 rv_asd \
            myo_dice myo_jc myo_hd95 myo_asd \
            lv_dice lv_jc lv_hd95 lv_asd \
            avg_dice avg_jc avg_hd95 avg_asd <<< "$metrics_str"

    local csv_header="exp,labelnum,model,stage,gpu_id,test_time,train_best_dice,train_best_hd95,RV_dice,RV_jc,RV_hd95,RV_asd,MYO_dice,MYO_jc,MYO_hd95,MYO_asd,LV_dice,LV_jc,LV_hd95,LV_asd,avg_dice,avg_jc,avg_hd95,avg_asd"

    local csv_row="${exp_name},${labelnum},${model},${stage},${gpu_id},${test_time},${train_best_dice},${train_best_hd95},${rv_dice},${rv_jc},${rv_hd95},${rv_asd},${myo_dice},${myo_jc},${myo_hd95},${myo_asd},${lv_dice},${lv_jc},${lv_hd95},${lv_asd},${avg_dice},${avg_jc},${avg_hd95},${avg_asd}"

    # 如果文件不存在，先写入表头
    if [ ! -f "$MASTER_CSV" ]; then
        echo "$csv_header" > "$MASTER_CSV"
        info "创建主 CSV 文件: ${MASTER_CSV}"
    fi

    # 追加数据行
    echo "$csv_row" >> "$MASTER_CSV"
    info "结果已追加到主 CSV: ${MASTER_CSV}"
}

# ====================================================================
# 自动生成 SUMMARY.md — 从 master_metrics.csv 渲染 Markdown 汇总表
# ====================================================================
generate_summary_md() {
    local summary_file="${TEST_RESULTS_DIR}/SUMMARY.md"

    if [ ! -f "$MASTER_CSV" ]; then
        warn "master_metrics.csv 不存在，跳过 SUMMARY.md 生成"
        return
    fi

    {
        echo "# BCP 测试结果汇总表"
        echo ""
        echo "> 自动生成于: $(date '+%Y-%m-%d %H:%M')"
        echo "> 数据来源: [\`master_metrics.csv\`](master_metrics.csv)"
        echo ""
        echo "| # | 实验 | 标签 | 模型 | 阶段 | GPU | 时间 | 训练 Dice | 训练 HD95 | RV Dice | RV HD95 | MYO Dice | MYO HD95 | LV Dice | LV HD95 | Avg Dice | Avg HD95 |"
        echo "|---|------|------|------|------|-----|------|-----------|-----------|---------|---------|----------|----------|---------|---------|----------|----------|"

        local row_num=0
        # 跳过表头行 (tail -n +2)
        tail -n +2 "$MASTER_CSV" | while IFS=',' read -r exp labelnum model stage gpu_id test_time \
            train_dice train_hd95 \
            rv_dice rv_jc rv_hd95 rv_asd \
            myo_dice myo_jc myo_hd95 myo_asd \
            lv_dice lv_jc lv_hd95 lv_asd \
            avg_dice avg_jc avg_hd95 avg_asd; do
            row_num=$((row_num + 1))

            # 提取时间中的 MM-DD HH:MM 部分
            local short_time
            short_time=$(echo "$test_time" | grep -oP '\d{2}-\d{2} \d{2}:\d{2}' || echo "$test_time" | grep -oP '\d{2}:\d{2}:\d{2}' || echo "$test_time")

            printf "| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.4f | %.4f |\n" \
                "$row_num" \
                "$exp" "$labelnum" "$model" "$stage" "$gpu_id" "$short_time" \
                "$train_dice" "$train_hd95" \
                "$(printf "%.4f" "$rv_dice")" "$(printf "%.4f" "$rv_hd95")" \
                "$(printf "%.4f" "$myo_dice")" "$(printf "%.4f" "$myo_hd95")" \
                "$(printf "%.4f" "$lv_dice")" "$(printf "%.4f" "$lv_hd95")" \
                "$avg_dice" "$avg_hd95"
        done

        echo ""
        echo "---"
        echo ""
        echo "*上次更新: $(date '+%Y-%m-%d %H:%M:%S')*"
    } > "$summary_file"

    info "SUMMARY.md 已生成: ${summary_file}"
}

# ====================================================================
# 从训练日志中提取最佳 Dice/HD95
# ====================================================================
extract_best_from_log() {
    local exp_name="$1"
    local labelnum="$2"
    local stage="$3"

    local log_file="${CODE_DIR}/model/BCP/ACDC_${exp_name}_${labelnum}_labeled/${stage}/log.txt"
    if [ ! -f "$log_file" ]; then
        echo ""
        return
    fi

    # 查找 "Best mean_dice : X.XXXX" 和 "Best mean_hd95 : XX.XX" 模式
    local best_dice best_hd95
    best_dice=$(grep -oP 'Best mean_dice\s*:\s*[\d.]+' "$log_file" | tail -1 | grep -oP '[\d.]+$')
    best_hd95=$(grep -oP 'Best mean_hd95\s*:\s*[\d.]+' "$log_file" | tail -1 | grep -oP '[\d.]+$')

    # 也查找 "BEST | dice=X.XXXX, hd95=XX.XX" 模式
    if [ -z "$best_dice" ]; then
        local best_line
        best_line=$(grep "BEST |" "$log_file" | tail -1)
        if [ -n "$best_line" ]; then
            best_dice=$(echo "$best_line" | grep -oP '(?<=dice=)[\d.]+')
            best_hd95=$(echo "$best_line" | grep -oP '(?<=hd95=)[\d.]+')
        fi
    fi

    echo "${best_dice:-N/A} ${best_hd95:-N/A}"
}

# ====================================================================
# 核心改进：固定路径问题
#
# 问题根源：
#   test_ACDC.py 中的路径（如 ./model/BCP/...）是相对于运行目录的。
#   它期望从 code/ 目录运行，因为模型和数据拆分路径都是相对于 code/ 的。
#
# 修复方案：
#   在执行 python 之前，切换到 CODE_DIR，确保所有相对路径解析正确。
#   使用 pushd/popd 确保目录切换安全，避免在子 shell 中 cd 失效。
# ====================================================================

run_test() {
    local gpu_id="$1"

    # 构建脚本参数（不含 --gpu，test_ACDC.py 不支持）
    local script_args=""
    script_args+="--exp ${EXP} "
    script_args+="--labelnum ${LABELNUM} "
    script_args+="--model ${MODEL} "
    script_args+="--stage_name ${STAGE_NAME} "
    script_args+="--root_path ${ROOT_PATH} "
    script_args+="--num_classes ${NUM_CLASSES} "
    for arg in "${EXTRA_ARGS[@]}"; do
        script_args+="${arg} "
    done

    info "──────────────────────────────────────────"
    info "运行测试: CUDA_VISIBLE_DEVICES=${gpu_id} python test_ACDC.py ${script_args}"
    info "工作目录: ${CODE_DIR}"
    info "──────────────────────────────────────────"

    # 记录开始标记
    {
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║  测试开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "║  GPU: ${gpu_id} (CUDA_VISIBLE_DEVICES)"
        echo "║  工作目录: ${CODE_DIR}"
        echo "║  命令: python test_ACDC.py ${script_args}"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
    } >> "$TEST_LOG"

    # 关键修复：使用 pushd/popd 确保在 code/ 目录下执行
    # 这样 test_ACDC.py 中的 ./model/ 和 ../data_split/ 路径都能正确解析
    pushd "$CODE_DIR" > /dev/null

    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    local test_output
    test_output=$(python test_ACDC.py ${script_args} 2>&1)
    local exit_code=$?

    popd > /dev/null

    # 写入日志
    echo "$test_output" >> "$TEST_LOG"
    echo "" >> "$TEST_LOG"
    echo "Exit code: $exit_code" >> "$TEST_LOG"

    echo "$test_output"
    return $exit_code
}

# ---------- 主流程 ----------

main() {
    echo_banner

    # 1. 激活 conda 环境
    activate_conda

    # 2. 选择 GPU
    if [ "$GPU_SPECIFIED" = true ]; then
        info "使用指定 GPU: ${GPU_ID}"
    else
        if ! select_best_gpu; then
            warn "初始没有满足条件的 GPU，进入等待模式..."
            wait_for_gpu
        fi
    fi

    # 3. 验证模型路径是否存在
    MODEL_DIR="${CODE_DIR}/model/BCP/ACDC_${EXP}_${LABELNUM}_labeled/${STAGE_NAME}"
    MODEL_FILE="${MODEL_DIR}/${MODEL}_best_model.pth"

    info "检查模型路径: ${MODEL_FILE}"
    if [ ! -f "$MODEL_FILE" ]; then
        warn "模型文件不存在: ${MODEL_FILE}"
        warn "尝试搜索可能的文件 ..."
        local found
        found=$(find "${CODE_DIR}/model/BCP/ACDC_${EXP}_${LABELNUM}_labeled" -name "*best_model*" 2>/dev/null | head -5)
        if [ -n "$found" ]; then
            info "找到以下模型文件:"
            echo "$found" | while read -r f; do echo "  - $f"; done
        else
            error "未找到任何 best_model 文件，请检查:"
            error "  ${CODE_DIR}/model/BCP/ACDC_${EXP}_${LABELNUM}_labeled/"
            ls -d "${CODE_DIR}"/model/BCP/*/ 2>/dev/null | while read dir; do
                echo "  - $dir"
            done
            exit 1
        fi
    else
        info "模型文件存在: ${MODEL_FILE}"
    fi

    # 4. 提取训练最佳指标
    local train_metrics
    train_metrics=$(extract_best_from_log "$EXP" "$LABELNUM" "$STAGE_NAME")
    local train_best_dice="${train_metrics%% *}"
    local train_best_hd95="${train_metrics##* }"
    if [ "$train_best_dice" != "N/A" ]; then
        info "训练最佳指标: dice=${train_best_dice}, hd95=${train_best_hd95}"
    fi

    # 5. 运行测试
    info "开始测试 ..."
    echo ""
    local test_output
    test_output=$(run_test "$GPU_ID")
    local exit_code=$?

    # 显示测试输出
    echo ""
    echo "──────────────────────────────────────────"
    echo "测试脚本输出:"
    echo "──────────────────────────────────────────"
    echo "$test_output"
    echo "──────────────────────────────────────────"
    echo ""

    # 6. 检测是否成功完成
    if [ $exit_code -eq 0 ]; then
        success "测试成功完成！"

        # 7. 定位 performance.txt
        PERFORMANCE_FILE="${CODE_DIR}/model/BCP/ACDC_${EXP}_${LABELNUM}_labeled/performance.txt"

        if [ ! -f "$PERFORMANCE_FILE" ]; then
            warn "performance.txt 未在预期路径找到: ${PERFORMANCE_FILE}"
            local found
            found=$(find "${CODE_DIR}/model/BCP/ACDC_${EXP}_${LABELNUM}_labeled" -maxdepth 2 -name "performance.txt" 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                PERFORMANCE_FILE="$found"
                info "在 ${PERFORMANCE_FILE} 找到 performance.txt"
            else
                PERFORMANCE_FILE=""
                warn "performance.txt 未找到，将仅从测试输出中提取指标"
            fi
        else
            info "performance.txt 路径: ${PERFORMANCE_FILE}"
        fi

        # 8. 显示指标表格到终端
        echo -e "${GREEN}${BOLD}性能指标:${NC}"
        parse_metrics_table "$test_output" "$PERFORMANCE_FILE"

        # 9. 记录结果
        record_results "$EXP" "$LABELNUM" "$MODEL" "$STAGE_NAME" "$GPU_ID" \
            "$test_output" "$PERFORMANCE_FILE" "$train_best_dice" "$train_best_hd95"

        # 10. 追加到主 CSV 汇总表
        if [ -n "$PERFORMANCE_FILE" ]; then
            append_to_master_csv "$EXP" "$LABELNUM" "$MODEL" "$STAGE_NAME" "$GPU_ID" \
                "$PERFORMANCE_FILE" "$train_best_dice" "$train_best_hd95"
        else
            warn "performance.txt 不存在，跳过 CSV 汇总记录"
        fi

        # 11. 更新 SUMMARY.md
        generate_summary_md

        echo ""
        echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}${BOLD}  测试完成！结果已保存${NC}"
        echo -e "${GREEN}${BOLD}  结果文件: ${RESULT_FILE}${NC}"
        echo -e "${GREEN}${BOLD}  日志文件: ${TEST_LOG}${NC}"
        echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════════${NC}"
        echo ""

        return 0
    else
        error "测试失败 (exit code: $exit_code)"
        info "日志文件: ${TEST_LOG}"

        # 记录失败信息
        > "$RESULT_FILE"
        {
            echo "=================================================================================="
            echo "                         测试实验结果记录（失败）"
            echo "=================================================================================="
            echo ""
            echo "  实验名称:        ${EXP}"
            echo "  模型:            ${MODEL}"
            echo "  标签数量:        ${LABELNUM}"
            echo "  阶段:            ${STAGE_NAME}"
            echo "  使用的 GPU:      ${GPU_ID}"
            echo "  测试时间:        $(date '+%Y-%m-%d %H:%M:%S')"
            echo "  状态:            失败 (exit code: $exit_code)"
            echo ""
            echo "----------------------------------------------------------------------------------"
            echo "  参数设置"
            echo "----------------------------------------------------------------------------------"
            echo "  --exp           ${EXP}"
            echo "  --labelnum      ${LABELNUM}"
            echo "  --model         ${MODEL}"
            echo "  --stage_name    ${STAGE_NAME}"
            echo ""
            echo "----------------------------------------------------------------------------------"
            echo "  测试脚本输出"
            echo "----------------------------------------------------------------------------------"
            echo "${test_output}"
            echo ""
            echo "=================================================================================="
            echo "  记录时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=================================================================================="
        } >> "$RESULT_FILE"

        warn "失败记录已保存到: ${RESULT_FILE}"

        # 提供常见错误的解决建议
        if echo "$test_output" | grep -qi "No module named"; then
            echo -e "${YELLOW}💡 提示: Python 模块缺失，请检查 conda 环境是否正确${NC}"
            echo -e "${YELLOW}   运行: conda activate ${CONDA_ENV}${NC}"
        fi
        if echo "$test_output" | grep -qi "FileNotFoundError"; then
            echo -e "${YELLOW}💡 提示: 文件路径错误，请检查:${NC}"
            echo -e "${YELLOW}   - root_path (当前: ${ROOT_PATH})${NC}"
            echo -e "${YELLOW}   - 模型文件路径 (当前: ${MODEL_FILE})${NC}"
            echo -e "${YELLOW}   - 工作目录是否为 ${CODE_DIR}${NC}"
        fi
        if echo "$test_output" | grep -qi "CUDA.*out of memory"; then
            echo -e "${YELLOW}💡 提示: GPU ${GPU_ID} 显存不足，可尝试:${NC}"
            echo -e "${YELLOW}   - 指定更大显存的 GPU: --gpu <id>${NC}"
            echo -e "${YELLOW}   - 降低 --mem 要求: --mem 2000${NC}"
        fi

        return 1
    fi
}

# ---------- 后台模式 ----------
if [ "$BACKGROUND" = true ]; then
    nohup bash -c "$(declare -f main activate_conda run_test record_results parse_metrics_table extract_best_from_log \
        extract_metrics_values append_to_master_csv generate_summary_md \
        select_best_gpu wait_for_gpu get_gpu_memory_free get_gpu_memory_total \
        get_gpus_sorted_by_free_memory echo_banner log info warn error success); \
        GPU_SPECIFIED=${GPU_SPECIFIED}; GPU_ID='${GPU_ID}'; \
        EXP='${EXP}'; LABELNUM='${LABELNUM}'; MODEL='${MODEL}'; \
        STAGE_NAME='${STAGE_NAME}'; ROOT_PATH='${ROOT_PATH}'; \
        NUM_CLASSES=${NUM_CLASSES}; MIN_FREE_MEMORY=${MIN_FREE_MEMORY}; \
        CODE_DIR='${CODE_DIR}'; PROJECT_DIR='${PROJECT_DIR}'; \
        CONDA_ENV='${CONDA_ENV}'; CONDA_BASE='${CONDA_BASE}'; \
        TEST_LOG='${TEST_LOG}'; RESULT_FILE='${RESULT_FILE}'; \
        TEST_RESULTS_DIR='${TEST_RESULTS_DIR}'; MASTER_CSV='${MASTER_CSV}'; \
        GPUS=(${GPUS[*]}); EXTRA_ARGS=(${EXTRA_ARGS[@]}); \
        main" >> "$TEST_LOG" 2>&1 &
    local PID=$!
    echo ""
    echo "=============================================="
    echo "测试已在后台启动 (PID: ${PID})"
    echo "日志文件: ${TEST_LOG}"
    echo "结果将保存到: ${RESULT_FILE}"
    echo "  查看实时日志: tail -f ${TEST_LOG}"
    echo "  查看 GPU 状态: gpustat"
    echo "  终止测试:     kill ${PID}"
    echo "=============================================="
else
    main
fi
