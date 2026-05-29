#!/bin/bash
# =============================================================================
# run_exp.sh — 智能实验运行脚本 v3.0
# 功能：
#   1. 自动激活 conda ssl 环境
#   2. 自动选择空闲显存最充足的 GPU 运行实验
#   3. 遇 CUDA OOM 自动切换到空闲显存次多的 GPU
#   4. 所有 GPU 显存均不足时，使用 gpustat 监控等待
#   5. 尊重他人进程，只利用空闲显存，绝不切断正在运行的任务
#   6. 支持前台运行和后台 nohup 运行
#   7. 所有运行日志统一输出到 logs/ 目录，命名规则：{实验名}_{时间戳}.log
#   8. ★ 新增 DDP 模式：自动检测 --ddp 参数并启动 torchrun
#
# 用法：
#   ./run_exp.sh <script.py> [args...]
#   ./run_exp.sh --mem 12000 <script.py> [args...]  # 指定最低空闲显存 (MiB)
#   ./run_exp.sh -b <script.py> [args...]           # 后台运行 (nohup)
#   ./run_exp.sh --torchrun --nproc 4 <script.py> [args...]  # DDP 多卡
#
# 示例：
#   # 单 GPU:
#   ./run_exp.sh code/ACDC_BCP_MAE_train.py --exp my_exp --labelnum 7
#   # DDP 4 GPU + AMP:
#   ./run_exp.sh --torchrun --nproc 4 code/BCP_CMC_fusion_ddp.py \
#       --batch_size 96 --labeled_bs 24 --amp --ddp
#   # 后台 DDP:
#   ./run_exp.sh -b --torchrun --nproc 4 code/ACDC_BCP_MAE_train_ddp.py \
#       --batch_size 96 --labeled_bs 24 --amp --ddp
# =============================================================================

set -o pipefail

# ---------- 配置 ----------
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${PROJECT_DIR}/code"
CONDA_ENV="ssl"
CONDA_BASE="${HOME}/miniconda3"

# GPU 列表（所有可用 GPU）
GPUS=(0 1 2 3)
# 最低空闲显存要求（MiB），脚本会根据实验自动估算，也可由 --mem 参数指定
MIN_FREE_MEMORY=8000
WAIT_INTERVAL=30             # 等待空闲 GPU 时的轮询间隔（秒）
OOM_CHECK_INTERVAL=10        # OOM 重试间隔（秒）

# DDP 模式标志
DDP_MODE=false
DDP_NPROC=4                  # torchrun --nproc_per_node

# ---------- 检测 DDP 模式 & 后台模式 & --mem 参数 ----------
BACKGROUND=false
DDP_MODE=false
DDP_NPROC=4

# 必须解析 --torchrun 和 --nproc 参数（在脚本名之前）
while true; do
    if [ "$1" = "--mem" ] || [ "$1" = "-m" ]; then
        MIN_FREE_MEMORY="$2"
        shift 2
    elif [ "$1" = "-b" ] || [ "$1" = "--background" ]; then
        BACKGROUND=true
        shift
    elif [ "$1" = "--torchrun" ] || [ "$1" = "-t" ]; then
        DDP_MODE=true
        shift
    elif [ "$1" = "--nproc" ] || [ "$1" = "-n" ]; then
        DDP_NPROC="$2"
        shift 2
    else
        break
    fi
done

# 兼容旧版 -b 后还可能跟 --mem
if [ "$1" = "--mem" ] || [ "$1" = "-m" ]; then
    MIN_FREE_MEMORY="$2"
    shift 2
fi

# ---------- 检查参数 ----------
if [ $# -lt 1 ]; then
    echo "用法: $0 [--mem <min_free_mib>] [-b] <script.py> [args...]"
    echo "  --mem <n>   最低空闲显存要求 (MiB)，默认 8000"
    echo "  -b          后台运行 (nohup)"
    echo "  script.py   要运行的 Python 脚本 (相对于 code/ 目录)"
    echo "  args...     传递给脚本的参数 (不要包含 --gpu，脚本会自动管理)"
    exit 1
fi

SCRIPT_PATH="$1"
shift
USER_ARGS="$@"

# ---------- 解析脚本路径 ----------
# 如果传的是相对路径如 code/xxx.py 或 直接 xxx.py
if [[ "$SCRIPT_PATH" == code/* ]]; then
    SCRIPT_REL="${SCRIPT_PATH#code/}"
    SCRIPT_ABS="${CODE_DIR}/${SCRIPT_REL}"
elif [[ "$SCRIPT_PATH" == /* ]]; then
    SCRIPT_ABS="$SCRIPT_PATH"
    SCRIPT_REL="$(basename "$SCRIPT_PATH")"
else
    SCRIPT_REL="$SCRIPT_PATH"
    SCRIPT_ABS="${CODE_DIR}/${SCRIPT_PATH}"
fi

if [ ! -f "$SCRIPT_ABS" ]; then
    echo "[错误] 脚本文件不存在: $SCRIPT_ABS"
    exit 1
fi

# ---------- 日志文件（统一输出到 logs/ 目录）----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 尝试从 --exp 参数提取实验名，提取不到则从脚本名推导
EXPERIMENT_NAME=$(echo "$USER_ARGS" | grep -oP '(?<=--exp\s)\S+')
if [ -z "$EXPERIMENT_NAME" ]; then
    # 从脚本文件名推导：去掉 .py 后缀，保留有意义的部分
    EXPERIMENT_NAME=$(basename "${SCRIPT_REL}" .py | sed 's/_train//; s/^BCP_//; s/^ACDC_//')
fi
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"

# ---------- 工具函数 ----------

log() {
    local level="$1"
    shift
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

info()    { log "INFO"    "$@"; }
warn()    { log "WARN"    "$@"; }
error()   { log "ERROR"   "$@"; }
success() { log "SUCCESS" "$@"; }

# 激活 conda 环境
activate_conda() {
    # 多种方式尝试激活 conda
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        # 尝试从 PATH 定位 conda
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

# 检查 CUDA 可用性
check_cuda() {
    python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"
}

# ---------- GPU 内存查询（基于 nvidia-smi，稳定可靠） ----------

get_gpu_memory_used() {
    local gpu_id="$1"
    nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

get_gpu_memory_total() {
    local gpu_id="$1"
    nvidia-smi --id="$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

get_gpu_memory_free() {
    local gpu_id="$1"
    local used total
    used=$(get_gpu_memory_used "$gpu_id")
    total=$(get_gpu_memory_total "$gpu_id")
    if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
        echo $((total - used))
    else
        echo 0
    fi
}

# 获取 GPU 上的进程列表（仅用于信息展示，不用于决策）
get_gpu_processes() {
    local gpu_id="$1"
    nvidia-smi --id="$gpu_id" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
}

# ---------- GPU 排序选择逻辑 ----------

# 获取所有 GPU 按空闲显存从大到小排序（只返回显存足够 >= MIN_FREE_MEMORY 的 GPU）
# 返回格式: "gpu_id:free_mem" 列表，按 free_mem 降序
get_gpus_sorted_by_free_memory() {
    local gpu_list=()
    for gpu_id in "${GPUS[@]}"; do
        local free_mem
        free_mem=$(get_gpu_memory_free "$gpu_id")
        info "  GPU ${gpu_id}: 空闲显存 ${free_mem} MiB / 总显存 $(get_gpu_memory_total "$gpu_id") MiB"
        if [ "$free_mem" -ge "$MIN_FREE_MEMORY" ]; then
            gpu_list+=("$gpu_id:$free_mem")
        fi
    done

    # 按空闲显存降序排序（冒泡）
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

# 显示 GPU 状态概览
show_gpu_status() {
    info "当前 GPU 使用状态 (最低空闲需求: ${MIN_FREE_MEMORY} MiB):"
    echo "  GPU  |  已用 / 总显存  |  空闲  |  进程" | tee -a "$LOG_FILE"
    echo "  ─────┼─────────────────┼────────┼──────" | tee -a "$LOG_FILE"
    for gpu_id in "${GPUS[@]}"; do
        local used total free
        used=$(get_gpu_memory_used "$gpu_id")
        total=$(get_gpu_memory_total "$gpu_id")
        free=$((total - used))
        local pids
        pids=$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ',' | sed 's/,$//')
        [ -z "$pids" ] && pids="(无)"
        local enough="✓"
        [ "$free" -lt "$MIN_FREE_MEMORY" ] && enough="✗"
        printf "  %-4s | %7s / %-7s | %5s %s | %s\n" \
            "GPU $gpu_id" "${used}MiB" "${total}MiB" "${free}MiB" "$enough" "$pids" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
}

# 检测 CUDA Out of Memory 错误
is_oom_error() {
    local exit_code="$1"
    local output="$2"

    # OOM 常见特征
    # 1. exit code 1 (Python 异常退出)
    # 2. 输出中包含 "out of memory" 或 "OUT_OF_MEMORY" 或 "OOM"
    # 3. 有时 exit code 是 134 (SIGABRT)

    if echo "$output" | grep -qiE "(out of memory|OUT_OF_MEMORY|OOM|CUDA error|CUDA out of memory)"; then
        return 0  # 是 OOM
    fi

    # exit code 134 通常是 OOM 导致的 SIGABRT
    if [ "$exit_code" -eq 134 ]; then
        return 0
    fi

    return 1  # 不是 OOM
}

run_on_gpu() {
    local gpu_id="$1"
    shift
    local gpu_args="$@"

    # ---------- DDP 模式：使用 torchrun 启动，需要至少 2 个 GPU ----------
    if [ "$DDP_MODE" = true ]; then
        # DDP 模式需要 DDP_NPROC 张 GPU
        # 检查是否有足够的连续 GPU 从 gpu_id 开始
        local available_gpus=""
        for ((i=0; i<DDP_NPROC; i++)); do
            local target_gpu=$((gpu_id + i))
            local free_mem=$(get_gpu_memory_free "$target_gpu")
            if [ "$free_mem" -lt "$MIN_FREE_MEMORY" ]; then
                warn "DDP: GPU ${target_gpu} 空闲 ${free_mem} MiB 不足 ${MIN_FREE_MEMORY} MiB"
                return 1  # 这组 GPU 不行
            fi
            available_gpus="${available_gpus}${target_gpu},"
        done
        available_gpus="${available_gpus%,}"

        # 去除用户参数中可能自带的 --gpu（脚本统一管理）
        local clean_args
        clean_args=$(echo " ${USER_ARGS} " | sed -E 's/ --gpu\s+\S+ / /g')
        # 也去除 --ddp（torchrun 本身已隐含 DDP）
        clean_args=$(echo " ${clean_args} " | sed -E 's/ --ddp / /g')

        info "──────────────────────────────────────────"
        info "🚀 DDP 模式: ${DDP_NPROC} GPU (IDs: ${available_gpus})"
        info "命令: torchrun --nproc_per_node=${DDP_NPROC} \\"
        info "          ${SCRIPT_REL} ${clean_args} --amp"
        info "──────────────────────────────────────────"

        export CUDA_VISIBLE_DEVICES="${available_gpus}"

        cd "$CODE_DIR"
        {
            echo ""
            echo "╔══════════════════════════════════════════════════════╗"
            echo "║  [DDP] 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "║  [DDP] GPU: ${available_gpus} (${DDP_NPROC} cards)"
            echo "║  [DDP] 命令: torchrun --nproc_per_node=${DDP_NPROC}"
            echo "║         ${SCRIPT_REL} ${clean_args} --amp"
            echo "╚══════════════════════════════════════════════════════╝"
            echo ""
        } >> "$LOG_FILE"

        torchrun --nproc_per_node=${DDP_NPROC} \
            "$SCRIPT_REL" ${clean_args} --amp >> "$LOG_FILE" 2>&1
        local exit_code=$?
        cd "$PROJECT_DIR"

        if [ $exit_code -eq 0 ]; then
            success "✅ [DDP] 实验成功完成！ (${DDP_NPROC} GPU)"
            return 0
        fi

        if is_oom_error "$exit_code" "$(tail -30 "$LOG_FILE")"; then
            warn "⚠️  [DDP] OOM on GPU set {${available_gpus}}"
            return 1
        else
            error "❌ [DDP] 异常退出 (exit: $exit_code)"
            tail -30 "$LOG_FILE" | while IFS= read -r line; do echo "  $line" | tee -a "$LOG_FILE"; done
            return 2
        fi
    fi

    # ---------- 单 GPU 模式 ----------

    # 去除用户参数中可能自带的 --gpu（由脚本统一管理）
    local clean_args
    clean_args=$(echo " ${USER_ARGS} " | sed -E 's/ --gpu\s+\S+ / /g')

    info "──────────────────────────────────────────"
    info "运行 GPU ${gpu_id}（空闲显存: $(get_gpu_memory_free "$gpu_id") MiB）"
    info "命令: python ${SCRIPT_REL} --gpu ${gpu_id} ${clean_args}"
    info "──────────────────────────────────────────"

    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    # 执行并同时捕获 stdout+stderr
    cd "$CODE_DIR"

    # 先记录启动信息
    {
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "║  GPU: ${gpu_id} (空闲: $(get_gpu_memory_free "$gpu_id") MiB)"
        echo "║  命令: python ${SCRIPT_REL} --gpu ${gpu_id} ${clean_args}"
        echo "╚══════════════════════════════════════════════════════╝"
        echo ""
    } >> "$LOG_FILE"

    python "$SCRIPT_REL" --gpu "${gpu_id}" ${clean_args} >> "$LOG_FILE" 2>&1
    local exit_code=$?
    cd "$PROJECT_DIR"

    if [ $exit_code -eq 0 ]; then
        success "✅ 实验成功完成！ (GPU ${gpu_id})"
        return 0
    fi

    # 检查是否为 OOM
    if is_oom_error "$exit_code" "$(tail -30 "$LOG_FILE")"; then
        warn "⚠️  GPU ${gpu_id} 遇到 CUDA OOM！尝试更优 GPU ..."

        # 记录 OOM 时该 GPU 的显存状态到日志
        {
            echo "--- OOM 时 GPU ${gpu_id} 状态 ---"
            nvidia-smi --id="$gpu_id" --query-gpu=memory.used,memory.total --format=csv,noheader
            echo "--- 进程列表 ---"
            nvidia-smi --id="$gpu_id" --query-compute-apps=pid,used_memory --format=csv,noheader
            echo "-------------------------------"
        } >> "$LOG_FILE"

        return 1  # OOM，尝试下一张
    else
        error "❌ 脚本异常退出 (exit code: $exit_code)，非 OOM 错误，停止尝试"
        error "最后 30 行日志:"
        tail -30 "$LOG_FILE" | while IFS= read -r line; do echo "  $line" | tee -a "$LOG_FILE"; done
        return 2  # 非 OOM 错误，停止
    fi
}

wait_for_available_gpu() {
    info "📡 所有 GPU 显存均不足 ${MIN_FREE_MEMORY} MiB，进入等待模式 ..."
    info "每 ${WAIT_INTERVAL} 秒检查一次 GPU 显存状态"

    while true; do
        info "检查 GPU 状态 ..."
        show_gpu_status

        local sorted_gpus
        sorted_gpus=($(get_gpus_sorted_by_free_memory))
        if [ ${#sorted_gpus[@]} -gt 0 ]; then
            local best="${sorted_gpus[0]}"
            local best_id="${best%%:*}"
            local best_free="${best##*:}"
            info "✅ 发现 GPU ${best_id} 空闲显存 ${best_free} MiB >= ${MIN_FREE_MEMORY} MiB，准备运行！"
            return 0
        fi

        info "⏳ 暂无 GPU 满足条件，${WAIT_INTERVAL} 秒后重试 ..."
        sleep "$WAIT_INTERVAL"
    done
}

# ---------- 主流程 ----------

main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           BCP 智能实验运行脚本 v2.1                             ║"
    echo "║  按空闲显存排序，自动选择最优 GPU                                ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  脚本:          ${SCRIPT_REL}"
    echo "║  参数:          ${USER_ARGS}"
    echo "║  最低空闲显存:   ${MIN_FREE_MEMORY} MiB"
    echo "║  日志:          ${LOG_FILE}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    # 1. 激活 conda 环境
    activate_conda

    # 2. 检查 CUDA
    if ! check_cuda; then
        error "CUDA 不可用，请检查 PyTorch 安装"
        exit 1
    fi
    info "CUDA 可用，PyTorch 版本: $(python3 -c 'import torch; print(torch.__version__)')"

    # 3. 初始 GPU 状态
    info "扫描 GPU 显存状态（最低需求: ${MIN_FREE_MEMORY} MiB）..."
    show_gpu_status

    # 4. 主循环：按空闲显存排序，从最优 GPU 开始尝试
    local total_rounds=4
    local round=0
    local best_gpu_attempted=""

    while [ $round -lt $total_rounds ]; do
        round=$((round + 1))
        info "══════════ 第 ${round}/${total_rounds} 轮 ══════════"

        # 获取按空闲显存降序排列的 GPU 列表
        local sorted_gpus
        sorted_gpus=($(get_gpus_sorted_by_free_memory))

        if [ ${#sorted_gpus[@]} -eq 0 ]; then
            warn "当前没有 GPU 满足 ${MIN_FREE_MEMORY} MiB 空闲显存要求"
            info "GPU 详细状态:"
            show_gpu_status
        else
            info "GPU 可用性排序（空闲显存从高到低）:"
            for entry in "${sorted_gpus[@]}"; do
                local g="${entry%%:*}"
                local f="${entry##*:}"
                info "  🏆 GPU ${g} — 空闲 ${f} MiB"
            done

            # 按排序依次尝试
            for entry in "${sorted_gpus[@]}"; do
                local gpu_id="${entry%%:*}"

                # 跳过已尝试过的 GPU（本轮避免重复）
                if [[ "$best_gpu_attempted" == *"$gpu_id"* ]]; then
                    continue
                fi

                run_on_gpu "$gpu_id"
                local ret=$?

                if [ $ret -eq 0 ]; then
                    exit 0  # 成功
                elif [ $ret -eq 2 ]; then
                    exit 1  # 非 OOM 错误
                fi
                # ret=1: OOM，记录尝试过
                best_gpu_attempted="${best_gpu_attempted},${gpu_id}"
            done
        fi

        # 准备下一轮
        if [ $round -lt $total_rounds ]; then
            info "═══════════════════════════════════════"
            info "⏳ ${OOM_CHECK_INTERVAL} 秒后重试（已尝试 GPU: ${best_gpu_attempted#,}）..."
            sleep "$OOM_CHECK_INTERVAL"
        fi
    done

    # 5. 所有轮次都失败，进入等待模式
    info "═══════════════════════════════════════"
    info "已尝试所有 GPU 共 ${total_rounds} 轮，全部 OOM 或显存不足。"
    wait_for_available_gpu

    # 6. 有空闲显存了，获取最新排序并运行
    local sorted_gpus
    sorted_gpus=($(get_gpus_sorted_by_free_memory))
    for entry in "${sorted_gpus[@]}"; do
        local gpu_id="${entry%%:*}"
        run_on_gpu "$gpu_id"
        local ret=$?
        if [ $ret -eq 0 ]; then
            exit 0
        elif [ $ret -eq 2 ]; then
            exit 1
        fi
    done

    # 7. 兜底
    error "❌ 无法找到可用的 GPU 运行实验"
    error "请手动检查 GPU 状态: gpustat"
    exit 1
}

# ---------- 执行 ----------
if [ "$BACKGROUND" = true ]; then
    # 后台运行模式
    nohup bash -c "$(declare -f main activate_conda check_cuda run_on_gpu is_oom_error get_gpu_memory_used get_gpu_memory_total get_gpu_memory_free get_gpu_processes get_gpus_sorted_by_free_memory show_gpu_status wait_for_available_gpu info warn error success log); SCRIPT_REL='${SCRIPT_REL}'; USER_ARGS='${USER_ARGS}'; LOG_FILE='${LOG_FILE}'; CODE_DIR='${CODE_DIR}'; PROJECT_DIR='${PROJECT_DIR}'; CONDA_ENV='${CONDA_ENV}'; CONDA_BASE='${CONDA_BASE}'; GPUS=(${GPUS[*]}); MIN_FREE_MEMORY=${MIN_FREE_MEMORY}; WAIT_INTERVAL=${WAIT_INTERVAL}; OOM_CHECK_INTERVAL=${OOM_CHECK_INTERVAL}; main" >> "$LOG_FILE" 2>&1 &
    local PID=$!
    echo ""
    echo "=============================================="
    info "实验已在后台启动 (PID: ${PID})"
    info "日志文件: ${LOG_FILE}"
    echo "  查看实时日志: tail -f ${LOG_FILE}"
    echo "  查看 GPU 状态: gpustat"
    echo "  终止实验:     kill ${PID}"
    echo "=============================================="
else
    # 前台运行
    main
fi
