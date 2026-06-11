#!/bin/bash
# 监控实验5 (cmc_mutual_weight=1.0) 训练进度脚本
# 功能：每5分钟检查一次，训练完成后自动运行 test_ACDC.py
# 使用方法：bash monitor_exp5.sh [检查间隔(秒)] [运行测试=y/n]
# 例如：bash monitor_exp5.sh 300 y

LOG_FILE="/home/hjj/ssq/my-bcp-experiments/code/run_mw10.log"
MONITOR_LOG="/home/hjj/ssq/my-bcp-experiments/code/monitor_exp5.log"
TEST_LOG="/home/hjj/ssq/my-bcp-experiments/code/test_mw10.log"
PYTHON="/home/hjj/anaconda3/envs/yll/bin/python"
TEST_SCRIPT="/home/hjj/ssq/my-bcp-experiments/code/test_ACDC.py"
MODEL_BASE="/home/hjj/ssq/my-bcp-experiments/code/model/BCP"

INTERVAL=${1:-300}  # 默认每5分钟（300秒）检查一次
RUN_TEST=${2:-y}    # 默认完成后运行测试
EXP_NAME="BCP_CMC_v1_mutual_mw10"
LABELNUM=7

echo "$(date) - 开始监控实验5 (mw=1.0)" | tee -a "$MONITOR_LOG"
echo "$(date) - 检查间隔: ${INTERVAL}s, 自动测试: ${RUN_TEST}" | tee -a "$MONITOR_LOG"
echo "" >> "$MONITOR_LOG"

check_completed=0

while [ $check_completed -eq 0 ]; do
    if [ ! -f "$LOG_FILE" ]; then
        echo "$(date) - ⚠️ 日志文件不存在: $LOG_FILE" | tee -a "$MONITOR_LOG"
        sleep $INTERVAL
        continue
    fi

    # 获取最后几行
    last_lines=$(tail -5 "$LOG_FILE")
    last_iter=$(echo "$last_lines" | grep -oP 'iteration \K[0-9]+' | tail -1)
    has_mean_dice=$(grep -c "mean_dice" "$LOG_FILE")
    has_30000=$(grep -c "iteration 30000" "$LOG_FILE")

    echo "$(date) - 当前 iteration: ${last_iter:-unknown}" | tee -a "$MONITOR_LOG"

    if [ "$has_30000" -gt 0 ] && [ "$has_mean_dice" -gt 0 ]; then
        mean_dice=$(grep "mean_dice" "$LOG_FILE" | tail -1 | grep -oP 'mean_dice : [0-9.]+')
        echo "$(date) - ✅ 训练完成! ${mean_dice}" | tee -a "$MONITOR_LOG"

        if [ "$RUN_TEST" = "y" ]; then
            cd /home/hjj/ssq/my-bcp-experiments/code
            echo "$(date) - 🚀 开始运行 test_ACDC.py (exp=${EXP_NAME})" | tee -a "$MONITOR_LOG"
            CUDA_VISIBLE_DEVICES=2 $PYTHON $TEST_SCRIPT \
                --exp $EXP_NAME \
                --labelnum $LABELNUM \
                --stage_name self_train \
                --gpu 2 2>&1 | tee -a "$TEST_LOG"
            echo "$(date) - ✅ 测试完成!" | tee -a "$MONITOR_LOG"

            # 读取测试结果
            perf_file="${MODEL_BASE}/ACDC_${EXP_NAME}_${LABELNUM}_labeled/performance.txt"
            if [ -f "$perf_file" ]; then
                echo "" | tee -a "$MONITOR_LOG"
                echo "========== 最终测试结果 ==========" | tee -a "$MONITOR_LOG"
                cat "$perf_file" | tee -a "$MONITOR_LOG"
                echo "=================================" | tee -a "$MONITOR_LOG"
            fi
        fi

        check_completed=1
    else
        # 检查是否还在 pretrain 阶段
        in_pretrain=$(tail -20 "$LOG_FILE" | grep -c "mix_dice")
        if [ "$in_pretrain" -gt 0 ]; then
            echo "$(date) - ▶️ Pretrain 阶段" | tee -a "$MONITOR_LOG"
        else
            echo "$(date) - ▶️ Self-train 阶段" | tee -a "$MONITOR_LOG"
        fi
        sleep $INTERVAL
    fi
done

echo "$(date) - 🎉 监控结束" | tee -a "$MONITOR_LOG"
