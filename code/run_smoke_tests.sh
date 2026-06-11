#!/bin/bash
# ============================================================
# 冒烟测试：小轮数验证 LA / Pancreas / ACDC label3 脚本正确性
#
# 使用方式：
#   先只跑一个测试，确认无误后再跑其他
#   bash run_smoke_tests.sh
#
# 注意：
#   - 小轮数 pre_iterations=200, self_max_iterations=500
#   - LA 小轮数 pre_max_iteration=100, self_max_iteration=300
#   - Pancreas 小轮数 pretraining_epochs=5, self_training_epochs=10
# ============================================================
PY=/home/hjj/anaconda3/envs/yll/bin/python
BASE_DIR="/home/hjj/ssq/my-bcp-experiments/code"
PANCREAS_DIR="/home/hjj/ssq/BCP_original/code/pancreas"

# ============================================================
# 待测试的脚本命令（先注释掉，手动取消注释执行）
# ============================================================

# --- Test 1: ACDC label3 NA_v2_s0.3 小轮数 ---
# CMD1="$PY BCP_CMC_v1_NA_v2_soft_mask.py \
#     --exp BCP_CMC_NA_v2_s0.3_label3_smoke \
#     --cmc_patch_size 8 \
#     --cmc_soft_sigma 0.3 \
#     --cmc_mutual_weight 0.5 \
#     --gpu 0 \
#     --labelnum 3 \
#     --pre_iterations 200 \
#     --max_iterations 500 \
#     --batch_size 12 \
#     --labeled_bs 6 \
#     2>&1 | tee run_smoke_acdc_label3.log"
# echo "Test 1: ACDC label3 smoke test"
# echo "$CMD1"
# echo ""

# --- Test 2: LA label4 NA_v2_s0.3 小轮数 ---
# CMD2="$PY LA_BCP_CMC_NA_v2_soft_mask.py \
#     --exp BCP_CMC_NA_v2_s0.3_smoke \
#     --cmc_patch_size 16 \
#     --cmc_soft_sigma 0.3 \
#     --cmc_mutual_weight 0.5 \
#     --batch_size 8 \
#     --labeled_bs 4 \
#     --gpu 0 \
#     --labelnum 4 \
#     --pre_max_iteration 100 \
#     --self_max_iteration 300 \
#     --max_samples 80 \
#     2>&1 | tee run_smoke_la_label4.log"
# echo "Test 2: LA label4 smoke test"
# echo "$CMD2"
# echo ""

# --- Test 3: Pancreas 10% NA_v2_s0.3 小轮数 ---
# 注：需要修改胰腺脚本中 epoch 数为小值
# CMD3="cd $PANCREAS_DIR && $PY train_pancreas_bcp_cmc_na_v2.py \
#     --gpu 0 \
#     --label_percent 10 \
#     --cmc_patch_size 16 \
#     --cmc_soft_sigma 0.3 \
#     --cmc_mutual_weight 0.5 \
#     --batch_size 2 \
#     --lr 1e-3 \
#     2>&1 | tee $BASE_DIR/run_smoke_pancreas_10pct.log"
# echo "Test 3: Pancreas 10% smoke test"
# echo "$CMD3"
# echo ""

echo "======================================"
echo "冒烟测试脚本 v1.0"
echo "============================="
echo ""
echo "使用方法：取消注释对应测试命令后执行"
echo ""
echo "推荐测试顺序："
echo "  1. ACDC label3（最快，验证 soft mask 在小样本下能否跑通）"
echo "  2. LA label4（验证 3D 软权重掩码是否正常）"
echo "  3. Pancreas 10%（验证胰腺数据加载和软权重掩码是否正常）"
echo ""
echo "使用前请确认目标 GPU 空闲（通过 nvidia-smi 查看）"
