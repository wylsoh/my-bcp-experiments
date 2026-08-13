#!/bin/bash
# 切换到脚本目录
cd /home/hjj/ssq/my-bcp-experiments/code

EXP=$1
SIGMA=$2
GPU=$3

# 用 yll 环境运行
/home/hjj/anaconda3/envs/yll/bin/python launch_la_self_train.py \
  --exp "$EXP" \
  --cmc_soft_sigma "$SIGMA" \
  --cmc_patch_size 8 \
  --gpu "$GPU" \
  --labelnum 8 \
  --self_max_iteration 15000 \
  > ./model/BCP/LA_"${EXP}"_8_labeled/self_train/nohup.log 2>&1
