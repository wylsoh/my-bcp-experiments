#!/usr/bin/env python3
"""
直接测试 NA-CMC V2 新代码：从 BCP 预训练模型加载，跳过 pre_train

使用方式：
  # GPU0 上跑 10% (batch_size=48 加速)
  python test_NA_v2_from_BCP_pretrain.py --gpu 0 --labelnum 10 --batch_size 48 --labeled_bs 24

  # GPU3 上跑 10% (batch_size=24 保守)
  python test_NA_v2_from_BCP_pretrain.py --gpu 3 --labelnum 10 --batch_size 24 --labeled_bs 12
"""
import logging
import os
import sys
import argparse

# ============ 参数 ============
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--labelnum', type=int, default=10,
    help='标注量：10=10% (382 slices), 20=20% (764 slices)')
parser.add_argument('--batch_size', type=int, default=48,
    help='总 batch size。注意 CMC 分支会额外前向传播几次，加大显存占用')
parser.add_argument('--labeled_bs', type=int, default=24,
    help='labeled batch size（监督信号数量）')
parser.add_argument('--max_iterations', type=int, default=40000)
parser.add_argument('--cmc_patch_size', type=int, default=8)
parser.add_argument('--cmc_soft_sigma', type=float, default=0.5)
parser.add_argument('--cmc_loss_weight', type=float, default=1.0)
parser.add_argument('--exp', type=str, default='MM_BCP')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import numpy as np
from MM_BCP_train import self_train

# ============ 手动构造 Args 对象 ============
class Args:
    root_path = '../data_split/MM'
    exp = args.exp
    model = 'unet'
    pre_iterations = 10000
    max_iterations = args.max_iterations
    batch_size = args.batch_size
    deterministic = 1
    base_lr = 0.01
    patch_size = [256, 256]
    seed = 1337
    num_classes = 8
    labeled_bs = args.labeled_bs
    labelnum = args.labelnum
    u_weight = 0.5
    gpu = args.gpu
    consistency = 0.1
    consistency_rampup = 200.0
    magnitude = 6.0
    s_param = 6
    # CMC 参数
    use_cmc = 1
    cmc_patch_size = args.cmc_patch_size
    cmc_warmup_iter = 5000
    cmc_init_shared = 0.4
    cmc_loss_weight = args.cmc_loss_weight
    cmc_mutual_weight = 0.5
    cmc_mutual_conf_thresh = 0.75
    conf_thresh_init = 0.90
    conf_thresh_final = 0.70
    cmc_soft_sigma = args.cmc_soft_sigma

a = Args()

# ============ 路径 ============
dataset_name = 'MM'
# 使用 BCP 预训练模型
pre_snapshot_path = f"./model/BCP/{dataset_name}_{a.exp}_{a.labelnum}_labeled/pre_train"
# 新 NA 实验路径（避免覆盖）
na_exp_name = f"NA_v2_from_BCP_ps{a.cmc_patch_size}_s{a.cmc_soft_sigma}_bs{a.batch_size}"
self_snapshot_path = f"./model/BCP/{dataset_name}_{a.exp}_{na_exp_name}_{a.labelnum}_labeled/self_train"

os.makedirs(self_snapshot_path, exist_ok=True)

# ============ 验证 pre_train 存在 ============
pre_model_path = os.path.join(pre_snapshot_path, 'unet_best_model.pth')
if not os.path.exists(pre_model_path):
    print(f"❌ 预训练模型不存在: {pre_model_path}")
    sys.exit(1)
print(f"✅ 使用 BCP 预训练模型: {pre_model_path}")
print(f"✅ 实验保存路径: {self_snapshot_path}")

# ============ 初始化 logging ============
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("=" * 60)
logging.info("  测试 NA-CMC V2 新代码：从 BCP 预训练加载")
logging.info("=" * 60)
logging.info(f"  预训练来源: {pre_model_path}")
logging.info(f"  labelnum={a.labelnum} | batch_size={a.batch_size} | labeled_bs={a.labeled_bs}")
logging.info(f"  cmc_patch_size={a.cmc_patch_size} | cmc_soft_sigma={a.cmc_soft_sigma}")
logging.info(f"  cmc_loss_weight={a.cmc_loss_weight} | max_iterations={a.max_iterations}")
logging.info(f"  GPU={a.gpu}")
for attr in ['root_path', 'exp', 'model', 'base_lr', 'num_classes', 'u_weight',
             'deterministic', 'seed', 'patch_size', 'cmc_warmup_iter', 'cmc_init_shared',
             'cmc_mutual_weight', 'cmc_mutual_conf_thresh', 'conf_thresh_init', 'conf_thresh_final',
             'consistency', 'consistency_rampup', 'magnitude', 's_param']:
    logging.info(f"  {attr}={getattr(a, attr)}")
logging.info("=" * 60)

# ============ 运行 self_train ============
self_train(a, pre_snapshot_path, self_snapshot_path)
