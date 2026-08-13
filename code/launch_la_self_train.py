"""
只启动 LA self_train（跳过 pre_train），用于已完成了 pre_train 的实验。
"""
import os
import sys
import argparse
import torch
import logging

# 将脚本目录加入 path，保证 import 正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LA_BCP_CMC_NA_v2_soft_mask import (
    self_train, patch_size, num_classes,
    args as original_args
)

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='BCP_CMC_NA_v2')
parser.add_argument('--labelnum', type=int, default=8)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--cmc_soft_sigma', type=float, default=0.3)
parser.add_argument('--cmc_patch_size', type=int, default=8)
parser.add_argument('--self_max_iteration', type=int, default=15000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--labeled_bs', type=int, default=4)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--max_samples', type=int, default=80)
# 其他参数使用默认值
FLAGS = parser.parse_args()

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# 复制原始 args 中的相关参数
original_args.exp = FLAGS.exp
original_args.labelnum = FLAGS.labelnum
original_args.gpu = FLAGS.gpu
original_args.cmc_soft_sigma = FLAGS.cmc_soft_sigma
original_args.cmc_patch_size = FLAGS.cmc_patch_size
original_args.self_max_iteration = FLAGS.self_max_iteration
original_args.batch_size = FLAGS.batch_size
original_args.labeled_bs = FLAGS.labeled_bs
original_args.seed = FLAGS.seed
original_args.deterministic = FLAGS.deterministic
original_args.base_lr = FLAGS.base_lr
original_args.max_samples = FLAGS.max_samples

# deterministic
import random
import numpy as np
import torch.backends.cudnn as cudnn
if FLAGS.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

pre_snapshot_path  = "./model/BCP/LA_{}_{}_labeled/pre_train".format(FLAGS.exp, FLAGS.labelnum)
self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(FLAGS.exp, FLAGS.labelnum)

if not os.path.exists(pre_snapshot_path):
    print(f"ERROR: pre_train path {pre_snapshot_path} does not exist!")
    sys.exit(1)

os.makedirs(self_snapshot_path, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=self_snapshot_path + "/log.txt",
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f"Starting self_train only (pre_train already done)")
logging.info(f"  exp={FLAGS.exp}, labelnum={FLAGS.labelnum}, sigma={FLAGS.cmc_soft_sigma}, p={FLAGS.cmc_patch_size}")
logging.info(f"  GPU={FLAGS.gpu}, self_max_iter={FLAGS.self_max_iteration}")
logging.info(f"  pre_snapshot={pre_snapshot_path}")
logging.info(f"  self_snapshot={self_snapshot_path}")

self_train(original_args, pre_snapshot_path, self_snapshot_path)
