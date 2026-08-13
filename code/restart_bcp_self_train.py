#!/usr/bin/env python3
"""从已完成的 MM pre_train 启动 BCP self_train (跳过 pre_train)"""

import sys, os, argparse
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() and __file__ else os.getcwd()
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import torch
import logging
from MM_BCP_train import self_train

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='MM_BCP')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--labelnum', type=int, default=10)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--labeled_bs', type=int, default=12)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--root_path', type=str, default='../data_split/MM')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--patch_size', type=str, default='[256, 256]')
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
parser.add_argument('--conf_thresh_init', type=float, default=0.9)
parser.add_argument('--conf_thresh_final', type=float, default=0.7)
parser.add_argument('--s_param', type=int, default=6)
parser.add_argument('--magnitude', type=float, default=6.0)
args = parser.parse_args()

args.patch_size = [256, 256]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import random, numpy as np
from torch.backends import cudnn
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

dataset_name = "MM"
if args.exp.endswith("_labeled"):
    pre_snapshot_path = "./model/BCP/{}_{}/pre_train".format(dataset_name, args.exp)
    self_snapshot_path = "./model/BCP/{}_{}/self_train".format(dataset_name, args.exp)
else:
    pre_snapshot_path = "./model/BCP/{}_{}_{}_labeled/pre_train".format(dataset_name, args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/{}_{}_{}_labeled/self_train".format(dataset_name, args.exp, args.labelnum)

print("pre_snapshot_path:", pre_snapshot_path)
print("self_snapshot_path:", self_snapshot_path)

# 不删除旧目录，保留已保存的checkpoint
os.makedirs(self_snapshot_path, exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

self_train(args, pre_snapshot_path, self_snapshot_path)
