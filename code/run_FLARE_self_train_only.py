#!/usr/bin/env python3
"""
直接启动 FLARE self-train，跳过 pre-train（使用已有 checkpoint）
"""
import os
import sys
import shutil
import argparse
import logging

# 复用原脚本的所有函数
from FLARE_BCP_CMC_NA_v2_soft_mask import (
    self_train, pre_train, args as original_args,
    pre_snapshot_path, self_snapshot_path,
    FLAREDataSets, DICE, CE_14, patch_size, num_classes
)

if __name__ == "__main__":
    print("=" * 60)
    print("FLARE Self-Train Only (跳过 Pre-Train)")
    print("=" * 60)
    
    pre_snapshot_path  = "./model/BCP/FLARE_{}_{}_labeled/pre_train".format(
        original_args.exp, original_args.labelnum)
    self_snapshot_path = "./model/BCP/FLARE_{}_{}_labeled/self_train".format(
        original_args.exp, original_args.labelnum)
    
    # 检查 pre-train checkpoint 是否存在
    pretrained_model = os.path.join(pre_snapshot_path,
                                    '{}_best_model.pth'.format(original_args.model))
    if not os.path.exists(pretrained_model):
        print("ERROR: Pre-trained model not found at {}".format(pretrained_model))
        print("Please run full training first (with pre-train)")
        sys.exit(1)
    
    print("Found pre-trained model: {}".format(pretrained_model))
    print("  exp={}, labelnum={}, soft_sigma={}, cmc_patch_size={}".format(
        original_args.exp, original_args.labelnum, 
        original_args.cmc_soft_sigma, original_args.cmc_patch_size))
    
    # 确保 self-train 目录存在
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)
    shutil.copy(__file__, self_snapshot_path)
    
    # 释放 CUDA 缓存
    import torch
    torch.cuda.empty_cache()
    
    # 设置 logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(original_args))
    
    # 直接跑 self-train
    self_train(original_args, pre_snapshot_path, self_snapshot_path)
