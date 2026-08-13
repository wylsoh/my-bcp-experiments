"""
FLARE 数据集测试脚本（13个腹部器官 + 背景 = 14 类）

用法：
    # 测试 self_train 阶段的最佳模型
    python test_FLARE.py --gpu 0 --stage_name self_train --labelnum 42

    # 测试 pre_train 阶段的最佳模型
    python test_FLARE.py --gpu 0 --stage_name pre_train --labelnum 42

    # 指定自定义 checkpoint
    python test_FLARE.py --gpu 0 --checkpoint /path/to/model.pth
"""
import os
import sys
import argparse
import numpy as np
import torch
import h5py
import nibabel as nib
from tqdm import tqdm

from networks.net_factory import net_factory
from utils.test_3d_patch import test_single_case, _calc_multiclass_metric

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/flare',
                    help='FLARE dataset root')
parser.add_argument('--exp', type=str, default='BCP_CMC_NA_v2',
                    help='experiment name')
parser.add_argument('--model', type=str, default='VNet',
                    help='model name')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--detail', type=int, default=1,
                    help='print metrics for every sample?')
parser.add_argument('--labelnum', type=int, default=42,
                    help='number of labeled samples (for path construction)')
parser.add_argument('--stage_name', type=str, default='self_train',
                    help='self_train or pre_train')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to custom checkpoint (overrides stage_name/labelnum)')
parser.add_argument('--save_result', type=int, default=1,
                    help='save predictions (nii.gz) to disk?')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

num_classes = 14
patch_size = (64, 128, 128)
stride_xy = 32
stride_z = 16

# ---------- 模型路径 ----------
if FLAGS.checkpoint is not None:
    save_model_path = FLAGS.checkpoint
    exp_dir = os.path.dirname(os.path.dirname(save_model_path))
    test_save_path = os.path.join(exp_dir, '{}_predictions'.format(FLAGS.model))
else:
    snapshot_path = "./model/BCP/FLARE_{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = os.path.join(
        snapshot_path, '{}_predictions'.format(FLAGS.model))
    save_model_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

print("=" * 60)
print("FLARE Test Configuration")
print("=" * 60)
print("  Model checkpoint: {}".format(save_model_path))
print("  Save path:        {}".format(test_save_path))
print("  num_classes:      {}".format(num_classes))
print("  patch_size:       {}".format(patch_size))
print("  stride_xy:        {}, stride_z: {}".format(stride_xy, stride_z))
print("=" * 60)

# ---------- FLARE 测试图像列表 ----------
base_dir = FLAGS.root_path
with open(os.path.join(base_dir, 'test.txt'), 'r') as f:
    image_list = f.readlines()
image_list = [item.strip() for item in image_list]
image_list = [
    os.path.join(base_dir, 'data', item, '2022.h5')
    for item in image_list
]
print("Testing on {} FLARE cases".format(len(image_list)))
print()

# 器官名称（13个腹部器官）
organ_names = [
    'Spleen',                  # 1
    'Right Kidney',            # 2
    'Left Kidney',             # 3
    'Gallbladder',             # 4
    'Liver',                   # 5
    'Stomach',                 # 6
    'Aorta',                   # 7
    'Inferior Vena Cava',      # 8
    'Portal Vein & Splenic Vein',  # 9
    'Pancreas',                # 10
    'Right Adrenal Gland',     # 11
    'Left Adrenal Gland',      # 12
    'Duodenum',                # 13
]


def test_flare():
    model = net_factory(net_type=FLAGS.model, in_chns=1,
                        class_num=num_classes, mode="test")
    checkpoint = torch.load(save_model_path)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Loaded weights from {}".format(save_model_path))
    model.eval()

    total_metric = None

    for ith, image_path in enumerate(
            tqdm(image_list, desc="Evaluating", ncols=80)):
        # 读取数据
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:].astype(np.uint8)
        h5f.close()

        # 滑动窗口推理
        prediction, score_map = test_single_case(
            model, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes)

        # 计算 per-class 指标
        single_metric = _calc_multiclass_metric(prediction, label, num_classes)
        if total_metric is None:
            total_metric = np.zeros_like(single_metric)
        total_metric += np.asarray(single_metric)

        # 保存预测结果
        if FLAGS.save_result:
            case_name = os.path.basename(os.path.dirname(image_path))
            nib.save(nib.Nifti1Image(prediction.astype(np.int16),
                                     np.eye(4)),
                     os.path.join(test_save_path,
                                  '{}_pred.nii.gz'.format(case_name)))

    # ================================================================
    # 汇总结果
    # ================================================================
    print("\n" + "=" * 70)
    print("FLARE Test Results ({} labeled cases)".format(FLAGS.labelnum))
    print("=" * 70)

    avg_metric = total_metric / len(image_list)

    # 每个器官的指标
    print("\n{:<30s} {:>8s} {:>8s} {:>10s}".format(
        'Organ', 'Dice', 'HD95', 'ASD'))
    print("-" * 60)
    for i in range(num_classes - 1):
        print("{:<30s} {:>8.4f} {:>8.2f} {:>10.4f}".format(
            organ_names[i],
            avg_metric[i, 0],  # Dice
            avg_metric[i, 2],  # HD95
            avg_metric[i, 3],  # ASD
        ))

    print("-" * 60)
    mean_dice = np.mean(avg_metric[:, 0])
    mean_hd95 = np.mean(avg_metric[:, 2])
    mean_asd  = np.mean(avg_metric[:, 3])
    print("{:<30s} {:>8.4f} {:>8.2f} {:>10.4f}".format(
        'MEAN (over 13 organs)', mean_dice, mean_hd95, mean_asd))

    # 保存结果到文件
    result_path = os.path.join(test_save_path, 'summary.txt')
    with open(result_path, 'w') as f:
        f.write("FLARE Test Results ({} labeled cases)\n".format(FLAGS.labelnum))
        f.write("Checkpoint: {}\n\n".format(save_model_path))
        f.write("{:<30s} {:>8s} {:>8s} {:>10s}\n".format(
            'Organ', 'Dice', 'HD95', 'ASD'))
        f.write("-" * 60 + "\n")
        for i in range(num_classes - 1):
            f.write("{:<30s} {:>8.4f} {:>8.2f} {:>10.4f}\n".format(
                organ_names[i],
                avg_metric[i, 0], avg_metric[i, 2], avg_metric[i, 3]))
        f.write("-" * 60 + "\n")
        f.write("{:<30s} {:>8.4f} {:>8.2f} {:>10.4f}\n".format(
            'MEAN (over 13 organs)', mean_dice, mean_hd95, mean_asd))
    print("\nResults saved to {}".format(result_path))

    return avg_metric


if __name__ == '__main__':
    metric = test_flare()
    print("\nDone.")
