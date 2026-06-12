#!/usr/bin/env python3
"""
ACDC NA_v2 训练数据流可视化 — 纯图像输出
对 patient031_frame02_slice_6 应用训练时的 RandomGenerator 数据增强，
然后模拟 BCP 混合 + CMC 软掩码视图分离 + 模型预测，
每个组件保存为纯净单张图像（无标题/标签/colorbar）。

用法:
  cd my-bcp-experiments/code
  conda run -n yll python visualize_acdc_dataflow.py

输出:
  visualizations/ACDC/dataflow_components/  — 各组件纯净 PNG
"""

import os, sys, copy
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

sys.path.insert(0, '.')

# ---- 导入 ----
from networks.net_factory import net_factory
from BCP_CMC_v1_NA_v2_soft_mask import generate_cmc_masks_v2_soft, generate_mask
from dataloaders.dataset import random_rot_flip, random_rotate, random_crop

# ─── 配置 ───
DATA_H5 = '../../BCP/data_split/ACDC/data/patient031_frame02.h5'
SLICE_IDX = 6
OUTPUT_DIR = '../../BCP/code/visualizations/ACDC'
COMPONENT_DIR = os.path.join(OUTPUT_DIR, 'dataflow_components')
os.makedirs(COMPONENT_DIR, exist_ok=True)

# NA_v2 最优模型路径 (s0.3_ps8, 7_labeled)
MODEL_PTH = './model/BCP/ACDC_BCP_CMC_NA_v2_s0.3_ps8_7_labeled/self_train/unet_best_model.pth'

# ─── 配色（与 qualitative_comparison 一致） ───
# RV=红, Myo=蓝, LV=绿
CLASS_COLORS_RGBA = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0.72),       # RV - 红
    2: (0, 0.27, 1, 0.72),    # MYO - 蓝
    3: (0, 0.80, 0, 0.65),    # LV - 绿
}
CLASS_COLORS_OPAQUE = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 1.0),        # RV - 红不透明
    2: (0, 0.27, 1, 1.0),     # MYO - 蓝不透明
    3: (0, 0.80, 0, 1.0),     # LV - 绿不透明
}
PATCH_SIZE = (256, 256)


def data_augment(image, label):
    """模拟 RandomGenerator，仅保留 random_crop（去掉了旋转和翻转）"""
    import random as rnd
    if rnd.random() > 0.5:
        image, label = random_crop(image, label)
    x, y = image.shape
    image = zoom(image, (PATCH_SIZE[0] / x, PATCH_SIZE[1] / y), order=0)
    label = zoom(label, (PATCH_SIZE[0] / x, PATCH_SIZE[1] / y), order=0)
    return image, label


def augment_with_seed(img, gt, seed):
    """用固定种子做增强，得到可重复的结果"""
    import random as rnd
    rnd.seed(seed)
    # 还需要设置 numpy 和 python 的 random seed 来保证 random_rot_flip 等内部的一致性
    old_state = rnd.getstate()
    np_old = np.random.get_state()
    rnd.seed(seed)
    np.random.seed(seed)
    aug_img, aug_gt = data_augment(img.copy(), gt.copy())
    rnd.setstate(old_state)
    np.random.set_state(np_old)
    return aug_img, aug_gt


def save_clean_image(data, filepath, cmap='gray', vmin=None, vmax=None):
    """保存纯净图像：无轴、无标题、无留白"""
    h, w = data.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100), dpi=100)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    print(f'  -> {filepath}')


def save_seg_overlay(image, seg, filepath, vmin=None, vmax=None):
    """保存分割叠加图（MRI 灰度 + 半透明色块）"""
    h, w = image.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100), dpi=100)
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    overlay = np.zeros((*image.shape, 4))
    for cls in [1, 2, 3]:
        mask = (seg == cls)
        rgba = CLASS_COLORS_RGBA[cls]
        for c in range(4):
            overlay[:, :, c] += mask * rgba[c]
    overlay = np.clip(overlay, 0, 1)
    ax.imshow(overlay)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    print(f'  -> {filepath}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ─── 1. 加载原始数据 ───
    print(f'\n[1/7] 加载数据: {DATA_H5}')
    with h5py.File(DATA_H5, 'r') as f:
        img_vol = f['image'][:]   # (D, H, W)
        lbl_vol = f['label'][:]   # (D, H, W)

    img_raw = img_vol[SLICE_IDX]  # (256, 216)
    gt_raw  = lbl_vol[SLICE_IDX]  # (256, 216)
    print(f'  原始尺寸: {img_raw.shape}, GT 类别: {sorted(set(gt_raw.flatten()))}')

    # ─── 2. 应用数据增强（模拟训练中的 RandomGenerator） ───
    print(f'\n[2/7] 应用数据增强 (RandomGenerator: rot_flip / rotate / crop + resize 256x256) ...')
    # 用不同种子为 4 个输入生成 4 个不同的增强版本
    seeds = [42, 123, 456, 789]
    aug_results = [augment_with_seed(img_raw, gt_raw, s) for s in seeds]
    # aug_results[i] = (aug_img, aug_gt)  每个都是 (256, 256)

    (img_a_np, lab_a_np)   = aug_results[0]  # labeled A
    (img_b_np, lab_b_np)   = aug_results[1]  # labeled B
    (uimg_a_np, ulab_a_np) = aug_results[2]  # unlabeled A
    (uimg_b_np, ulab_b_np) = aug_results[3]  # unlabeled B

    for i, (img, gt) in enumerate(aug_results):
        print(f'  aug[{i}] shape={img.shape}, GT 类别: {sorted(set(gt.flatten()))}')

    # ─── 3. 转 torch tensor ───
    print(f'\n[3/7] 构造 batch 张量...')
    to_tensor = lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()
    to_label  = lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).long().cuda()

    img_a  = to_tensor(img_a_np)
    img_b  = to_tensor(img_b_np)
    uimg_a = to_tensor(uimg_a_np)
    uimg_b = to_tensor(uimg_b_np)
    lab_a  = to_label(lab_a_np)
    lab_b  = to_label(lab_b_np)

    # ─── 4. BCP 掩码与混合 ───
    print(f'\n[4/7] BCP 掩码与混合...')
    img_mask, loss_mask = generate_mask(img_a)
    net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
    net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)

    # ─── 5. CMC 软掩码 ───
    print(f'\n[5/7] CMC 软掩码 (patch_size=8, sigma=0.5)...')
    mask_a_ab, mask_b_ab = generate_cmc_masks_v2_soft(
        uimg_a, cmc_patch_size=8, shared_ratio=0.0, soft_sigma=0.5)
    uimg_a_viewA = uimg_a * mask_a_ab
    uimg_a_viewB = uimg_a * mask_b_ab

    # ─── 6. 加载模型做推理 ───
    print(f'\n[6/7] 加载模型推理: {MODEL_PTH}')
    net = net_factory(net_type='unet', in_chns=1, class_num=4)
    checkpoint = torch.load(MODEL_PTH)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()
    net.cuda()
    print('  模型加载成功')

    with torch.no_grad():
        out_main = net(img_a)
        if isinstance(out_main, (list, tuple)):
            out_main = out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()  # (256, 256)
    print(f'  预测类别: {sorted(set(pred.flatten()))}')

    # ─── 7. 转为 numpy ───
    print(f'\n[7/7] 转为 numpy 并保存...')
    to_np = lambda x: x.cpu().squeeze().numpy()

    img_mask_np       = to_np(img_mask)
    loss_mask_np      = to_np(loss_mask)
    net_input_unl_np  = to_np(net_input_unl)
    net_input_l_np    = to_np(net_input_l)

    mask_a_ab_np = to_np(mask_a_ab)
    mask_b_ab_np = to_np(mask_b_ab)
    uimg_a_viewA_np = to_np(uimg_a_viewA)
    uimg_a_viewB_np = to_np(uimg_a_viewB)

    # 统一 vmin/vmax（灰度医学图像）
    all_imgs = [img_a_np, img_b_np, uimg_a_np, uimg_b_np,
                net_input_unl_np, net_input_l_np,
                uimg_a_viewA_np, uimg_a_viewB_np]
    all_vals = np.concatenate([x.flatten() for x in all_imgs if x.size > 0])
    vmin_g = np.percentile(all_vals, 1)
    vmax_g = np.percentile(all_vals, 99)

    # ─── 保存纯净图像 ───
    # === 输入图像（增强后） ===
    for name, data in [('img_a', img_a_np), ('img_b', img_b_np),
                       ('uimg_a', uimg_a_np), ('uimg_b', uimg_b_np)]:
        save_clean_image(data, os.path.join(COMPONENT_DIR, f'{name}.png'),
                         cmap='gray', vmin=vmin_g, vmax=vmax_g)

    # === 输入对应的 GT 标签（增强后） ===
    for name, data in [('lab_a', lab_a_np), ('lab_b', lab_b_np),
                       ('ulab_a', ulab_a_np), ('ulab_b', ulab_b_np)]:
        overlay = np.zeros((*data.shape, 4))
        for cls in [1, 2, 3]:
            mask = (data == cls)
            rgba = CLASS_COLORS_OPAQUE[cls]
            for c in range(4):
                overlay[:, :, c] += mask * rgba[c]
        overlay = np.clip(overlay, 0, 1)
        save_clean_image(overlay, os.path.join(COMPONENT_DIR, f'{name}.png'))

    # GT 原图叠加
    save_seg_overlay(img_a_np, lab_a_np,
                     os.path.join(COMPONENT_DIR, 'gt_overlay.png'),
                     vmin=vmin_g, vmax=vmax_g)
    # 纯色块 GT
    gt_flat = np.zeros((*lab_a_np.shape, 4))
    for cls in [1, 2, 3]:
        mask = (lab_a_np == cls)
        rgba = CLASS_COLORS_OPAQUE[cls]
        for c in range(4):
            gt_flat[:, :, c] += mask * rgba[c]
    gt_flat = np.clip(gt_flat, 0, 1)
    save_clean_image(gt_flat, os.path.join(COMPONENT_DIR, 'gt.png'))

    # === 预测结果 ===
    pred_flat = np.zeros((*pred.shape, 4))
    for cls in [1, 2, 3]:
        mask = (pred == cls)
        rgba = CLASS_COLORS_OPAQUE[cls]
        for c in range(4):
            pred_flat[:, :, c] += mask * rgba[c]
    pred_flat = np.clip(pred_flat, 0, 1)
    save_clean_image(pred_flat, os.path.join(COMPONENT_DIR, 'pred.png'))
    save_seg_overlay(img_a_np, pred, os.path.join(COMPONENT_DIR, 'pred_overlay.png'),
                     vmin=vmin_g, vmax=vmax_g)

    # === BCP 组件 ===
    save_clean_image(img_mask_np, os.path.join(COMPONENT_DIR, 'bcp_mask.png'),
                     cmap='RdBu_r', vmin=0, vmax=1)
    save_clean_image(loss_mask_np, os.path.join(COMPONENT_DIR, 'bcp_loss_mask.png'),
                     cmap='RdBu_r', vmin=0, vmax=1)
    save_clean_image(net_input_unl_np, os.path.join(COMPONENT_DIR, 'bcp_mix_unl.png'),
                     cmap='gray', vmin=vmin_g, vmax=vmax_g)
    save_clean_image(net_input_l_np, os.path.join(COMPONENT_DIR, 'bcp_mix_l.png'),
                     cmap='gray', vmin=vmin_g, vmax=vmax_g)

    # === CMC 组件 ===
    save_clean_image(mask_a_ab_np, os.path.join(COMPONENT_DIR, 'cmc_mask_a.png'),
                     cmap='Reds', vmin=0, vmax=1)
    save_clean_image(mask_b_ab_np, os.path.join(COMPONENT_DIR, 'cmc_mask_b.png'),
                     cmap='Blues', vmin=0, vmax=1)
    save_clean_image(uimg_a_viewA_np, os.path.join(COMPONENT_DIR, 'cmc_viewA.png'),
                     cmap='gray', vmin=vmin_g, vmax=vmax_g)
    save_clean_image(uimg_a_viewB_np, os.path.join(COMPONENT_DIR, 'cmc_viewB.png'),
                     cmap='gray', vmin=vmin_g, vmax=vmax_g)

    print(f'\n全部完成! 组件保存在: {COMPONENT_DIR}/')
    print(f'生成的文件:')
    for f in sorted(os.listdir(COMPONENT_DIR)):
        fpath = os.path.join(COMPONENT_DIR, f)
        size = os.path.getsize(fpath)
        print(f'  {f:25s}  {size//1024:4d} KB')


if __name__ == '__main__':
    # 首先设置全局随机种子
    import random as rnd
    rnd.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
