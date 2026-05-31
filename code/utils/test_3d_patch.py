import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import os


def getLargestCC(segmentation):
    """取最大连通分量作为后处理（NMS）"""
    labels = label(segmentation)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = segmentation
    return largestCC


# =====================================================================
# 验证入口：LA 2 分类（仅 Dice）
# =====================================================================

def var_all_case_LA(model, num_classes, patch_size=(112, 112, 80),
                    stride_xy=18, stride_z=4):
    """LA 数据集验证（二分类，只算 Dice）"""
    base_dir = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'data_split', 'LA')
    list_path = os.path.join(base_dir, 'test.list')
    data_dir  = os.path.join(base_dir, '2018LA_Seg_Training Set')

    with open(list_path, 'r') as f:
        image_list = f.readlines()
    image_list = [
        os.path.join(data_dir, item.strip(), 'mri_norm2.h5')
        for item in image_list
    ]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(
            model, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def var_all_case_LA_plus(model_l, model_r, num_classes,
                          patch_size=(112, 112, 80),
                          stride_xy=18, stride_z=4):
    """LA 双模型验证"""
    base_dir = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'data_split', 'LA')
    list_path = os.path.join(base_dir, 'test.list')
    data_dir  = os.path.join(base_dir, '2018LA_Seg_Training Set')

    with open(list_path, 'r') as f:
        image_list = f.readlines()
    image_list = [
        os.path.join(data_dir, item.strip(), 'mri_norm2.h5')
        for item in image_list
    ]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_plus(
            model_l, model_r, image, stride_xy, stride_z,
            patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


# =====================================================================
# 通用全量验证入口
# =====================================================================

def test_all_case(model, image_list, num_classes,
                  patch_size=(112, 112, 80),
                  stride_xy=18, stride_z=4,
                  save_result=True, test_save_path=None,
                  preproc_fn=None, metric_detail=0, nms=0):
    """
    通用 3D 滑动窗口验证（自动适配二分类 / 多分类）。

    返回:
        num_classes == 2 → 1D array [dice, jc, hd95, asd] （二分类，与旧版兼容）
        num_classes > 2  → 2D array [num_classes-1, 4]   （逐类指标）
          每行 = [dice, jc, hd95, asd]
    """
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = None
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            model, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)

        if num_classes > 2:
            single_metric = _calc_multiclass_metric(
                prediction, label, num_classes)
        else:
            single_metric = _calc_binary_metric(prediction, label)

        if metric_detail:
            _print_metric(ith, single_metric, num_classes)

        if total_metric is None:
            total_metric = np.zeros_like(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_pred.nii.gz" % ith)
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


def test_all_case_plus(model_l, model_r, image_list, num_classes,
                       patch_size=(112, 112, 80),
                       stride_xy=18, stride_z=4,
                       save_result=True, test_save_path=None,
                       preproc_fn=None, metric_detail=0, nms=0):
    """双模型集成验证（test_all_case 的双模型版本）"""
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = None
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(
            model_l, model_r, image, stride_xy, stride_z,
            patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)

        if num_classes > 2:
            single_metric = _calc_multiclass_metric(
                prediction, label, num_classes)
        else:
            single_metric = _calc_binary_metric(prediction, label)

        if metric_detail:
            _print_metric(ith, single_metric, num_classes)

        if total_metric is None:
            total_metric = np.zeros_like(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_pred.nii.gz" % ith)
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32),
                                np.eye(4)),
                test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


# =====================================================================
# 单案例滑动窗口推理（核心）
# =====================================================================

def test_single_case(model, image, stride_xy, stride_z,
                     patch_size, num_classes=1):
    """
    单案例 3D 滑动窗口推理，兼容任意 num_classes（≥2）。

    image shape: [H, W, D] (numpy)
    模型输入:   [1, 1, ps0, ps1, ps2] (torch, batch=1, channel=1)
    模型输出:   y1 [1, C, ps0, ps1, ps2] → softmax → y [1, C, ps0, ps1, ps2]
    累加:       score_map[C, xs:xs+ps0, ys:ys+ps1, zs:zs+ps2] += y[0, :, :, :, :]
    """
    w, h, d = image.shape
    add_pad = False
    w_pad = max(patch_size[0] - w, 0)
    h_pad = max(patch_size[1] - h, 0)
    d_pad = max(patch_size[2] - d, 0)
    if w_pad > 0 or h_pad > 0 or d_pad > 0:
        add_pad = True
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                                   ys:ys + patch_size[1],
                                   zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0),
                    axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    y1, _ = model(test_patch)
                    y = F.softmax(y1, dim=1)
                # y: [1, C, ps0, ps1, ps2] → [0] → [C, ps0, ps1, ps2]
                y = y.cpu().data.numpy()[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0],
                             ys:ys + patch_size[1],
                             zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0).astype(np.int)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h,
                              dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w,
                                 hl_pad:hl_pad + h,
                                 dl_pad:dl_pad + d]
    return label_map, score_map


def test_single_case_plus(model_l, model_r, image, stride_xy, stride_z,
                           patch_size, num_classes=1):
    """双模型集成滑动窗口推理"""
    w, h, d = image.shape
    add_pad = False
    w_pad = max(patch_size[0] - w, 0)
    h_pad = max(patch_size[1] - h, 0)
    d_pad = max(patch_size[2] - d, 0)
    if w_pad > 0 or h_pad > 0 or d_pad > 0:
        add_pad = True
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                                   ys:ys + patch_size[1],
                                   zs:zs + patch_size[2]]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0),
                    axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    y1_l, _ = model_l(test_patch)
                    y1_r, _ = model_r(test_patch)
                    y1 = (y1_l + y1_r) / 2
                    y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0],
                             ys:ys + patch_size[1],
                             zs:zs + patch_size[2]] += y
                cnt[xs:xs + patch_size[0],
                    ys:ys + patch_size[1],
                    zs:zs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0).astype(np.int)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h,
                              dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w,
                                 hl_pad:hl_pad + h,
                                 dl_pad:dl_pad + d]
    return label_map, score_map


# =====================================================================
# 指标计算
# =====================================================================

def _calc_binary_metric(pred, gt):
    """二分类指标: (dice, jc, hd95, asd) 或全 0。"""
    if np.sum(pred) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)
    hd   = metric.binary.hd95(pred, gt)
    asd  = metric.binary.asd(pred, gt)
    return np.array([dice, jc, hd, asd])


def _calc_multiclass_metric(pred, gt, num_classes):
    """
    多分类 per-class 指标（参考 GuidedNet）。
    对每个类别 i ∈ [1, num_classes-1] 计算 Dice / JC / HD95 / ASD。
    返回 shape [num_classes-1, 4]。
    """
    metrics = []
    for i in range(1, num_classes):
        pred_i = (pred == i).astype(np.int)
        gt_i   = (gt == i).astype(np.int)
        if pred_i.sum() > 0 and gt_i.sum() > 0:
            dice = metric.binary.dc(pred_i, gt_i)
            jc   = metric.binary.jc(pred_i, gt_i)
            hd   = metric.binary.hd95(pred_i, gt_i)
            asd  = metric.binary.asd(pred_i, gt_i)
        else:
            dice = jc = hd = asd = 0.0
        metrics.append([dice, jc, hd, asd])
    return np.array(metrics)


def _print_metric(ith, single_metric, num_classes):
    """打印单样本指标"""
    if num_classes > 2:
        dice_str = ', '.join(f'{s[0]:.5f}' for s in single_metric)
        print(f'{ith:02d},\tDice=[{dice_str}]')
    else:
        print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            ith, single_metric[0], single_metric[1],
            single_metric[2], single_metric[3]))


# =====================================================================
# 兼容旧版 API（保持向后兼容）
# =====================================================================

def calculate_metric_percase(pred, gt):
    """二分类指标（旧版 API）"""
    return _calc_binary_metric(pred, gt)
