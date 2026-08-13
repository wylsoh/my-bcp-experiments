#!/usr/bin/env python3
"""
BCP Baseline + NA-CMC V2 for MM (MMWHS) Dataset
=================================================
基于原始 BCP 基线，直接嵌入 NA-CMC V2 网络改动（软权重掩码 + 软权重损失）。

使用方式：
  # 纯 BCP（不加 CMC）
  python MM_BCP_train.py --gpu 0 --labelnum 5 --use_cmc 0

  # BCP + NA-CMC V2（默认）
  python MM_BCP_train.py --gpu 0 --labelnum 10 --use_cmc 1 \\
      --cmc_patch_size 8 --cmc_soft_sigma 0.5 --cmc_loss_weight 1.0

参数修复说明（相较于旧版 NA_v2 独立文件）：
  - max_iterations 默认从 30000 → 40000（与 BCP 基线一致）
  - cmc_patch_size argparse 默认 = 函数默认 = 8（统一）
  - batch_size / labeled_bs 在 pre_train 和 self_train 中保持一致
  - 新增 --use_cmc 开关，可单独关闭 CMC 分支做消融
"""
import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net
from utils import losses, ramps, val_2d

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/MM')
parser.add_argument('--exp', type=str, default='MM_BCP')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--max_iterations', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--labeled_bs', type=int, default=12)
parser.add_argument('--labelnum', type=int, default=5)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
parser.add_argument('--magnitude', type=float, default=6.0)
parser.add_argument('--s_param', type=int, default=6)
# ---------- CMC 开关 ----------
parser.add_argument('--use_cmc', type=int, default=1,
    help='是否启用 CMC 分支（0=纯 BCP，1=BCP+NA-CMC V2）')
# ---------- NA-CMC V2 参数 ----------
parser.add_argument('--cmc_patch_size',         type=int,   default=8)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=5000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
parser.add_argument('--cmc_soft_sigma', type=float, default=0.5,
    help='平滑强度，控制过渡带宽度。0=接近二值，1=最大平滑')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=args.num_classes)


# ================================================================
# 公用工具函数
# ================================================================
def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))

def get_2DLargestCC(segmentation, num_classes=8):
    """通用版 LargestCC，支持任意类别数"""
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, num_classes):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        n_batch = sum(class_list)
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()

def get_masks(output, nms=0, num_classes=8):
    """通用版 masks，支持任意类别数"""
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_2DLargestCC(probs, num_classes)
    return probs

def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136, "14": 256,
                    "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "MM" in dataset:
        ref_dict = {"1": 38, "2": 76, "5": 191, "10": 382, "15": 3823, "20": 764}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return ref_dict[str(patiens_num)]


# ================================================================
# NA-CMC V2 函数（直接嵌入 BCP 基线）
# ================================================================
def get_progressive_shared_ratio(current_iter, warmup_iter, init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter, init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

def generate_cmc_masks_v2_soft(img, cmc_patch_size=8, shared_ratio=0.0, soft_sigma=0.5):
    """
    NA-CMC V2 软权重掩码生成。

    用 bilinear 插值代替 nearest，再用 avg_pool 平滑，
    使掩码边界从 0/1 变为渐变的 [0,1] 连续值。

    Args:
        img             : [B, C, H, W]
        cmc_patch_size  : patch 大小
        shared_ratio    : 热身阶段共享比例
        soft_sigma      : 平滑强度，avg_pool 核大小 = max(int(patch_size * sigma) | 1, 1)
    Returns:
        mask_a, mask_b  : [B, 1, H, W] float [0,1]
    """
    B, C, H, W = img.shape
    n = H // cmc_patch_size
    masks_a, masks_b = [], []

    smooth_k = max(int(cmc_patch_size * soft_sigma) | 1, 1)
    smooth_pad = smooth_k // 2

    for _ in range(B):
        base = (torch.rand(n, n) > 0.5).float()

        if shared_ratio > 0.0:
            shared = torch.rand(n, n) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa = (base == 0).float()
            pb = (base == 1).float()

        pa_t = pa.view(1, 1, n, n)
        pb_t = pb.view(1, 1, n, n)

        # Step 1: bilinear 上采样（产生连续过渡）
        pa_up = F.interpolate(pa_t, size=(H, W), mode='bilinear', align_corners=False)
        pb_up = F.interpolate(pb_t, size=(H, W), mode='bilinear', align_corners=False)

        # Step 2: avg_pool 平滑（扩大过渡带宽度）
        if smooth_k > 1:
            pa_up = F.avg_pool2d(pa_up, kernel_size=smooth_k, stride=1, padding=smooth_pad)
            pb_up = F.avg_pool2d(pb_up, kernel_size=smooth_k, stride=1, padding=smooth_pad)

        masks_a.append(pa_up.squeeze(0))
        masks_b.append(pb_up.squeeze(0))

    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device))


def cmc_mutual_loss_soft(out_viewA, out_viewB, plab_teacher,
                          conf_mask, mask_a_soft, mask_b_soft,
                          mutual_conf_thresh, mutual_weight):
    """
    NA-CMC V2 软权重损失。

    - anchor: mask 值作为像素权重
    - mutual: (1-mb)*ma 代替硬 excl，边界像素获得渐进权重

    Args:
        out_viewA/B     : [B, C, H, W] 模型输出
        plab_teacher    : [B, H, W] long，教师硬标签
        conf_mask       : [B, H, W] float，置信度掩码
        mask_a/b_soft   : [B, 1, H, W] float [0,1]
    Returns:
        loss_anchor + mutual_weight * loss_mutual, loss_anchor, loss_mutual
    """
    ma = mask_a_soft.squeeze(1)
    mb = mask_b_soft.squeeze(1)

    # Anchor loss：软权重
    w = conf_mask
    denom_a = (w * ma).sum() + 1e-6
    denom_b = (w * mb).sum() + 1e-6
    la = F.cross_entropy(out_viewA, plab_teacher, reduction='none')
    lb = F.cross_entropy(out_viewB, plab_teacher, reduction='none')
    loss_anchor = ((la * ma * w).sum() / denom_a +
                   (lb * mb * w).sum() / denom_b) / 2.0

    # Mutual loss：软 exclusive 权重
    with torch.no_grad():
        prob_a = F.softmax(out_viewA, dim=1)
        prob_b = F.softmax(out_viewB, dim=1)
        conf_va = prob_a.max(dim=1).values
        conf_vb = prob_b.max(dim=1).values
        plab_va = prob_a.argmax(dim=1).long()
        plab_vb = prob_b.argmax(dim=1).long()

    w_b = (1.0 - mb) * ma * (conf_va > mutual_conf_thresh).float()
    w_a = (1.0 - ma) * mb * (conf_vb > mutual_conf_thresh).float()

    denom_mut_b = w_b.sum() + 1e-6
    denom_mut_a = w_a.sum() + 1e-6
    l_b_from_a = (F.cross_entropy(out_viewB, plab_va, reduction='none') * w_b).sum() / denom_mut_b
    l_a_from_b = (F.cross_entropy(out_viewA, plab_vb, reduction='none') * w_a).sum() / denom_mut_a
    loss_mutual = (l_b_from_a + l_a_from_b) / 2.0

    return loss_anchor + mutual_weight * loss_mutual, loss_anchor, loss_mutual


# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    global dice_loss
    base_lr = args.base_lr
    num_classes = args.num_classes
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# Self-train：BCP + 可选 NA-CMC V2 分支
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    global dice_loss
    base_lr = args.base_lr
    num_classes = args.num_classes
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    if args.use_cmc:
        smooth_k = max(int(args.cmc_patch_size * args.cmc_soft_sigma) | 1, 1)
        logging.info("Start self_training (BCP + NA-CMC V2 Soft Mask)")
        logging.info("soft_sigma={}  →  avg_pool 核 {}px，过渡带宽约 {}px".format(
            args.cmc_soft_sigma, smooth_k, smooth_k // 2))
    else:
        logging.info("Start self_training (BCP only, no CMC)")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a  = volume_batch[:labeled_sub_bs]
            img_b  = volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            ulab_b = label_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]

            # ----------------------------------------------------------
            # BCP 分支
            # ----------------------------------------------------------
            with torch.no_grad():
                pre_a  = ema_model(uimg_a)
                pre_b  = ema_model(uimg_b)
                plab_a = get_masks(pre_a, nms=1, num_classes=num_classes)
                plab_b = get_masks(pre_b, nms=1, num_classes=num_classes)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label  = lab_b * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l   = model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask,
                                         u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_bcp = (unl_dice + unl_ce + l_dice + l_ce) / 2

            loss = loss_bcp
            logging_extra = ""  # CMC 分支无额外日志

            # ----------------------------------------------------------
            # NA-CMC V2 分支（可选）
            # ----------------------------------------------------------
            if args.use_cmc:
                shared_ratio = get_progressive_shared_ratio(
                    iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
                current_conf_thresh = get_adaptive_threshold(
                    iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

                # ① 软权重掩码
                mask_a_ab, mask_b_ab = generate_cmc_masks_v2_soft(
                    uimg_a, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)
                mask_a_cd, mask_b_cd = generate_cmc_masks_v2_soft(
                    uimg_b, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)

                # ② 图像遮挡（加权融合）
                uimg_a_viewA = uimg_a * mask_a_ab
                uimg_a_viewB = uimg_a * mask_b_ab
                uimg_b_viewC = uimg_b * mask_a_cd
                uimg_b_viewD = uimg_b * mask_b_cd

                out_ab = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
                out_cd = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))
                out_a_viewA, out_a_viewB = out_ab[:unlabeled_sub_bs], out_ab[unlabeled_sub_bs:]
                out_b_viewC, out_b_viewD = out_cd[:unlabeled_sub_bs], out_cd[unlabeled_sub_bs:]

                with torch.no_grad():
                    conf_a = F.softmax(pre_a, dim=1).max(dim=1).values
                    conf_b = F.softmax(pre_b, dim=1).max(dim=1).values
                    conf_mask_a = (conf_a > current_conf_thresh).float()
                    conf_mask_b = (conf_b > current_conf_thresh).float()
                    plab_teacher_a = plab_a.long()
                    plab_teacher_b = plab_b.long()

                # ③ 软权重损失
                loss_cmc_a, la_anchor, la_mutual = cmc_mutual_loss_soft(
                    out_a_viewA, out_a_viewB, plab_teacher_a, conf_mask_a,
                    mask_a_ab, mask_b_ab, args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
                loss_cmc_b, lb_anchor, lb_mutual = cmc_mutual_loss_soft(
                    out_b_viewC, out_b_viewD, plab_teacher_b, conf_mask_b,
                    mask_a_cd, mask_b_cd, args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
                loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

                # 监控：过渡带占比
                with torch.no_grad():
                    ma = mask_a_ab.squeeze(1)
                    border_ratio = ((ma > 0.1) & (ma < 0.9)).float().mean().item()

                cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
                loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

                logging_extra = "cmc=%.4f (anchor=%.4f mutual=%.4f) | border=%.3f conf_t=%.2f" % (
                    loss_cmc.item(), (la_anchor + lb_anchor).item() / 2,
                    (la_mutual + lb_mutual).item() / 2, border_ratio, current_conf_thresh)

                # TensorBoard 写入
                writer.add_scalar('info/loss_cmc',     loss_cmc,     iter_num)
                writer.add_scalar('info/loss_anchor',  (la_anchor + lb_anchor) / 2, iter_num)
                writer.add_scalar('info/loss_mutual',  (la_mutual + lb_mutual) / 2, iter_num)
                writer.add_scalar('info/cmc_rampup',   cmc_rampup,   iter_num)
                writer.add_scalar('info/shared_ratio', shared_ratio, iter_num)
                writer.add_scalar('info/conf_threshold', current_conf_thresh, iter_num)
                writer.add_scalar('na_cmc/soft_border_ratio', border_ratio, iter_num)
                writer.add_scalar('na_cmc/mask_mean', ma.mean().item(), iter_num)

                if iter_num % 20 == 0:
                    writer.add_image('cmc/MaskA_soft', mask_a_ab[0], iter_num)
                    writer.add_image('cmc/MaskB_soft', mask_b_ab[0], iter_num)
                    writer.add_image('cmc/ViewA', uimg_a_viewA[0, 0:1], iter_num)
                    writer.add_image('cmc/ViewB', uimg_a_viewB[0, 0:1], iter_num)
                    pred_a_vis = torch.argmax(torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                    pred_b_vis = torch.argmax(torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                    writer.add_image('cmc/PredViewA', pred_a_vis[0].float() * 50, iter_num)
                    writer.add_image('cmc/PredViewB', pred_b_vis[0].float() * 50, iter_num)

            # ----------------------------------------------------------
            # 通用优化步骤
            # ----------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss,     iter_num)
            writer.add_scalar('info/loss_bcp',   loss_bcp, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            if logging_extra:
                logging.info('iter %d | loss=%.4f bcp=%.4f %s' %
                             (iter_num, loss.item(), loss_bcp.item(), logging_extra))
            else:
                logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' %
                             (iter_num, loss, unl_dice + l_dice, unl_ce + l_ce))

            if iter_num % 20 == 0:
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label[1, ...].unsqueeze(0) * 50
                writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
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

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(__file__, self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
