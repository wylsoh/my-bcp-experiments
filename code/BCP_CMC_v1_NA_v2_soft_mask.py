"""
BCP + CMC v1 - NA-CMC V2：软权重掩码（Soft Mask）
=============================================================
改动说明（相对于原始 BCP_CMC_v1_mutual.py）：
  ① generate_cmc_masks → generate_cmc_masks_v2_soft
     - nearest 插值 → bilinear 插值，产生连续过渡值
     - 额外 avg_pool 平滑掩码边缘，扩大过渡带宽度
     - 掩码值域从 {0,1} 变为 [0,1]
  ② cmc_mutual_loss → cmc_mutual_loss_soft
     - anchor 损失：mask 值本身作为像素级权重（掩码越高权重越大）
     - mutual 损失：用 (1-mb)*ma 代替硬性 excl_a，自然融合边界区域
     - 不再需要硬截断，边界处的像素获得渐进权重而非被忽略
  ③ 新增参数：--cmc_soft_sigma（过渡带平滑强度）

关键优势：
  边界像素权重从 0/1 变为 0.1~0.9，使得模型在边界处同时接受来自两个
  视图的监督信号，梯度更平滑，分割边缘的连贯性更强。
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
parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
parser.add_argument('--exp', type=str, default='BCP_CMC_NA_v2')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--labeled_bs', type=int, default=12)
parser.add_argument('--labelnum', type=int, default=7)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
parser.add_argument('--magnitude', type=float, default=6.0)
parser.add_argument('--s_param', type=int, default=6)
# ---------- CMC 参数 ----------
parser.add_argument('--cmc_patch_size',         type=int,   default=8)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=5000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
# ---------- NA-CMC V2 新增参数 ----------
parser.add_argument('--cmc_soft_sigma', type=float, default=0.5,
    help='平滑强度，控制过渡带宽度。0=接近二值，1=最大平滑。'
         '实际核大小 = int(patch_size * sigma) | 1。推荐范围 [0.3, 0.8]')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

# ================================================================
# 原始 BCP 函数（逐字复制）
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

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
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
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
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
    batch_size, channel, img_x, img_y = img.shape
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
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
        return
    return ref_dict[str(patiens_num)]

def get_progressive_shared_ratio(current_iter, warmup_iter, init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter, init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

# ================================================================
# NA-CMC V2 核心：软权重掩码生成
# ================================================================
def generate_cmc_masks_v2_soft(img, cmc_patch_size=16, shared_ratio=0.0, soft_sigma=0.5):
    """
    NA-CMC V2：软权重掩码

    将 nearest 插值替换为 bilinear，再用 avg_pool 平滑，
    使掩码边界从硬性 0/1 变为平滑过渡值 [0,1]。

    边界处的像素值示例（patch_size=16, sigma=0.5）：
      中心区域: ~1.0（完全可见）
      边界1px带: ~0.9（高权重，向对方视图延伸）
      边界4px带: ~0.5（等权重，两视图均感知）
      边界8px带: ~0.1（低权重，接近对方视图）
      对方区域: ~0.0（基本不可见）

    Args:
        img         : [B, C, H, W]
        cmc_patch_size: patch 大小
        shared_ratio: 热身阶段共享比例
        soft_sigma  : 平滑强度（0~1），越大过渡带越宽。
                      实际核大小 = max(int(patch_size * sigma) | 1, 1)

    Returns:
        mask_a : [B, 1, H, W] float [0,1]，连续值软掩码
        mask_b : [B, 1, H, W] float [0,1]，连续值软掩码
    """
    B, C, H, W = img.shape
    n = H // cmc_patch_size
    masks_a, masks_b = [], []

    # avg_pool 核大小（像素空间）
    smooth_k = max(int(cmc_patch_size * soft_sigma) | 1, 1)  # 保证为奇数
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

        # ---- Step 1: bilinear 上采样（产生连续过渡） ----
        # bilinear 在 patch 边界处自然产生插值，不同于 nearest 的硬切
        pa_up = F.interpolate(pa_t, size=(H, W), mode='bilinear', align_corners=False)
        pb_up = F.interpolate(pb_t, size=(H, W), mode='bilinear', align_corners=False)

        # ---- Step 2: avg_pool 平滑（扩大过渡带宽度） ----
        # avg_pool 在像素空间平均邻域，使过渡带从 1px 扩展到 smooth_k/2 px
        if smooth_k > 1:
            pa_up = F.avg_pool2d(pa_up, kernel_size=smooth_k,
                                  stride=1, padding=smooth_pad)
            pb_up = F.avg_pool2d(pb_up, kernel_size=smooth_k,
                                  stride=1, padding=smooth_pad)

        masks_a.append(pa_up.squeeze(0))   # [1,H,W]
        masks_b.append(pb_up.squeeze(0))

    return (torch.stack(masks_a).to(img.device),   # [B,1,H,W]
            torch.stack(masks_b).to(img.device))


# ================================================================
# NA-CMC V2 核心：软权重损失函数
# ================================================================
def cmc_mutual_loss_soft(out_viewA, out_viewB, plab_teacher,
                          conf_mask, mask_a_soft, mask_b_soft,
                          mutual_conf_thresh, mutual_weight):
    """
    NA-CMC V2 损失：掩码值 [0,1] 直接作为像素权重

    相比原始 CMC：
    - anchor：mask 值越高，该像素在 anchor 损失中权重越大
    - mutual：用 (1-mb)*ma 和 (1-ma)*mb 代替硬性 excl_a/excl_b
              边界像素（ma≈mb≈0.5）同时获得来自两个方向的监督信号

    Args:
        out_viewA/B       : [B, C, H, W] 模型输出
        plab_teacher      : [B, H, W] long，教师硬标签
        conf_mask         : [B, H, W] float，教师置信度掩码
        mask_a_soft/b_soft: [B, 1, H, W] float [0,1]，软权重掩码
        mutual_conf_thresh: 互教置信度阈值
        mutual_weight     : 互教损失权重
    """
    ma = mask_a_soft.squeeze(1)   # [B, H, W]，视图A的像素权重
    mb = mask_b_soft.squeeze(1)   # [B, H, W]，视图B的像素权重

    # ---- Anchor 损失：用 soft mask 值加权 ----
    # 高 mask 值的区域（视图完整可见处）贡献更大的监督信号
    w = conf_mask
    denom_a = (w * ma).sum() + 1e-6
    denom_b = (w * mb).sum() + 1e-6
    la = F.cross_entropy(out_viewA, plab_teacher, reduction='none')  # [B,H,W]
    lb = F.cross_entropy(out_viewB, plab_teacher, reduction='none')
    loss_anchor = ((la * ma * w).sum() / denom_a +
                   (lb * mb * w).sum() / denom_b) / 2.0

    # ---- Mutual 损失：软权重版本 ----
    # w_b = (1-mb) * ma：A 完全可见而 B 完全不可见的区域最大，
    #                    A 可见 B 也可见的边界区域有中等权重（这是关键变化！）
    with torch.no_grad():
        prob_a = F.softmax(out_viewA, dim=1)
        prob_b = F.softmax(out_viewB, dim=1)
        conf_va = prob_a.max(dim=1).values
        conf_vb = prob_b.max(dim=1).values
        plab_va = prob_a.argmax(dim=1).long()
        plab_vb = prob_b.argmax(dim=1).long()

    # 软 exclusive 权重：(1-mb)*ma → A独有区域为1，边界区域~0.25，B独有区域为0
    w_b = (1.0 - mb) * ma * (conf_va > mutual_conf_thresh).float()
    w_a = (1.0 - ma) * mb * (conf_vb > mutual_conf_thresh).float()

    denom_mut_b = w_b.sum() + 1e-6
    denom_mut_a = w_a.sum() + 1e-6
    l_b_from_a = (F.cross_entropy(out_viewB, plab_va, reduction='none') * w_b
                  ).sum() / denom_mut_b
    l_a_from_b = (F.cross_entropy(out_viewA, plab_vb, reduction='none') * w_a
                  ).sum() / denom_mut_a
    loss_mutual = (l_b_from_a + l_a_from_b) / 2.0

    return loss_anchor + mutual_weight * loss_mutual, loss_anchor, loss_mutual


# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
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
            logging.info('iteration %d: loss: %f' % (iter_num, loss))
            if iter_num % 20 == 0:
                writer.add_image('pre_train/Mixed_Image', net_input[1, 0:1, :, :], iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model)))
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# Self-train：CMC 分支使用软权重掩码和软权重损失
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
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
    smooth_k = max(int(args.cmc_patch_size * args.cmc_soft_sigma) | 1, 1)
    logging.info("Start self_training (BCP + NA-CMC V2 Soft Mask)")
    logging.info("soft_sigma={}  →  avg_pool 核 {}px，过渡带宽约 {}px".format(
        args.cmc_soft_sigma, smooth_k, smooth_k // 2))
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

            # BCP 部分（完全不变）
            with torch.no_grad():
                pre_a  = ema_model(uimg_a)
                pre_b  = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l   = model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask,
                                         u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_bcp = ((unl_dice + l_dice) + (unl_ce + l_ce)) / 2

            # ------------------------------------------------------------------
            # NA-CMC V2 分支：软权重掩码 + 软权重损失
            # ------------------------------------------------------------------
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

            # ① 软权重掩码
            mask_a_ab, mask_b_ab = generate_cmc_masks_v2_soft(
                uimg_a, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)
            mask_a_cd, mask_b_cd = generate_cmc_masks_v2_soft(
                uimg_b, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)

            # ② 图像遮挡（用软掩码直接乘，视觉上是加权融合，非硬遮挡）
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

            # ③ 软权重损失（改动最大的地方）
            loss_cmc_a, la_anchor, la_mutual = cmc_mutual_loss_soft(
                out_a_viewA, out_a_viewB, plab_teacher_a, conf_mask_a,
                mask_a_ab, mask_b_ab, args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
            loss_cmc_b, lb_anchor, lb_mutual = cmc_mutual_loss_soft(
                out_b_viewC, out_b_viewD, plab_teacher_b, conf_mask_b,
                mask_a_cd, mask_b_cd, args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # 监控：掩码边界带宽度（有效过渡像素数占比）
            with torch.no_grad():
                ma = mask_a_ab.squeeze(1)
                # 过渡带定义：mask 值在 (0.1, 0.9) 之间的像素
                border_ratio = ((ma > 0.1) & (ma < 0.9)).float().mean().item()

            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss',        loss,         iter_num)
            writer.add_scalar('info/loss_bcp',          loss_bcp,     iter_num)
            writer.add_scalar('info/loss_cmc',          loss_cmc,     iter_num)
            writer.add_scalar('info/loss_anchor',       (la_anchor + lb_anchor) / 2, iter_num)
            writer.add_scalar('info/loss_mutual',       (la_mutual + lb_mutual) / 2, iter_num)
            writer.add_scalar('info/cmc_rampup',        cmc_rampup,   iter_num)
            writer.add_scalar('info/shared_ratio',      shared_ratio, iter_num)
            writer.add_scalar('info/conf_threshold',    current_conf_thresh, iter_num)
            # V2 新增监控：过渡带面积、掩码均值
            writer.add_scalar('na_cmc/soft_border_ratio', border_ratio, iter_num)
            writer.add_scalar('na_cmc/mask_mean', ma.mean().item(), iter_num)

            logging.info(
                'iter %d | loss=%.4f bcp=%.4f cmc=%.4f '
                '(anchor=%.4f mutual=%.4f) | border=%.3f conf_t=%.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 la_anchor.item(), la_mutual.item(), border_ratio, current_conf_thresh))

            if iter_num % 20 == 0:
                writer.add_image('train/Un_Image', net_input_unl[1, 0:1, :, :], iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                # 可视化软掩码（灰度图，0.0~1.0，边界处有渐变）
                writer.add_image('cmc/MaskA_soft', mask_a_ab[0], iter_num)
                writer.add_image('cmc/MaskB_soft', mask_b_ab[0], iter_num)
                writer.add_image('cmc/ViewA', uimg_a_viewA[0, 0:1], iter_num)
                writer.add_image('cmc/ViewB', uimg_a_viewB[0, 0:1], iter_num)
                pred_a_vis = torch.argmax(torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                pred_b_vis = torch.argmax(torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                writer.add_image('cmc/PredViewA', pred_a_vis[0].float() * 50, iter_num)
                writer.add_image('cmc/PredViewB', pred_b_vis[0].float() * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                                   iter_num, round(best_performance, 4))))
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model)))
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

    pre_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(__file__, self_snapshot_path)

    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
