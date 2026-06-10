"""
BCP + CMC v1 - NA-CMC V3：显式边界一致性损失
=============================================================
改动说明（相对于原始 BCP_CMC_v1_mutual.py）：
  ① generate_cmc_masks 不变（保持原始 nearest 插值）
  ② cmc_mutual_loss 新增 L_border 项：
     - 检测掩码边界区域（max_pool(mask_a) XOR mask_a）
     - 在边界区域：两视图的预测概率差异（KL 散度）最小化
     - 这直接约束：无论遮挡方式如何，边界两侧的预测应保持一致
  ③ 新增参数：--cmc_border_loss_weight（默认 1.5）

核心原理：
  L_border 不依赖教师标签，是纯粹的一致性约束。
  边界像素处：预测(视图A) ≈ 预测(视图B)，强制边界两侧语义连贯。

计算成本几乎为零（仅多一次 max_pool 和 KL div 计算）。
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
parser.add_argument('--exp', type=str, default='BCP_CMC_NA_v3')
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
# ---------- NA-CMC V3 新增参数 ----------
parser.add_argument('--cmc_border_loss_weight', type=float, default=1.5,
    help='边界一致性损失权重 lambda_border，推荐范围 [0.5, 3.0]。'
         '0 = 关闭（退化为原始 CMC），1.5 = 默认')
parser.add_argument('--cmc_border_kl_temp', type=float, default=1.0,
    help='KL 散度温度系数，>1 使概率分布更平滑，推荐 1.0（不改变）')
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

# 原始 CMC 掩码生成（不变）
def generate_cmc_masks(img, cmc_patch_size=16, shared_ratio=0.0):
    B, C, H, W = img.shape
    n = H // cmc_patch_size
    masks_a, masks_b = [], []
    for _ in range(B):
        base = (torch.rand(n, n) > 0.5).float()
        if shared_ratio > 0.0:
            shared = torch.rand(n, n) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa = (base == 0).float()
            pb = (base == 1).float()
        pa = F.interpolate(pa.view(1, 1, n, n), size=(H, W), mode='nearest').squeeze(0)
        pb = F.interpolate(pb.view(1, 1, n, n), size=(H, W), mode='nearest').squeeze(0)
        masks_a.append(pa)
        masks_b.append(pb)
    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device))

def get_progressive_shared_ratio(current_iter, warmup_iter, init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter, init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress


# ================================================================
# NA-CMC V3 核心：边界检测 + 显式边界一致性损失
# ================================================================
def get_patch_boundary_mask(mask_binary, patch_size=16):
    """
    检测 patch 边界区域：mask 内部变化的区域（patch 边界带）

    方法：在 patch 空间做 max_pool（膨胀）后与原始 mask 做 XOR，
    得到每个 patch 向外扩张 1 圈的边界带。

    等价原理：膨胀后 - 原始 = 边界扩张区域

    Args:
        mask_binary: [B, 1, H, W] float {0,1}，原始二值掩码
        patch_size  : patch 大小（像素）

    Returns:
        boundary: [B, H, W] float {0,1}，边界带（patch 边界处为 1）
    """
    # 在像素空间做 max_pool，核大小 = patch_size（覆盖整个 patch）
    # 这样每个 mask=1 的 patch 会向四周扩张 patch_size/2 像素
    k = patch_size
    pad = k // 2
    # 注意：偶数 kernel_size 时 max_pool2d 输出尺寸可能比输入多 1 像素
    # 先获取原始尺寸，再 crop 保证输出形状与输入一致
    H, W = mask_binary.shape[2], mask_binary.shape[3]
    mask_dilated = F.max_pool2d(mask_binary, kernel_size=k, stride=1, padding=pad)
    if mask_dilated.shape[2] != H or mask_dilated.shape[3] != W:
        mask_dilated = mask_dilated[..., :H, :W]
    # 边界带 = 膨胀后出现 1 但原始是 0 的像素（即 patch 周围一圈）
    boundary = (mask_dilated - mask_binary).clamp(0, 1)
    return boundary.squeeze(1)  # [B, H, W]


def cmc_mutual_loss_v3_border(out_viewA, out_viewB, plab_teacher,
                               conf_mask, mask_a, mask_b,
                               mutual_weight, border_loss_weight,
                               patch_size, kl_temp=1.0):
    """
    NA-CMC V3 损失：原始 CMC 损失 + 显式边界一致性损失

    L_total = L_anchor + λ_mutual * L_mutual + λ_border * L_border

    L_border 是纯一致性损失（不依赖教师标签）：
      在 mask 边界区域，要求两个视图的预测概率分布尽量一致。
      实现为对称 KL 散度（JSD 近似）:
        L_border = mean(KL(p_A || p_B) + KL(p_B || p_A)) * border_mask

    直觉：patch 边界两侧的像素，应该预测相同的类别，无论它被哪个视图遮挡。
         这直接约束了模型的边缘连贯性。

    Args:
        out_viewA/B        : [B, C, H, W] 模型输出 logit
        plab_teacher       : [B, H, W] long，教师硬标签
        conf_mask          : [B, H, W] float，教师置信度掩码
        mask_a/b           : [B, 1, H, W] float {0,1}，互补二值掩码
        mutual_weight      : 互教损失权重
        border_loss_weight : 边界一致性损失权重（新增）
        patch_size         : patch 大小（用于边界检测）
        kl_temp            : KL 散度温度（>1 使分布更平滑）
    """
    # ---- 原始 anchor 损失（完全不变）----
    w = conf_mask
    denom = w.sum() + 1e-6
    la = F.cross_entropy(out_viewA, plab_teacher, reduction='none')
    lb = F.cross_entropy(out_viewB, plab_teacher, reduction='none')
    loss_anchor = ((la + lb) * w).sum() / denom / 2.0

    # ---- 原始 mutual 损失（完全不变）----
    with torch.no_grad():
        prob_a = F.softmax(out_viewA, dim=1)
        prob_b = F.softmax(out_viewB, dim=1)
        conf_va = prob_a.max(dim=1).values
        conf_vb = prob_b.max(dim=1).values
        plab_va = prob_a.argmax(dim=1).long()
        plab_vb = prob_b.argmax(dim=1).long()
    excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))
    excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))
    w_b = excl_a * (conf_va > args.cmc_mutual_conf_thresh).float()
    w_a = excl_b * (conf_vb > args.cmc_mutual_conf_thresh).float()
    l_b_from_a = (F.cross_entropy(out_viewB, plab_va, reduction='none') * w_b
                  ).sum() / (w_b.sum() + 1e-6)
    l_a_from_b = (F.cross_entropy(out_viewA, plab_vb, reduction='none') * w_a
                  ).sum() / (w_a.sum() + 1e-6)
    loss_mutual = (l_b_from_a + l_a_from_b) / 2.0

    # ---- 新增：边界一致性损失 ----
    if border_loss_weight > 0:
        # Step 1：检测边界带（两个掩码各自的边界合并）
        # mask_a 的边界：A patch 向外扩张到 B 区域的部分
        # mask_b 的边界：B patch 向外扩张到 A 区域的部分
        border_a = get_patch_boundary_mask(mask_a, patch_size)  # [B,H,W]
        border_b = get_patch_boundary_mask(mask_b, patch_size)  # [B,H,W]
        # 合并：两个掩码的边界取并集
        border_mask = (border_a + border_b).clamp(0, 1)          # [B,H,W]

        # Step 2：计算两视图在边界处的预测概率
        # 注意：这里使用当前的 out_viewA/B，梯度会同时流向两个视图
        # 这是关键：两视图都会被约束在边界处保持一致
        prob_A_border = F.softmax(out_viewA / kl_temp, dim=1)    # [B,C,H,W]
        prob_B_border = F.softmax(out_viewB / kl_temp, dim=1)    # [B,C,H,W]

        # Step 3：对称 KL 散度（也叫 JSD 的无缩放版本）
        # KL(A||B) = sum(p_A * log(p_A/p_B))
        # 用 F.kl_div(log_B, p_A) = sum(p_A * (log p_A - log p_B))
        log_prob_A = torch.log(prob_A_border + 1e-8)             # [B,C,H,W]
        log_prob_B = torch.log(prob_B_border + 1e-8)             # [B,C,H,W]

        # KL(A→B)：用 B 的分布近似 A，梯度流向 A
        kl_a_to_b = F.kl_div(log_prob_B, prob_A_border.detach(),
                              reduction='none').sum(dim=1)         # [B,H,W]
        # KL(B→A)：用 A 的分布近似 B，梯度流向 B
        kl_b_to_a = F.kl_div(log_prob_A, prob_B_border.detach(),
                              reduction='none').sum(dim=1)         # [B,H,W]

        # 对称 KL（两个方向都计算，确保双向约束）
        sym_kl = (kl_a_to_b + kl_b_to_a) / 2.0                   # [B,H,W]

        # 只在边界带内计算
        border_count = border_mask.sum() + 1e-6
        loss_border = (sym_kl * border_mask).sum() / border_count
    else:
        loss_border = torch.tensor(0.0, device=out_viewA.device)
        border_mask = torch.zeros_like(conf_mask)

    # ---- 总损失 ----
    loss_total = loss_anchor + mutual_weight * loss_mutual + border_loss_weight * loss_border

    return loss_total, loss_anchor, loss_mutual, loss_border, border_mask


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
# Self-train
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
    logging.info("Start self_training (BCP + NA-CMC V3 Border Consistency Loss)")
    logging.info("border_loss_weight={}  kl_temp={}".format(
        args.cmc_border_loss_weight, args.cmc_border_kl_temp))
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
            # NA-CMC V3 分支：掩码生成不变，损失函数新增边界一致性项
            # ------------------------------------------------------------------
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

            # ① 掩码生成与原始完全一致
            mask_a_ab, mask_b_ab = generate_cmc_masks(uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks(uimg_b, args.cmc_patch_size, shared_ratio)

            uimg_a_viewA = uimg_a * mask_a_ab
            uimg_a_viewB = uimg_a * mask_b_ab
            uimg_b_viewC = uimg_b * mask_a_cd
            uimg_b_viewD = uimg_b * mask_b_cd

            # ② forward（与原始完全一致）
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

            # ③ 带边界一致性的损失（核心改动）
            loss_cmc_a, la_anchor, la_mutual, la_border, border_mask_a = \
                cmc_mutual_loss_v3_border(
                    out_a_viewA, out_a_viewB, plab_teacher_a, conf_mask_a,
                    mask_a_ab, mask_b_ab,
                    args.cmc_mutual_weight, args.cmc_border_loss_weight,
                    args.cmc_patch_size, args.cmc_border_kl_temp)

            loss_cmc_b, lb_anchor, lb_mutual, lb_border, border_mask_b = \
                cmc_mutual_loss_v3_border(
                    out_b_viewC, out_b_viewD, plab_teacher_b, conf_mask_b,
                    mask_a_cd, mask_b_cd,
                    args.cmc_mutual_weight, args.cmc_border_loss_weight,
                    args.cmc_patch_size, args.cmc_border_kl_temp)

            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # 监控：边界带覆盖率
            border_ratio = ((border_mask_a + border_mask_b) / 2).mean().item()

            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            # 日志
            loss_anchor = (la_anchor + lb_anchor) / 2
            loss_mutual = (la_mutual + lb_mutual) / 2
            loss_border = (la_border + lb_border) / 2
            writer.add_scalar('info/total_loss',        loss,         iter_num)
            writer.add_scalar('info/loss_bcp',          loss_bcp,     iter_num)
            writer.add_scalar('info/loss_cmc',          loss_cmc,     iter_num)
            writer.add_scalar('info/loss_anchor',       loss_anchor,  iter_num)
            writer.add_scalar('info/loss_mutual',       loss_mutual,  iter_num)
            writer.add_scalar('info/loss_border',       loss_border,  iter_num)   # 新增
            writer.add_scalar('info/cmc_rampup',        cmc_rampup,   iter_num)
            writer.add_scalar('info/shared_ratio',      shared_ratio, iter_num)
            writer.add_scalar('info/conf_threshold',    current_conf_thresh, iter_num)
            writer.add_scalar('na_cmc/border_ratio',    border_ratio, iter_num)   # 新增
            logging.info(
                'iter %d | loss=%.4f bcp=%.4f cmc=%.4f '
                '(anchor=%.4f mutual=%.4f border=%.4f) | '
                'border_ratio=%.3f conf_t=%.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 loss_anchor.item(), loss_mutual.item(), loss_border.item(),
                 border_ratio, current_conf_thresh))

            if iter_num % 20 == 0:
                writer.add_image('train/Un_Image', net_input_unl[1, 0:1, :, :], iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                writer.add_image('cmc/MaskA',       mask_a_ab[0],    iter_num)
                writer.add_image('cmc/MaskB',       mask_b_ab[0],    iter_num)
                # 可视化边界带（白色区域即为 KL 损失作用的区域）
                writer.add_image('cmc/BorderZone',  border_mask_a[0:1].float(), iter_num)
                writer.add_image('cmc/ViewA',  uimg_a_viewA[0, 0:1], iter_num)
                writer.add_image('cmc/ViewB',  uimg_a_viewB[0, 0:1], iter_num)
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
