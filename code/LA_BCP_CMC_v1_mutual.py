"""
LA BCP + CMC v1：互补掩码互教一致性（3D 版本）

基于原始 LA_BCP_train.py，完整保留所有 BCP 函数，
仅在 self_train 末尾追加 CMC v1 互教分支。

路径修正：
  root_path : /data/byh_data/SSNet_data/LA  →  ../data_split/LA
  shutil.copy('../code/LA_BCP_train.py', ...)  →  shutil.copy(__file__, ...)

3D 适配说明：
  ACDC 是 2D [B,C,H,W]，LA 是 3D [B,C,H,W,D]
  generate_cmc_masks_3d 使用 3D 最近邻插值生成体素级互补掩码
  模型输出为 (output, features) 元组，CMC 分支只取 output
  伪标签为二值 [B,H,W,D]，对应 2 类 LA 分割任务
"""
from asyncore import write
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage.measure import label
from torch.utils.data import DataLoader

from utils import losses, ramps, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, update_ema_variables

# ================================================================
# 参数（原始 LA BCP 参数完全保留，追加 CMC v1 专用参数）
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data_split/LA',         # ← 路径已修正
                    help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP_CMC_v1_mutual')
parser.add_argument('--model', type=str, default='VNet')
parser.add_argument('--pre_max_iteration', type=int,  default=2000)
parser.add_argument('--self_max_iteration', type=int,  default=15000)
parser.add_argument('--max_samples', type=int,  default=80)
parser.add_argument('--labeled_bs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--base_lr', type=float,  default=0.01)
parser.add_argument('--deterministic', type=int,  default=1)
parser.add_argument('--labelnum', type=int,  default=8)
parser.add_argument('--gpu', type=str,  default='1')
parser.add_argument('--seed', type=int,  default=1337)
parser.add_argument('--consistency', type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
parser.add_argument('--magnitude', type=float,  default=10.0)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--mask_ratio', type=float, default=2/3)
parser.add_argument('--u_alpha', type=float, default=2.0)
parser.add_argument('--loss_weight', type=float, default=0.5)
# ---------- CMC v1 专用参数 ----------
parser.add_argument('--cmc_patch_size',        type=int,   default=16,
                    help='3D CMC 网格块大小（x/y/z 方向统一），需整除 H/W/D')
parser.add_argument('--cmc_warmup_iter',       type=int,   default=2000,
                    help='CMC shared_ratio 退火步数（与 self_max_iteration 匹配）')
parser.add_argument('--cmc_init_shared',       type=float, default=0.4,
                    help='热身初期两视图共享块比例')
parser.add_argument('--cmc_loss_weight',       type=float, default=1.0,
                    help='CMC 总损失乘子')
parser.add_argument('--cmc_mutual_weight',     type=float, default=0.5,
                    help='互教损失权重：L_CMC = L_anchor + λ * L_mutual')
parser.add_argument('--cmc_mutual_conf_thresh',type=float, default=0.75,
                    help='互教最低置信度门限')
parser.add_argument('--conf_thresh_init',      type=float, default=0.90,
                    help='教师置信度阈值初始值（保守）')
parser.add_argument('--conf_thresh_final',     type=float, default=0.70,
                    help='教师置信度阈值最终值（宽松）')
args = parser.parse_args()

# ================================================================
# 全局设置（与原始 LA BCP 完全一致）
# ================================================================
train_data_path = args.root_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations  = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size  = (112, 112, 80)
num_classes = 2

# ================================================================
# 原始 LA BCP 函数（逐字复制，未做任何修改）
# ================================================================
def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# ================================================================
# CMC v1 新增辅助函数（3D 适配）
# ================================================================
def generate_cmc_masks_3d(img, cmc_patch_size=16, shared_ratio=0.0):
    """
    生成 3D 互补网格掩码对

    将 [B, C, H, W, D] 体积划分为 cmc_patch_size^3 的均匀网格块，
    随机将每块分配给视图A或视图B。

    Args:
        img            : [B, C, H, W, D] 输入体积
        cmc_patch_size : 网格块大小，需同时整除 H, W, D
        shared_ratio   : 共享块比例（热身阶段用，0=纯互补）

    Returns:
        mask_a : [B, 1, H, W, D]，值域 {0,1}
        mask_b : [B, 1, H, W, D]，值域 {0,1}
        shared_ratio=0 时 mask_a + mask_b = 1（逐体素）
    """
    B, C, H, W, D = img.shape
    p = cmc_patch_size
    assert H % p == 0 and W % p == 0 and D % p == 0, \
        f"H={H}, W={W}, D={D} 必须均能被 cmc_patch_size={p} 整除"
    n_h, n_w, n_d = H // p, W // p, D // p
    masks_a, masks_b = [], []
    for _ in range(B):
        base = (torch.rand(n_h, n_w, n_d) > 0.5).float()
        if shared_ratio > 0.0:
            shared = torch.rand(n_h, n_w, n_d) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa = (base == 0).float()
            pb = (base == 1).float()
        # 3D 最近邻插值到完整尺寸 [1, 1, n_h, n_w, n_d] → [1, H, W, D]
        pa = F.interpolate(
            pa.view(1, 1, n_h, n_w, n_d),
            size=(H, W, D), mode='nearest'
        ).squeeze(0)   # [1, H, W, D]
        pb = F.interpolate(
            pb.view(1, 1, n_h, n_w, n_d),
            size=(H, W, D), mode='nearest'
        ).squeeze(0)
        masks_a.append(pa)
        masks_b.append(pb)
    return (torch.stack(masks_a).to(img.device),   # [B, 1, H, W, D]
            torch.stack(masks_b).to(img.device))

def get_progressive_shared_ratio(current_iter, warmup_iter,
                                  init_ratio=0.4, final_ratio=0.0):
    """shared_ratio 线性退火：init_ratio → final_ratio"""
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter,
                            init_threshold=0.90, final_threshold=0.70):
    """置信度阈值随训练线性降低"""
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

# ================================================================
# Pre-train（与原始 LA BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs   = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size,
                                           args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs]
            label_batch  = sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs],  label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch  = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce   = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce',   loss_ce,   iter_num)
            writer.add_scalar('pre/loss_all',  loss,      iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' %
                         (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice',      dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice,   iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()

# ================================================================
# Self-train：BCP 部分与原始完全一致，追加 CMC v1 互教分支
# ================================================================
def self_train(args, pre_snapshot_path, self_snapshot_path):
    model     = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs   = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size,
                                           args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path,
                                    '{}_best_model.pth'.format(args.model))
    load_net(model,     pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num  = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a,   img_b   = volume_batch[:sub_bs],            volume_batch[sub_bs:args.labeled_bs]
            lab_a,   lab_b   = label_batch[:sub_bs],             label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = (volume_batch[args.labeled_bs:args.labeled_bs + sub_bs],
                                volume_batch[args.labeled_bs + sub_bs:])

            # ==============================================================
            # BCP 部分（与原始 LA BCP 完全一致）
            # ==============================================================
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a   * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b   * (1 - img_mask)
            mixl_lab = lab_a   * img_mask + plab_a  * (1 - img_mask)
            mixu_lab = plab_b  * img_mask + lab_b   * (1 - img_mask)

            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            loss_l = mix_loss(outputs_l, lab_a,  plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b,  loss_mask,
                              u_weight=args.u_weight, unlab=True)
            loss_bcp = loss_l + loss_u

            # ==============================================================
            # CMC v1 互教分支（新增）
            #
            # 与 ACDC 版的差异：
            #   1. 掩码是 3D：[B, 1, H, W, D]
            #   2. 模型输出需解包：out, _ = model(input)
            #   3. 伪标签是二值：get_cut_mask 返回 [B, H, W, D] LongTensor
            #   4. 教师置信度取 class-1 概率（二值分割）
            #
            # 损失结构（与 ACDC v1 相同）：
            #   L_CMC = L_anchor + λ * L_mutual
            #   L_anchor  : 两视图分别对齐教师硬标签（置信度加权）
            #   L_mutual  : A 的高置信预测监督 B 的盲区，反之亦然
            # ==============================================================
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, self_max_iterations,
                args.conf_thresh_init, args.conf_thresh_final)

            # 1) 生成 3D 互补掩码，构建两视图（unimg_a 和 unimg_b 各一对）
            mask_a_ab, mask_b_ab = generate_cmc_masks_3d(
                unimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d(
                unimg_b, args.cmc_patch_size, shared_ratio)

            uimg_a_viewA = unimg_a * mask_a_ab   # [B, 1, H, W, D]
            uimg_a_viewB = unimg_a * mask_b_ab
            uimg_b_viewC = unimg_b * mask_a_cd
            uimg_b_viewD = unimg_b * mask_b_cd

            # 2) 批量 forward（两视图 concat 后一次前向，节省 kernel launch）
            out_ab_all, _ = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
            out_cd_all, _ = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))
            out_a_viewA = out_ab_all[:sub_bs]
            out_a_viewB = out_ab_all[sub_bs:]
            out_b_viewC = out_cd_all[:sub_bs]
            out_b_viewD = out_cd_all[sub_bs:]

            # 3) 教师置信度掩码（复用已有 unoutput_a/b，无额外 EMA 调用）
            #    LA 为二值分割，取 class-1 的 softmax 概率作为置信度
            with torch.no_grad():
                # teacher_hard：[B, H, W, D] LongTensor，与 BCP 的 plab_a/plab_b 相同
                teacher_hard_a = plab_a.long()
                teacher_hard_b = plab_b.long()
                # 置信度：max(p_0, p_1)，高于阈值才参与损失计算
                conf_a = F.softmax(unoutput_a, dim=1).max(dim=1).values  # [B, H, W, D]
                conf_b = F.softmax(unoutput_b, dim=1).max(dim=1).values
                conf_mask_a = (conf_a > current_conf_thresh).float()
                conf_mask_b = (conf_b > current_conf_thresh).float()

            def cmc_mutual_loss_3d(out_vA, out_vB,
                                   teacher_hard, conf_mask,
                                   mask_a, mask_b):
                """
                3D CMC v1：锚点损失 + 互补互教损失

                Args:
                    out_vA/vB    : [B, C, H, W, D]  学生两视图的原始 logit
                    teacher_hard : [B, H, W, D]      教师硬标签（Long）
                    conf_mask    : [B, H, W, D]      教师置信度二值掩码（Float）
                    mask_a/b     : [B, 1, H, W, D]   互补可见掩码
                """
                w     = conf_mask                        # [B, H, W, D]
                denom = w.sum() + 1e-6

                # L_anchor：两视图均对齐教师硬标签
                la = F.cross_entropy(out_vA, teacher_hard, reduction='none')  # [B,H,W,D]
                lb = F.cross_entropy(out_vB, teacher_hard, reduction='none')
                loss_anchor = ((la + lb) * w).sum() / denom / 2.0

                # L_mutual：盲区互教（梯度只流向被监督方）
                with torch.no_grad():
                    prob_a = F.softmax(out_vA, dim=1)
                    prob_b = F.softmax(out_vB, dim=1)
                    conf_va = prob_a.max(dim=1).values          # [B, H, W, D]
                    conf_vb = prob_b.max(dim=1).values
                    # 硬标签：对 2 类取 argmax
                    plab_va = prob_a.argmax(dim=1).long()       # [B, H, W, D]
                    plab_vb = prob_b.argmax(dim=1).long()

                # excl_a：仅 A 可见的区域（B 的盲区）
                excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))  # [B,H,W,D]
                excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))

                # A → B：用 A 高置信预测监督 B 的盲区
                w_b = excl_a * (conf_va > args.cmc_mutual_conf_thresh).float()
                l_b_from_a = (
                    F.cross_entropy(out_vB, plab_va, reduction='none') * w_b
                ).sum() / (w_b.sum() + 1e-6)

                # B → A：用 B 高置信预测监督 A 的盲区
                w_a = excl_b * (conf_vb > args.cmc_mutual_conf_thresh).float()
                l_a_from_b = (
                    F.cross_entropy(out_vA, plab_vb, reduction='none') * w_a
                ).sum() / (w_a.sum() + 1e-6)

                loss_mutual = (l_b_from_a + l_a_from_b) / 2.0
                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            loss_cmc_a = cmc_mutual_loss_3d(
                out_a_viewA, out_a_viewB,
                teacher_hard_a, conf_mask_a, mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss_3d(
                out_b_viewC, out_b_viewD,
                teacher_hard_b, conf_mask_b, mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ==============================================================
            # 总损失
            # ==============================================================
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            iter_num += 1
            writer.add_scalar('Self/consistency',   consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l',        loss_l,             iter_num)
            writer.add_scalar('Self/loss_u',        loss_u,             iter_num)
            writer.add_scalar('Self/loss_bcp',      loss_bcp,           iter_num)
            writer.add_scalar('Self/loss_cmc',      loss_cmc,           iter_num)
            writer.add_scalar('Self/loss_all',      loss,               iter_num)
            writer.add_scalar('Self/cmc_rampup',    cmc_rampup,         iter_num)
            writer.add_scalar('Self/shared_ratio',  shared_ratio,       iter_num)
            writer.add_scalar('Self/conf_threshold', current_conf_thresh, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, bcp: %03f, cmc: %03f, '
                'shared: %.2f, conf_t: %.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 shared_ratio, current_conf_thresh))

            update_ema_variables(model, ema_model, 0.99)

            # LR 阶梯衰减（与原始 LA BCP 完全一致）
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,
                        '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice',      dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice,   iter_num)
                model.train()

            # 可视化（与原始 LA BCP 完全一致）
            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = outputs_l.size()
                snapshot_img = torch.zeros(
                    size=(D, 3, 3 * H + 3 * ins_width, W + ins_width),
                    dtype=torch.float32)
                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2*H + ins_width:2*H + 2*ins_width, :] = 1
                snapshot_img[:, :, 3*H + 2*ins_width:3*H + 3*ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1
                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out   = outputs_l_soft[0, 1, ...].permute(2, 0, 1)
                target    = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)
                snapshot_img[:, 0, :H, :W] = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)
                snapshot_img[:, 1, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 2, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 0, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 0, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                snapshot_img[:, 1, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                snapshot_img[:, 2, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out   = outputs_u_soft[0, 1, ...].permute(2, 0, 1)
                target    = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)
                snapshot_img[:, 0, :H, :W] = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-8)
                snapshot_img[:, 1, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 2, :H, :W] = snapshot_img[:, 0, :H, :W]
                snapshot_img[:, 0, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2*H + ins_width, :W] = target
                snapshot_img[:, 0, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                snapshot_img[:, 1, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                snapshot_img[:, 2, 2*H + 2*ins_width:3*H + 2*ins_width, :W] = seg_out
                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break
        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()

# ================================================================
# 主入口（路径已修正）
# ================================================================
if __name__ == "__main__":
    pre_snapshot_path  = "./model/BCP/LA_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum)
    print("Starting LA BCP + CMC v1 training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            import shutil
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(__file__, self_snapshot_path)   # ← 路径已修正

    # Pre-train
    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self-train
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
