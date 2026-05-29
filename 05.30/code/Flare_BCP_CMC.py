"""
Flare BCP + CMC v1：互补掩码互教一致性（3D，14类）

数据集  : Flare（BraTS2019 Loader，读 {root}/{case}/2022.h5）
网络    : guidedNet（flare_networks/net_factory_3d.py），返回 {"pred", "rep"}
工具    : flare_utils/（tools, loss, losses, ramps）
CMC逻辑 : 与 LA_BCP_CMC_v1_mutual.py 完全一致，适配14类
"""
import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import h5py

# ---------- 数据集 ----------
from dataloaders.brats2019 import (
    BraTS2019, RandomCrop, RandomRotFlip, ToTensor,
    TwoStreamBatchSampler, RandomNoise
)
# ---------- 网络（flare 专用） ----------
from flare_networks.net_factory_3d import net_factory_3d
# ---------- 工具（flare 专用） ----------
from flare_utils import losses as flare_losses
from flare_utils import ramps as flare_ramps
from flare_utils import tools
from flare_utils.loss import RobustCrossEntropyLoss

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data_split/flare',
                    help='Flare 数据集根目录，下含 train.txt val.txt 和各 case 子目录')
parser.add_argument('--exp',   type=str,  default='BCP_CMC_v1_mutual_flare')
parser.add_argument('--model', type=str,  default='guidedNet')
# -- 迭代
parser.add_argument('--max_iterations',    type=int,   default=30000)
# -- 批次
parser.add_argument('--batch_size',        type=int,   default=4)
parser.add_argument('--labeled_bs',        type=int,   default=2)
# -- 数据量
parser.add_argument('--labeled_num',       type=int,   default=21,
                    help='有标签样本数')
parser.add_argument('--data_num',          type=int,   default=378,
                    help='总样本数')
# -- 训练超参
parser.add_argument('--base_lr',           type=float, default=0.1)
parser.add_argument('--patch_size',        type=list,  default=[64, 128, 128])
parser.add_argument('--seed',              type=int,   default=1337)
parser.add_argument('--deterministic',     type=int,   default=1)
parser.add_argument('--gpu',               type=str,   default='0')
# -- BCP / 一致性
parser.add_argument('--consistency',       type=float, default=0.1)
parser.add_argument('--consistency_rampup',type=float, default=200.0)
parser.add_argument('--ema_decay',         type=float, default=0.99)
parser.add_argument('--u_weight',          type=float, default=0.5)
parser.add_argument('--mask_ratio',        type=float, default=2/3)
# -- KTCPS
parser.add_argument('--lanmuda',           type=float, default=1.0)
# -- CMC v1 专用
parser.add_argument('--cmc_patch_size',         type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=5000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size  = tuple(args.patch_size)   # (64, 128, 128)
num_classes = 14

# ================================================================
# 工具函数
# ================================================================
def EMA(cur_weight, past_weight, momentum=0.9):
    return momentum * past_weight + (1 - momentum) * cur_weight

def get_current_consistency_weight(epoch):
    return args.consistency * flare_ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

# ================================================================
# KTCPS（与原始 flare 脚本完全一致）
# ================================================================
class KTCPS:
    def __init__(self, num_cls, momentum=0.95):
        self.num_cls   = num_cls
        self.momentum  = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P     = (num_each_class.max() + 1e-8) / (num_each_class + 1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, trainloader):
        num_each_class = np.zeros(self.num_cls)
        ids_list = trainloader.dataset.image_list[:42]
        for data_id in ids_list:
            h5f   = h5py.File(
                os.path.join(args.root_path, data_id, '2022.h5'), 'r')
            label = h5f['label'][:]
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights      = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label, label):
        pseudo_label  = torch.argmax(
            pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy   = pseudo_label.data.cpu().numpy()
        gt_numpy      = label.data.cpu().numpy()
        label_numpy   = np.squeeze(label_numpy, axis=1)
        mask          = (label_numpy == gt_numpy)
        label_numpy   = np.where(mask, label_numpy, 0)
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            tmp, _ = np.histogram(
                label_numpy[i].reshape(-1), range(self.num_cls + 1))
            num_each_class += tmp
        cur_weights  = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights

# ================================================================
# BCP 3D 掩码（随机长方体，mask_ratio 控制遮挡比例）
# ================================================================
def generate_mask_3d(img):
    """
    img: [B, 1, D, H, W]
    返回 mask [D,H,W] long，loss_mask [B,D,H,W] long
    """
    B, C, D, H, W = img.shape
    loss_mask = torch.ones(B, D, H, W).cuda()
    mask      = torch.ones(D, H, W).cuda()
    pd = int(D * args.mask_ratio)
    ph = int(H * args.mask_ratio)
    pw = int(W * args.mask_ratio)
    d0 = np.random.randint(0, max(D - pd, 1))
    h0 = np.random.randint(0, max(H - ph, 1))
    w0 = np.random.randint(0, max(W - pw, 1))
    mask     [d0:d0+pd, h0:h0+ph, w0:w0+pw] = 0
    loss_mask[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] = 0
    return mask.long(), loss_mask.long()

# ================================================================
# 3D Dice Loss（内联，不依赖 losses.py 的 2D repeat 逻辑）
# ================================================================
def dice_loss_3d(score, target, mask=None):
    """
    score  : [B, C, D, H, W]  softmax 概率
    target : [B, 1, D, H, W]  long
    mask   : [B, 1, D, H, W]  float（可为 None）
    """
    B, C, D, H, W = score.shape
    target_onehot = torch.zeros_like(score)
    target_onehot.scatter_(1, target.long(), 1)
    if mask is not None:
        score         = score         * mask.float()
        target_onehot = target_onehot * mask.float()
    intersect = (score * target_onehot).sum(dim=(2, 3, 4))
    denom     = (score + target_onehot).sum(dim=(2, 3, 4))
    dice      = (2.0 * intersect + 1e-5) / (denom + 1e-5)
    return 1.0 - dice.mean()

def mix_loss_3d(output, img_l, patch_l, mask,
                l_weight=1.0, u_weight=0.5, unlab=False):
    """
    output  : [B, C, D, H, W]
    img_l / patch_l : [B, D, H, W] long
    mask    : [B, D, H, W] long
    """
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l   = img_l.long()
    patch_l = patch_l.long()
    output_soft  = F.softmax(output, dim=1)
    image_weight, patch_weight = (u_weight, l_weight) if unlab else (l_weight, u_weight)
    patch_mask = 1 - mask

    loss_dice  = dice_loss_3d(output_soft, img_l.unsqueeze(1),
                              mask.unsqueeze(1))   * image_weight
    loss_dice += dice_loss_3d(output_soft, patch_l.unsqueeze(1),
                              patch_mask.unsqueeze(1)) * patch_weight

    loss_ce    = image_weight  * (CE(output, img_l)   * mask      ).sum() \
                               / (mask.sum()       + 1e-16)
    loss_ce   += patch_weight  * (CE(output, patch_l) * patch_mask).sum() \
                               / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce

def get_flare_masks(output):
    """14 类 argmax → [B, D, H, W] LongTensor"""
    return torch.argmax(F.softmax(output, dim=1), dim=1)

# ================================================================
# CMC v1：3D 互补掩码
# ================================================================
def generate_cmc_masks_3d(img, cmc_patch_size=16, shared_ratio=0.0):
    """
    img: [B, 1, D, H, W]
    返回 mask_a, mask_b: [B, 1, D, H, W] float {0,1}
    在 H×W 平面生成互补网格，沿 D 复制
    """
    B, C, D, H, W = img.shape
    p   = cmc_patch_size
    n_h = max(H // p, 1)
    n_w = max(W // p, 1)
    masks_a, masks_b = [], []
    for _ in range(B):
        base = (torch.rand(n_h, n_w) > 0.5).float()
        if shared_ratio > 0.0:
            shared = torch.rand(n_h, n_w) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa = (base == 0).float()
            pb = (base == 1).float()
        # 上采样到 H×W，再沿 D 扩展
        pa = F.interpolate(
            pa.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')  # [1,1,H,W]
        pb = F.interpolate(
            pb.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')
        pa = pa.unsqueeze(2).expand(1, 1, D, H, W)   # [1,1,D,H,W]
        pb = pb.unsqueeze(2).expand(1, 1, D, H, W)
        masks_a.append(pa)
        masks_b.append(pb)
    return (torch.cat(masks_a, dim=0).to(img.device),
            torch.cat(masks_b, dim=0).to(img.device))

def get_progressive_shared_ratio(current_iter, warmup_iter,
                                  init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter,
                            init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

# ================================================================
# 主训练函数
# ================================================================
def train(args, snapshot_path):
    base_lr      = args.base_lr
    batch_size   = args.batch_size
    max_iterations = args.max_iterations

    # -- 网络（两个独立模型，不同初始化）
    net1 = net_factory_3d(
        net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    net2 = net_factory_3d(
        net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1 = nn.DataParallel(model1)
    model2 = nn.DataParallel(model2)
    model1.train()
    model2.train()

    # -- 数据
    db_train = BraTS2019(
        base_dir=args.root_path,
        split='train',
        num=None,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            RandomNoise(),
            ToTensor(),
        ])
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs   = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.data_num))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        batch_size, batch_size - args.labeled_bs)
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # -- KTCPS 权重初始化
    ktcps    = KTCPS(num_classes, momentum=0.99)
    weight_A = ktcps.init_weights(trainloader)
    weight_B = ktcps.init_weights(trainloader)

    iter_num  = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator  = tqdm(range(max_epoch), ncols=70)
    start_time = time.time()

    labeled_sub_bs   = args.labeled_bs // 2
    unlabeled_sub_bs = (batch_size - args.labeled_bs) // 2

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()   # [B,1,D,H,W]
            label_batch  = sampled_batch['label'].cuda()   # [B,D,H,W]

            # ── 前向（guidedNet 返回字典） ──────────────────────────────
            out1      = model1(volume_batch)
            outputs1  = out1["pred"]
            rep1      = out1["rep"]
            soft1     = torch.softmax(outputs1, dim=1)

            out2      = model2(volume_batch)
            outputs2  = out2["pred"]
            rep2      = out2["rep"]
            soft2     = torch.softmax(outputs2, dim=1)

            # ── 有监督损失（CE + Dice，仅 labeled 部分） ─────────────────
            sup_loss1 = 0.5 * (
                ce_loss(outputs1[:args.labeled_bs],
                        label_batch[:args.labeled_bs].long()) +
                dice_loss_3d(soft1[:args.labeled_bs],
                             label_batch[:args.labeled_bs].unsqueeze(1))
            )
            sup_loss2 = 0.5 * (
                ce_loss(outputs2[:args.labeled_bs],
                        label_batch[:args.labeled_bs].long()) +
                dice_loss_3d(soft2[:args.labeled_bs],
                             label_batch[:args.labeled_bs].unsqueeze(1))
            )

            # ── GMM（与原始 flare 脚本完全一致） ─────────────────────────
            feat1      = rep1[:args.labeled_bs]
            mask1_gt   = label_batch[:args.labeled_bs]
            cls_label1 = torch.stack(
                [torch.arange(num_classes)] * args.labeled_bs).cuda()
            cur_cls1   = tools.build_cur_cls_label(mask1_gt, num_classes)
            pred_cl1   = tools.clean_mask(outputs1[:args.labeled_bs], cls_label1, True)
            vecs1, proto_loss1 = tools.cal_protypes(feat1, mask1_gt, num_classes)
            res1       = tools.GMM(feat1, vecs1, pred_cl1, mask1_gt, cur_cls1)
            gmm_loss1  = tools.cal_gmm_loss(
                soft1[:args.labeled_bs], res1, cur_cls1, mask1_gt) + proto_loss1

            feat1_u    = rep1[args.labeled_bs:]
            pred_cl1_u = tools.clean_mask_predict(outputs1[args.labeled_bs:], True)
            res1_u     = tools.GMM_predict(feat1_u, vecs1, pred_cl1_u)
            gmm_loss1_u = tools.gmm_loss(soft1[args.labeled_bs:], res1_u, cur_cls1)

            feat2      = rep2[:args.labeled_bs]
            mask2_gt   = label_batch[:args.labeled_bs]
            cls_label2 = torch.stack(
                [torch.arange(num_classes)] * args.labeled_bs).cuda()
            cur_cls2   = tools.build_cur_cls_label(mask2_gt, num_classes)
            pred_cl2   = tools.clean_mask(outputs2[:args.labeled_bs], cls_label2, True)
            vecs2, proto_loss2 = tools.cal_protypes(feat2, mask2_gt, num_classes)
            res2       = tools.GMM(feat2, vecs2, pred_cl2, mask2_gt, cur_cls2)
            gmm_loss2  = tools.cal_gmm_loss(
                soft2[:args.labeled_bs], res2, cur_cls2, mask2_gt) + proto_loss2

            feat2_u    = rep2[args.labeled_bs:]
            pred_cl2_u = tools.clean_mask_predict(outputs2[args.labeled_bs:], True)
            res2_u     = tools.GMM_predict(feat2_u, vecs2, pred_cl2_u)
            gmm_loss2_u = tools.gmm_loss(soft2[args.labeled_bs:], res2_u, cur_cls2)

            # ── GMM 一致性损失 ────────────────────────────────────────────
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss   = consistency_weight * torch.mean(
                (torch.softmax(res1, dim=1) - torch.softmax(res2, dim=1)) ** 2)

            # ── KTCPS ────────────────────────────────────────────────────
            weight_A = ktcps.get_ema_weights(
                outputs1[:args.labeled_bs].detach(),
                label_batch[:args.labeled_bs].detach())
            weight_B = ktcps.get_ema_weights(
                outputs2[:args.labeled_bs].detach(),
                label_batch[:args.labeled_bs].detach())
            weight_A = weight_A.cpu().numpy()
            weight_B = weight_B.cpu().numpy()

            unsup_A = RobustCrossEntropyLoss(weight=weight_A)
            unsup_B = RobustCrossEntropyLoss(weight=weight_B)
            max_A   = torch.argmax(outputs1.detach(), dim=1, keepdim=True).long()
            max_B   = torch.argmax(outputs2.detach(), dim=1, keepdim=True).long()
            loss_cps = unsup_B(outputs1, max_B) + unsup_A(outputs2, max_A)

            # ── CMC v1 互教分支 ───────────────────────────────────────────
            shared_ratio        = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations,
                args.conf_thresh_init, args.conf_thresh_final)

            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]

            with torch.no_grad():
                pre_a = model1(uimg_a)["pred"]
                pre_b = model2(uimg_b)["pred"]
                plab_teacher_a = get_flare_masks(pre_a).long()
                plab_teacher_b = get_flare_masks(pre_b).long()
                conf_a = F.softmax(pre_a, dim=1).max(dim=1).values
                conf_b = F.softmax(pre_b, dim=1).max(dim=1).values
                conf_mask_a = (conf_a > current_conf_thresh).float()
                conf_mask_b = (conf_b > current_conf_thresh).float()

            mask_a_ab, mask_b_ab = generate_cmc_masks_3d(
                uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d(
                uimg_b, args.cmc_patch_size, shared_ratio)

            out_ab_A = model1(uimg_a * mask_a_ab)["pred"]
            out_ab_B = model1(uimg_a * mask_b_ab)["pred"]
            out_cd_C = model2(uimg_b * mask_a_cd)["pred"]
            out_cd_D = model2(uimg_b * mask_b_cd)["pred"]

            def cmc_mutual_loss_3d(out_A, out_B,
                                   teacher_hard, conf_mask,
                                   mask_a, mask_b):
                CE_none = nn.CrossEntropyLoss(reduction='none')
                w     = conf_mask
                denom = w.sum() + 1e-6
                la = CE_none(out_A, teacher_hard)
                lb = CE_none(out_B, teacher_hard)
                loss_anchor = ((la + lb) * w).sum() / denom / 2.0

                with torch.no_grad():
                    prob_A  = F.softmax(out_A, dim=1)
                    prob_B  = F.softmax(out_B, dim=1)
                    conf_vA = prob_A.max(dim=1).values
                    conf_vB = prob_B.max(dim=1).values
                    plab_vA = prob_A.argmax(dim=1).long()
                    plab_vB = prob_B.argmax(dim=1).long()

                excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))
                excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))
                w_b = excl_a * (conf_vA > args.cmc_mutual_conf_thresh).float()
                w_a = excl_b * (conf_vB > args.cmc_mutual_conf_thresh).float()
                l_b_from_a = (CE_none(out_B, plab_vA) * w_b
                              ).sum() / (w_b.sum() + 1e-6)
                l_a_from_b = (CE_none(out_A, plab_vB) * w_a
                              ).sum() / (w_a.sum() + 1e-6)
                loss_mutual = (l_b_from_a + l_a_from_b) / 2.0
                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            loss_cmc_a = cmc_mutual_loss_3d(
                out_ab_A, out_ab_B,
                plab_teacher_a, conf_mask_a, mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss_3d(
                out_cd_C, out_cd_D,
                plab_teacher_b, conf_mask_b, mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ── 总损失 ────────────────────────────────────────────────────
            cmc_rampup  = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            model1_loss = sup_loss1 + args.lanmuda * (
                gmm_loss1 + gmm_loss1_u + consistency_loss)
            model2_loss = sup_loss2 + args.lanmuda * (
                gmm_loss2 + gmm_loss2_u + consistency_loss)
            loss = (model1_loss + model2_loss
                    + consistency_weight * loss_cps
                    + args.cmc_loss_weight * cmc_rampup * loss_cmc)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            iter_num += 1

            # LR poly 衰减
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer1.param_groups:
                pg['lr'] = lr_
            for pg in optimizer2.param_groups:
                pg['lr'] = lr_

            # TensorBoard
            writer.add_scalar('lr',                  lr_,          iter_num)
            writer.add_scalar('loss/total',          loss,         iter_num)
            writer.add_scalar('loss/model1',         model1_loss,  iter_num)
            writer.add_scalar('loss/model2',         model2_loss,  iter_num)
            writer.add_scalar('loss/gmm1',           gmm_loss1,    iter_num)
            writer.add_scalar('loss/gmm2',           gmm_loss2,    iter_num)
            writer.add_scalar('loss/gmm1_u',         gmm_loss1_u,  iter_num)
            writer.add_scalar('loss/gmm2_u',         gmm_loss2_u,  iter_num)
            writer.add_scalar('loss/cps',            loss_cps,     iter_num)
            writer.add_scalar('loss/cmc',            loss_cmc,     iter_num)
            writer.add_scalar('cmc/rampup',          cmc_rampup,   iter_num)
            writer.add_scalar('cmc/shared_ratio',    shared_ratio, iter_num)
            writer.add_scalar('cmc/conf_thresh',     current_conf_thresh, iter_num)
            writer.add_scalar('consistency/weight',  consistency_weight,  iter_num)

            logging.info(
                'iter %d : loss=%.4f m1=%.4f m2=%.4f '
                'cps=%.4f cmc=%.4f shared=%.2f conf=%.2f' % (
                    iter_num, loss.item(),
                    model1_loss.item(), model2_loss.item(),
                    loss_cps.item(),    loss_cmc.item(),
                    shared_ratio,       current_conf_thresh))

            # 模型保存（后半段每 1000 步）
            if iter_num > max_iterations * 0.5 and iter_num % 1000 == 0:
                torch.save(model1.state_dict(),
                    os.path.join(snapshot_path,
                        'model1_iter_{}.pth'.format(iter_num)))
                torch.save(model2.state_dict(),
                    os.path.join(snapshot_path,
                        'model2_iter_{}.pth'.format(iter_num)))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    total_time = time.time() - start_time
    print(f"Training finished. Total time: {total_time:.1f}s")

# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark   = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark   = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/flare/{}_{}".format(args.exp, args.model)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)