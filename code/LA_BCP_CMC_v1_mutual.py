"""
BCP + CMC v1：互补掩码互教一致性
数据集：LA（3D）
网络：VNet（输出单个 tensor，非字典）
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
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

# ---------- 数据集（LA 专用）----------
from dataloaders.dataset import (
    LAHeart, RandomRotFlip, RandomCrop, ToTensor, TwoStreamBatchSampler
)
# ---------- 网络 ----------
from networks.net_factory import net_factory
# ---------- 工具 ----------
from utils import losses, ramps, test_3d_patch
from utils.BCP_utils import context_mask, mix_loss, update_ema_variables

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
# -- 路径
parser.add_argument('--root_path', type=str,
                    default='../data_split/LA',
                    help='LA 数据集根目录，下含 2018LA_Seg_Training Set/ train.list test.list')
parser.add_argument('--exp',   type=str, default='BCP_CMC_v1_mutual')
parser.add_argument('--model', type=str, default='VNet')
# -- 迭代
parser.add_argument('--pre_iterations',  type=int, default=2000)
parser.add_argument('--max_iterations',  type=int, default=15000)
# -- 批次
parser.add_argument('--batch_size',  type=int, default=4)
parser.add_argument('--labeled_bs',  type=int, default=2)
# -- 训练超参
parser.add_argument('--base_lr',    type=float, default=0.01)
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80])
parser.add_argument('--num_classes',type=int,   default=2)
parser.add_argument('--labelnum',   type=int,   default=8,
                    help='有标签样本数量（LA 共 80 个训练样本）')
parser.add_argument('--max_samples',type=int,   default=80)
parser.add_argument('--seed',       type=int,   default=1337)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--gpu',        type=str,   default='0')
# -- BCP 超参
parser.add_argument('--u_weight',       type=float, default=0.5)
parser.add_argument('--mask_ratio',     type=float, default=2/3)
parser.add_argument('--consistency',    type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
# -- CMC v1 专用参数
parser.add_argument('--cmc_patch_size',        type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',        type=int,   default=2000)
parser.add_argument('--cmc_init_shared',        type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',        type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',      type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh', type=float, default=0.75)
parser.add_argument('--conf_thresh_init',       type=float, default=0.90)
parser.add_argument('--conf_thresh_final',      type=float, default=0.70)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size = tuple(args.patch_size)   # (112, 112, 80)
num_classes = args.num_classes        # 2
dice_loss = losses.DiceLoss(n_classes=num_classes)

def dice_loss_3d(score, target, mask=None):
    """
    纯内联 3D Dice Loss，不依赖 losses.py 里的 2D repeat 逻辑。
    score  : [B, C, D, H, W]  softmax 之后的概率
    target : [B, 1, D, H, W]  long → 转 one-hot
    mask   : [B, 1, D, H, W]  float，1=参与计算，0=忽略（可为 None）
    """
    B, C, D, H, W = score.shape
    # one-hot: [B, C, D, H, W]
    target_onehot = torch.zeros_like(score)
    target_onehot.scatter_(1, target.long(), 1)

    if mask is not None:
        score         = score         * mask.float()
        target_onehot = target_onehot * mask.float()

    intersect = (score * target_onehot).sum(dim=(2, 3, 4))   # [B, C]
    denom     = (score + target_onehot).sum(dim=(2, 3, 4))   # [B, C]
    dice      = (2.0 * intersect + 1e-5) / (denom + 1e-5)
    return 1.0 - dice.mean()


# ================================================================
# 通用工具函数
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

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state     = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


# ================================================================
# LA 专用：生成 3D BCP 掩码
# ================================================================
def generate_mask_3d(img):
    """
    输入 img: [B, 1, D, H, W]
    返回 mask: [D, H, W]  long，loss_mask: [B, D, H, W]  long
    遮住约 2/3 体积的随机长方体区域。
    """
    B, C, D, H, W = img.shape
    loss_mask = torch.ones(B, D, H, W).cuda()
    mask      = torch.ones(D, H, W).cuda()

    patch_d = int(D * args.mask_ratio)
    patch_h = int(H * args.mask_ratio)
    patch_w = int(W * args.mask_ratio)

    d0 = np.random.randint(0, D - patch_d)
    h0 = np.random.randint(0, H - patch_h)
    w0 = np.random.randint(0, W - patch_w)

    mask     [d0:d0+patch_d, h0:h0+patch_h, w0:w0+patch_w] = 0
    loss_mask[:, d0:d0+patch_d, h0:h0+patch_h, w0:w0+patch_w] = 0
    return mask.long(), loss_mask.long()


def mix_loss_3d(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    """
    3D 版 mix_loss，与 2D 版逻辑相同，支持 [B, C, D, H, W] 输出。
    img_l / patch_l: [B, D, H, W]  long
    mask / patch_mask: [B, D, H, W]  long
    """
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l   = img_l.type(torch.int64)
    patch_l = patch_l.type(torch.int64)
    output_soft  = F.softmax(output, dim=1)
    image_weight, patch_weight = (u_weight, l_weight) if unlab else (l_weight, u_weight)
    patch_mask = 1 - mask

    loss_dice  = dice_loss_3d(output_soft, img_l.unsqueeze(1),   mask.unsqueeze(1))   * image_weight
    loss_dice += dice_loss_3d(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight

    loss_ce    = image_weight * (CE(output, img_l)   * mask      ).sum() / (mask.sum()       + 1e-16)
    loss_ce   += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def get_LA_masks(output, nms=0):
    """
    LA 二分类：取 argmax 得到 [B, D, H, W] 的硬标签。
    nms=1 时保留最大连通域（可选）。
    """
    probs = F.softmax(output, dim=1)
    _, pred = torch.max(probs, dim=1)       # [B, D, H, W]
    return pred


# ================================================================
# CMC v1：3D 互补掩码生成
# ================================================================
def generate_cmc_masks_3d(img, cmc_patch_size=16, shared_ratio=0.0):
    """
    输入 img: [B, 1, D, H, W]
    返回 mask_a, mask_b: [B, 1, D, H, W]  float，值域 {0,1}
    在 H×W 平面上生成互补网格，沿 D 方向复制（3D 切片共享同一平面掩码）。
    """
    B, C, D, H, W = img.shape
    n_h = H // cmc_patch_size
    n_w = W // cmc_patch_size
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

        # 上采样到 H×W
        pa = F.interpolate(pa.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')  # [1,1,H,W]
        pb = F.interpolate(pb.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')

        # 沿 D 方向扩展 → [1, 1, D, H, W]
        pa = pa.unsqueeze(2).expand(1, 1, D, H, W)
        pb = pb.unsqueeze(2).expand(1, 1, D, H, W)
        masks_a.append(pa)
        masks_b.append(pb)

    return (torch.cat(masks_a, dim=0).to(img.device),   # [B,1,D,H,W]
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
# Pre-train（3D LA）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr      = args.base_lr
    max_iterations = args.pre_iterations
    labeled_sub_bs   = int(args.labeled_bs / 2)

    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=num_classes, mode='train').cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = LAHeart(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )
    labeled_idxs   = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training (LA 3D)")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num    = 0
    best_dice   = 0.0
    max_epoch   = max_iterations // len(trainloader) + 1
    iterator    = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()  # [B,1,D,H,W]
            label_batch  = sampled_batch['label'].cuda()  # [B,D,H,W]

            img_a  = volume_batch[:labeled_sub_bs]
            img_b  = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = generate_mask_3d(img_a)  # 3D 掩码

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl, _ = model(net_input)   # VNet 返回 (pred, feature)

            loss_dice, loss_ce = mix_loss_3d(
                out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/loss_total', loss,      iter_num)
            writer.add_scalar('pre/loss_dice',  loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce',    loss_ce,   iter_num)
            logging.info('pre iter %d: loss=%.4f dice=%.4f ce=%.4f' %
                         (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 200 == 0:
                model.eval()
                dice_val = test_3d_patch.var_all_case_LA(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_val > best_dice:
                    best_dice = round(dice_val, 4)
                    save_net_opt(model, optimizer,
                        os.path.join(snapshot_path,
                            'iter_{}_dice_{}.pth'.format(iter_num, best_dice)))
                    save_net_opt(model, optimizer,
                        os.path.join(snapshot_path,
                            '{}_best_model.pth'.format(args.model)))
                    logging.info("saved best model  dice={}".format(best_dice))
                writer.add_scalar('pre/val_dice', dice_val, iter_num)
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# Self-train：BCP（3D）+ CMC v1 互教分支
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr        = args.base_lr
    max_iterations = args.max_iterations
    pre_trained_model = os.path.join(pre_snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
    labeled_sub_bs   = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model     = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes, mode='train').cuda()
    ema_model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes, mode='train').cuda()
    for param in ema_model.parameters():
        param.detach_()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = LAHeart(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )
    labeled_idxs   = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (LA 3D BCP + CMC v1)")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    iter_num  = 0
    best_dice = 0.0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator  = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()   # [B,1,D,H,W]
            label_batch  = sampled_batch['label'].cuda()   # [B,D,H,W]

            img_a  = volume_batch[:labeled_sub_bs]
            img_b  = volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]

            # ==============================================================
            # BCP 3D 部分
            # ==============================================================
            with torch.no_grad():
                pre_a_out, _ = ema_model(uimg_a)
                pre_b_out, _ = ema_model(uimg_b)
                plab_a = get_LA_masks(pre_a_out)   # [sub_bs, D, H, W]
                plab_b = get_LA_masks(pre_b_out)

                img_mask, loss_mask = generate_mask_3d(img_a)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)

            out_unl, _ = model(net_input_unl)
            out_l,   _ = model(net_input_l)

            unl_dice, unl_ce = mix_loss_3d(out_unl, plab_a, lab_a, loss_mask,
                                            u_weight=args.u_weight, unlab=True)
            l_dice,   l_ce   = mix_loss_3d(out_l,   lab_b,  plab_b, loss_mask,
                                            u_weight=args.u_weight)

            loss_bcp = (unl_dice + unl_ce + l_dice + l_ce) / 2

            # ==============================================================
            # CMC v1 互教分支（3D）
            # ==============================================================
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

            mask_a_ab, mask_b_ab = generate_cmc_masks_3d(
                uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d(
                uimg_b, args.cmc_patch_size, shared_ratio)

            uimg_a_viewA = uimg_a * mask_a_ab
            uimg_a_viewB = uimg_a * mask_b_ab
            uimg_b_viewC = uimg_b * mask_a_cd
            uimg_b_viewD = uimg_b * mask_b_cd

            out_ab_A, _ = model(uimg_a_viewA)
            out_ab_B, _ = model(uimg_a_viewB)
            out_cd_C, _ = model(uimg_b_viewC)
            out_cd_D, _ = model(uimg_b_viewD)

            with torch.no_grad():
                conf_a = F.softmax(pre_a_out, dim=1).max(dim=1).values  # [B,D,H,W]
                conf_b = F.softmax(pre_b_out, dim=1).max(dim=1).values
                conf_mask_a = (conf_a > current_conf_thresh).float()
                conf_mask_b = (conf_b > current_conf_thresh).float()
                plab_teacher_a = plab_a.long()
                plab_teacher_b = plab_b.long()

            def cmc_mutual_loss_3d(out_A, out_B, plab_teacher,
                                   conf_mask, mask_a, mask_b):
                """
                3D CMC 互教损失
                out_A/B: [B, C, D, H, W]
                plab_teacher / conf_mask: [B, D, H, W]
                mask_a/b: [B, 1, D, H, W]
                """
                CE = nn.CrossEntropyLoss(reduction='none')
                w     = conf_mask
                denom = w.sum() + 1e-6
                la = CE(out_A, plab_teacher)   # [B,D,H,W]
                lb = CE(out_B, plab_teacher)
                loss_anchor = ((la + lb) * w).sum() / denom / 2.0

                with torch.no_grad():
                    prob_A  = F.softmax(out_A, dim=1)
                    prob_B  = F.softmax(out_B, dim=1)
                    conf_vA = prob_A.max(dim=1).values
                    conf_vB = prob_B.max(dim=1).values
                    plab_vA = prob_A.argmax(dim=1).long()
                    plab_vB = prob_B.argmax(dim=1).long()

                # mask_a/b: [B,1,D,H,W] → squeeze → [B,D,H,W]
                excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))
                excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))
                w_b = excl_a * (conf_vA > args.cmc_mutual_conf_thresh).float()
                w_a = excl_b * (conf_vB > args.cmc_mutual_conf_thresh).float()

                l_b_from_a = (CE(out_B, plab_vA) * w_b).sum() / (w_b.sum() + 1e-6)
                l_a_from_b = (CE(out_A, plab_vB) * w_a).sum() / (w_a.sum() + 1e-6)
                loss_mutual = (l_b_from_a + l_a_from_b) / 2.0

                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            loss_cmc_a = cmc_mutual_loss_3d(out_ab_A, out_ab_B,
                                             plab_teacher_a, conf_mask_a,
                                             mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss_3d(out_cd_C, out_cd_D,
                                             plab_teacher_b, conf_mask_b,
                                             mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ==============================================================
            # 总损失
            # ==============================================================
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            # -- 学习率 poly 衰减
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            # -- 日志
            writer.add_scalar('info/total_loss',   loss,      iter_num)
            writer.add_scalar('info/loss_bcp',     loss_bcp,  iter_num)
            writer.add_scalar('info/loss_cmc',     loss_cmc,  iter_num)
            writer.add_scalar('info/cmc_rampup',   cmc_rampup, iter_num)
            writer.add_scalar('info/shared_ratio', shared_ratio, iter_num)
            writer.add_scalar('info/conf_thresh',  current_conf_thresh, iter_num)
            writer.add_scalar('info/lr',           lr_,       iter_num)
            logging.info(
                'iter %d: loss=%.4f bcp=%.4f cmc=%.4f shared=%.2f conf_t=%.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 shared_ratio, current_conf_thresh))

            # -- 验证
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                dice_val = test_3d_patch.var_all_case_LA(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4)
                writer.add_scalar('info/val_dice', dice_val, iter_num)
                if dice_val > best_dice:
                    best_dice = round(dice_val, 4)
                    torch.save(model.state_dict(),
                        os.path.join(snapshot_path,
                            'iter_{}_dice_{}.pth'.format(iter_num, best_dice)))
                    torch.save(model.state_dict(),
                        os.path.join(snapshot_path,
                            '{}_best_model.pth'.format(args.model)))
                    logging.info("saved best  dice={}".format(best_dice))
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
        cudnn.benchmark  = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path  = "./model/BCP/LA_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum)

    for p in [pre_snapshot_path, self_snapshot_path]:
        os.makedirs(p, exist_ok=True)
    shutil.copy(__file__, self_snapshot_path)

    # -- Pre-train 日志
    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # -- Self-train 日志（重置 handler）
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)