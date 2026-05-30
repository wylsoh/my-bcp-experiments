"""
LA CMC v1 only：去掉 BCP mix 分支，只保留 CMC 互教一致性（3D，两阶段）
pre_train : 纯监督（有标签 CE + Dice）
self_train: 有标签监督 + CMC v1 互教（无标签）
数据集  : LA（LAHeart，读 {root}/data/{case}/mri_norm2.h5）
网络    : VNet（net_factory，返回 (pred, feature) 元组）
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

from dataloaders.dataset import (
    LAHeart, RandomRotFlip, RandomCrop, ToTensor, TwoStreamBatchSampler
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.test_3d_patch import test_all_case

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path',          type=str,   default='../data_split/LA')
parser.add_argument('--exp',                type=str,   default='CMC_v1_only')
parser.add_argument('--model',              type=str,   default='VNet')
parser.add_argument('--pre_max_iteration',  type=int,   default=2000)
parser.add_argument('--self_max_iteration', type=int,   default=15000)
parser.add_argument('--labelnum',           type=int,   default=8,
                    help='有标签样本数（LA 共 80 个训练样本）')
parser.add_argument('--max_samples',        type=int,   default=80)
parser.add_argument('--labeled_bs',         type=int,   default=2)
parser.add_argument('--batch_size',         type=int,   default=4)
parser.add_argument('--base_lr',            type=float, default=0.01)
parser.add_argument('--patch_size',         type=list,  default=[112, 112, 80])
parser.add_argument('--num_classes',        type=int,   default=2)
parser.add_argument('--seed',               type=int,   default=1337)
parser.add_argument('--deterministic',      type=int,   default=1)
parser.add_argument('--gpu',                type=str,   default='0')
parser.add_argument('--consistency',        type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
# CMC v1
parser.add_argument('--cmc_patch_size',         type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=2000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
patch_size          = tuple(args.patch_size)   # (112, 112, 80)
num_classes         = args.num_classes         # 2
pre_max_iterations  = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr             = args.base_lr
train_data_path     = args.root_path

if args.deterministic:
    cudnn.benchmark     = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

# ================================================================
# 测试数据列表（test.list）
# ================================================================
with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
    test_image_list = f.readlines()
test_image_list = [
    os.path.join(args.root_path, 'data', item.strip(), 'mri_norm2.h5')
    for item in test_image_list
]

# ================================================================
# 工具函数
# ================================================================
def save_net_opt(net, optimizer, path):
    torch.save({'net': net.state_dict(), 'opt': optimizer.state_dict()}, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    net.load_state_dict(torch.load(str(path))['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    ms, es = model.state_dict(), ema_model.state_dict()
    ema_model.load_state_dict(
        {k: alpha * es[k] + (1 - alpha) * ms[k] for k in ms})

# ================================================================
# 3D Dice Loss（内联）
# ================================================================
def dice_loss_3d(score, target):
    """score: [B,C,D,H,W]  target: [B,1,D,H,W] long"""
    oh = torch.zeros_like(score).scatter_(1, target.long(), 1)
    inter = (score * oh).sum(dim=(2, 3, 4))
    denom = (score + oh).sum(dim=(2, 3, 4))
    return (1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5))).mean()

# ================================================================
# CMC v1：3D 互补掩码与损失
# ================================================================
def generate_cmc_masks_3d(img, cmc_patch_size=16, shared_ratio=0.0):
    """返回 mask_a, mask_b: [B,1,D,H,W] float {0,1}"""
    B, C, D, H, W = img.shape
    p = cmc_patch_size
    n_h, n_w = max(H // p, 1), max(W // p, 1)
    masks_a, masks_b = [], []
    for _ in range(B):
        base = (torch.rand(n_h, n_w) > 0.5).float()
        if shared_ratio > 0.0:
            shared = torch.rand(n_h, n_w) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa, pb = (base == 0).float(), (base == 1).float()
        pa = F.interpolate(pa.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')
        pb = F.interpolate(pb.view(1, 1, n_h, n_w), size=(H, W), mode='nearest')
        masks_a.append(pa.unsqueeze(2).expand(1, 1, D, H, W))
        masks_b.append(pb.unsqueeze(2).expand(1, 1, D, H, W))
    return (torch.cat(masks_a, dim=0).to(img.device),
            torch.cat(masks_b, dim=0).to(img.device))

def get_progressive_shared_ratio(cur, warmup, init=0.4, final=0.0):
    if warmup <= 0 or cur >= warmup:
        return float(final)
    return init + (final - init) * float(cur) / float(warmup)

def get_adaptive_threshold(cur, maxiter, init=0.90, final=0.70):
    return init + (final - init) * min(1.0, float(cur) / float(maxiter))

def cmc_mutual_loss_3d(out_A, out_B, teacher_hard, conf_mask, mask_a, mask_b):
    CE = nn.CrossEntropyLoss(reduction='none')
    w     = conf_mask
    denom = w.sum() + 1e-6
    loss_anchor = (((CE(out_A, teacher_hard) + CE(out_B, teacher_hard)) * w
                    ).sum() / denom / 2.0)
    with torch.no_grad():
        prob_A  = F.softmax(out_A, dim=1)
        prob_B  = F.softmax(out_B, dim=1)
        conf_vA = prob_A.max(dim=1).values
        conf_vB = prob_B.max(dim=1).values
        plab_vA = prob_A.argmax(dim=1).long()
        plab_vB = prob_B.argmax(dim=1).long()
    excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))
    excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))
    w_b    = excl_a * (conf_vA > args.cmc_mutual_conf_thresh).float()
    w_a    = excl_b * (conf_vB > args.cmc_mutual_conf_thresh).float()
    l_mutual = (
        (CE(out_B, plab_vA) * w_b).sum() / (w_b.sum() + 1e-6) +
        (CE(out_A, plab_vB) * w_a).sum() / (w_a.sum() + 1e-6)
    ) / 2.0
    return loss_anchor + args.cmc_mutual_weight * l_mutual

# ================================================================
# 公共数据加载
# ================================================================
def build_loader():
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = LAHeart(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(), RandomCrop(patch_size), ToTensor()
        ])
    )
    labeled_idxs   = list(range(0, args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs)
    return DataLoader(db_train, batch_sampler=batch_sampler,
                      num_workers=4, pin_memory=True,
                      worker_init_fn=worker_init_fn)

# ================================================================
# Pre-train：纯监督
# ================================================================
def pre_train(args, snapshot_path):
    model       = net_factory(net_type=args.model, in_chns=1,
                              class_num=num_classes, mode='train').cuda()
    trainloader = build_loader()
    optimizer   = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    writer    = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num  = 0
    best_dice = 0.0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator  = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch  = sampled_batch['label'][:args.labeled_bs].cuda()

            outputs, _ = model(volume_batch)
            loss_ce    = F.cross_entropy(outputs, label_batch.long())
            loss_dice  = DICE(outputs, label_batch)
            loss       = (loss_ce + loss_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/loss_ce',   loss_ce,   iter_num)
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_all',  loss,      iter_num)
            logging.info('pre iter %d: loss=%.4f ce=%.4f dice=%.4f' %
                         (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 200 == 0:
                model.eval()
                pred_path = os.path.join(snapshot_path,
                                         '{}_predictions/'.format(args.model))
                os.makedirs(pred_path, exist_ok=True)
                avg_metric = test_all_case(
                    model, test_image_list, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4,
                    save_result=False, test_save_path=pred_path)
                dice_val = float(avg_metric[0])
                writer.add_scalar('pre/val_dice', dice_val, iter_num)
                if dice_val > best_dice:
                    best_dice = round(float(dice_val), 4)
                    save_net_opt(model, optimizer,
                        os.path.join(snapshot_path,
                            'iter_{}_dice_{}.pth'.format(iter_num, best_dice)))
                    save_net_opt(model, optimizer,
                        os.path.join(snapshot_path,
                            '{}_best_model.pth'.format(args.model)))
                    logging.info("saved best  dice={}".format(best_dice))
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()

# ================================================================
# Self-train：有标签监督 + CMC v1
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    model     = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes, mode='train').cuda()
    ema_model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes, mode='train').cuda()
    for param in ema_model.parameters():
        param.detach_()

    pretrained = os.path.join(pre_snapshot_path,
                              '{}_best_model.pth'.format(args.model))
    load_net(model,     pretrained)
    load_net(ema_model, pretrained)
    logging.info("Loaded from {}".format(pretrained))

    trainloader = build_loader()
    optimizer   = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    ema_model.train()
    writer    = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num  = 0
    best_dice = 0.0
    lr_       = base_lr
    max_epoch = self_max_iterations // len(trainloader) + 1
    unl_sub_bs = (args.batch_size - args.labeled_bs) // 2
    iterator  = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            # ── 有标签监督损失 ─────────────────────────────────────────
            vol_l  = volume_batch[:args.labeled_bs]
            lab_l  = label_batch[:args.labeled_bs]
            out_l, _ = model(vol_l)
            soft_l   = F.softmax(out_l, dim=1)
            loss_sup = (F.cross_entropy(out_l, lab_l.long()) +
                        dice_loss_3d(soft_l, lab_l.unsqueeze(1))) / 2

            # ── CMC v1（无标签部分） ───────────────────────────────────
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unl_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unl_sub_bs:]

            shared_ratio        = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, self_max_iterations,
                args.conf_thresh_init, args.conf_thresh_final)

            with torch.no_grad():
                unout_a, _ = ema_model(uimg_a)
                unout_b, _ = ema_model(uimg_b)
                teacher_a   = unout_a.argmax(dim=1).long()
                teacher_b   = unout_b.argmax(dim=1).long()
                conf_mask_a = (F.softmax(unout_a, dim=1).max(dim=1).values
                               > current_conf_thresh).float()
                conf_mask_b = (F.softmax(unout_b, dim=1).max(dim=1).values
                               > current_conf_thresh).float()

            mask_a_ab, mask_b_ab = generate_cmc_masks_3d(
                uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d(
                uimg_b, args.cmc_patch_size, shared_ratio)

            out_ab_A, _ = model(uimg_a * mask_a_ab)
            out_ab_B, _ = model(uimg_a * mask_b_ab)
            out_cd_C, _ = model(uimg_b * mask_a_cd)
            out_cd_D, _ = model(uimg_b * mask_b_cd)

            loss_cmc = (
                cmc_mutual_loss_3d(out_ab_A, out_ab_B,
                                   teacher_a, conf_mask_a,
                                   mask_a_ab, mask_b_ab) +
                cmc_mutual_loss_3d(out_cd_C, out_cd_D,
                                   teacher_b, conf_mask_b,
                                   mask_a_cd, mask_b_cd)
            ) / 2.0

            # ── 总损失 ────────────────────────────────────────────────
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_sup + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            # LR 阶梯衰减
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_

            writer.add_scalar('self/loss_sup',     loss_sup,            iter_num)
            writer.add_scalar('self/loss_cmc',     loss_cmc,            iter_num)
            writer.add_scalar('self/loss_all',     loss,                iter_num)
            writer.add_scalar('self/cmc_rampup',   cmc_rampup,          iter_num)
            writer.add_scalar('self/shared_ratio', shared_ratio,        iter_num)
            writer.add_scalar('self/conf_thresh',  current_conf_thresh,  iter_num)
            writer.add_scalar('self/consistency',  consistency_weight,   iter_num)
            logging.info(
                'self iter %d: loss=%.4f sup=%.4f cmc=%.4f '
                'shared=%.2f conf=%.2f' %
                (iter_num, loss.item(), loss_sup.item(), loss_cmc.item(),
                 shared_ratio, current_conf_thresh))

            if iter_num % 200 == 0:
                model.eval()
                pred_path = os.path.join(snapshot_path,
                                         '{}_predictions/'.format(args.model))
                os.makedirs(pred_path, exist_ok=True)
                avg_metric = test_all_case(
                    model, test_image_list, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=18, stride_z=4,
                    save_result=False, test_save_path=pred_path)
                dice_val = float(avg_metric[0])
                writer.add_scalar('self/val_dice', dice_val, iter_num)
                if dice_val > best_dice:
                    best_dice = round(float(dice_val), 4)
                    torch.save(model.state_dict(),
                        os.path.join(snapshot_path,
                            'iter_{}_dice_{}.pth'.format(iter_num, best_dice)))
                    torch.save(model.state_dict(),
                        os.path.join(snapshot_path,
                            '{}_best_model.pth'.format(args.model)))
                    logging.info("saved best  dice={}".format(best_dice))
                model.train()

            if iter_num >= self_max_iterations:
                break
        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()

# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    pre_snapshot_path  = "./model/BCP/LA_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/LA_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum)

    for p in [pre_snapshot_path, self_snapshot_path]:
        os.makedirs(p, exist_ok=True)
    shutil.copy(__file__, self_snapshot_path)

    logging.basicConfig(
        filename=os.path.join(pre_snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=os.path.join(self_snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

# python LA_CMC_v1_only.py --labelnum 8 --gpu 0
