"""
Flare BCP + CMC v1：互补掩码互教一致性（3D，两阶段）
数据集  : Flare（BraTS2019 Loader，读 {root}/{case}/2022.h5）
网络    : VNet（net_factory，返回 (pred, feature) 元组）
工具    : utils/（losses, ramps, test_3d_patch）
验证    : test_all_case，读 test.txt
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

from dataloaders.brats2019 import (
    BraTS2019, RandomCrop, RandomRotFlip, ToTensor,
    TwoStreamBatchSampler, RandomNoise
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.test_3d_patch import test_all_case
from utils.BCP_utils import context_mask, mix_loss, update_ema_variables

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path',   type=str,   default='../data_split/flare',
                    help='Flare 数据集根目录')
parser.add_argument('--exp',         type=str,   default='BCP_CMC_v1_mutual')
parser.add_argument('--model',       type=str,   default='VNet')
parser.add_argument('--pre_max_iteration',  type=int,   default=2000)
parser.add_argument('--self_max_iteration', type=int,   default=15000)
parser.add_argument('--labeled_num', type=int,   default=21)
parser.add_argument('--data_num',    type=int,   default=378)
parser.add_argument('--labeled_bs',  type=int,   default=2)
parser.add_argument('--batch_size',  type=int,   default=4)
parser.add_argument('--base_lr',     type=float, default=0.01)
parser.add_argument('--patch_size',  type=list,  default=[64, 128, 128])
parser.add_argument('--num_classes', type=int,   default=14)
parser.add_argument('--seed',        type=int,   default=1337)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--gpu',         type=str,   default='0')
parser.add_argument('--consistency', type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
parser.add_argument('--u_weight',    type=float, default=0.5)
parser.add_argument('--mask_ratio',  type=float, default=2/3)
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
patch_size  = tuple(args.patch_size)
num_classes = args.num_classes

pre_max_iterations  = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr             = args.base_lr
train_data_path     = args.root_path

if args.deterministic:
    cudnn.benchmark   = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

# ================================================================
# 测试数据列表（test.txt）
# ================================================================
with open(os.path.join(args.root_path, 'test.txt'), 'r') as f:
    test_image_list = f.readlines()
test_image_list = [
    os.path.join(args.root_path, item.strip(), '2022.h5')
    for item in test_image_list
]

# ================================================================
# 工具函数
# ================================================================
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

def update_model_ema(model, ema_model, alpha):
    model_state     = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

# ================================================================
# BCP 3D 掩码
# ================================================================
def generate_mask_3d(img):
    """img: [B,1,D,H,W] → mask [D,H,W] long，loss_mask [B,D,H,W] long"""
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
# 3D Dice Loss（内联）
# ================================================================
def dice_loss_3d(score, target, mask=None):
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
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.long(), patch_l.long()
    output_soft = F.softmax(output, dim=1)
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

def get_pseudo_label(output):
    """argmax → [B,D,H,W] LongTensor"""
    return torch.argmax(F.softmax(output, dim=1), dim=1)

# ================================================================
# CMC v1：3D 互补掩码
# ================================================================
def generate_cmc_masks_3d(img, cmc_patch_size=16, shared_ratio=0.0):
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
        pa = F.interpolate(pa.view(1, 1, n_h, n_w),
                           size=(H, W), mode='nearest')   # [1,1,H,W]
        pb = F.interpolate(pb.view(1, 1, n_h, n_w),
                           size=(H, W), mode='nearest')
        pa = pa.unsqueeze(2).expand(1, 1, D, H, W)        # [1,1,D,H,W]
        pb = pb.unsqueeze(2).expand(1, 1, D, H, W)
        masks_a.append(pa)
        masks_b.append(pb)
    return (torch.cat(masks_a, dim=0).to(img.device),
            torch.cat(masks_b, dim=0).to(img.device))

def get_progressive_shared_ratio(current_iter, warmup_iter,
                                  init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * \
           float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter,
                            init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

def cmc_mutual_loss_3d(out_A, out_B, teacher_hard, conf_mask, mask_a, mask_b):
    """
    out_A/B      : [B,C,D,H,W]
    teacher_hard : [B,D,H,W]  long
    conf_mask    : [B,D,H,W]  float
    mask_a/b     : [B,1,D,H,W] float
    """
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
    l_b_from_a = (CE_none(out_B, plab_vA) * w_b).sum() / (w_b.sum() + 1e-6)
    l_a_from_b = (CE_none(out_A, plab_vB) * w_a).sum() / (w_a.sum() + 1e-6)
    loss_mutual = (l_b_from_a + l_a_from_b) / 2.0
    return loss_anchor + args.cmc_mutual_weight * loss_mutual

# ================================================================
# 公共数据加载
# ================================================================
def build_loader():
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BraTS2019(
        base_dir=train_data_path,
        split='train',
        num=None,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            RandomNoise(),
            ToTensor(),
        ])
    )
    labeled_idxs   = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.data_num))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn)
    return trainloader

# ================================================================
# Pre-train
# ================================================================
def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=num_classes, mode='train').cuda()
    trainloader = build_loader()
    optimizer   = optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    writer   = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num  = 0
    best_dice = 0.0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    sub_bs    = args.labeled_bs // 2
    iterator  = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch  = sampled_batch['label'][:args.labeled_bs].cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs],  label_batch[sub_bs:]

            with torch.no_grad():
                img_mask, loss_mask = generate_mask_3d(img_a)

            net_input   = img_a * img_mask + img_b * (1 - img_mask)
            label_mixed = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(net_input)
            loss_ce    = F.cross_entropy(outputs, label_mixed.long())
            loss_dice  = DICE(outputs, label_mixed)
            loss       = (loss_ce + loss_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce',   loss_ce,   iter_num)
            writer.add_scalar('pre/loss_all',  loss,      iter_num)
            logging.info('pre iter %d: loss=%.4f dice=%.4f ce=%.4f' %
                         (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 200 == 0:
                model.eval()
                test_save_path = os.path.join(
                    snapshot_path, '{}_predictions/'.format(args.model))
                os.makedirs(test_save_path, exist_ok=True)
                avg_metric = test_all_case(
                    model, test_image_list, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=32, stride_z=16,
                    save_result=False, test_save_path=test_save_path)
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
                    logging.info("saved best model  dice={}".format(best_dice))
                model.train()

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()

# ================================================================
# Self-train：BCP 3D + CMC v1
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

    model.train()
    ema_model.train()
    writer    = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num  = 0
    best_dice = 0.0
    lr_       = base_lr
    max_epoch = self_max_iterations // len(trainloader) + 1
    sub_bs    = args.labeled_bs // 2
    unl_sub_bs = (args.batch_size - args.labeled_bs) // 2
    iterator  = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            img_a   = volume_batch[:sub_bs]
            img_b   = volume_batch[sub_bs:args.labeled_bs]
            lab_a   = label_batch[:sub_bs]
            lab_b   = label_batch[sub_bs:args.labeled_bs]
            uimg_a  = volume_batch[args.labeled_bs:args.labeled_bs + unl_sub_bs]
            uimg_b  = volume_batch[args.labeled_bs + unl_sub_bs:]

            # ── BCP 3D ────────────────────────────────────────────────
            with torch.no_grad():
                unout_a, _ = ema_model(uimg_a)
                unout_b, _ = ema_model(uimg_b)
                plab_a = get_pseudo_label(unout_a)
                plab_b = get_pseudo_label(unout_b)
                img_mask, loss_mask = generate_mask_3d(img_a)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a   * img_mask + uimg_a * (1 - img_mask)
            mixu_img = uimg_b  * img_mask + img_b  * (1 - img_mask)
            mixl_lab = lab_a   * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b  * img_mask + lab_b  * (1 - img_mask)

            out_l, _ = model(mixl_img)
            out_u, _ = model(mixu_img)

            loss_l_dice, loss_l_ce = mix_loss_3d(
                out_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u_dice, loss_u_ce = mix_loss_3d(
                out_u, plab_b, lab_b, loss_mask,
                u_weight=args.u_weight, unlab=True)
            loss_bcp = (loss_l_dice + loss_l_ce + loss_u_dice + loss_u_ce) / 2

            # ── CMC v1 ────────────────────────────────────────────────
            shared_ratio        = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, self_max_iterations,
                args.conf_thresh_init, args.conf_thresh_final)

            mask_a_ab, mask_b_ab = generate_cmc_masks_3d(
                uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d(
                uimg_b, args.cmc_patch_size, shared_ratio)

            out_ab_A, _ = model(uimg_a * mask_a_ab)
            out_ab_B, _ = model(uimg_a * mask_b_ab)
            out_cd_C, _ = model(uimg_b * mask_a_cd)
            out_cd_D, _ = model(uimg_b * mask_b_cd)

            with torch.no_grad():
                conf_a = F.softmax(unout_a, dim=1).max(dim=1).values
                conf_b = F.softmax(unout_b, dim=1).max(dim=1).values
                conf_mask_a = (conf_a > current_conf_thresh).float()
                conf_mask_b = (conf_b > current_conf_thresh).float()
                teacher_a   = plab_a.long()
                teacher_b   = plab_b.long()

            loss_cmc_a = cmc_mutual_loss_3d(
                out_ab_A, out_ab_B, teacher_a, conf_mask_a,
                mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss_3d(
                out_cd_C, out_cd_D, teacher_b, conf_mask_b,
                mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ── 总损失 ────────────────────────────────────────────────
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

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

            writer.add_scalar('self/loss_bcp',     loss_bcp,           iter_num)
            writer.add_scalar('self/loss_cmc',     loss_cmc,           iter_num)
            writer.add_scalar('self/loss_all',     loss,               iter_num)
            writer.add_scalar('self/cmc_rampup',   cmc_rampup,         iter_num)
            writer.add_scalar('self/shared_ratio', shared_ratio,       iter_num)
            writer.add_scalar('self/conf_thresh',  current_conf_thresh, iter_num)
            writer.add_scalar('self/consistency',  consistency_weight,  iter_num)
            logging.info(
                'self iter %d: loss=%.4f bcp=%.4f cmc=%.4f '
                'shared=%.2f conf=%.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 shared_ratio, current_conf_thresh))

            if iter_num % 200 == 0:
                model.eval()
                test_save_path = os.path.join(
                    snapshot_path, '{}_predictions/'.format(args.model))
                os.makedirs(test_save_path, exist_ok=True)
                avg_metric = test_all_case(
                    model, test_image_list, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=32, stride_z=16,
                    save_result=False, test_save_path=test_save_path)
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
    pre_snapshot_path  = "./model/BCP/flare_{}_{}_labeled/pre_train".format(
        args.exp, args.labeled_num)
    self_snapshot_path = "./model/BCP/flare_{}_{}_labeled/self_train".format(
        args.exp, args.labeled_num)

    for p in [pre_snapshot_path, self_snapshot_path]:
        os.makedirs(p, exist_ok=True)
    shutil.copy(__file__, self_snapshot_path)

    # Pre-train
    logging.basicConfig(
        filename=os.path.join(pre_snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self-train
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=os.path.join(self_snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

# python flare_BCP_CMC_v1_mutual.py --labeled_num 21 --gpu 0