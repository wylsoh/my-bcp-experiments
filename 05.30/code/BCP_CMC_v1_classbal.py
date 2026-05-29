"""
BCP + CMC v1-ClassBal：类别平衡互补掩码互教一致性

在 BCP_CMC_v1_mutual.py 基础上，仅替换掩码生成函数：
  generate_cmc_masks          →  generate_cmc_masks_class_balanced

其余所有代码（BCP 部分、CMC 互教损失、训练流程）完全不变。

类别平衡掩码的动机：
  原始随机网格掩码的问题：随机分配可能导致某个前景类别（如心肌）
  的像素全部落入视图B，视图A完全看不到该类，互教信号退化。
  对 ACDC 的小目标类别（RV=类别1, MYO=类别2, LV=类别3）尤其明显。

类别平衡掩码的策略：
  1. 先随机生成网格块分配（与 v1 相同）
  2. 对每个前景类别 c ∈ {1,2,3}：
     - 计算每个网格块内类别 c 的像素占比（avg_pool2d 实现）
     - 若类别 c 在视图A或视图B的覆盖率低于 min_class_ratio：
       将包含该类的所有网格块设为"共享块"（两视图均可见）
  3. 共享块不破坏互补性的语义，只是从"互斥"变为"均可见"

新增参数：
  --cmc_min_class_ratio  前景类别最低覆盖率（默认 0.3）
                         低于此值则将该类所在块设为共享
                         建议范围：0.2~0.5

需要传入伪标签：
  mask 生成时需要 plab_a/plab_b（EMA 教师的预测，已在 BCP 部分计算好）
  无额外计算开销
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

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net
from utils import losses, ramps, val_2d

# ================================================================
# 参数（与 v1 完全一致，新增 cmc_min_class_ratio）
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
parser.add_argument('--exp', type=str, default='BCP_CMC_v1_classbal')
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
# ---------- CMC v1 参数 ----------
parser.add_argument('--cmc_patch_size',         type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=5000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
# ---------- 类别平衡专用参数（新增）----------
parser.add_argument('--cmc_min_class_ratio', type=float, default=0.3,
                    help='每个前景类别在单个视图中的最低覆盖率。'
                         '低于此值时将该类所在块设为共享块。'
                         '0=不做平衡（退化为 v1），1=所有含前景类的块都共享。'
                         '推荐范围 0.2~0.5。')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

# ================================================================
# 原始 BCP 函数（逐字复制，未做任何修改）
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

# ================================================================
# CMC 辅助函数
# ================================================================
def get_progressive_shared_ratio(current_iter, warmup_iter, init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)

def get_adaptive_threshold(current_iter, max_iter, init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress

# ================================================================
# 核心改动：generate_cmc_masks_class_balanced
# （替换 v1 的 generate_cmc_masks，其余代码完全不变）
# ================================================================
def generate_cmc_masks_class_balanced(img, pseudo_label,
                                       cmc_patch_size=16,
                                       shared_ratio=0.0,
                                       min_class_ratio=0.3):
    """
    类别平衡互补网格掩码生成器

    在随机网格分配的基础上，检查每个前景类别在两个视图中的覆盖率，
    若覆盖率不足则将该类所在的块设为共享块（两视图均可见）。

    与 v1 的 generate_cmc_masks 的区别：
      v1 : base = random → pa = (base==0), pb = (base==1)，纯随机互补
      本函数 : 在 v1 基础上，对覆盖不足的前景类别块做补救性共享

    平衡机制示意（以类别2/心肌为例）：
      随机分配后：心肌80%在视图B，20%在视图A
      min_class_ratio=0.3：视图A不足 → 将心肌所在块全部设为共享
      结果：心肌100%在视图B，100%在视图A（两视图均可见）

    平衡后 shared_ratio 会自适应增大（取决于前景类的分布），
    可通过 TensorBoard 中的 cmc/actual_shared_ratio 监控。

    Args:
        img             : [B, C, H, W] 无标签图像（仅用于获取 B/H/W）
        pseudo_label    : [B, H, W]    EMA 教师伪标签（Long 或 Float）
                          直接复用 BCP 分支已算好的 plab_a/plab_b
        cmc_patch_size  : 网格块大小（需整除 H/W）
        shared_ratio    : 基础共享比例（热身阶段用，与 v1 含义相同）
        min_class_ratio : 前景类别最低覆盖率阈值
                          低于此值触发平衡补救，设 0 可退化为 v1

    Returns:
        mask_a          : [B, 1, H, W]，值域 {0,1}
        mask_b          : [B, 1, H, W]，值域 {0,1}
        actual_shared   : float，本批次平均实际共享块比例（用于日志）
    """
    B, C, H, W = img.shape
    p = cmc_patch_size
    n = H // p
    masks_a, masks_b = [], []
    shared_ratios = []

    for b in range(B):
        plab = pseudo_label[b].float()   # [H, W]，在 CUDA 上

        # ---- Step 1: 初始随机分配（与 v1 完全相同）----
        base   = (torch.rand(n, n) > 0.5)                       # [n,n] bool，CPU
        shared = (torch.rand(n, n) < shared_ratio)              # [n,n] bool，CPU

        # ---- Step 2: 类别平衡补救 ----
        for c in range(1, 4):   # 前景类别 1(RV), 2(MYO), 3(LV)
            # 计算每个网格块内类别 c 的像素比例 [n, n]
            # avg_pool2d: 将 [1,1,H,W] 下采样到 [1,1,n,n]
            class_bin = (plab == c).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            class_per_block = F.avg_pool2d(
                class_bin, kernel_size=p, stride=p
            ).squeeze()   # [n, n]，CUDA

            if class_per_block.sum() < 1e-6:
                continue   # 该样本无此类，跳过

            # 将块级覆盖图移回 CPU 与 base/shared 做布尔运算
            cpb_cpu = class_per_block.cpu()          # [n, n]
            has_class = cpb_cpu > 0                  # [n, n] bool

            # 当前分配下各视图对类别 c 的覆盖率
            view_a = base.logical_not() | shared     # base==0 or shared → A可见
            view_b = base | shared                   # base==1 or shared → B可见

            total   = cpb_cpu.sum().item()
            cover_a = (cpb_cpu * view_a.float()).sum().item() / (total + 1e-6)
            cover_b = (cpb_cpu * view_b.float()).sum().item() / (total + 1e-6)

            # 覆盖不足则将该类所在块设为共享
            if cover_a < min_class_ratio or cover_b < min_class_ratio:
                shared = shared | has_class

        # ---- Step 3: 生成最终掩码 ----
        pa = (base.logical_not() | shared).float()   # A可见：base==0 or shared
        pb = (base | shared).float()                 # B可见：base==1 or shared

        actual_shared_ratio = shared.float().mean().item()
        shared_ratios.append(actual_shared_ratio)

        # 上采样到完整分辨率
        pa = F.interpolate(
            pa.view(1, 1, n, n), size=(H, W), mode='nearest'
        ).squeeze(0)   # [1, H, W]
        pb = F.interpolate(
            pb.view(1, 1, n, n), size=(H, W), mode='nearest'
        ).squeeze(0)

        masks_a.append(pa)
        masks_b.append(pb)

    actual_shared = float(np.mean(shared_ratios))
    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device),
            actual_shared)

# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = (int(args.labeled_bs / 2),
                                         int((args.batch_size - args.labeled_bs) / 2))

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer   = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

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
            gt_mixl   = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl  = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            writer.add_scalar('info/total_loss', loss,      iter_num)
            writer.add_scalar('info/mix_dice',   loss_dice, iter_num)
            writer.add_scalar('info/mix_ce',     loss_ce,   iter_num)
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' %
                         (iter_num, loss, loss_dice, loss_ce))
            if iter_num % 20 == 0:
                writer.add_image('pre_train/Mixed_Image', net_input[1, 0:1], iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                writer.add_image('pre_train/Mixed_GroundTruth',
                                 gt_mixl[1, ...].unsqueeze(0) * 50, iter_num)
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"],
                        model, classes=num_classes)
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
                    save_net_opt(model, optimizer,
                        os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                            iter_num, round(best_performance, 4))))
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
# Self-train：BCP 部分与原始完全一致，CMC 分支改用类别平衡掩码
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr        = args.base_lr
    num_classes    = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,
                                     '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = (int(args.labeled_bs / 2),
                                         int((args.batch_size - args.labeled_bs) / 2))

    model     = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (BCP + CMC v1-ClassBal)")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("min_class_ratio={}".format(args.cmc_min_class_ratio))

    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
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

            # ==============================================================
            # BCP 部分（与原始 BCP 完全一致）
            # ==============================================================
            with torch.no_grad():
                pre_a  = ema_model(uimg_a)
                pre_b  = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label   = lab_b  * img_mask + ulab_b * (1 - img_mask)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l   = model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask,
                                         u_weight=args.u_weight, unlab=True)
            l_dice,   l_ce   = mix_loss(out_l, lab_b, plab_b, loss_mask,
                                         u_weight=args.u_weight)
            loss_ce   = unl_ce   + l_ce
            loss_dice = unl_dice + l_dice
            loss_bcp  = (loss_dice + loss_ce) / 2

            # ==============================================================
            # CMC 分支：类别平衡互补掩码（核心改动在此）
            # ==============================================================
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

            # 类别平衡掩码：传入 plab_a/plab_b 指导掩码生成
            # 返回 actual_shared：本批次实际共享块比例（含平衡补救后的增量）
            mask_a_ab, mask_b_ab, shared_a = generate_cmc_masks_class_balanced(
                uimg_a, plab_a,
                cmc_patch_size=args.cmc_patch_size,
                shared_ratio=shared_ratio,
                min_class_ratio=args.cmc_min_class_ratio
            )
            mask_a_cd, mask_b_cd, shared_b = generate_cmc_masks_class_balanced(
                uimg_b, plab_b,
                cmc_patch_size=args.cmc_patch_size,
                shared_ratio=shared_ratio,
                min_class_ratio=args.cmc_min_class_ratio
            )
            actual_shared = (shared_a + shared_b) / 2.0

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

            # 互教损失（与 v1 完全相同）
            def cmc_mutual_loss(out_viewA, out_viewB, plab_teacher,
                                conf_mask, mask_a, mask_b):
                w     = conf_mask
                denom = w.sum() + 1e-6
                la = F.cross_entropy(out_viewA, plab_teacher, reduction='none')
                lb = F.cross_entropy(out_viewB, plab_teacher, reduction='none')
                loss_anchor = ((la + lb) * w).sum() / denom / 2.0

                with torch.no_grad():
                    prob_a  = F.softmax(out_viewA, dim=1)
                    prob_b  = F.softmax(out_viewB, dim=1)
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
                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            loss_cmc_a = cmc_mutual_loss(out_a_viewA, out_a_viewB,
                                          plab_teacher_a, conf_mask_a, mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss(out_b_viewC, out_b_viewD,
                                          plab_teacher_b, conf_mask_b, mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ==============================================================
            # 总损失（与 v1 完全一致）
            # ==============================================================
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss',         loss,               iter_num)
            writer.add_scalar('info/loss_bcp',           loss_bcp,           iter_num)
            writer.add_scalar('info/mix_dice',           loss_dice,          iter_num)
            writer.add_scalar('info/mix_ce',             loss_ce,            iter_num)
            writer.add_scalar('info/loss_cmc',           loss_cmc,           iter_num)
            writer.add_scalar('info/cmc_rampup',         cmc_rampup,         iter_num)
            writer.add_scalar('info/conf_threshold',     current_conf_thresh, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight,  iter_num)
            # 类别平衡专用指标：base shared_ratio vs 实际 actual_shared_ratio
            writer.add_scalar('cmc/base_shared_ratio',   shared_ratio,        iter_num)
            writer.add_scalar('cmc/actual_shared_ratio', actual_shared,       iter_num)
            # actual - base = 平衡补救带来的额外共享量，越大说明类别越不均衡
            writer.add_scalar('cmc/balance_delta',
                              actual_shared - shared_ratio, iter_num)

            logging.info(
                'iteration %d: loss: %f, bcp: %f, cmc: %f, '
                'shared_base: %.2f, shared_actual: %.2f, conf_t: %.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 shared_ratio, actual_shared, current_conf_thresh))

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
                # CMC 可视化
                writer.add_image('cmc/Original',    uimg_a[0, 0:1],       iter_num)
                writer.add_image('cmc/ViewA',       uimg_a_viewA[0, 0:1], iter_num)
                writer.add_image('cmc/ViewB',       uimg_a_viewB[0, 0:1], iter_num)
                writer.add_image('cmc/MaskA',       mask_a_ab[0],         iter_num)
                writer.add_image('cmc/MaskB',       mask_b_ab[0],         iter_num)
                # 教师伪标签（供对比：哪些类被平衡保护）
                writer.add_image('cmc/PseudoLabel',
                                 plab_a[0].unsqueeze(0).float() * 50,     iter_num)
                pred_a_vis = torch.argmax(
                    torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                pred_b_vis = torch.argmax(
                    torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                writer.add_image('cmc/PredViewA', pred_a_vis[0].float() * 50, iter_num)
                writer.add_image('cmc/PredViewB', pred_b_vis[0].float() * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"],
                        model, classes=num_classes)
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

    pre_snapshot_path  = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum)
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
