"""
消融实验：仅 CMC v1（去掉 BCP 双向 copy-paste）

对比关系：
  原始 BCP          : loss = loss_bcp
  BCP + CMC v1      : loss = loss_bcp + cmc_weight * loss_cmc
  本文件（消融）     : loss = loss_sup  + cmc_weight * loss_cmc
                             ↑ 替换为普通监督损失（无 copy-paste）

self_train 变化：
  删除: generate_mask / net_input_unl / net_input_l / mix_loss / loss_bcp
  新增: 有标签数据的标准 CE+Dice 监督（loss_sup），作为 BCP 的替代基线
  保留: EMA 教师前向（pre_a/pre_b，CMC 仍需要）/ CMC v1 互教分支

pre_train 保持不变（与所有对比实验共享同一预训练权重，保证公平对比）
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
# 参数（与 BCP_CMC_v1_mutual.py 完全一致，仅 exp 名不同）
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
parser.add_argument('--exp', type=str, default='BCP_CMC_v1_ablation_cmc_only')
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
parser.add_argument('--cmc_patch_size',         type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',         type=int,   default=5000)
parser.add_argument('--cmc_init_shared',         type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',         type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',       type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',  type=float, default=0.75)
parser.add_argument('--conf_thresh_init',        type=float, default=0.90)
parser.add_argument('--conf_thresh_final',       type=float, default=0.70)
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
# CMC v1 辅助函数（与 BCP_CMC_v1_mutual.py 完全一致）
# ================================================================
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
# Pre-train（与原始 BCP 完全一致，保证消融对比公平）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

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
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' %
                         (iter_num, loss, loss_dice, loss_ce))
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
# Self-train：去掉 BCP，仅保留 CMC v1 + 有标签监督
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

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
    logging.info("Start self_training (CMC v1 only, no BCP)")
    logging.info("{} iterations per epoch".format(len(trainloader)))

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
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]

            # ==============================================================
            # 有标签数据标准监督（替代 BCP copy-paste，无混合操作）
            # CE + Dice，与 pre_train 的单路监督一致
            # ==============================================================
            out_lab = model(torch.cat([img_a, img_b], dim=0))       # [labeled_bs, C, H, W]
            lab_all = torch.cat([lab_a, lab_b], dim=0).long()       # [labeled_bs, H, W]
            out_soft = F.softmax(out_lab, dim=1)

            loss_sup_ce   = nn.CrossEntropyLoss()(out_lab, lab_all)
            loss_sup_dice = dice_loss(out_soft, lab_all.unsqueeze(1),
                                      torch.ones_like(lab_all).unsqueeze(1))
            loss_sup = (loss_sup_ce + loss_sup_dice) / 2

            # ==============================================================
            # EMA 教师前向（CMC 需要，与 BCP_CMC_v1 保持一致）
            # ==============================================================
            with torch.no_grad():
                pre_a  = ema_model(uimg_a)
                pre_b  = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # ==============================================================
            # CMC v1 互教分支（与 BCP_CMC_v1_mutual.py 完全一致）
            # ==============================================================
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, max_iterations, args.conf_thresh_init, args.conf_thresh_final)

            mask_a_ab, mask_b_ab = generate_cmc_masks(uimg_a, args.cmc_patch_size, shared_ratio)
            mask_a_cd, mask_b_cd = generate_cmc_masks(uimg_b, args.cmc_patch_size, shared_ratio)

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

            def cmc_mutual_loss(out_viewA, out_viewB, plab_teacher,
                                conf_mask, mask_a, mask_b):
                w = conf_mask
                denom = w.sum() + 1e-6
                la = F.cross_entropy(out_viewA, plab_teacher, reduction='none')
                lb = F.cross_entropy(out_viewB, plab_teacher, reduction='none')
                loss_anchor = ((la + lb) * w).sum() / denom / 2.0

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
                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            loss_cmc_a = cmc_mutual_loss(out_a_viewA, out_a_viewB,
                                          plab_teacher_a, conf_mask_a, mask_a_ab, mask_b_ab)
            loss_cmc_b = cmc_mutual_loss(out_b_viewC, out_b_viewD,
                                          plab_teacher_b, conf_mask_b, mask_a_cd, mask_b_cd)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            # ==============================================================
            # 总损失：有标签监督 + CMC（无 BCP）
            # ==============================================================
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_sup + args.cmc_loss_weight * cmc_rampup * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss',         loss,               iter_num)
            writer.add_scalar('info/loss_sup',           loss_sup,           iter_num)
            writer.add_scalar('info/loss_sup_ce',        loss_sup_ce,        iter_num)
            writer.add_scalar('info/loss_sup_dice',      loss_sup_dice,      iter_num)
            writer.add_scalar('info/loss_cmc',           loss_cmc,           iter_num)
            writer.add_scalar('info/cmc_rampup',         cmc_rampup,         iter_num)
            writer.add_scalar('info/shared_ratio',       shared_ratio,       iter_num)
            writer.add_scalar('info/conf_threshold',     current_conf_thresh, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight,  iter_num)
            logging.info(
                'iteration %d: loss: %f, sup: %f, cmc: %f, shared: %.2f, conf_t: %.2f' %
                (iter_num, loss.item(), loss_sup.item(), loss_cmc.item(),
                 shared_ratio, current_conf_thresh))

            if iter_num % 20 == 0:
                writer.add_image('cmc/ViewA', uimg_a_viewA[0, 0:1], iter_num)
                writer.add_image('cmc/ViewB', uimg_a_viewB[0, 0:1], iter_num)
                pred_a_vis = torch.argmax(torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                pred_b_vis = torch.argmax(torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                writer.add_image('cmc/PredViewA', pred_a_vis[0].float() * 50, iter_num)
                writer.add_image('cmc/PredViewB', pred_b_vis[0].float() * 50, iter_num)
                # 有标签监督可视化
                out_vis = torch.argmax(torch.softmax(out_lab, dim=1), dim=1, keepdim=True)
                writer.add_image('sup/Image',      img_a[0, 0:1],                  iter_num)
                writer.add_image('sup/Prediction', out_vis[0].float() * 50,        iter_num)
                writer.add_image('sup/GT',         lab_a[0].unsqueeze(0).float() * 50, iter_num)

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
