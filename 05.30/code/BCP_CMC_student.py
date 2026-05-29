"""
BCP + CMC 半监督医学图像分割实验脚本

实验命名: BCP_CMC_student
指标输出: Dice, Jaccard, HD95, ASD

核心创新：互补掩码一致性 (CMC, Complementary Mask Consistency)
==============================================================
与 BCP_MAE_student 的对比：

  MAE 方案（非对称）:
    学生 ← 遮挡图像  →  pred_masked
    教师 ← 完整图像  →  pseudo_label
    损失: CE(pred_masked, pseudo_label)  [仅在遮挡区域计算]

  CMC 方案（对称，本脚本）:
    视图A = img ⊙ M_a       学生 → pred_A
    视图B = img ⊙ M_b = img ⊙ (1-M_a)  学生（同一模型）→ pred_B
    教师  ← 完整图像         → pseudo_label（锚点监督）

    损失组成：
    1. L_anchor : pred_A 和 pred_B 均向教师伪标签对齐（教师置信度加权）
    2. L_mutual : 盲区互教
       - A独有区域（B看不见）中，A的高置信预测监督B
       - B独有区域（A看不见）中，B的高置信预测监督A

  CMC 优势：
    ① 对称性：两个视图地位平等，均为"不完整输入"，避免 MAE 的学生/教师信息量不对称
    ② 额外自监督信号：互教损失不依赖教师质量，纯粹来自两视图的一致性约束
    ③ 渐进热身：从部分重叠（init_shared=0.4）逐步收敛到纯互补（任务难度递增）

训练流程：
  Phase 1 — Pre-train  (10k iter): 仅有标签数据，BCP copy-paste 监督
  Phase 2 — Self-train (30k iter): 有标签数据 BCP + 无标签数据 CMC

效率说明：
  CMC 每迭代步在无标签分支增加 2 次学生 forward（通过 batch concat 节省一次 forward call）
  总 forward 次数：BCP 2次 + CMC batched 2次（每无标签子批） + EMA 2次 = 6次/step
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
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net
from utils.train_utils import (
    load_net, load_net_opt, save_net_opt,
    update_model_ema, get_current_consistency_weight,
    poly_lr, mix_loss, patients_to_slices
)
from utils.mask_generator import BCPMaskGenerator
from utils.pseudo_label_utils import (
    get_ACDC_masks, get_confidence_mask, get_adaptive_threshold
)
from utils.metric_utils import (
    test_single_volume_all_metrics, log_validation_metrics
)
# CMC 专用工具（本实验新增，见 utils/cmc_utils.py）
from utils.cmc_utils import CMCGridMaskGenerator, cmc_consistency_loss


# ================================================================
# 参数定义
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path',           type=str,   default='../data_split/ACDC')
parser.add_argument('--exp',                 type=str,   default='BCP_CMC_student')
parser.add_argument('--model',               type=str,   default='unet')
parser.add_argument('--pre_iterations',      type=int,   default=10000)
parser.add_argument('--max_iterations',      type=int,   default=30000)
parser.add_argument('--batch_size',          type=int,   default=24)
parser.add_argument('--deterministic',       type=int,   default=1)
parser.add_argument('--base_lr',             type=float, default=0.01)
parser.add_argument('--patch_size',          type=list,  default=[256, 256])
parser.add_argument('--seed',                type=int,   default=1337)
parser.add_argument('--num_classes',         type=int,   default=4)
parser.add_argument('--labeled_bs',          type=int,   default=12,  help='有标签子批大小')
parser.add_argument('--labelnum',            type=int,   default=7,   help='有标签患者数')
parser.add_argument('--u_weight',            type=float, default=0.5, help='无标签基础权重')
parser.add_argument('--gpu',                 type=str,   default='0')
parser.add_argument('--consistency',         type=float, default=0.1)
parser.add_argument('--consistency_rampup',  type=float, default=200.0)

# ---- 教师置信度阈值（与 MAE 脚本保持一致）----
parser.add_argument('--conf_thresh_init',    type=float, default=0.90,
                    help='自训练初期教师伪标签置信度阈值（较高，更保守）')
parser.add_argument('--conf_thresh_final',   type=float, default=0.70,
                    help='自训练后期教师伪标签置信度阈值（逐步放宽）')

# ---- CMC 专用参数 ----
parser.add_argument('--cmc_patch_size',      type=int,   default=16,
                    help='CMC 网格块大小（像素）')
parser.add_argument('--cmc_warmup_iter',     type=int,   default=5000,
                    help='CMC 渐进热身步数：shared_ratio 从 init 线性退火到 0')
parser.add_argument('--cmc_init_shared',     type=float, default=0.4,
                    help='热身初期两视图共享块比例（0=纯互补，0.4=40%块双方均可见）')
parser.add_argument('--cmc_loss_weight',     type=float, default=1.0,
                    help='CMC 总损失乘子（与 consistency_weight 相乘后作用于总损失）')
parser.add_argument('--cmc_mutual_weight',   type=float, default=0.5,
                    help='互教损失在 CMC 总损失中的权重: L_CMC = L_anchor + λ·L_mutual')
parser.add_argument('--cmc_mutual_conf_thresh', type=float, default=0.75,
                    help='互补互教的最低置信度门限（低于此的预测不用于教对方）')

args = parser.parse_args()


# ================================================================
# Pre-train 阶段：仅有标签数据，与原始 BCP 完全一致
# ================================================================
def pre_train(args, snapshot_path):
    base_lr    = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs = int(args.labeled_bs / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices: {}, labeled slices: {}".format(total_slices, labeled_slice))

    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num        = 0
    max_epoch       = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd          = 100.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            img_a  = volume_batch[:labeled_sub_bs]
            img_b  = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)
            gt_mixl    = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input  = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl   = model(net_input)

            loss_dice, loss_ce = mix_loss(
                out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True
            )
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/total_loss', loss, iter_num)
            writer.add_scalar('pre/mix_dice',   loss_dice, iter_num)
            writer.add_scalar('pre/mix_ce',     loss_ce,   iter_num)
            logging.info(
                'pre iter %d: loss=%.4f, dice=%.4f, ce=%.4f' %
                (iter_num, loss.item(), loss_dice.item(), loss_ce.item())
            )

            if iter_num % 20 == 0:
                image   = net_input[1, 0:1, :, :]
                writer.add_image('pre/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre/Mixed_Pred', outputs[1, ...] * 50, iter_num)
                writer.add_image('pre/Mixed_GT',   gt_mixl[1, ...].unsqueeze(0) * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        model, classes=num_classes, patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)

                mean_dice, mean_jaccard, mean_hd95, mean_asd = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="pre_val"
                )

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path,
                                              'iter_{}_dice_{}.pth'.format(
                                                  iter_num, round(best_performance, 4))))
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path,
                                              '{}_best_model.pth'.format(args.model)))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info("Pre-training done. Best dice: {:.4f}".format(best_performance))


# ================================================================
# Self-train 阶段：BCP + CMC 联合训练（核心改进）
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr         = args.base_lr
    num_classes     = args.num_classes
    max_iterations  = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model = os.path.join(
        pre_snapshot_path, '{}_best_model.pth'.format(args.model)
    )
    labeled_sub_bs   = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    # ---- 模型初始化 ----
    model     = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices: {}, labeled slices: {}".format(total_slices, labeled_slice))

    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # 加载预训练权重（学生用 optimizer 状态，教师仅加载网络权重）
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded pre-trained weights from {}".format(pre_trained_model))

    # ---- CMC 掩码生成器初始化 ----
    cmc_mask_gen = CMCGridMaskGenerator(
        img_size=args.patch_size[0],
        patch_size=args.cmc_patch_size
    )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (BCP + CMC)")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info(
        "CMC config: patch_size={}, warmup={}, init_shared={:.2f}, "
        "mutual_weight={:.2f}, mutual_conf_thresh={:.2f}".format(
            args.cmc_patch_size, args.cmc_warmup_iter, args.cmc_init_shared,
            args.cmc_mutual_weight, args.cmc_mutual_conf_thresh
        )
    )

    model.train()
    ema_model.train()

    iter_num         = 0
    max_epoch        = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd          = 100.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            # ---- 数据拆分 ----
            img_a  = volume_batch[:labeled_sub_bs]
            img_b  = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a  = label_batch[:labeled_sub_bs]
            lab_b  = label_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            ulab_b = label_batch[args.labeled_bs + unlabeled_sub_bs:]

            # ============================================================
            # Part 1: BCP 双向 copy-paste（保持与原始 BCP 完全一致）
            # ============================================================
            with torch.no_grad():
                # EMA 教师处理完整无标签图像 → 生成伪标签（同时复用于 CMC Part 2）
                pre_a = ema_model(uimg_a)   # 完整图像, [B, C, H, W] 原始 logit
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)   # NMS 清理后的硬标签 [B, H, W]
                plab_b = get_ACDC_masks(pre_b, nms=1)

                img_mask, loss_mask = BCPMaskGenerator.generate(img_a)

                # 可视化用混合标签
                unl_label_vis = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label_vis   = lab_b  * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(
                iter_num // 150, args.consistency, args.consistency_rampup
            )

            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l   = img_b  * img_mask + uimg_b * (1 - img_mask)

            out_unl = model(net_input_unl)
            out_l   = model(net_input_l)

            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask,
                                         u_weight=args.u_weight, unlab=True)
            l_dice,   l_ce   = mix_loss(out_l,   lab_b,  plab_b, loss_mask,
                                         u_weight=args.u_weight)

            loss_bcp_dice = unl_dice + l_dice
            loss_bcp_ce   = unl_ce   + l_ce
            loss_bcp      = (loss_bcp_dice + loss_bcp_ce) / 2

            # ============================================================
            # Part 2: CMC 互补掩码一致性分支（本脚本核心创新）
            #
            # 流程：
            #   ① 生成互补掩码对 M_a, M_b（M_a + M_b = 1 在纯互补模式下）
            #   ② 同一学生模型处理两个互补视图 → pred_viewA, pred_viewB
            #   ③ EMA 教师完整图像输出（已在 Part 1 计算）→ 锚点伪标签
            #   ④ 三重损失：anchor（两视图→教师）+ mutual（盲区互教）
            #
            # 内存优化：
            #   将两个互补视图 concat 为一个大 batch，仅做一次 forward call
            #   [B, 1, H, W] cat [B, 1, H, W] → [2B, 1, H, W]，节省一次前向
            # ============================================================

            # ---- 2.1 渐进式共享比例（热身阶段从部分重叠逐步退火到纯互补）----
            # 早期 shared_ratio 较高（如0.4），两视图各看到~70%图像，任务宽松
            # 热身后 shared_ratio=0，每视图仅看到~50%图像，难度最大
            current_shared_ratio = CMCGridMaskGenerator.get_progressive_shared_ratio(
                current_iter=iter_num,
                warmup_iter=args.cmc_warmup_iter,
                init_ratio=args.cmc_init_shared,
                final_ratio=0.0
            )
            cmc_mask_gen.set_shared_ratio(current_shared_ratio)

            # ---- 2.2 生成互补掩码对（uimg_a 和 uimg_b 各独立一套）----
            mask_a_ab, mask_b_ab = cmc_mask_gen.generate(uimg_a.shape[0], uimg_a.device)
            mask_a_cd, mask_b_cd = cmc_mask_gen.generate(uimg_b.shape[0], uimg_b.device)

            # ---- 2.3 构建互补视图 ----
            # uimg_a 的两个互补视图
            uimg_a_viewA = uimg_a * mask_a_ab   # 保留 M_a 区域
            uimg_a_viewB = uimg_a * mask_b_ab   # 保留互补 M_b 区域
            # uimg_b 的两个互补视图
            uimg_b_viewC = uimg_b * mask_a_cd
            uimg_b_viewD = uimg_b * mask_b_cd

            # ---- 2.4 批量化 forward（两互补视图 concat，减少一次 forward）----
            # 对 uimg_a：concat([viewA, viewB]) → [2*usbs, 1, H, W]
            usbs = unlabeled_sub_bs
            out_cmc_ab = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
            out_cmc_cd = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))

            # 拆分回各视图的输出
            out_a_viewA, out_a_viewB = out_cmc_ab[:usbs], out_cmc_ab[usbs:]
            out_b_viewC, out_b_viewD = out_cmc_cd[:usbs], out_cmc_cd[usbs:]

            # ---- 2.5 自适应教师置信度阈值（随训练进行逐步降低，扩大监督范围）----
            current_conf_thresh = get_adaptive_threshold(
                current_iter=iter_num,
                max_iter=max_iterations,
                init_threshold=args.conf_thresh_init,
                final_threshold=args.conf_thresh_final
            )

            # ---- 2.6 教师伪标签 + 置信度掩码（复用 Part 1 的 EMA 前向结果）----
            with torch.no_grad():
                # get_confidence_mask 返回：(硬伪标签, 置信度浮点掩码, 均值置信度)
                plab_a_full, conf_mask_a, mean_conf_a = get_confidence_mask(
                    pre_a, threshold=current_conf_thresh
                )
                plab_b_full, conf_mask_b, mean_conf_b = get_confidence_mask(
                    pre_b, threshold=current_conf_thresh
                )
                # 用 NMS 清理后的伪标签覆盖（与 BCP 保持一致）
                plab_a_full = get_ACDC_masks(pre_a, nms=1).long()
                plab_b_full = get_ACDC_masks(pre_b, nms=1).long()

            # ---- 2.7 CMC 三重损失 ----
            # 对 uimg_a 的两个互补视图计算损失
            loss_anchor_a, loss_mutual_a, stats_a = cmc_consistency_loss(
                pred_a=out_a_viewA,
                pred_b=out_a_viewB,
                teacher_plab=plab_a_full,
                teacher_conf_mask=conf_mask_a,
                mask_a=mask_a_ab,
                mask_b=mask_b_ab,
                n_classes=num_classes,
                mutual_conf_thresh=args.cmc_mutual_conf_thresh
            )
            # 对 uimg_b 的两个互补视图计算损失
            loss_anchor_b, loss_mutual_b, stats_b = cmc_consistency_loss(
                pred_a=out_b_viewC,
                pred_b=out_b_viewD,
                teacher_plab=plab_b_full,
                teacher_conf_mask=conf_mask_b,
                mask_a=mask_a_cd,
                mask_b=mask_b_cd,
                n_classes=num_classes,
                mutual_conf_thresh=args.cmc_mutual_conf_thresh
            )

            # 汇总两个无标签子批的损失
            loss_anchor = (loss_anchor_a + loss_anchor_b) / 2
            loss_mutual = (loss_mutual_a + loss_mutual_b) / 2

            # CMC 总损失：锚点 + 互教
            loss_cmc = loss_anchor + args.cmc_mutual_weight * loss_mutual

            # ============================================================
            # Part 3: 总损失 = BCP + CMC（带 ramp-up 权重）
            # ============================================================
            # CMC ramp-up：训练初期权重从 0 逐步增大到 cmc_loss_weight
            # 避免早期低质量预测干扰有标签数据的监督
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            cmc_weight = args.cmc_loss_weight * cmc_rampup

            loss = loss_bcp + consistency_weight * cmc_weight * loss_cmc

            # ---- 优化 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            current_lr = poly_lr(optimizer, base_lr, iter_num, max_iterations)

            # ============================================================
            # 日志记录
            # ============================================================
            writer.add_scalar('info/total_loss',        loss,               iter_num)
            writer.add_scalar('info/loss_bcp',          loss_bcp,           iter_num)
            writer.add_scalar('info/loss_bcp_dice',     loss_bcp_dice,      iter_num)
            writer.add_scalar('info/loss_bcp_ce',       loss_bcp_ce,        iter_num)
            writer.add_scalar('info/loss_cmc',          loss_cmc,           iter_num)
            writer.add_scalar('info/loss_anchor',       loss_anchor,        iter_num)
            writer.add_scalar('info/loss_mutual',       loss_mutual,        iter_num)
            writer.add_scalar('info/cmc_weight',        cmc_weight,         iter_num)
            writer.add_scalar('info/shared_ratio',      current_shared_ratio, iter_num)
            writer.add_scalar('info/conf_threshold',    current_conf_thresh, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/lr',                current_lr,         iter_num)
            # CMC 细粒度统计（来自 stats_a）
            writer.add_scalar('cmc/mean_conf_a',        stats_a['mean_conf_a'],        iter_num)
            writer.add_scalar('cmc/mean_conf_b',        stats_a['mean_conf_b'],        iter_num)
            writer.add_scalar('cmc/high_conf_ratio_a',  stats_a['high_conf_ratio_a'],  iter_num)
            writer.add_scalar('cmc/high_conf_ratio_b',  stats_a['high_conf_ratio_b'],  iter_num)
            writer.add_scalar('cmc/valid_a_learn_px',   stats_a['valid_a_learn_px'],   iter_num)
            writer.add_scalar('cmc/valid_b_learn_px',   stats_a['valid_b_learn_px'],   iter_num)
            writer.add_scalar('cmc/teacher_conf_a',     mean_conf_a,                   iter_num)
            writer.add_scalar('cmc/teacher_conf_b',     mean_conf_b,                   iter_num)

            logging.info(
                'iter %d | total=%.4f | bcp=%.4f | cmc=%.4f '
                '(anchor=%.4f mutual=%.4f) | cmc_w=%.3f | '
                'shared=%.2f | conf_t=%.2f | lr=%.5f' % (
                    iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                    loss_anchor.item(), loss_mutual.item(), cmc_weight,
                    current_shared_ratio, current_conf_thresh, current_lr
                )
            )

            # ============================================================
            # TensorBoard 可视化
            # ============================================================
            if iter_num % 20 == 0:
                # BCP 分支可视化
                writer.add_image('bcp/Un_Input',  net_input_unl[0, 0:1], iter_num)
                writer.add_image('bcp/Un_GT',     unl_label_vis[0].unsqueeze(0).float() * 50, iter_num)
                pred_unl = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('bcp/Un_Pred',   pred_unl[0].float() * 50, iter_num)

                writer.add_image('bcp/L_Input',   net_input_l[0, 0:1], iter_num)
                writer.add_image('bcp/L_GT',      l_label_vis[0].unsqueeze(0).float() * 50, iter_num)
                pred_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('bcp/L_Pred',    pred_l[0].float() * 50, iter_num)

                # CMC 分支可视化
                # 原始图 / 视图A（M_a遮挡后）/ 视图B（M_b遮挡后，互补）
                writer.add_image('cmc/Original',    uimg_a[0, 0:1],         iter_num)
                writer.add_image('cmc/ViewA_Input', uimg_a_viewA[0, 0:1],  iter_num)
                writer.add_image('cmc/ViewB_Input', uimg_a_viewB[0, 0:1],  iter_num)
                writer.add_image('cmc/MaskA',       mask_a_ab[0, 0:1],     iter_num)
                writer.add_image('cmc/MaskB',       mask_b_ab[0, 0:1],     iter_num)

                pred_viewA = torch.argmax(torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                pred_viewB = torch.argmax(torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                writer.add_image('cmc/PredViewA',   pred_viewA[0].float() * 50, iter_num)
                writer.add_image('cmc/PredViewB',   pred_viewB[0].float() * 50, iter_num)
                writer.add_image('cmc/PseudoLabel', plab_a_full[0].unsqueeze(0).float() * 50, iter_num)
                writer.add_image('cmc/ConfMask',    conf_mask_a[0].unsqueeze(0), iter_num)

                # 可视化互教掩码（A独有区域 中 A高置信的子区域）
                excl_a_vis = mask_a_ab.squeeze(1)
                high_a_vis = (torch.softmax(out_a_viewA, dim=1).max(dim=1).values
                            > args.cmc_mutual_conf_thresh).float()
                mutual_vis = (excl_a_vis[0] * high_a_vis[0]).unsqueeze(0)  # [1, H, W]
                writer.add_image('cmc/MutualTeachingZone_A', mutual_vis, iter_num)

            # ============================================================
            # 验证（Dice, Jaccard, HD95, ASD 四指标）
            # ============================================================
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        model, classes=num_classes, patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)

                mean_dice, mean_jaccard, mean_hd95, mean_asd = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="val"
                )

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path,
                                            'iter_{}_dice_{}.pth'.format(
                                                iter_num, round(best_performance, 4))))
                    torch.save(model.state_dict(),
                               os.path.join(snapshot_path,
                                            '{}_best_model.pth'.format(args.model)))
                    logging.info("=> Saved best model, iter={}, dice={:.4f}".format(
                        iter_num, best_performance
                    ))

                if mean_hd95 < best_hd:
                    best_hd = mean_hd95

                logging.info('BEST | dice={:.4f}, hd95={:.2f}'.format(
                    best_performance, best_hd
                ))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info("=" * 60)
    logging.info("Self-training finished.")
    logging.info("Best mean_dice : {:.4f}".format(best_performance))
    logging.info("Best mean_hd95 : {:.2f}".format(best_hd))
    logging.info("=" * 60)


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark   = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum
    )
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum
    )
    for path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    shutil.copy(__file__, self_snapshot_path)

    # ---- Pre-train 阶段 ----
    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # ---- Self-train 阶段（重置 handler 避免日志重复）----
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)