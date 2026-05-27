"""
BCP + CMC-Fusion 半监督医学图像分割实验脚本

实验命名: BCP_CMC_fusion_student
指标输出: Dice, Jaccard, HD95, ASD

核心思想：互补掩码预测融合一致性（CMC-Fusion）
==============================================

与 BCP_CMC_student（互教版）的唯一区别在于无标签分支的损失设计：

  互教版（v1）：
    L_CMC = L_anchor + λ_mutual * L_mutual
    L_mutual：A 的预测监督 B 的盲区，B 的预测监督 A 的盲区
    缺陷：互教信号来自学生，早期质量差；与 L_anchor 存在冗余

  融合版（v2，本脚本）：
    L_CMC = L_anchor + λ_fusion * L_fusion
    L_anchor：每视图独立对齐教师硬伪标签（pointwise 约束）
    L_fusion：两视图概率均值对齐教师软概率（pairwise 约束）
             H(p_teacher, (p_A + p_B) / 2)
    优势：
      ① 信号质量完全依赖 EMA 教师，不依赖学生的预测质量
      ② 直接利用互补结构的数学性质：avg(view_A, view_B) ≈ full
      ③ 由 Jensen 不等式，L_fusion 比 L_anchor 约束更紧，不冗余
      ④ 梯度同时流向两视图，自然强迫两者协调

参数对比（相较互教版）：
  删除：--cmc_mutual_weight, --cmc_mutual_conf_thresh
  新增：--cmc_fusion_weight（融合损失权重，默认 1.0）
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
from utils.cmc_utils import CMCGridMaskGenerator, cmc_fusion_consistency_loss


# ================================================================
# 参数定义
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path',           type=str,   default='../data_split/ACDC')
parser.add_argument('--exp',                 type=str,   default='BCP_CMC_fusion_student')
parser.add_argument('--model',               type=str,   default='unet')
parser.add_argument('--pre_iterations',      type=int,   default=10000)
parser.add_argument('--max_iterations',      type=int,   default=30000)
parser.add_argument('--batch_size',          type=int,   default=24)
parser.add_argument('--deterministic',       type=int,   default=1)
parser.add_argument('--base_lr',             type=float, default=0.01)
parser.add_argument('--patch_size',          type=list,  default=[256, 256])
parser.add_argument('--seed',                type=int,   default=1337)
parser.add_argument('--num_classes',         type=int,   default=4)
parser.add_argument('--labeled_bs',          type=int,   default=12)
parser.add_argument('--labelnum',            type=int,   default=7)
parser.add_argument('--u_weight',            type=float, default=0.5)
parser.add_argument('--gpu',                 type=str,   default='0')
parser.add_argument('--consistency',         type=float, default=0.1)
parser.add_argument('--consistency_rampup',  type=float, default=200.0)

# 教师置信度阈值（自适应退火）
parser.add_argument('--conf_thresh_init',    type=float, default=0.90,
                    help='自训练初期教师置信度阈值（保守）')
parser.add_argument('--conf_thresh_final',   type=float, default=0.70,
                    help='自训练后期教师置信度阈值（逐步放宽）')

# CMC 通用参数
parser.add_argument('--cmc_patch_size',      type=int,   default=16,
                    help='互补掩码的网格块大小（像素）')
parser.add_argument('--cmc_warmup_iter',     type=int,   default=5000,
                    help='shared_ratio 从 init 退火到 0 的步数')
parser.add_argument('--cmc_init_shared',     type=float, default=0.4,
                    help='热身初期两视图共享块比例')
parser.add_argument('--cmc_loss_weight',     type=float, default=1.0,
                    help='CMC 总损失乘子（与 consistency_weight 相乘）')

# CMC-Fusion 专用参数（替换互教版的 mutual_weight 和 mutual_conf_thresh）
parser.add_argument('--cmc_fusion_weight',   type=float, default=1.0,
                    help='融合损失权重：L_CMC = L_anchor + cmc_fusion_weight * L_fusion')

args = parser.parse_args()


# ================================================================
# Pre-train 阶段：仅有标签数据，与原始 BCP 完全一致
# ================================================================
def pre_train(args, snapshot_path):
    base_lr     = args.base_lr
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
    iter_num         = 0
    max_epoch        = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)
            gt_mixl   = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl  = model(net_input)

            loss_dice, loss_ce = mix_loss(
                out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True
            )
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/total_loss', loss,      iter_num)
            writer.add_scalar('pre/mix_dice',   loss_dice, iter_num)
            writer.add_scalar('pre/mix_ce',     loss_ce,   iter_num)
            logging.info('pre iter %d: loss=%.4f, dice=%.4f, ce=%.4f' %
                         (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                writer.add_image('pre/Mixed_Image', net_input[1, 0:1],      iter_num)
                out_vis = torch.argmax(
                    torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre/Mixed_Pred', out_vis[1, ...] * 50,    iter_num)
                writer.add_image('pre/Mixed_GT',
                                 gt_mixl[1, ...].unsqueeze(0) * 50,         iter_num)

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

                mean_dice, *_ = log_validation_metrics(
                    metric_list, num_classes, iter_num,
                    writer, logging, prefix="pre_val"
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
# Self-train 阶段：BCP + CMC-Fusion 联合训练
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr        = args.base_lr
    num_classes    = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model = os.path.join(
        pre_snapshot_path, '{}_best_model.pth'.format(args.model)
    )
    labeled_sub_bs   = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

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

    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded pre-trained weights from {}".format(pre_trained_model))

    # CMC 掩码生成器
    cmc_mask_gen = CMCGridMaskGenerator(
        img_size=args.patch_size[0],
        patch_size=args.cmc_patch_size
    )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (BCP + CMC-Fusion)")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info(
        "CMC-Fusion config: patch_size={}, warmup={}, init_shared={:.2f}, "
        "cmc_loss_weight={:.2f}, fusion_weight={:.2f}".format(
            args.cmc_patch_size, args.cmc_warmup_iter, args.cmc_init_shared,
            args.cmc_loss_weight, args.cmc_fusion_weight
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
            # Part 1: BCP 双向 copy-paste（完全保持原始逻辑）
            # ============================================================
            with torch.no_grad():
                # EMA 教师处理完整无标签图像
                # pre_a / pre_b 是原始 logit，后续 CMC-Fusion 也复用这个结果
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1).long()
                plab_b = get_ACDC_masks(pre_b, nms=1).long()

                img_mask, loss_mask = BCPMaskGenerator.generate(img_a)

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
            # Part 2: CMC-Fusion 分支
            #
            # 流程：
            #   ① 生成互补掩码对（渐进式 shared_ratio 热身）
            #   ② concat 两视图为一个大 batch，一次 forward 得到两视图输出
            #   ③ 调用 cmc_fusion_consistency_loss：
            #      - L_anchor：每视图独立对齐教师硬标签
            #      - L_fusion：两视图概率均值对齐教师软概率
            #
            # 教师 logit（pre_a / pre_b）直接复用 Part 1 的计算结果
            # ============================================================

            # ---- 2.1 渐进式 shared_ratio ----
            current_shared_ratio = CMCGridMaskGenerator.get_progressive_shared_ratio(
                current_iter=iter_num,
                warmup_iter=args.cmc_warmup_iter,
                init_ratio=args.cmc_init_shared,
                final_ratio=0.0
            )
            cmc_mask_gen.set_shared_ratio(current_shared_ratio)

            # ---- 2.2 生成互补掩码对 ----
            mask_a_ab, mask_b_ab = cmc_mask_gen.generate(uimg_a.shape[0], uimg_a.device)
            mask_a_cd, mask_b_cd = cmc_mask_gen.generate(uimg_b.shape[0], uimg_b.device)

            # ---- 2.3 构建互补视图 ----
            uimg_a_viewA = uimg_a * mask_a_ab
            uimg_a_viewB = uimg_a * mask_b_ab
            uimg_b_viewC = uimg_b * mask_a_cd
            uimg_b_viewD = uimg_b * mask_b_cd

            # ---- 2.4 批量化 forward（两视图 concat，减少一次 kernel launch）----
            usbs = unlabeled_sub_bs
            out_cmc_ab = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
            out_cmc_cd = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))

            out_a_viewA, out_a_viewB = out_cmc_ab[:usbs], out_cmc_ab[usbs:]
            out_b_viewC, out_b_viewD = out_cmc_cd[:usbs], out_cmc_cd[usbs:]

            # ---- 2.5 自适应置信度阈值 ----
            current_conf_thresh = get_adaptive_threshold(
                current_iter=iter_num,
                max_iter=max_iterations,
                init_threshold=args.conf_thresh_init,
                final_threshold=args.conf_thresh_final
            )

            # ---- 2.6 教师置信度掩码（复用 Part 1 的 EMA 结果）----
            with torch.no_grad():
                _, conf_mask_a, mean_conf_a = get_confidence_mask(
                    pre_a, threshold=current_conf_thresh
                )
                _, conf_mask_b, mean_conf_b = get_confidence_mask(
                    pre_b, threshold=current_conf_thresh
                )
            # 注意：teacher_logit 直接传 pre_a / pre_b（原始 logit）
            # cmc_fusion_consistency_loss 内部会做 softmax 以获得软标签

            # ---- 2.7 CMC-Fusion 损失 ----
            loss_anchor_a, loss_fusion_a, stats_a = cmc_fusion_consistency_loss(
                pred_a=out_a_viewA,
                pred_b=out_a_viewB,
                teacher_logit=pre_a,
                teacher_conf_mask=conf_mask_a,
                n_classes=num_classes
            )
            loss_anchor_b, loss_fusion_b, stats_b = cmc_fusion_consistency_loss(
                pred_a=out_b_viewC,
                pred_b=out_b_viewD,
                teacher_logit=pre_b,
                teacher_conf_mask=conf_mask_b,
                n_classes=num_classes
            )

            loss_anchor = (loss_anchor_a + loss_anchor_b) / 2
            loss_fusion = (loss_fusion_a + loss_fusion_b) / 2

            # CMC 总损失
            loss_cmc = loss_anchor + args.cmc_fusion_weight * loss_fusion

            # ============================================================
            # Part 3: 总损失
            # ============================================================
            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            cmc_weight = args.cmc_loss_weight * cmc_rampup

            loss = loss_bcp + consistency_weight * cmc_weight * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)
            current_lr = poly_lr(optimizer, base_lr, iter_num, max_iterations)
            # 这里和原始设计不同

            # ============================================================
            # TensorBoard 日志
            # ============================================================
            writer.add_scalar('info/total_loss',         loss,               iter_num)
            writer.add_scalar('info/loss_bcp',           loss_bcp,           iter_num)
            writer.add_scalar('info/loss_bcp_dice',      loss_bcp_dice,      iter_num)
            writer.add_scalar('info/loss_bcp_ce',        loss_bcp_ce,        iter_num)
            writer.add_scalar('info/loss_cmc',           loss_cmc,           iter_num)
            writer.add_scalar('info/loss_anchor',        loss_anchor,        iter_num)
            writer.add_scalar('info/loss_fusion',        loss_fusion,        iter_num)
            writer.add_scalar('info/cmc_weight',         cmc_weight,         iter_num)
            writer.add_scalar('info/shared_ratio',       current_shared_ratio, iter_num)
            writer.add_scalar('info/conf_threshold',     current_conf_thresh,  iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight,   iter_num)
            writer.add_scalar('info/lr',                 current_lr,           iter_num)
            # CMC-Fusion 细粒度统计
            writer.add_scalar('cmc/conf_a_mean',         stats_a['conf_a_mean'],        iter_num)
            writer.add_scalar('cmc/conf_b_mean',         stats_a['conf_b_mean'],        iter_num)
            writer.add_scalar('cmc/fused_conf_mean',     stats_a['fused_conf_mean'],    iter_num)
            # agree_ratio：两视图预测一致的像素比例，期望随训练上升
            writer.add_scalar('cmc/agree_ratio',         stats_a['agree_ratio'],        iter_num)
            writer.add_scalar('cmc/teacher_conf_a',      mean_conf_a,                   iter_num)
            writer.add_scalar('cmc/teacher_conf_b',      mean_conf_b,                   iter_num)

            logging.info(
                'iter %d | total=%.4f | bcp=%.4f | '
                'anchor=%.4f fusion=%.4f | '
                'cmc_w=%.3f | shared=%.2f | conf_t=%.2f | '
                'agree=%.3f | lr=%.5f' % (
                    iter_num, loss.item(), loss_bcp.item(),
                    loss_anchor.item(), loss_fusion.item(),
                    cmc_weight, current_shared_ratio, current_conf_thresh,
                    stats_a['agree_ratio'], current_lr
                )
            )

            # ============================================================
            # TensorBoard 可视化
            # ============================================================
            if iter_num % 20 == 0:
                # BCP 分支
                writer.add_image('bcp/Un_Input', net_input_unl[0, 0:1], iter_num)
                writer.add_image('bcp/Un_GT',
                                 unl_label_vis[0].unsqueeze(0).float() * 50, iter_num)
                pred_unl = torch.argmax(
                    torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('bcp/Un_Pred', pred_unl[0].float() * 50, iter_num)

                # CMC-Fusion 分支
                writer.add_image('cmc/Original',    uimg_a[0, 0:1],        iter_num)
                writer.add_image('cmc/ViewA_Input', uimg_a_viewA[0, 0:1],  iter_num)
                writer.add_image('cmc/ViewB_Input', uimg_a_viewB[0, 0:1],  iter_num)
                writer.add_image('cmc/MaskA',       mask_a_ab[0],          iter_num)
                writer.add_image('cmc/MaskB',       mask_b_ab[0],          iter_num)

                pred_viewA = torch.argmax(
                    torch.softmax(out_a_viewA, dim=1), dim=1, keepdim=True)
                pred_viewB = torch.argmax(
                    torch.softmax(out_a_viewB, dim=1), dim=1, keepdim=True)
                writer.add_image('cmc/PredViewA', pred_viewA[0].float() * 50, iter_num)
                writer.add_image('cmc/PredViewB', pred_viewB[0].float() * 50, iter_num)

                # 融合预测可视化
                with torch.no_grad():
                    prob_fused_vis = (
                        torch.softmax(out_a_viewA, dim=1) +
                        torch.softmax(out_a_viewB, dim=1)
                    ) / 2.0
                    pred_fused_vis = prob_fused_vis.argmax(dim=1, keepdim=True)
                writer.add_image('cmc/PredFused',
                                 pred_fused_vis[0].float() * 50, iter_num)

                # 教师参考
                with torch.no_grad():
                    teacher_pred_vis = pre_a.argmax(dim=1, keepdim=True)
                writer.add_image('cmc/TeacherPred',
                                 teacher_pred_vis[0].float() * 50, iter_num)
                writer.add_image('cmc/ConfMask',
                                 conf_mask_a[0].unsqueeze(0), iter_num)

                # 两视图一致性图（高亮一致区域）
                with torch.no_grad():
                    agree_vis = (
                        pred_viewA[0] == pred_viewB[0]
                    ).float()
                writer.add_image('cmc/AgreeMap', agree_vis, iter_num)

            # ============================================================
            # 验证（Dice, Jaccard, HD95, ASD）
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
                        iter_num, best_performance))

                if mean_hd95 < best_hd:
                    best_hd = mean_hd95

                logging.info('BEST | dice={:.4f}, hd95={:.2f}'.format(
                    best_performance, best_hd))
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
        cudnn.benchmark    = False
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

    # Pre-train
    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
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
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)