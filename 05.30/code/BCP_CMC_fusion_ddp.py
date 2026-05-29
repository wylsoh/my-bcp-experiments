"""
===============================================================================
BCP + CMC-Fusion — DDP + AMP 最优配置版本
===============================================================================

实验命名: BCP_CMC_fusion_student
指标输出: Dice, Jaccard, HD95, ASD

与原始 BCP_CMC_fusion.py 的核心区别:
  [DDP]   torchrun 多卡启动，DistributedTwoStreamBatchSampler
  [AMP]   autocast + GradScaler 混合精度训练
  [cudnn] cudnn.benchmark = True（固定输入尺寸）
  [SyncBN] SyncBatchNorm 自动转换（多卡时）
  [Save]  仅 rank 0 保存 checkpoint 和日志
  [EMA]   仅 rank 0 更新 EMA（各 rank 模型权重相同）

启动方式:
    # 4 GPU 最优配置（batch_size=96, labeled_bs=24, AMP）
    torchrun --nproc_per_node=4 BCP_CMC_fusion_ddp.py \
        --root_path ../data_split/ACDC \
        --exp BCP_CMC_fusion_student_ddp \
        --pre_iterations 10000 \
        --max_iterations 30000 \
        --batch_size 96 --labeled_bs 24 \
        --amp --ddp

    # 单 GPU 传统模式
    python BCP_CMC_fusion_ddp.py \
        --batch_size 24 --labeled_bs 12 --gpu 0

    # 2 GPU 调试
    torchrun --nproc_per_node=2 BCP_CMC_fusion_ddp.py \
        --batch_size 48 --labeled_bs 12 --amp --ddp \
        --pre_iterations 1000 --max_iterations 2000
===============================================================================

核心算法（与原始版一致）:
  ① Part 1: BCP 双向 copy-paste（监督分支）
  ② Part 2: CMC-Fusion（互补掩码预测融合）→ L_anchor + λ * L_fusion
  ③ Part 3: EMA 教师生成伪标签 + 自适应置信度阈值
  ④ 总损失: L = L_bcp + λ_consistency * λ_cmc * L_cmc
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

from dataloaders.dataset import (
    BaseDataSets, RandomGenerator, TwoStreamBatchSampler
)
from networks.net_factory import BCP_net
from utils.train_utils import (
    load_net, load_net_opt, save_net_opt,
    update_model_ema, get_current_consistency_weight,
    poly_lr, mix_loss, patients_to_slices,
    init_distributed_mode, get_rank, get_world_size, is_main_process,
    get_amp_autocast, get_grad_scaler, unwrap_ddp, convert_model_to_syncbn,
    save_checkpoint, log_info, log_scalar, log_images
)
from utils.mask_generator import BCPMaskGenerator
from utils.pseudo_label_utils import (
    get_ACDC_masks, get_confidence_mask, get_adaptive_threshold
)
from utils.metric_utils import (
    test_single_volume_all_metrics, log_validation_metrics
)
from utils.cmc_utils import CMCGridMaskGenerator, cmc_fusion_consistency_loss
from ddp_train_adapter import (
    add_ddp_args, ddp_wrap_model, create_ddp_dataloader, AMPTrainer,
    print_ddp_config
)


# ================================================================
# 参数定义
# ================================================================
parser = argparse.ArgumentParser()
parser = add_ddp_args(parser)  # 注入 --ddp, --amp, --gpu, --sync_bn

parser.add_argument('--root_path',           type=str,   default='../data_split/ACDC')
parser.add_argument('--exp',                 type=str,   default='BCP_CMC_fusion_student_ddp')
parser.add_argument('--model',               type=str,   default='unet')
parser.add_argument('--pre_iterations',      type=int,   default=10000)
parser.add_argument('--max_iterations',      type=int,   default=30000)
parser.add_argument('--batch_size',          type=int,   default=96)
parser.add_argument('--deterministic',       type=int,   default=0)  # 默认关闭 deterministic
parser.add_argument('--base_lr',             type=float, default=0.01)
parser.add_argument('--patch_size',          type=list,  default=[256, 256])
parser.add_argument('--seed',                type=int,   default=1337)
parser.add_argument('--num_classes',         type=int,   default=4)
parser.add_argument('--labeled_bs',          type=int,   default=24)
parser.add_argument('--labelnum',            type=int,   default=7)
parser.add_argument('--u_weight',            type=float, default=0.5)
parser.add_argument('--consistency',         type=float, default=0.1)
parser.add_argument('--consistency_rampup',  type=float, default=200.0)

# 教师置信度阈值（自适应退火）
parser.add_argument('--conf_thresh_init',    type=float, default=0.90)
parser.add_argument('--conf_thresh_final',   type=float, default=0.70)

# CMC 通用参数
parser.add_argument('--cmc_patch_size',      type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',     type=int,   default=5000)
parser.add_argument('--cmc_init_shared',     type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',     type=float, default=1.0)

# CMC-Fusion 专用参数
parser.add_argument('--cmc_fusion_weight',   type=float, default=1.0)

# EMA 更新间隔（DDP 减少通信）
parser.add_argument('--ema_update_freq',     type=int,   default=5)

# 从外部 checkpoint 恢复训练（跳过 pre-train）
parser.add_argument('--load_path',           type=str,   default=None,
                    help='从已有 checkpoint 恢复训练，跳过 pre-train')

args = parser.parse_args()


# ================================================================
# 预训练阶段
# ================================================================
def pre_train(args):
    """预训练阶段 — DDP + AMP 兼容版"""
    base_lr     = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    labeled_sub_bs = int(args.labeled_bs / 2)
    device = args.device
    rank = args.rank
    world_size = args.world_size

    # --- 模型创建（CPU 上，ddp_wrap_model 负责 to(device) + DDP）---
    model = BCP_net(in_chns=1, class_num=num_classes)
    model = ddp_wrap_model(model, args)
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + rank + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    if is_main_process():
        log_info(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    # --- DDP 兼容 DataLoader ---
    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
        num_workers=4, pin_memory=True,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    writer = SummaryWriter(snapshot_path + '/log') if is_main_process() else None

    if is_main_process():
        logging.info("Start pre_training (DDP + AMP)")
        logging.info(f"{len(trainloader)} iterations per epoch")

    model.train()
    iter_num         = 0
    max_epoch        = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    # DDP: sync BN  stats before training
    if world_size > 1:
        torch.distributed.barrier()

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch  = sampled_batch['label'].to(device, non_blocking=True)

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)
            gt_mixl   = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input = img_a * img_mask + img_b * (1 - img_mask)

            # AMP forward
            with ampt.autocast():
                out_mixl = model(net_input)
                loss_dice, loss_ce = mix_loss(
                    out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True
                )
                loss = (loss_dice + loss_ce) / 2

            # AMP backward
            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)
            iter_num += 1

            if is_main_process():
                writer.add_scalar('pre/total_loss', loss,      iter_num)
                writer.add_scalar('pre/mix_dice',   loss_dice, iter_num)
                writer.add_scalar('pre/mix_ce',     loss_ce,   iter_num)

                if iter_num % 20 == 0:
                    logging.info(f'pre iter {iter_num}: loss={loss.item():.4f}, '
                                 f'dice={loss_dice.item():.4f}, ce={loss_ce.item():.4f}')
                    writer.add_image('pre/Mixed_Image', net_input[1, 0:1], iter_num)
                    out_vis = torch.argmax(
                        torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                    writer.add_image('pre/Mixed_Pred', out_vis[1, ...] * 50, iter_num)
                    writer.add_image('pre/Mixed_GT',
                                     gt_mixl[1, ...].unsqueeze(0) * 50, iter_num)

            # Validation（仅在 rank 0）
            if iter_num > 0 and iter_num % 200 == 0 and is_main_process():
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        unwrap_ddp(model), classes=num_classes,
                        patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)
                mean_dice, *_ = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="pre_val"
                )
                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path,
                                              f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth'))
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path,
                                              f'{args.model}_best_model.pth'))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            if is_main_process():
                iterator.close()
            break

    if is_main_process():
        save_net_opt(model, optimizer,
                     os.path.join(snapshot_path, 'pre_train_model.pth'))
        writer.close()
        logging.info(f"Pre-training done. Best dice: {best_performance:.4f}")


# ================================================================
# 自训练阶段
# ================================================================
def self_train(args, pre_snapshot_path):
    """自训练阶段 — DDP + AMP 兼容版"""
    base_lr        = args.base_lr
    num_classes    = args.num_classes
    max_iterations = args.max_iterations
    device = args.device
    rank = args.rank
    world_size = args.world_size

    pre_trained_model = os.path.join(
        pre_snapshot_path, f'{args.model}_best_model.pth'
    )
    labeled_sub_bs   = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    # --- 模型（CPU 创建 → 自动 DDP 包装）---
    model     = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)
    model     = ddp_wrap_model(model, args)
    ema_model = ema_model.to(device)  # EMA 不 DDP 包装
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + rank + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices  = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    if is_main_process():
        log_info(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
        num_workers=4, pin_memory=True,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    # 加载预训练权重（支持外部 load_path 覆盖）
    if args.load_path is not None:
        pre_trained_model = args.load_path
        if is_main_process():
            log_info(f"Override load path: {pre_trained_model}")
    if is_main_process():
        log_info(f"Loading pre-trained from {pre_trained_model}")
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)

    # CMC 掩码生成器
    cmc_mask_gen = CMCGridMaskGenerator(
        img_size=args.patch_size[0],
        patch_size=args.cmc_patch_size
    )

    writer = SummaryWriter(snapshot_path + '/log') if is_main_process() else None

    if is_main_process():
        logging.info("Start self_training (BCP + CMC-Fusion, DDP+AMP)")
        logging.info(f"{len(trainloader)} iterations per epoch")
        logging.info(
            f"CMC-Fusion: patch_size={args.cmc_patch_size}, "
            f"warmup={args.cmc_warmup_iter}, init_shared={args.cmc_init_shared:.2f}, "
            f"cmc_loss_weight={args.cmc_loss_weight:.2f}, "
            f"fusion_weight={args.cmc_fusion_weight:.2f}"
        )

    model.train()
    ema_model.train()

    iter_num         = 0
    max_epoch        = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd          = 100.0
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    # DDP barrier
    if world_size > 1:
        torch.distributed.barrier()

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch  = sampled_batch['label'].to(device, non_blocking=True)

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
            # Part 1: BCP 双向 copy-paste
            # ============================================================
            with torch.no_grad():
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

            # AMP forward
            with ampt.autocast():
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
                # ============================================================
                current_shared_ratio = CMCGridMaskGenerator.get_progressive_shared_ratio(
                    current_iter=iter_num,
                    warmup_iter=args.cmc_warmup_iter,
                    init_ratio=args.cmc_init_shared,
                    final_ratio=0.0
                )
                cmc_mask_gen.set_shared_ratio(current_shared_ratio)

                mask_a_ab, mask_b_ab = cmc_mask_gen.generate(uimg_a.shape[0], uimg_a.device)
                mask_a_cd, mask_b_cd = cmc_mask_gen.generate(uimg_b.shape[0], uimg_b.device)

                uimg_a_viewA = uimg_a * mask_a_ab
                uimg_a_viewB = uimg_a * mask_b_ab
                uimg_b_viewC = uimg_b * mask_a_cd
                uimg_b_viewD = uimg_b * mask_b_cd

                usbs = unlabeled_sub_bs
                out_cmc_ab = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
                out_cmc_cd = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))

                out_a_viewA, out_a_viewB = out_cmc_ab[:usbs], out_cmc_ab[usbs:]
                out_b_viewC, out_b_viewD = out_cmc_cd[:usbs], out_cmc_cd[usbs:]

                current_conf_thresh = get_adaptive_threshold(
                    current_iter=iter_num, max_iter=max_iterations,
                    init_threshold=args.conf_thresh_init,
                    final_threshold=args.conf_thresh_final
                )

                with torch.no_grad():
                    _, conf_mask_a, mean_conf_a = get_confidence_mask(
                        pre_a, threshold=current_conf_thresh
                    )
                    _, conf_mask_b, mean_conf_b = get_confidence_mask(
                        pre_b, threshold=current_conf_thresh
                    )

                loss_anchor_a, loss_fusion_a, stats_a = cmc_fusion_consistency_loss(
                    pred_a=out_a_viewA, pred_b=out_a_viewB,
                    teacher_logit=pre_a, teacher_conf_mask=conf_mask_a,
                    n_classes=num_classes
                )
                loss_anchor_b, loss_fusion_b, stats_b = cmc_fusion_consistency_loss(
                    pred_a=out_b_viewC, pred_b=out_b_viewD,
                    teacher_logit=pre_b, teacher_conf_mask=conf_mask_b,
                    n_classes=num_classes
                )

                loss_anchor = (loss_anchor_a + loss_anchor_b) / 2
                loss_fusion = (loss_fusion_a + loss_fusion_b) / 2
                loss_cmc = loss_anchor + args.cmc_fusion_weight * loss_fusion

                # ============================================================
                # Part 3: 总损失
                # ============================================================
                cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
                cmc_weight = args.cmc_loss_weight * cmc_rampup
                loss = loss_bcp + consistency_weight * cmc_weight * loss_cmc

            # AMP backward
            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)

            iter_num += 1

            # EMA 仅在 rank 0 更新（各 rank 模型相同）
            if is_main_process() and iter_num % args.ema_update_freq == 0:
                update_model_ema(model, ema_model, 0.99)

            current_lr = poly_lr(optimizer, base_lr, iter_num, max_iterations)

            # ============================================================
            # 日志（仅在 rank 0）
            # ============================================================
            if is_main_process():
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
                writer.add_scalar('cmc/conf_a_mean',         stats_a['conf_a_mean'],        iter_num)
                writer.add_scalar('cmc/conf_b_mean',         stats_a['conf_b_mean'],        iter_num)
                writer.add_scalar('cmc/fused_conf_mean',     stats_a['fused_conf_mean'],    iter_num)
                writer.add_scalar('cmc/agree_ratio',         stats_a['agree_ratio'],        iter_num)
                writer.add_scalar('cmc/teacher_conf_a',      mean_conf_a,                   iter_num)
                writer.add_scalar('cmc/teacher_conf_b',      mean_conf_b,                   iter_num)

                if iter_num % 20 == 0:
                    logging.info(
                        f'iter {iter_num} | total={loss.item():.4f} | '
                        f'bcp={loss_bcp.item():.4f} | '
                        f'anchor={loss_anchor.item():.4f} fusion={loss_fusion.item():.4f} | '
                        f'cmc_w={cmc_weight:.3f} | shared={current_shared_ratio:.2f} | '
                        f'conf_t={current_conf_thresh:.2f} | '
                        f'agree={stats_a["agree_ratio"]:.3f} | lr={current_lr:.5f}'
                    )

                if iter_num % 20 == 0:
                    writer.add_image('bcp/Un_Input', net_input_unl[0, 0:1], iter_num)
                    writer.add_image('bcp/Un_GT',
                                     unl_label_vis[0].unsqueeze(0).float() * 50, iter_num)
                    pred_unl = torch.argmax(
                        torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                    writer.add_image('bcp/Un_Pred', pred_unl[0].float() * 50, iter_num)
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
                    with torch.no_grad():
                        prob_fused_vis = (
                            torch.softmax(out_a_viewA, dim=1) +
                            torch.softmax(out_a_viewB, dim=1)
                        ) / 2.0
                        pred_fused_vis = prob_fused_vis.argmax(dim=1, keepdim=True)
                    writer.add_image('cmc/PredFused',
                                     pred_fused_vis[0].float() * 50, iter_num)
                    with torch.no_grad():
                        teacher_pred_vis = pre_a.argmax(dim=1, keepdim=True)
                    writer.add_image('cmc/TeacherPred',
                                     teacher_pred_vis[0].float() * 50, iter_num)
                    writer.add_image('cmc/ConfMask',
                                     conf_mask_a[0].unsqueeze(0), iter_num)
                    with torch.no_grad():
                        agree_vis = (
                            pred_viewA[0] == pred_viewB[0]
                        ).float()
                    writer.add_image('cmc/AgreeMap', agree_vis, iter_num)

            # ============================================================
            # 验证（仅在 rank 0）
            # ============================================================
            if iter_num > 0 and iter_num % 200 == 0 and is_main_process():
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        unwrap_ddp(model), classes=num_classes,
                        patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)
                mean_dice, mean_jaccard, mean_hd95, mean_asd = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="val"
                )
                if mean_dice > best_performance:
                    best_performance = mean_dice
                    torch.save(unwrap_ddp(model).state_dict(),
                               os.path.join(snapshot_path,
                                            f'iter_{iter_num}_dice_{round(best_performance, 4)}.pth'))
                    torch.save(unwrap_ddp(model).state_dict(),
                               os.path.join(snapshot_path,
                                            f'{args.model}_best_model.pth'))
                    logging.info(f"=> Saved best model, iter={iter_num}, dice={best_performance:.4f}")
                if mean_hd95 < best_hd:
                    best_hd = mean_hd95
                logging.info(f'BEST | dice={best_performance:.4f}, hd95={best_hd:.2f}')
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations and is_main_process():
            iterator.close()
            break

    if is_main_process():
        torch.save(unwrap_ddp(model).state_dict(),
                   os.path.join(snapshot_path, 'final_model.pth'))
        writer.close()
        logging.info("=" * 60)
        logging.info(f"Self-training finished.")
        logging.info(f"Best mean_dice : {best_performance:.4f}")
        logging.info(f"Best mean_hd95 : {best_hd:.2f}")
        logging.info("=" * 60)


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    # 解析 DDP 环境
    use_ddp = args.ddp and 'RANK' in os.environ
    use_amp = args.amp

    if use_ddp:
        # DDP 模式
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=world_size
        )
        args.local_rank = local_rank
        args.rank = rank
        args.world_size = world_size
        args.device = torch.device(f'cuda:{local_rank}')

        # 各 rank 不同种子
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)

        torch.distributed.barrier()
        print_ddp_config(args)
    else:
        # 单 GPU 模式
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.local_rank = int(args.gpu)
        args.rank = 0
        args.world_size = 1
        args.device = torch.device('cuda:0')

        if args.deterministic:
            cudnn.benchmark    = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark    = True   # 固定输入尺寸 → 加速
            cudnn.deterministic = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print_ddp_config(args)

    # 输出路径
    pre_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum
    )
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum
    )
    for path in [pre_snapshot_path, self_snapshot_path]:
        if is_main_process() and not os.path.exists(path):
            os.makedirs(path)

    # DDP: 确保路径已创建
    if use_ddp:
        torch.distributed.barrier()

    if is_main_process():
        shutil.copy(__file__, self_snapshot_path)

    # 日志
    if is_main_process():
        logging.basicConfig(
            filename=pre_snapshot_path + "/log.txt",
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    # Pre-train（有 load_path 时跳过）
    if args.load_path is not None:
        snapshot_path = pre_snapshot_path
        if is_main_process():
            log_info(f"Skip pre-train, resume from {args.load_path}")
    else:
        snapshot_path = pre_snapshot_path
        pre_train(args)

    # Self-train (重新配置日志)
    snapshot_path = self_snapshot_path
    if is_main_process():
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
    self_train(args, pre_snapshot_path)

    # DDP 清理
    if use_ddp:
        torch.distributed.destroy_process_group()
