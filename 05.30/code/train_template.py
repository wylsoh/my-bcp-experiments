"""
===============================================================================
BCP 实验训练模板 — DDP + AMP 最优配置
===============================================================================

本模板展示了在 4×NVIDIA L20 (46GB) 上使用 DDP + AMP 训练的标准模式。

使用方式:
    # 4 GPU 训练
    torchrun --nproc_per_node=4 train_template.py \
        --batch_size=96 --labeled_bs=24 --amp --ddp \
        --dataset ACDC --base_dir /path/to/ACDC

    # 单 GPU 训练（兼容旧模式）
    python train_template.py \
        --batch_size=24 --labeled_bs=6 --gpu 0

    # 2 GPU 训练
    torchrun --nproc_per_node=2 train_template.py \
        --batch_size=48 --labeled_bs=12 --amp --ddp

优化说明（vs 原始代码调整点）:
    1. DDP 多卡: torchrun 启动，DistributedSampler + DistributedTwoStreamBatchSampler
    2. AMP 混合精度: autocast + GradScaler，4×GPU throughput 提升 2-3 倍
    3. cudnn.benchmark=True: cuDNN auto-tuner 加速固定尺寸卷积
    4. SyncBN: SyncBatchNorm.convert_sync_batchnorm，大 batch 稳定 BN
    5. 学习率缩放: batch_size 增大 N 倍时，lr 可缩放 sqrt(N) 或保持 poly lr
    6. EMA 仅在 rank 0 更新（所有 rank 权重相同，EMA 只需一份）
    7. 日志/保存仅在 rank 0
===============================================================================
"""
import argparse
import os
import sys
import logging
import shutil
import math
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

# ======================== 项目内部模块 ========================
from networks.net_factory import net_factory, BCP_net
from dataloaders.dataset import (
    BaseDataSets, RandomGenerator, TwoStreamBatchSampler,
    DistributedTwoStreamBatchSampler, get_distributed_sampler
)
from utils.train_utils import (
    init_distributed_mode, get_rank, get_world_size, is_main_process,
    get_amp_autocast, get_grad_scaler, convert_model_to_syncbn,
    unwrap_ddp, update_model_ema, load_net, load_net_opt, save_net_opt,
    get_current_consistency_weight, poly_lr, warmup_poly_lr,
    mix_loss, mae_consistency_loss, patients_to_slices,
    save_checkpoint, log_info, log_scalar, log_images
)

# ⚠️ 以下是每个实验特有的模块，按需 import
# from utils.cmc_utils import ...
# from utils.corrmatch_utils import ...
# from utils.mask_generator import ...
# from utils.pseudo_label_utils import ...


# ======================== 参数解析 ========================

def get_args():
    parser = argparse.ArgumentParser(description='BCP DDP+AMP 训练模板')

    # ---------- 分布式训练 ----------
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='启用分布式训练 (torchrun)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='启用 AMP 混合精度训练')
    parser.add_argument('--gpu', type=str, default='0',
                        help='单 GPU 模式时指定 GPU ID')

    # ---------- 数据 ----------
    parser.add_argument('--dataset', type=str, default='ACDC',
                        choices=['ACDC', 'LA', 'Prostate'],
                        help='数据集名称')
    parser.add_argument('--base_dir', type=str,
                        default='../data/ACDC',
                        help='数据集根目录')
    parser.add_argument('--label_num', type=str, default='7',
                        help='有标签患者数 (affects slice count)')
    parser.add_argument('--max_iterations', type=int, default=30000,
                        help='自训练最大迭代数')
    parser.add_argument('--pre_iterations', type=int, default=10000,
                        help='预训练迭代数')

    # ---------- 网络 ----------
    parser.add_argument('--batch_size', type=int, default=96,
                        help='全局 batch size (跨所有 GPU)')
    parser.add_argument('--labeled_bs', type=int, default=24,
                        help='全局有标签 batch size')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='分割类别数')
    parser.add_argument('--patch_size', type=int, nargs='+',
                        default=[256, 256],
                        help='输入图像尺寸')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='是否固定随机种子')

    # ---------- 优化器 ----------
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='启用 warmup')
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='warmup 迭代数')

    # ---------- 一致性损失 ----------
    parser.add_argument('--consistency', type=float, default=0.1,
                        help='一致性损失权重')
    parser.add_argument('--consistency_rampup', type=float, default=200.0,
                        help='一致性权重 rampup 周期')

    # ---------- 保存与日志 ----------
    parser.add_argument('--snapshot_path', type=str,
                        default='./model/ACDC_template',
                        help='模型保存路径')
    parser.add_argument('--load_path', type=str, default=None,
                        help='续训 checkpoint 路径')

    return parser.parse_args()


# ======================== 种子与 cuDNN ========================

def set_seed(seed=1337, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 注意

    if deterministic:
        # 确定性模式 — 可复现但更慢
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 最优性能 — 不可完全复现
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ======================== 数据集 ========================

def get_acdc_train_loader(args):
    """创建 ACDC 训练集 DataLoader（兼容 DDP 和单 GPU）"""
    num_slices = patients_to_slices(args.dataset, int(args.label_num))
    num_labeled_slices = num_slices // 2  # 有标签切片数

    train_dataset = BaseDataSets(
        base_dir=args.base_dir,
        split='train',
        transform=RandomGenerator(args.patch_size)
    )

    # 分有标签 / 无标签索引
    labeled_idxs = list(range(0, num_labeled_slices))
    unlabeled_idxs = list(range(num_labeled_slices, num_slices))

    if args.ddp:
        # DDP 模式: 使用 DistributedTwoStreamBatchSampler
        batch_sampler = DistributedTwoStreamBatchSampler(
            primary_indices=labeled_idxs,
            secondary_indices=unlabeled_idxs,
            batch_size=args.batch_size,
            secondary_batch_size=args.batch_size - args.labeled_bs,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
        )
        log_info(f"DDP DataLoader: global_bs={args.batch_size}, "
                 f"per_gpu_bs={args.batch_size // get_world_size()}, "
                 f"labeled_bs={args.labeled_bs}")
    else:
        # 单 GPU 模式
        batch_sampler = TwoStreamBatchSampler(
            primary_indices=labeled_idxs,
            secondary_indices=unlabeled_idxs,
            batch_size=args.batch_size,
            secondary_batch_size=args.batch_size - args.labeled_bs,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
        )

    return loader, labeled_idxs, unlabeled_idxs


# ======================== 模型创建 ========================

def create_model(args):
    """
    创建模型，应用 SyncBN，move 到目标设备，包装 DDP。
    该函数兼容单 GPU 和 DDP 两种模式。
    """
    # --- 1. 确定目标设备 ---
    if args.ddp:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    # --- 2. 在 CPU 上创建模型（net_factory 已移除 .cuda()） ---
    student = BCP_net(in_chns=1, class_num=args.num_classes, ema=False)
    teacher = BCP_net(in_chns=1, class_num=args.num_classes, ema=True)

    # --- 3. SyncBN (仅在 DDP 多卡时生效) ---
    if args.ddp:
        student = convert_model_to_syncbn(student)

    # --- 4. move 到目标设备 ---
    student = student.to(device)
    teacher = teacher.to(device)

    # --- 5. 教师模型 sync teacher params = student params ---
    for param_t, param_s in zip(teacher.parameters(), student.parameters()):
        param_t.data.copy_(param_s.data)

    # --- 6. DDP 包装 student (teacher 不做 DDP，只用于 eval 推理) ---
    if args.ddp:
        student = DDP(
            student,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,  # BCP 无 unused params
            broadcast_buffers=True,
        )

    total_params = sum(p.numel() for p in unwrap_ddp(student).parameters())
    log_info(f"模型参数总量: {total_params / 1e6:.2f}M")

    return student, teacher, device


# ======================== 预训练阶段 ========================

def pre_train(args, student, teacher, loader, optimizer, scaler, device):
    """
    预训练阶段: 仅使用有标签数据训练 student 模型。
    
    DDP 模式下:
    - 每个 rank 独立 forward/backward
    - loss 在 DDP 内部自动做 all-reduce
    - 日志/保存仅在 rank 0
    - EMA 更新仅在 rank 0（且用 unwrap_ddp 拿到原始参数）
    """
    writer = SummaryWriter(args.snapshot_path) if is_main_process() else None
    log_info("=" * 50)
    log_info("预训练阶段开始")
    log_info("=" * 50)

    student.train()
    teacher.eval()

    iter_num = 0
    max_epoch = args.pre_iterations

    while iter_num < max_epoch:
        for sampled_batch in loader:
            # --- DDP: sampler 设置 epoch（确保每个 epoch 不同 shuffle） ---
            if args.ddp and hasattr(loader, 'batch_sampler'):
                pass  # DistributedTwoStreamBatchSampler 内部自带 shuffle

            # --- 数据移动到 GPU ---
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            # --- AMP 上下文 ---
            with get_amp_autocast(args.amp):
                outputs = student(volume_batch)
                # 有标签监督损失
                loss_seg = F.cross_entropy(outputs, label_batch)

            # --- 反向传播（GradScaler 处理 FP16 梯度缩放） ---
            if args.amp and scaler is not None:
                scaler.scale(loss_seg).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_seg.backward()
                optimizer.step()
            optimizer.zero_grad()

            # --- 学习率更新 ---
            if args.warmup:
                warmup_poly_lr(optimizer, args.base_lr, iter_num,
                               max_epoch, args.warmup_iters)
            else:
                poly_lr(optimizer, args.base_lr, iter_num, max_epoch)

            iter_num += 1

            # --- 日志（仅在 rank 0） ---
            if iter_num % 100 == 0 and is_main_process():
                writer.add_scalar('pre_train/loss', loss_seg.item(), iter_num)
                lr_ = optimizer.param_groups[0]['lr']
                writer.add_scalar('pre_train/lr', lr_, iter_num)
                log_info(
                    f"iter {iter_num:06d} / {max_epoch:06d} | "
                    f"loss: {loss_seg.item():.4f} | lr: {lr_:.6f}")

            # --- checkpoint（仅在 rank 0） ---
            if iter_num % 2000 == 0 and is_main_process():
                save_checkpoint(student, optimizer,
                                f"{args.snapshot_path}/pre_{iter_num}.pth")

            if iter_num >= max_epoch:
                break

    # --- 保存预训练最终模型 ---
    if is_main_process():
        save_checkpoint(student, optimizer,
                        f"{args.snapshot_path}/pre_train_model.pth")
    log_info("预训练阶段完成")


# ======================== 自训练阶段 ========================

def self_train(args, student, teacher, loader, optimizer, scaler, device):
    """
    自训练阶段: BCP + 一致性正则化训练。
    
    与原始代码的核心区别:
    1. AMP: forward 在 autocast 内，backward 通过 GradScaler
    2. DDP: loss 自动 all-reduce，更新 optimizer 时自动同步
    3. EMA: 仅在 rank 0 且每隔 N 步更新（所有 GPU 模型参数相同）
    """
    writer = SummaryWriter(args.snapshot_path) if is_main_process() else None
    log_info("=" * 50)
    log_info("自训练阶段开始")
    log_info("=" * 50)

    student.train()
    teacher.eval()

    iter_num = 0
    max_epoch = args.max_iterations

    # EMA 更新间隔（每 N 步更新一次减少通信开销）
    ema_update_freq = 5 if args.ddp else 1

    while iter_num < max_epoch:
        for sampled_batch in loader:
            # --- 数据 ---
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            # --- 有标签 / 无标签分离 ---
            label_batch_src = label_batch[:args.labeled_bs]
            # 对无标签数据，使用 BCP 策略生成 teacher 伪标签
            unlabeled_batch = volume_batch[args.labeled_bs:]

            # ====================================================
            # 此处插入各种实验特有的 BCP / CMC / MAE 逻辑：
            # - BCP: 生成 mask, patch, mix 输入
            # - CMC: 互补掩码一致性
            # - MAE: 掩码自编码器 + 一致性
            # 参考 BCP_CMC_fusion.py / ACDC_BCP_MAE_train.py 等
            # ====================================================

            # --- AMP 上下文 ---
            with get_amp_autocast(args.amp):
                # --- 教师推理（无梯度，no_grad） ---
                with torch.no_grad():
                    teacher_pred = teacher(volume_batch)
                    teacher_prob = F.softmax(teacher_pred, dim=1)
                    teacher_pseudo, teacher_conf = torch.max(teacher_prob, dim=1)
                    confidence_mask = (teacher_conf > 0.95).float()

                # --- BCP Student forward ---
                # net_input = ... (BCP mixed input)
                # outputs = student(net_input)
                # loss = mix_loss(outputs, ...)

                # --- 支撑损失（监督分支） ---
                sup_output = student(volume_batch[:args.labeled_bs])
                loss_sup = F.cross_entropy(sup_output, label_batch_src)

                # --- 总损失 ---
                consistency_weight = get_current_consistency_weight(
                    iter_num, args.consistency, args.consistency_rampup)
                loss = loss_sup  # + consistency_weight * unsup_loss

            # --- 反向传播 ---
            if args.amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            # --- 学习率 ---
            poly_lr(optimizer, args.base_lr, iter_num, max_epoch)

            # --- EMA 更新（仅在 rank 0，每隔 ema_update_freq 步） ---
            if iter_num % ema_update_freq == 0:
                update_model_ema(student, teacher, alpha=0.99)

            iter_num += 1

            # --- 日志 ---
            if iter_num % 200 == 0 and is_main_process():
                writer.add_scalar('train/loss', loss.item(), iter_num)
                writer.add_scalar('train/loss_sup', loss_sup.item(), iter_num)
                writer.add_scalar('train/consistency_weight',
                                  consistency_weight, iter_num)
                lr_ = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', lr_, iter_num)
                log_info(
                    f"iter {iter_num:06d} / {max_epoch:06d} | "
                    f"loss: {loss.item():.4f} | "
                    f"loss_sup: {loss_sup.item():.4f} | "
                    f"lr: {lr_:.6f}")

            if iter_num % 4000 == 0 and is_main_process():
                save_checkpoint(student, optimizer,
                                f"{args.snapshot_path}/iter_{iter_num}.pth")

            if iter_num >= max_epoch:
                break

    if is_main_process():
        save_checkpoint(student, optimizer,
                        f"{args.snapshot_path}/final_model.pth")
    log_info("自训练阶段完成")


# ======================== 主函数 ========================

def main():
    args = get_args()

    # --- DDP 初始化 ---
    if args.ddp:
        use_ddp = init_distributed_mode()
        if not use_ddp:
            args.ddp = False
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.gpu))
    else:
        args.local_rank = int(args.gpu)
        # 单 GPU 模式下设置可见设备
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # --- 打印配置 ---
    if is_main_process():
        print(f"{'='*60}")
        print(f"配置: DDP={args.ddp}, AMP={args.amp}, "
              f"batch_size={args.batch_size}, labeled_bs={args.labeled_bs}")
        print(f"{'='*60}")

    # --- 随机种子 ---
    if args.deterministic:
        set_seed(1337, deterministic=True)
    else:
        set_seed(1337, deterministic=False)

    # --- 创建 checkpoint 目录 ---
    if is_main_process() and not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    # --- DDP barrier 确保目录已创建 ---
    if args.ddp:
        torch.distributed.barrier()

    # --- DataLoader ---
    loader, labeled_idxs, unlabeled_idxs = get_acdc_train_loader(args)

    # --- 模型 ---
    student, teacher, device = create_model(args)

    # --- 优化器 ---
    optimizer = optim.SGD(
        unwrap_ddp(student).parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )

    # --- GradScaler (AMP) ---
    # 注意: BF16 不需要 scaler，但 torch.bfloat16 需要 GPU Ampere+ 架构
    # L20 (Ada Lovelace) 支持 bf16
    scaler = get_grad_scaler(args.amp)

    # --- 载入 checkpoint 续训 ---
    if args.load_path is not None:
        load_net_opt(student, optimizer, args.load_path)
        log_info(f"已载入 checkpoint: {args.load_path}")

    # --- 训练 ---
    pre_train(args, student, teacher, loader, optimizer, scaler, device)
    self_train(args, student, teacher, loader, optimizer, scaler, device)

    # --- 清理 ---
    if args.ddp:
        torch.distributed.destroy_process_group()
    log_info("训练完成！")


if __name__ == "__main__":
    main()
