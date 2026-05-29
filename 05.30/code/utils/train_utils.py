"""
通用训练工具函数
包含：DDP初始化、模型保存加载、EMA更新、学习率调度、损失函数、AMP混合精度
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import itertools
from utils import losses, ramps
from torch.utils.data.sampler import Sampler

dice_loss = losses.DiceLoss(n_classes=4)


# ============================================================================
# DDP (Distributed Data Parallel) 基础设施
# ============================================================================

def is_dist_avail_and_initialized():
    """检查分布式环境是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """获取当前进程 rank，单卡模式下返回 0"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取总进程数，单卡模式下返回 1"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """判断是否为主进程 (rank 0)"""
    return get_rank() == 0


def init_distributed_mode():
    """
    使用 torchrun 环境变量初始化 DDP。
    
    在单 GPU 模式下（未检测到 RANK 环境变量），降级为单卡训练。
    
    必须在所有模型和数据创建之前调用。会设置当前 CUDA 设备、
    初始化进程组，并在 barrier 同步后打印初始化信息。
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("[DDP] 未检测到 torchrun 环境变量，使用单 GPU 模式")
        return False

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    dist.barrier()

    if rank == 0:
        print(f"[DDP] 初始化成功: rank={rank}, world_size={world_size}, "
              f"local_rank={local_rank}, device={torch.cuda.get_device_name(local_rank)}")
    return True


def gather_tensor(tensor, dst=0):
    """收集所有进程的 tensor 到 rank 0（用于日志/验证）"""
    if not is_dist_avail_and_initialized():
        return tensor
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.gather(tensor, gather_list=tensor_list if get_rank() == dst else None, dst=dst)
    if get_rank() == dst:
        return torch.cat(tensor_list, dim=0)
    return None


def all_reduce_mean(tensor):
    """对所有进程的 tensor 做 all-reduce 平均"""
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor


def all_reduce_sum(tensor):
    """对所有进程的 tensor 做 all-reduce 求和"""
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ============================================================================
# AMP (Automatic Mixed Precision) 基础设施
# ============================================================================

def get_amp_autocast(enabled=True):
    """
    获取 AMP 自动混合精度上下文管理器。
    
    用法:
        with get_amp_autocast(args.amp):
            output = model(input)
            loss = criterion(output, target)
    
    Args:
        enabled: 是否启用 FP16/BF16 混合精度
    Returns:
        torch.cuda.amp.autocast 上下文管理器
    """
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(enabled=True):
    """
    获取 GradScaler，用于 FP16 混合精度训练时防止梯度下溢。
    
    用法:
        scaler = get_grad_scaler(args.amp)
        with get_amp_autocast(args.amp):
            output = model(input)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    注意：BF16 精度不需要 scaler，仅 FP16 需要。
    
    Args:
        enabled: 创建 GradScaler 对象，仅在 FP16 训练时生效
    Returns:
        GradScaler 实例，enabled=False 时返回 None
    """
    if enabled:
        return torch.cuda.amp.GradScaler()
    return None


def convert_model_to_syncbn(model):
    """
    将模型的 BatchNorm 层替换为 SyncBatchNorm。
    
    在 DDP 多卡训练中，SyncBN 跨 GPU 同步 BN 统计量，
    对大 batch size 训练至关重要。需在 DDP 包装之前调用。
    
    Args:
        model: 原始模型
    Returns:
        SyncBatchNorm 版本的模型
    """
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def unwrap_ddp(model):
    """
    安全地解包 DDP 包装的模型，返回原始模型。
    
    Args:
        model: 可能被 DDP 包装的模型
    Returns:
        原始 nn.Module
    """
    if hasattr(model, 'module'):
        return model.module
    return model


# ============================================================================
# 模型保存与加载 (DDP 兼容)
# ============================================================================

def load_net(net, path):
    """
    加载模型权重，自动处理 DDP state_dict 的 'module.' 前缀。
    
    兼容以下情况:
    - 保存时无 DDP 包装 → 加载时也无 DDP
    - 保存时有 'module.' 前缀 → 加载时自动剥离
    - 保存时无前缀 → 加载到 DDP 模型时自动添加
    """
    state = torch.load(str(path), map_location='cpu')
    state_dict = state['net']

    is_ddp_wrapped = hasattr(net, 'module')

    new_state_dict = {}
    for k, v in state_dict.items():
        if is_ddp_wrapped and not k.startswith('module.'):
            new_state_dict['module.' + k] = v
        elif not is_ddp_wrapped and k.startswith('module.'):
            new_state_dict[k[7:]] = v  # strip 'module.'
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)
    if get_rank() == 0:
        print(f"[load_net] 已从 {path} 加载模型权重")


def load_net_opt(net, optimizer, path):
    """
    加载模型和优化器状态，自动处理 DDP 前缀。
    """
    state = torch.load(str(path), map_location='cpu')
    state_dict = state['net']

    is_ddp_wrapped = hasattr(net, 'module')

    new_state_dict = {}
    for k, v in state_dict.items():
        if is_ddp_wrapped and not k.startswith('module.'):
            new_state_dict['module.' + k] = v
        elif not is_ddp_wrapped and k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)
    optimizer.load_state_dict(state['opt'])
    if get_rank() == 0:
        print(f"[load_net_opt] 已从 {path} 加载模型和优化器")


def save_net_opt(net, optimizer, path):
    """
    保存模型和优化器状态。
    
    在 DDP 模式下，仅 rank 0 执行保存操作，避免文件写冲突。
    保存时自动解包 DDP 模型，确保保存的是原始 state_dict（无 module. 前缀）。
    """
    if not is_main_process():
        return

    raw_net = unwrap_ddp(net)
    state = {
        'net': raw_net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))
    print(f"[save_net_opt] 已保存到 {path}")


# ============================================================================
# EMA (Exponential Moving Average) 教师模型
# ============================================================================

def update_model_ema(model, ema_model, alpha):
    """
    EMA 教师模型参数更新。
    
    θ_ema = α * θ_ema + (1 - α) * θ_student
    
    自动解包 DDP 包装的 student 模型。EMA 模型始终不进行 DDP 包装，
    因为它仅在 eval 模式下用于生成伪标签。
    
    Args:
        model: student 模型（可能被 DDP 包装）
        ema_model: EMA 教师模型（原始 nn.Module，不 DDP 包装）
        alpha: EMA 衰减系数，通常 0.99 ~ 0.999
    """
    raw_model = unwrap_ddp(model)
    model_state = raw_model.state_dict()
    ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


# ============================================================================
# 学习率调度
# ============================================================================

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=200.0):
    """一致性权重 sigmoid ramp-up"""
    return 5 * consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def poly_lr(optimizer, base_lr, current_iter, max_iter, power=0.9):
    """
    Poly 学习率衰减。
    
    lr = base_lr * (1 - iter/max_iter) ** power
    
    在 DDP 模式下，所有 rank 使用相同的 lr（通过 optimizer 同步）。
    """
    lr = base_lr * (1.0 - current_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_poly_lr(optimizer, base_lr, current_iter, max_iter,
                   warmup_iters=500, warmup_ratio=0.1, power=0.9):
    """
    带 warmup 的 Poly 学习率衰减。
    
    前 warmup_iters 轮: lr 从 base_lr * warmup_ratio 线性增长到 base_lr
    之后: poly 衰减
    
    Args:
        optimizer: 优化器
        base_lr: 基础学习率
        current_iter: 当前迭代数
        max_iter: 最大迭代数
        warmup_iters: warmup 迭代数
        warmup_ratio: warmup 起始学习率比例
        power: poly 衰减指数
    """
    if current_iter < warmup_iters:
        # Linear warmup
        lr = base_lr * warmup_ratio + (base_lr - base_lr * warmup_ratio) * \
             float(current_iter) / float(max(1, warmup_iters))
    else:
        # Poly decay (resume from warmup_iters)
        progress = float(current_iter - warmup_iters) / float(max(1, max_iter - warmup_iters))
        lr = base_lr * (1.0 - progress) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================================
# 损失函数
# ============================================================================

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    """
    BCP 混合损失：在图像区域和粘贴区域分别计算 Dice + CE 损失。
    
    Args:
        output: 网络输出 logits [B, C, H, W]
        img_l: 图像区域标签 [B, H, W]
        patch_l: 粘贴区域标签 [B, H, W]
        mask: BCP 掩码 [B, H, W], 1=图像区域, 0=粘贴区域
        l_weight: 有标签分支权重
        u_weight: 无标签分支权重
        unlab: 是否为无标签分支（交换权重）
    
    Returns:
        loss_dice, loss_ce 两个标量
    """
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


def mae_consistency_loss(student_output, teacher_pseudo_label,
                         confidence_mask, visible_mask, n_classes=4):
    """
    MAE 一致性损失：学生在遮挡输入上的预测 ↔ 教师在完整输入上的伪标签。
    
    仅在 [被遮挡区域 ∩ 高置信度区域] 计算损失。
    
    Args:
        student_output: [B, C, H, W] 学生模型 logits
        teacher_pseudo_label: [B, H, W] 教师伪标签
        confidence_mask: [B, H, W] 置信度掩码
        visible_mask: [B, 1, H, W] MAE 可见区域 (1=可见, 0=被遮挡)
        n_classes: 类别数
    
    Returns:
        loss: 标量损失
        num_valid_pixels: 有效像素数
    """
    CE = nn.CrossEntropyLoss(reduction='none')

    invisible_mask = 1.0 - visible_mask.squeeze(1)  # [B, H, W], 被遮挡区域=1
    combined_mask = invisible_mask * confidence_mask  # 被遮挡 ∩ 高置信度

    pseudo_label = teacher_pseudo_label.long()

    # CE 损失
    loss_ce = (CE(student_output, pseudo_label) * combined_mask).sum() / \
              (combined_mask.sum() + 1e-16)

    # Dice 损失
    output_soft = F.softmax(student_output, dim=1)
    loss_dice_val = dice_loss(output_soft, pseudo_label.unsqueeze(1),
                              combined_mask.unsqueeze(1))

    loss = (loss_ce + loss_dice_val) / 2.0
    num_valid = combined_mask.sum().item()

    return loss, num_valid


# ============================================================================
# 数据集工具
# ============================================================================

def patients_to_slices(dataset, patiens_num):
    """数据集患者数到切片数的映射"""
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ref_dict[str(patiens_num)]


# ============================================================================
# 日志工具
# ============================================================================

def log_info(msg, rank=0):
    """仅在主进程打印日志"""
    if get_rank() == rank:
        print(f"[INFO] {msg}")


def log_scalar(writer, tag, value, step, rank=0):
    """仅在主进程记录 TensorBoard 标量"""
    if get_rank() == rank and writer is not None:
        writer.add_scalar(tag, value, step)


def log_images(writer, tag, images, step, rank=0):
    """仅在主进程记录 TensorBoard 图像"""
    if get_rank() == rank and writer is not None:
        writer.add_images(tag, images, step)


def save_checkpoint(net, optimizer, path, is_best=False):
    """
    统一的 checkpoint 保存函数。
    
    在 DDP 模式下只保存 rank 0 的模型（所有 rank 权重相同）。
    
    Args:
        net: 模型 (可能 DDP 包装)
        optimizer: 优化器
        path: 保存路径
        is_best: 是否为最佳模型（用于命名）
    """
    save_net_opt(net, optimizer, path)
    if is_best and is_main_process():
        best_path = str(path).replace('.pth', '_best.pth')
        torch.save(unwrap_ddp(net).state_dict(), best_path)
        print(f"[save_checkpoint] 最佳模型已保存到 {best_path}")
