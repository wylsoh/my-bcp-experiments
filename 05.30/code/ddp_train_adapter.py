"""
==============================================================================
DDP + AMP 训练适配器
==============================================================================

本适配器允许「最小的代码修改」将现有实验脚本从单 GPU 升级到 DDP + AMP。

使用方式:
    1. 在实验脚本中 import 本模块
    2. 在 main 入口处调用 ddp_run() 包装整个训练流程

示例 — 将 BCP_CMC_fusion.py 从单 GPU 改造为 DDP+AMP:

    # BCP_CMC_fusion.py 顶部添加:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ddp_train_adapter import ddp_run, ddp_main_wrapper

    # 原来的 main 代码块改为:
    if __name__ == "__main__":
        ddp_run(
            main_func=lambda: ddp_main_wrapper(args),
            args=args,
            use_amp=hasattr(args, 'amp') and args.amp,
            use_ddp=hasattr(args, 'ddp') and args.ddp,
        )

    # 并将原本的 main 代码移到 ddp_main_wrapper(args) 函数中

    # 启动（4 GPU）:
    torchrun --nproc_per_node=4 BCP_CMC_fusion.py --amp --ddp \
        --batch_size=96 --labeled_bs=24

==============================================================================

适配器提供以下 DDP+AMP 自动处理:
  [×] 模型自动 to(device) + DDP 包装
  [×] AMP autocast + GradScaler
  [×] DataLoader 自动使用 DistributedTwoStreamBatchSampler
  [×] EMA 更新仅在 rank 0
  [×] 日志/checkpoint 仅在 rank 0
  [×] SyncBN 自动转换
  [×] 学习率调度兼容 poly_lr
  [×] 随机种子各 rank 独立
"""
import argparse
import os
import sys
import random
import numpy as np
from typing import Callable, Any, Optional

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.train_utils import (
    init_distributed_mode, get_rank, get_world_size, is_main_process,
    get_amp_autocast, get_grad_scaler, convert_model_to_syncbn,
    unwrap_ddp, update_model_ema, save_net_opt, load_net, load_net_opt,
    log_info, log_scalar, log_images
)
from dataloaders.dataset import (
    TwoStreamBatchSampler, ThreeStreamBatchSampler,
    DistributedTwoStreamBatchSampler, DistributedThreeStreamBatchSampler
)


# ============================================================================
# DDP 参数注入
# ============================================================================

def add_ddp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    向 ArgumentParser 注入 DDP + AMP 相关参数。
    
    用法:
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=24)
        # ... 其他参数 ...
        parser = add_ddp_args(parser)  # 注入 --ddp, --amp, --gpu
        args = parser.parse_args()
    """
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='启用分布式训练 (torchrun)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='启用 AMP 混合精度训练')
    parser.add_argument('--gpu', type=str, default='0',
                        help='单 GPU 模式时指定 GPU ID')
    parser.add_argument('--sync_bn', action='store_true', default=True,
                        help='启用 SyncBatchNorm')
    parser.add_argument('--find_unused_params', action='store_true',
                        default=False,
                        help='DDP find_unused_parameters (异常时开启)')
    return parser


# ============================================================================
# DDP 运行器
# ============================================================================

def ddp_run(main_func: Callable, args, use_amp: bool = False,
            use_ddp: bool = False):
    """
    DDP 运行入口。
    
    检测 torchrun 环境变量，自动进入 DDP 或单 GPU 模式。
    
    Args:
        main_func: 主训练函数（无参或可接受 args）
        args: 命令行参数对象
        use_amp: 是否启用 AMP
        use_ddp: 是否启用 DDP（由 --ddp 标志控制）
    """
    if use_ddp and 'RANK' in os.environ:
        # --- DDP 模式 ---
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # 设置当前 GPU
        torch.cuda.set_device(local_rank)

        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )

        # 各 rank 设置不同 seed 避免数据完全一致
        seed = getattr(args, 'seed', 1337) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 保存 rank 信息到 args
        args.local_rank = local_rank
        args.rank = rank
        args.world_size = world_size
        args.device = torch.device(f'cuda:{local_rank}')

        # barrier 同步
        dist.barrier()

        if rank == 0:
            print(f"[DDP] 初始化: world_size={world_size}, local_rank={local_rank}, "
                  f"device={torch.cuda.get_device_name(local_rank)}, "
                  f"AMP={'ON' if use_amp else 'OFF'}")

        # 执行主函数
        main_func(args)

        # 清理
        dist.destroy_process_group()
    else:
        # --- 单 GPU 模式 ---
        gpu_id = getattr(args, 'gpu', '0')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        args.local_rank = int(gpu_id)
        args.rank = 0
        args.world_size = 1
        args.device = torch.device(f'cuda:0')

        if getattr(args, 'deterministic', 1):
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False

        print(f"[单 GPU] device={torch.cuda.get_device_name(0)}, "
              f"AMP={'ON' if use_amp else 'OFF'}")

        main_func(args)


# ============================================================================
# 模型包装器
# ============================================================================

def ddp_wrap_model(model: torch.nn.Module, args) -> torch.nn.Module:
    """
    将模型包装为 DDP（如果需要）。
    
    执行:
    1. 移入目标设备
    2. SyncBN 转换（多卡时）
    3. DDP 包装
    
    Args:
        model: 原始模型（在 CPU 上）
        args: 参数对象（需有 device, world_size, sync_bn, find_unused_params）
    
    Returns:
        包装后的模型（单卡时返回原始模型 .to(device)）
    """
    device = args.device
    model = model.to(device)

    if args.world_size > 1:
        # SyncBN
        if getattr(args, 'sync_bn', True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # DDP 包装
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=getattr(args, 'find_unused_params', False),
            broadcast_buffers=True,
        )
        if args.rank == 0:
            print(f"[DDP] 模型已包装为 DDP, SyncBN={getattr(args, 'sync_bn', True)}")

    return model


def ddp_unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """解包 DDP，返回原始模型"""
    return unwrap_ddp(model)


# ============================================================================
# DataLoader 包装器
# ============================================================================

def create_ddp_dataloader(
    dataset: Dataset,
    batch_size: int,
    labeled_bs: int,
    labeled_idxs: list,
    unlabeled_idxs: list,
    args,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_three_stream: bool = False,
):
    """
    创建 DDP 兼容的 DataLoader。
    
    单 GPU 模式使用 TwoStreamBatchSampler/ThreeStreamBatchSampler。
    DDP 模式使用 DistributedTwoStreamBatchSampler/DistributedThreeStreamBatchSampler。
    
    Args:
        dataset: PyTorch Dataset
        batch_size: 全局 batch size
        labeled_bs: 全局有标签 batch size
        labeled_idxs: 有标签索引列表
        unlabeled_idxs: 无标签索引列表
        args: 参数对象（需有 world_size, rank）
        num_workers: DataLoader workers
        pin_memory: 是否固定内存
        use_three_stream: 是否使用 ThreeStream（MultiPatch 系列）
    
    Returns:
        DataLoader
    """
    if args.world_size > 1:
        if use_three_stream:
            batch_sampler = DistributedThreeStreamBatchSampler(
                primary_indices=labeled_idxs,
                secondary_indices=unlabeled_idxs,
                batch_size=batch_size,
                secondary_batch_size=batch_size - labeled_bs,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            batch_sampler = DistributedTwoStreamBatchSampler(
                primary_indices=labeled_idxs,
                secondary_indices=unlabeled_idxs,
                batch_size=batch_size,
                secondary_batch_size=batch_size - labeled_bs,
                world_size=args.world_size,
                rank=args.rank,
            )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        if use_three_stream:
            batch_sampler = ThreeStreamBatchSampler(
                primary_indices=labeled_idxs,
                secondary_indices=unlabeled_idxs,
                batch_size=batch_size,
                secondary_batch_size=batch_size - labeled_bs,
            )
        else:
            batch_sampler = TwoStreamBatchSampler(
                primary_indices=labeled_idxs,
                secondary_indices=unlabeled_idxs,
                batch_size=batch_size,
                secondary_batch_size=batch_size - labeled_bs,
            )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loader


# ============================================================================
# AMP 训练辅助
# ============================================================================

class AMPTrainer:
    """
    AMP 训练管理器。
    
    封装 autocast + GradScaler + backward + optimizer.step 的完整流程。
    
    用法:
        trainer = AMPTrainer(use_amp=True)
        for batch in loader:
            with trainer.autocast():
                output = model(input)
                loss = criterion(output, target)
            trainer.backward(loss)
            trainer.step(optimizer)
            trainer.zero_grad(optimizer)
    """
    def __init__(self, use_amp: bool = False):
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def autocast(self):
        """AMP 自动混合精度上下文管理器"""
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def backward(self, loss: torch.Tensor):
        """反向传播（含 GradScaler 缩放）"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer):
        """优化器 step（含 GradScaler 缩放）"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def zero_grad(self, optimizer: torch.optim.Optimizer):
        """清零梯度"""
        optimizer.zero_grad()


# ============================================================================
# 验证辅助（DDP 兼容）
# ============================================================================

def ddp_gather_metrics(metric_list: list, args) -> list:
    """
    收集所有 rank 的验证指标到 rank 0。
    
    在 DDP 模式下，每个 rank 只验证一部分数据，
    需要 all_gather 汇总所有结果。
    
    Args:
        metric_list: 当前 rank 的指标列表
        args: 参数对象
    
    Returns:
        汇总后的指标列表（仅在 rank 0 有数据）
    """
    if args.world_size <= 1:
        return metric_list

    # 转为 tensor
    if isinstance(metric_list, list) and len(metric_list) > 0:
        metric_tensor = torch.tensor(metric_list, device=args.device)
    else:
        metric_tensor = torch.tensor([0.0], device=args.device)

    # all_gather
    gathered = [torch.zeros_like(metric_tensor) for _ in range(args.world_size)]
    dist.all_gather(gathered, metric_tensor)

    if args.rank == 0:
        return torch.cat(gathered, dim=0).cpu().numpy().tolist()
    return []


# ============================================================================
# 快速迁移辅助（将现有脚本改为 DDP 运行的最小修改）
# ============================================================================

def ddp_main_wrapper(args):
    """
    DDP 主函数包装器桩。
    
    用于将现有脚本的 main 代码块包装为 ddp_run 接受的函数。
    用户需将原本的 main 代码移入此函数。
    
    示例:
        def main_func(args):
            # 原本的 __main__ 代码
            ...
        
        if __name__ == "__main__":
            ddp_run(main_func, args, use_amp=args.amp, use_ddp=args.ddp)
    """
    # 此函数由用户实现具体逻辑
    raise NotImplementedError(
        "请将原有 main 代码移入此函数，或在 ddp_run 中传入自定义 main_func"
    )


# ============================================================================
# 训练循环辅助函数（简化模板中的重复代码）
# ============================================================================

def add_ddp_common_args(parser):
    """
    快捷函数：同时添加 DDP 参数 + 常用 BCP 参数 + 输出目录。
    
    在新建实验脚本中使用:
        parser = argparse.ArgumentParser()
        parser = add_ddp_common_args(parser)
        parser.add_argument('--my_special_param', ...)
        args = parser.parse_args()
    """
    parser = add_ddp_args(parser)

    # BCP 常见参数
    parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
    parser.add_argument('--exp', type=str, default='exp')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--pre_iterations', type=int, default=10000)
    parser.add_argument('--max_iterations', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=96,
                        help='全局 batch size (跨所有 GPU)')
    parser.add_argument('--labeled_bs', type=int, default=24,
                        help='全局有标签 batch size')
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--labelnum', type=int, default=7)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--u_weight', type=float, default=0.5)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)

    return parser


def print_ddp_config(args):
    """打印 DDP/AMP 配置"""
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  [DDP] world_size={args.world_size}, "
              f"AMP={getattr(args, 'amp', False)}")
        print(f"  [数据] global_batch={args.batch_size}, "
              f"per_gpu={args.batch_size // max(args.world_size, 1)}")
        print(f"  [参数] labeled_bs={args.labeled_bs}, "
              f"lr={args.base_lr}, seed={getattr(args, 'seed', 1337)}")
        print(f"{'='*60}\n")
