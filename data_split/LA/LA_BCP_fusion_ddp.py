"""
LA-BCP 半监督分割训练脚本 -- 支持 DDP + AMP

本脚本专为 LA（左心房）3D 数据集设计，基于 BCP（双向 copy-paste）框架。
与 ACDC 版本的核心区别:
  1. 使用 LAHeart 读取 3D .h5 体积数据
  2. 使用 VNet 3D 网络替代 UNet
  3. 使用阈值型伪标签 (get_cut_mask) 替代 argmax 型
  4. 按病人索引而非切片索引划分有标签/无标签数据
  5. 使用 3D sliding window 验证

启动方式:
    # 4 GPU 训练 (batch_size=8 为单卡值, DDP 自动乘卡数)
    torchrun --nproc_per_node=4 LA_BCP_fusion_ddp.py \
        --root_path ../data_split/LA \
        --exp LA_BCP_fusion \
        --pre_iterations 2000 \
        --max_iterations 15000 \
        --batch_size 8 --labeled_bs 4 \
        --amp --ddp

    # 单 GPU 训练
    python LA_BCP_fusion_ddp.py \
        --batch_size 8 --labeled_bs 4 --gpu 0

    # 从已有权重恢复训练
    python LA_BCP_fusion_ddp.py \
        --batch_size 8 --labeled_bs 4 --gpu 0 \
        --load_path ./model/BCP/LA_BCP_fusion_8_labeled/pre_train/VNet_best_model.pth
"""
import argparse
import logging
import os
import random
import shutil
import sys

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from medpy import metric
from skimage.measure import label
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (
    LAHeart, RandomRotFlip, RandomCrop, ToTensor
)
from networks.net_factory import net_factory
from utils.train_utils import (
    load_net, load_net_opt, save_net_opt,
    update_model_ema, poly_lr,
    init_distributed_mode, get_rank, get_world_size, is_main_process,
    get_amp_autocast, get_grad_scaler, unwrap_ddp, convert_model_to_syncbn,
    save_checkpoint, log_info, log_scalar, log_images
)
from utils.BCP_utils import mix_loss, update_ema_variables
from utils.losses import mask_DiceLoss
from ddp_train_adapter import (
    add_ddp_args, ddp_wrap_model, create_ddp_dataloader, AMPTrainer,
    print_ddp_config
)


# ================================================================
# LA 专用: 阈值型伪标签生成 (3D 适配)
# ================================================================
def get_cut_mask(out, thres=0.5, nms=0):
    """基于阈值的伪标签生成, 仅提取左心房类别 (class 1)

    Args:
        out:  模型输出 logits, shape (B, C, D, H, W)
        thres: 概率阈值, 默认 0.5
        nms:   是否执行 LargestCC 后处理

    Returns:
        masks: 二值伪标签, shape (B, D, H, W), 值域 {0, 1}
    """
    probs = F.softmax(out, dim=1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, ...].contiguous()
    if nms == 1:
        masks = largest_cc_3d(masks)
    return masks


def largest_cc_3d(segmentation):
    """3D 最大连通分量提取

    Args:
        segmentation: 二值分割, shape (N, D, H, W)

    Returns:
        largest_cc: 仅保留最大连通分量, shape (N, D, H, W)
    """
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largest_cc = n_prob
        batch_list.append(largest_cc)
    return torch.tensor(np.array(batch_list), device=segmentation.device)


# ================================================================
# LA 专用: device-aware BCP mask 生成 (BCP_utils.context_mask 的 DDP 兼容版)
# ================================================================
def context_mask_device(img, mask_ratio, device):
    """BCP 随机裁剪掩码生成 -- device-aware 版本

    原始 BCP_utils.context_mask 中 .cuda() 硬编码, 本函数修复该问题以支持 DDP。

    Args:
        img:        输入图像, shape (B, C, D, H, W)
        mask_ratio: 裁剪比例 (如 2/3)
        device:     目标设备 (torch.device)

    Returns:
        mask:      二值掩码, 1=保留, 0=裁剪, shape (D, H, W)
        loss_mask: loss 加权掩码, shape (B, D, H, W)
    """
    batch_size, _, img_d, img_h, img_w = img.shape
    loss_mask = torch.ones(batch_size, img_d, img_h, img_w, device=device)
    mask = torch.ones(img_d, img_h, img_w, device=device)
    patch_d = int(img_d * mask_ratio)
    patch_h = int(img_h * mask_ratio)
    patch_w = int(img_w * mask_ratio)
    w = np.random.randint(0, img_w - patch_w)
    h = np.random.randint(0, img_h - patch_h)
    d = np.random.randint(0, img_d - patch_d)
    mask[d:d+patch_d, h:h+patch_h, w:w+patch_w] = 0
    loss_mask[:, d:d+patch_d, h:h+patch_h, w:w+patch_w] = 0
    return mask.long(), loss_mask.long()


# ================================================================
# LA 专用: 3D sliding window 验证
# ================================================================
def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes):
    """单个体积的 sliding window 推理

    Args:
        model:      模型
        image:      3D 图像, shape (D, H, W)
        stride_xy:  xy 平面步长
        stride_z:   z 方向步长
        patch_size: (D, H, W) 裁剪尺寸
        num_classes: 类别数

    Returns:
        prediction: 分割结果, shape (D, H, W)
        score_map:  softmax 概率图, shape (C, D, H, W)
    """
    D, H, W = image.shape
    pad_d = max(patch_size[0] - D, 0)
    pad_h = max(patch_size[1] - H, 0)
    pad_w = max(patch_size[2] - W, 0)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    D, H, W = image.shape
    sx = stride_xy
    sz = stride_z
    x_arr = list(range(0, W - patch_size[2], sx)) + [W - patch_size[2]]
    y_arr = list(range(0, H - patch_size[1], sx)) + [H - patch_size[1]]
    z_arr = list(range(0, D - patch_size[0], sz)) + [D - patch_size[0]]

    score_map = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.int16)

    model.eval()
    with torch.no_grad():
        for z in z_arr:
            for y in y_arr:
                for x in x_arr:
                    patch = image[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
                    pred = model(patch)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    pred = F.softmax(pred, dim=1).squeeze(0).cpu().numpy()
                    score_map[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += pred
                    count_map[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1

    score_map = score_map / np.maximum(count_map, 1)
    prediction = np.argmax(score_map, axis=0)

    if pad_d > 0:
        prediction = prediction[:-pad_d, :, :]
        score_map = score_map[:, :-pad_d, :, :]
    if pad_h > 0:
        prediction = prediction[:, :-pad_h, :]
        score_map = score_map[:, :, :-pad_h, :]
    if pad_w > 0:
        prediction = prediction[:, :, :-pad_w]
        score_map = score_map[:, :, :, :-pad_w]

    return prediction, score_map


def validate_3d(model, root_path, num_classes=2, patch_size=(112, 112, 80),
                stride_xy=18, stride_z=4):
    """LA 3D 验证 -- 读取 root_path/test.list, 计算平均 Dice

    Args:
        model:       模型 (已 unwrap_ddp)
        root_path:   数据根目录, 包含 test.list 和 2018LA_Seg_Training Set/
        num_classes: 类别数 (LA 为 2)
        patch_size:  sliding window 尺寸
        stride_xy:   xy 步长
        stride_z:    z 步长

    Returns:
        avg_dice: 测试集平均 Dice
    """
    test_list_path = os.path.join(root_path, 'test.list')
    if not os.path.exists(test_list_path):
        log_info(f"[WARNING] test.list not found at {test_list_path}, skip validation")
        return 0.0

    with open(test_list_path, 'r') as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    model.eval()
    total_dice = 0.0
    valid_count = 0

    for img_id in image_ids:
        h5_path = os.path.join(
            root_path, "2018LA_Seg_Training Set", img_id, "mri_norm2.h5"
        )
        if not os.path.exists(h5_path):
            log_info(f"[WARNING] {h5_path} not found, skip")
            continue
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]
            label_gt = f['label'][:]

        prediction, _ = test_single_case(
            model, image, stride_xy, stride_z, patch_size, num_classes
        )
        if np.sum(prediction) == 0:
            dice = 0.0
        else:
            dice = metric.binary.dc(prediction, label_gt)
        total_dice += dice
        valid_count += 1

    avg_dice = total_dice / max(valid_count, 1)
    return avg_dice


# ================================================================
# 参数定义
# ================================================================
parser = argparse.ArgumentParser()
parser = add_ddp_args(parser)

# LA 数据集参数
parser.add_argument('--root_path',           type=str,   default='../data_split/LA')
parser.add_argument('--exp',                 type=str,   default='LA_BCP_fusion')
parser.add_argument('--model',               type=str,   default='VNet')
parser.add_argument('--pre_iterations',      type=int,   default=2000)
parser.add_argument('--max_iterations',      type=int,   default=15000)
parser.add_argument('--max_samples',         type=int,   default=80)
parser.add_argument('--batch_size',          type=int,   default=8)
parser.add_argument('--labeled_bs',          type=int,   default=4)
parser.add_argument('--deterministic',       type=int,   default=0)
parser.add_argument('--base_lr',             type=float, default=0.01)
parser.add_argument('--patch_size',          type=list,  default=[112, 112, 80])
parser.add_argument('--seed',                type=int,   default=1337)
parser.add_argument('--num_classes',         type=int,   default=2)
parser.add_argument('--labelnum',            type=int,   default=8)
parser.add_argument('--u_weight',            type=float, default=0.5)
parser.add_argument('--mask_ratio',          type=float, default=2/3)
parser.add_argument('--consistency',         type=float, default=1.0)
parser.add_argument('--consistency_rampup',  type=float, default=40.0)
parser.add_argument('--load_path',           type=str,   default=None,
                    help='从已有 checkpoint 恢复训练, 跳过 pre-train')

# 验证参数
parser.add_argument('--stride_xy',           type=int,   default=18)
parser.add_argument('--stride_z',            type=int,   default=4)

args = parser.parse_args()


# ================================================================
# 预训练阶段
# ================================================================
def pre_train(args, snapshot_path):
    """LA 预训练阶段 -- 仅使用有标签数据训练, 含 DDP + AMP"""
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    device = args.device
    world_size = args.world_size
    labeled_sub_bs = int(args.labeled_bs / 2)

    # 创建模型
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = ddp_wrap_model(model, args)
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # LA 数据加载
    db_train = LAHeart(
        base_dir=args.root_path, split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor(),
        ])
    )

    # 病人级索引: 前 labelnum 个有标签
    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))

    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
        num_workers=4, pin_memory=True,
    )

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    writer = SummaryWriter(os.path.join(snapshot_path, 'log')) if is_main_process() else None

    if is_main_process():
        log_info("Start LA pre-training (DDP + AMP)")
        log_info(f"Iterations: {max_iterations}, labeled patients: {args.labelnum}")

    DICE = mask_DiceLoss(nclass=num_classes)
    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    if world_size > 1:
        torch.distributed.barrier()

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            # 仅使用有标签部分
            volume_batch = volume_batch[:args.labeled_bs]
            label_batch = label_batch[:args.labeled_bs]

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:]

            img_mask, loss_mask = context_mask_device(img_a, args.mask_ratio, device)
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            with ampt.autocast():
                outputs = model(net_input)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss_ce = F.cross_entropy(outputs, gt_mixl.long())
                loss_dice = DICE(outputs, gt_mixl.long())
                loss = (loss_ce + loss_dice) / 2

            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)
            iter_num += 1

            if is_main_process():
                writer.add_scalar('pre/total_loss', loss, iter_num)
                writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
                writer.add_scalar('pre/loss_ce', loss_ce, iter_num)

                if iter_num % 20 == 0:
                    log_info(
                        f"pre iter {iter_num}: loss={loss.item():.4f}, "
                        f"dice={loss_dice.item():.4f}, ce={loss_ce.item():.4f}"
                    )

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
        log_info(f"Pre-training done.")


# ================================================================
# 自训练阶段
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    """LA 自训练阶段 -- BCP 半监督 + 有标签/无标签混合训练, 含 DDP + AMP"""
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    device = args.device
    world_size = args.world_size
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    # 创建 Student + Teacher 模型
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()

    # 加载预训练权重
    pretrained_path = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    if os.path.exists(pretrained_path):
        load_net(model, pretrained_path)
        load_net(ema_model, pretrained_path)
        if is_main_process():
            log_info(f"Loaded pretrained weights from {pretrained_path}")
    else:
        if is_main_process():
            log_info(f"[WARNING] No pretrained weights at {pretrained_path}, training from scratch")

    model = ddp_wrap_model(model, args)
    ema_model = ddp_wrap_model(ema_model, args)
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # LA 数据加载
    db_train = LAHeart(
        base_dir=args.root_path, split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(args.patch_size),
            ToTensor(),
        ])
    )

    labeled_idxs = list(range(args.labelnum))
    unlabeled_idxs = list(range(args.labelnum, args.max_samples))

    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
        num_workers=4, pin_memory=True,
    )

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    writer = SummaryWriter(os.path.join(snapshot_path, 'log')) if is_main_process() else None

    if is_main_process():
        log_info("Start LA self-training (DDP + AMP)")
        log_info(f"Iterations: {max_iterations}")

    model.train()
    ema_model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 1e8
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    if world_size > 1:
        torch.distributed.barrier()

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            # 拆分有标签 / 无标签
            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]

            # EMA 教师生成伪标签
            with torch.no_grad():
                unoutput_a = ema_model(uimg_a)
                unoutput_b = ema_model(uimg_b)
                if isinstance(unoutput_a, (tuple, list)):
                    unoutput_a = unoutput_a[0]
                if isinstance(unoutput_b, (tuple, list)):
                    unoutput_b = unoutput_b[0]
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask_device(img_a, args.mask_ratio, device)

            # BCP 双向 copy-paste
            mixl_img = img_a * img_mask + uimg_a * (1 - img_mask)
            mixu_img = uimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)

            with ampt.autocast():
                outputs_l = model(mixl_img)
                outputs_u = model(mixu_img)
                if isinstance(outputs_l, (tuple, list)):
                    outputs_l = outputs_l[0]
                if isinstance(outputs_u, (tuple, list)):
                    outputs_u = outputs_u[0]

                loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask,
                                  u_weight=args.u_weight)
                loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask,
                                  u_weight=args.u_weight, unlab=True)
                loss = loss_l + loss_u

            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)
            iter_num += 1

            # EMA 更新 (仅 rank 0, 频率降低以减少通信)
            if is_main_process() and iter_num % 5 == 0:
                update_ema_variables(unwrap_ddp(model), unwrap_ddp(ema_model), 0.99)

            # 学习率调整 (每 2500 iter 降 10x)
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if is_main_process():
                writer.add_scalar('self/loss_total', loss, iter_num)
                writer.add_scalar('self/loss_l', loss_l, iter_num)
                writer.add_scalar('self/loss_u', loss_u, iter_num)

                if iter_num % 20 == 0:
                    log_info(
                        f"self iter {iter_num}: total={loss.item():.4f}, "
                        f"l={loss_l.item():.4f}, u={loss_u.item():.4f}"
                    )

            # 验证 (仅 rank 0)
            if iter_num > 0 and iter_num % 200 == 0 and is_main_process():
                dice = validate_3d(
                    unwrap_ddp(model), args.root_path,
                    num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=args.stride_xy, stride_z=args.stride_z
                )
                writer.add_scalar('val/dice', dice, iter_num)

                if dice > best_performance:
                    best_performance = dice
                    torch.save(
                        unwrap_ddp(model).state_dict(),
                        os.path.join(snapshot_path,
                                     f'iter_{iter_num}_dice_{round(dice, 4)}.pth')
                    )
                    torch.save(
                        unwrap_ddp(model).state_dict(),
                        os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    )
                    log_info(f"Saved best model, iter={iter_num}, dice={dice:.4f}")

                log_info(f"Best dice so far: {best_performance:.4f}")
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            if is_main_process():
                iterator.close()
            break

    if is_main_process():
        torch.save(unwrap_ddp(model).state_dict(),
                   os.path.join(snapshot_path, 'final_model.pth'))
        writer.close()
        log_info(f"Self-training finished. Best dice: {best_performance:.4f}")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    use_ddp = args.ddp and 'RANK' in os.environ
    use_amp = args.amp

    if use_ddp:
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

        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        torch.distributed.barrier()
        print_ddp_config(args)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.local_rank = int(args.gpu)
        args.rank = 0
        args.world_size = 1
        args.device = torch.device('cuda:0')

        if args.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
            cudnn.deterministic = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print_ddp_config(args)

    # 输出路径
    pre_snapshot_path = f"./model/BCP/LA_{args.exp}_{args.labelnum}_labeled/pre_train"
    self_snapshot_path = f"./model/BCP/LA_{args.exp}_{args.labelnum}_labeled/self_train"
    for path in [pre_snapshot_path, self_snapshot_path]:
        if is_main_process() and not os.path.exists(path):
            os.makedirs(path)
    if use_ddp:
        torch.distributed.barrier()
    if is_main_process():
        shutil.copy(__file__, self_snapshot_path)

    # 日志
    if is_main_process():
        logging.basicConfig(
            filename=os.path.join(pre_snapshot_path, "log.txt"),
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    # 执行训练
    if args.load_path is not None:
        if is_main_process():
            log_info(f"Skip pre-train, resume from {args.load_path}")
    else:
        pre_train(args, pre_snapshot_path)

    # Self-train (重新配置日志)
    if is_main_process():
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=os.path.join(self_snapshot_path, "log.txt"),
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    self_train(args, pre_snapshot_path, self_snapshot_path)

    if use_ddp:
        torch.distributed.destroy_process_group()
