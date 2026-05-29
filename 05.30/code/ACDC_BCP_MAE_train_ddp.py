"""
===============================================================================
BCP + MAE — DDP + AMP 最优配置版本
===============================================================================

核心差异对比:
  [原始] batch_size=24, labeled_bs=12, 单 GPU, no AMP, cudnn.benchmark=False
  [本版] batch_size=96, labeled_bs=24, 4×GPU, AMP, cudnn.benchmark=True

启动:
    torchrun --nproc_per_node=4 ACDC_BCP_MAE_train_ddp.py \
        --batch_size 96 --labeled_bs 24 --amp --ddp \
        --pre_iterations 10000 --max_iterations 30000
===============================================================================
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

from dataloaders.dataset import (
    BaseDataSets, RandomGenerator, TwoStreamBatchSampler
)
from networks.net_factory import BCP_net
from utils.train_utils import (
    load_net, load_net_opt, save_net_opt,
    update_model_ema, get_current_consistency_weight,
    poly_lr, mix_loss, mae_consistency_loss, patients_to_slices,
    is_main_process
)
from utils.mask_generator import MAEGridMaskGenerator, BCPMaskGenerator
from utils.pseudo_label_utils import (
    get_ACDC_masks, get_confidence_mask, get_adaptive_threshold
)
from utils.metric_utils import (
    test_single_volume_all_metrics, log_validation_metrics
)
from ddp_train_adapter import (
    add_ddp_args, ddp_wrap_model, create_ddp_dataloader, AMPTrainer,
    print_ddp_config
)

# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser = add_ddp_args(parser)
parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
parser.add_argument('--exp', type=str, default='BCP_MAE_student_warmup_iter_5k_ddp')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--deterministic', type=int, default=0)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--labeled_bs', type=int, default=24)
parser.add_argument('--labelnum', type=int, default=7)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
# MAE
parser.add_argument('--mae_patch_size', type=int, default=16)
parser.add_argument('--mae_mask_ratio', type=float, default=0.65)
parser.add_argument('--mae_warmup_iter', type=int, default=5000)
parser.add_argument('--mae_loss_weight', type=float, default=1.0)
parser.add_argument('--conf_thresh_init', type=float, default=0.90)
parser.add_argument('--conf_thresh_final', type=float, default=0.70)
args = parser.parse_args()


# ================================================================
# 预训练
# ================================================================
def pre_train(args):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    labeled_sub_bs = int(args.labeled_bs / 2)
    device = args.device

    model = BCP_net(in_chns=1, class_num=num_classes)
    model = ddp_wrap_model(model, args)
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + args.rank + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train", num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    writer = SummaryWriter(snapshot_path + '/log') if is_main_process() else None
    if is_main_process():
        logging.info("Start pre_training (DDP+AMP)")

    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)
            net_input = img_a * img_mask + img_b * (1 - img_mask)

            with ampt.autocast():
                out_mixl = model(net_input)
                loss_dice, loss_ce = mix_loss(
                    out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
                loss = (loss_dice + loss_ce) / 2

            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)
            iter_num += 1

            if is_main_process() and iter_num % 20 == 0:
                writer.add_scalar('pre/total_loss', loss, iter_num)
                logging.info(f'pre iter {iter_num}: loss={loss.item():.4f}')

            if iter_num > 0 and iter_num % 200 == 0 and is_main_process():
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        unwrap_ddp(model), classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)
                mean_dice, *_ = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="pre_val")
                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_net_opt(model, optimizer,
                                 os.path.join(snapshot_path, f'{args.model}_best_model.pth'))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    if is_main_process():
        save_net_opt(model, optimizer,
                     os.path.join(snapshot_path, 'pre_train_model.pth'))
        writer.close()
        logging.info(f"Pre-training done. Best dice: {best_performance:.4f}")


# ================================================================
# 自训练
# ================================================================
def self_train(args, pre_snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    device = args.device
    pre_trained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)
    model = ddp_wrap_model(model, args)
    ema_model = ema_model.to(device)
    ampt = AMPTrainer(args.amp)

    def worker_init_fn(worker_id):
        random.seed(args.seed + args.rank + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path, split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    trainloader = create_ddp_dataloader(
        db_train, args.batch_size, args.labeled_bs,
        labeled_idxs, unlabeled_idxs, args,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        unwrap_ddp(model).parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    if is_main_process():
        logging.info(f"Loading pre-trained from {pre_trained_model}")
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)

    # MAE 掩码生成器
    mae_mask_gen = MAEGridMaskGenerator(
        img_size=args.patch_size[0],
        patch_size=args.mae_patch_size,
        mask_ratio=args.mae_mask_ratio,
        warmup_iter=args.mae_warmup_iter
    )

    writer = SummaryWriter(snapshot_path + '/log') if is_main_process() else None
    if is_main_process():
        logging.info("Start self_training (BCP + MAE, DDP+AMP)")

    model.train()
    ema_model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100.0
    iterator = tqdm(range(max_epoch), ncols=70) if is_main_process() else range(max_epoch)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].to(device, non_blocking=True)
            label_batch = sampled_batch['label'].to(device, non_blocking=True)

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]

            # ---- 教师伪标签 ----
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1).long()
                plab_b = get_ACDC_masks(pre_b, nms=1).long()
                img_mask, loss_mask = BCPMaskGenerator.generate(img_a)

            consistency_weight = get_current_consistency_weight(
                iter_num // 150, args.consistency, args.consistency_rampup)

            # ---- BCP 分支 ----
            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)

            with ampt.autocast():
                out_unl = model(net_input_unl)
                out_l = model(net_input_l)
                unl_dice, unl_ce = mix_loss(
                    out_unl, plab_a, lab_a, loss_mask,
                    u_weight=args.u_weight, unlab=True)
                l_dice, l_ce = mix_loss(
                    out_l, lab_b, plab_b, loss_mask,
                    u_weight=args.u_weight)
                loss_bcp = (unl_dice + unl_ce + l_dice + l_ce) / 4

                # ---- MAE 分支 ----
                current_mae_ratio = mae_mask_gen.get_current_ratio(iter_num)
                visible_mask_a = mae_mask_gen.generate(uimg_a.shape[0], uimg_a.device)
                visible_mask_b = mae_mask_gen.generate(uimg_b.shape[0], uimg_b.device)
                mae_input_a = uimg_a * visible_mask_a
                mae_input_b = uimg_b * visible_mask_b
                mae_out_a = model(mae_input_a)
                mae_out_b = model(mae_input_b)

                # 置信度掩码
                current_conf_thresh = get_adaptive_threshold(
                    iter_num, max_iterations,
                    args.conf_thresh_init, args.conf_thresh_final)
                _, conf_mask_a, _ = get_confidence_mask(pre_a, threshold=current_conf_thresh)
                _, conf_mask_b, _ = get_confidence_mask(pre_b, threshold=current_conf_thresh)

                # MAE 一致性损失（仅在遮挡区域 ∩ 高置信度区域）
                mae_loss_a, valid_a = mae_consistency_loss(
                    mae_out_a, plab_a, conf_mask_a, visible_mask_a, num_classes)
                mae_loss_b, valid_b = mae_consistency_loss(
                    mae_out_b, plab_b, conf_mask_b, visible_mask_b, num_classes)
                loss_mae = (mae_loss_a + mae_loss_b) / 2

                loss = loss_bcp + consistency_weight * args.mae_loss_weight * loss_mae

            ampt.backward(loss)
            ampt.step(optimizer)
            ampt.zero_grad(optimizer)
            iter_num += 1

            if is_main_process() and iter_num % 5 == 0:
                update_model_ema(model, ema_model, 0.99)

            current_lr = poly_lr(optimizer, base_lr, iter_num, max_iterations)

            if is_main_process():
                if iter_num % 20 == 0:
                    writer.add_scalar('info/total_loss', loss, iter_num)
                    writer.add_scalar('info/loss_bcp', loss_bcp, iter_num)
                    writer.add_scalar('info/loss_mae', loss_mae, iter_num)
                    writer.add_scalar('info/mae_ratio', current_mae_ratio, iter_num)
                    writer.add_scalar('info/conf_thresh', current_conf_thresh, iter_num)
                    writer.add_scalar('info/lr', current_lr, iter_num)
                    logging.info(
                        f'iter {iter_num} | total={loss.item():.4f} | '
                        f'bcp={loss_bcp.item():.4f} | mae={loss_mae.item():.4f} | '
                        f'mae_r={current_mae_ratio:.2f} | lr={current_lr:.5f}')

                if iter_num % 20 == 0:
                    writer.add_image('bcp/Un_Input', net_input_unl[0, 0:1], iter_num)
                    pred_unl = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                    writer.add_image('bcp/Un_Pred', pred_unl[0].float() * 50, iter_num)
                    writer.add_image('mae/VisibleMask', visible_mask_a[0], iter_num)
                    writer.add_image('mae/Masked_Input', mae_input_a[0, 0:1], iter_num)

            if iter_num > 0 and iter_num % 200 == 0 and is_main_process():
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        val_batch["image"], val_batch["label"],
                        unwrap_ddp(model), classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list /= len(db_val)
                mean_dice, _, mean_hd95, _ = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="val")
                if mean_dice > best_performance:
                    best_performance = mean_dice
                    torch.save(unwrap_ddp(model).state_dict(),
                               os.path.join(snapshot_path, f'{args.model}_best_model.pth'))
                if mean_hd95 < best_hd:
                    best_hd = mean_hd95
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    if is_main_process():
        torch.save(unwrap_ddp(model).state_dict(),
                   os.path.join(snapshot_path, 'final_model.pth'))
        writer.close()
        logging.info(f"Best dice={best_performance:.4f}, hd95={best_hd:.2f}")


# ================================================================
# 主函数
# ================================================================
if __name__ == "__main__":
    use_ddp = args.ddp and 'RANK' in os.environ

    if use_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=world_size)
        args.local_rank = local_rank
        args.rank = rank
        args.world_size = world_size
        args.device = torch.device(f'cuda:{local_rank}')
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        torch.distributed.barrier()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.local_rank = int(args.gpu)
        args.rank = 0
        args.world_size = 1
        args.device = torch.device('cuda:0')
        cudnn.benchmark = True
        cudnn.deterministic = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print_ddp_config(args)

    global snapshot_path
    pre_snapshot_path = f"./model/BCP/ACDC_{args.exp}_{args.labelnum}_labeled/pre_train"
    self_snapshot_path = f"./model/BCP/ACDC_{args.exp}_{args.labelnum}_labeled/self_train"
    for path in [pre_snapshot_path, self_snapshot_path]:
        if is_main_process() and not os.path.exists(path):
            os.makedirs(path)
    if use_ddp:
        torch.distributed.barrier()

    snapshot_path = pre_snapshot_path
    if is_main_process():
        shutil.copy(__file__, self_snapshot_path)
        logging.basicConfig(
            filename=pre_snapshot_path + "/log.txt",
            level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    pre_train(args)
    snapshot_path = self_snapshot_path

    if is_main_process():
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=self_snapshot_path + "/log.txt",
            level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

    self_train(args, pre_snapshot_path)

    if use_ddp:
        torch.distributed.destroy_process_group()
