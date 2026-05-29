"""
BCP + MAE + CorrMatch 半监督医学图像分割实验脚本

核心创新:
  1. BCP 双向 copy-paste (基线)
  2. MAE 网格遮挡: 学生看遮挡图, 教师看完整图
  3. CorrMatch 相关性传播: 高置信度伪标签 → 传播到低置信度区域
     解决 MAE 遮挡区域边界处伪标签覆盖不足的问题

实验命名: BCP_MAE_CorrMatch
指标输出: Dice, Jaccard, HD95, ASD
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

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net

from utils.train_utils import (
    load_net, load_net_opt, save_net_opt,
    update_model_ema, get_current_consistency_weight,
    poly_lr, mix_loss, mae_consistency_loss, patients_to_slices
)
from utils.mask_generator import MAEGridMaskGenerator, BCPMaskGenerator
from utils.pseudo_label_utils import (
    get_ACDC_masks, get_confidence_mask, get_adaptive_threshold
)
from utils.metric_utils import (
    test_single_volume_all_metrics, log_validation_metrics
)
from utils.corrmatch_utils import (
    CorrelationPropagator, FeatureExtractorHook,
    register_feature_hooks, get_features_from_model
)

# ================================================================
# 参数定义
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/ACDC', help='数据根目录')
parser.add_argument('--exp', type=str, default='BCP_MAE_CorrMatch', help='实验名称')
parser.add_argument('--model', type=str, default='unet', help='模型名称')
parser.add_argument('--pre_iterations', type=int, default=10000, help='预训练最大迭代数')
parser.add_argument('--max_iterations', type=int, default=30000, help='自训练最大迭代数')
parser.add_argument('--batch_size', type=int, default=24, help='批大小')
parser.add_argument('--deterministic', type=int, default=1, help='确定性训练')
parser.add_argument('--base_lr', type=float, default=0.01, help='基础学习率')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='输入尺寸')
parser.add_argument('--seed', type=int, default=1337, help='随机种子')
parser.add_argument('--num_classes', type=int, default=4, help='类别数')
parser.add_argument('--labeled_bs', type=int, default=12, help='有标签子批大小')
parser.add_argument('--labelnum', type=int, default=7, help='有标签患者数')
parser.add_argument('--u_weight', type=float, default=0.5, help='无标签基础权重')
parser.add_argument('--gpu', type=str, default='0', help='GPU编号')
parser.add_argument('--consistency', type=float, default=0.1, help='一致性权重')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='一致性ramp-up')

# MAE 参数
parser.add_argument('--mae_patch_size', type=int, default=16, help='MAE网格块大小')
parser.add_argument('--mae_mask_ratio', type=float, default=0.65, help='MAE最终遮挡比例')
parser.add_argument('--mae_warmup_iter', type=int, default=3000, help='MAE渐进热身步数')
parser.add_argument('--mae_loss_weight', type=float, default=1.0, help='MAE一致性损失权重')
parser.add_argument('--conf_thresh_init', type=float, default=0.90, help='置信度初始阈值')
parser.add_argument('--conf_thresh_final', type=float, default=0.70, help='置信度最终阈值')

# CorrMatch 参数
parser.add_argument('--corr_temperature', type=float, default=0.1, help='相关性温度参数')
parser.add_argument('--corr_prop_threshold', type=float, default=0.6, help='传播后置信度阈值')
parser.add_argument('--corr_downsample', type=int, default=4, help='相关性计算下采样倍数')
parser.add_argument('--corr_warmup_iter', type=int, default=5000, help='CorrMatch启动迭代数')
parser.add_argument('--corr_loss_weight', type=float, default=0.5, help='传播区域损失权重')

args = parser.parse_args()


# ================================================================
# Pre-train（与原始BCP一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs = int(args.labeled_bs / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs
    )
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
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            writer.add_scalar('pre/total_loss', loss, iter_num)
            writer.add_scalar('pre/mix_dice', loss_dice, iter_num)
            writer.add_scalar('pre/mix_ce', loss_ce, iter_num)
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

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
                    metric_i = test_single_volume_all_metrics(
                        sampled_batch["image"], sampled_batch["label"],
                        model, classes=num_classes, patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice, mean_jaccard, mean_hd95, mean_asd = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="pre_val"
                )

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info("Pre-training finished. Best dice: {:.4f}".format(best_performance))


# ================================================================
# Self-train: BCP + MAE + CorrMatch
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
    print("Total slices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    # MAE 组件
    mae_mask_gen = MAEGridMaskGenerator(
        img_size=args.patch_size[0],
        patch_size=args.mae_patch_size,
        mask_ratio=args.mae_mask_ratio
    )

    # CorrMatch 组件
    corr_propagator = CorrelationPropagator(
        num_classes=num_classes,
        temperature=args.corr_temperature,
        propagation_threshold=args.corr_prop_threshold,
        downsample_factor=args.corr_downsample
    )

    # 教师模型特征提取 hook
    ema_hook, ema_handle, hook_layer_name = register_feature_hooks(ema_model)
    logging.info("Feature hook registered at layer: {}".format(hook_layer_name))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (BCP + MAE + CorrMatch)")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("MAE: patch={}, ratio={}, warmup={}".format(
        args.mae_patch_size, args.mae_mask_ratio, args.mae_warmup_iter))
    logging.info("CorrMatch: temp={}, prop_thresh={}, ds={}, warmup={}".format(
        args.corr_temperature, args.corr_prop_threshold, args.corr_downsample, args.corr_warmup_iter))

    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            # ---- 数据拆分 ----
            img_a = volume_batch[:labeled_sub_bs]
            img_b = volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a = label_batch[:labeled_sub_bs]
            lab_b = label_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            uimg_b = volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            ulab_a = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            ulab_b = label_batch[args.labeled_bs + unlabeled_sub_bs:]

            # ============================================================
            # Part 1: BCP 原始双向 copy-paste
            # ============================================================

            # ★ 已修复: 显式调用组件内部的 BCPMaskGenerator.generate
            img_mask, loss_mask = BCPMaskGenerator.generate(img_a)

            with torch.no_grad():
                # 教师前向 + 提取中间特征
                ema_hook.clear()
                pre_a = ema_model(uimg_a)
                feat_a = ema_hook.features

                ema_hook.clear()
                pre_b = ema_model(uimg_b)
                feat_b = ema_hook.features

                # 初始伪标签
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)

                # 可视化用混合标签
                unl_label_vis = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label_vis = lab_b * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(
                iter_num // 150, args.consistency, args.consistency_rampup
            )

            # BCP 混合输入
            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)

            out_unl = model(net_input_unl)
            out_l = model(net_input_l)

            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask,
                                         u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask,
                                     u_weight=args.u_weight)

            loss_bcp_dice = unl_dice + l_dice
            loss_bcp_ce = unl_ce + l_ce
            loss_bcp = (loss_bcp_dice + loss_bcp_ce) / 2

            # ============================================================
            # Part 2: MAE 一致性分支
            # ============================================================
            current_mask_ratio = mae_mask_gen.get_progressive_ratio(
                current_iter=iter_num,
                warmup_iter=args.mae_warmup_iter,
                min_ratio=0.25,
                max_ratio=args.mae_mask_ratio
            )
            mae_mask_gen.mask_ratio = current_mask_ratio

            current_conf_thresh = get_adaptive_threshold(
                current_iter=iter_num,
                max_iter=max_iterations,
                init_threshold=args.conf_thresh_init,
                final_threshold=args.conf_thresh_final
            )

            mae_mask_a = mae_mask_gen.generate(uimg_a.shape[0], uimg_a.device)
            mae_mask_b = mae_mask_gen.generate(uimg_b.shape[0], uimg_b.device)

            uimg_a_masked = uimg_a * mae_mask_a
            uimg_b_masked = uimg_b * mae_mask_b

            out_a_masked = model(uimg_a_masked)
            out_b_masked = model(uimg_b_masked)

            with torch.no_grad():
                _, conf_mask_a, mean_conf_a = get_confidence_mask(pre_a, threshold=current_conf_thresh)
                _, conf_mask_b, mean_conf_b = get_confidence_mask(pre_b, threshold=current_conf_thresh)
                plab_a_full = get_ACDC_masks(pre_a, nms=1)
                plab_b_full = get_ACDC_masks(pre_b, nms=1)

            # ============================================================
            # Part 3: CorrMatch 相关性传播
            # ============================================================
            use_corrmatch = (iter_num >= args.corr_warmup_iter)
            prop_ratio_a, prop_ratio_b = 0.0, 0.0

            if use_corrmatch and feat_a is not None and feat_b is not None:
                with torch.no_grad():
                    teacher_probs_a = F.softmax(pre_a, dim=1)
                    teacher_probs_b = F.softmax(pre_b, dim=1)

                    refined_plab_a, refined_conf_a, prop_ratio_a = corr_propagator.propagate(
                        features=feat_a,
                        pseudo_label=plab_a_full,
                        confidence_mask=conf_mask_a,
                        teacher_probs=teacher_probs_a
                    )
                    refined_plab_b, refined_conf_b, prop_ratio_b = corr_propagator.propagate(
                        features=feat_b,
                        pseudo_label=plab_b_full,
                        confidence_mask=conf_mask_b,
                        teacher_probs=teacher_probs_b
                    )

                    new_region_a = torch.clamp(refined_conf_a - conf_mask_a.float(), 0, 1)
                    new_region_b = torch.clamp(refined_conf_b - conf_mask_b.float(), 0, 1)
            else:
                refined_plab_a = plab_a_full
                refined_plab_b = plab_b_full
                refined_conf_a = conf_mask_a
                refined_conf_b = conf_mask_b
                new_region_a = torch.zeros_like(conf_mask_a)
                new_region_b = torch.zeros_like(conf_mask_b)

            # ============================================================
            # Part 4: MAE 一致性损失
            # ============================================================
            loss_mae_a, valid_px_a = mae_consistency_loss(
                student_output=out_a_masked,
                teacher_pseudo_label=refined_plab_a,
                confidence_mask=conf_mask_a,
                visible_mask=mae_mask_a,
                n_classes=num_classes
            )
            loss_mae_b, valid_px_b = mae_consistency_loss(
                student_output=out_b_masked,
                teacher_pseudo_label=refined_plab_b,
                confidence_mask=conf_mask_b,
                visible_mask=mae_mask_b,
                n_classes=num_classes
            )
            loss_mae = (loss_mae_a + loss_mae_b) / 2

            loss_prop_a, prop_px_a = mae_consistency_loss(
                student_output=out_a_masked,
                teacher_pseudo_label=refined_plab_a,
                confidence_mask=new_region_a,
                visible_mask=mae_mask_a,
                n_classes=num_classes
            )
            loss_prop_b, prop_px_b = mae_consistency_loss(
                student_output=out_b_masked,
                teacher_pseudo_label=refined_plab_b,
                confidence_mask=new_region_b,
                visible_mask=mae_mask_b,
                n_classes=num_classes
            )
            loss_prop = (loss_prop_a + loss_prop_b) / 2

            # ============================================================
            # Part 5: 总损失
            # ============================================================
            mae_rampup = min(1.0, iter_num / max(args.mae_warmup_iter, 1))
            mae_weight = args.mae_loss_weight * mae_rampup

            corr_rampup = min(1.0, max(0.0, (iter_num - args.corr_warmup_iter)) / max(args.mae_warmup_iter, 1))
            corr_weight = args.corr_loss_weight * corr_rampup

            loss = (loss_bcp
                    + consistency_weight * mae_weight * loss_mae
                    + consistency_weight * corr_weight * loss_prop)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            current_lr = poly_lr(optimizer, base_lr, iter_num, max_iterations)

            # ============================================================
            # 日志
            # ============================================================
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_bcp', loss_bcp, iter_num)
            writer.add_scalar('info/loss_bcp_dice', loss_bcp_dice, iter_num)
            writer.add_scalar('info/loss_bcp_ce', loss_bcp_ce, iter_num)
            writer.add_scalar('info/loss_mae', loss_mae, iter_num)
            writer.add_scalar('info/loss_prop', loss_prop, iter_num)
            writer.add_scalar('info/mae_weight', mae_weight, iter_num)
            writer.add_scalar('info/corr_weight', corr_weight, iter_num)
            writer.add_scalar('info/mask_ratio', current_mask_ratio, iter_num)
            writer.add_scalar('info/conf_threshold', current_conf_thresh, iter_num)
            writer.add_scalar('info/mean_conf_a', mean_conf_a, iter_num)
            writer.add_scalar('info/mean_conf_b', mean_conf_b, iter_num)
            writer.add_scalar('info/valid_pixels_a', valid_px_a, iter_num)
            writer.add_scalar('info/valid_pixels_b', valid_px_b, iter_num)
            writer.add_scalar('info/prop_ratio_a', prop_ratio_a, iter_num)
            writer.add_scalar('info/prop_ratio_b', prop_ratio_b, iter_num)
            writer.add_scalar('info/prop_pixels_a', prop_px_a, iter_num)
            writer.add_scalar('info/prop_pixels_b', prop_px_b, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/lr', current_lr, iter_num)
            writer.add_scalar('info/use_corrmatch', float(use_corrmatch), iter_num)

            logging.info(
                'iter %d: loss=%.4f, bcp=%.4f, mae=%.4f, prop=%.4f, '
                'mae_w=%.3f, corr_w=%.3f, mask_r=%.2f, conf_t=%.2f, '
                'conf_a=%.3f, conf_b=%.3f, prop_r_a=%.3f, prop_r_b=%.3f, lr=%f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_mae.item(), loss_prop.item(),
                 mae_weight, corr_weight, current_mask_ratio, current_conf_thresh,
                 mean_conf_a, mean_conf_b, prop_ratio_a, prop_ratio_b, current_lr)
            )

            # ============================================================
            # TensorBoard 可视化
            # ============================================================
            if iter_num % 20 == 0:
                # BCP
                image = net_input_unl[1, 0:1, :, :]
                writer.add_image('train/BCP_Un_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image('train/BCP_Un_Prediction', outputs[1, ...] * 50, iter_num)
                labs = unl_label_vis[1, ...].unsqueeze(0) * 50
                writer.add_image('train/BCP_Un_GroundTruth', labs, iter_num)

                image_l = net_input_l[1, 0:1, :, :]
                writer.add_image('train/BCP_L_Image', image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/BCP_L_Prediction', outputs_l[1, ...] * 50, iter_num)
                labs_l = l_label_vis[1, ...].unsqueeze(0) * 50
                writer.add_image('train/BCP_L_GroundTruth', labs_l, iter_num)

                # MAE
                writer.add_image('train/MAE_Original', uimg_a[0, 0:1, :, :], iter_num)
                writer.add_image('train/MAE_MaskedInput', uimg_a_masked[0, 0:1, :, :], iter_num)
                writer.add_image('train/MAE_VisibleMask', mae_mask_a[0, 0:1, :, :], iter_num)
                pred_mae = torch.argmax(torch.softmax(out_a_masked, dim=1), dim=1, keepdim=True)
                writer.add_image('train/MAE_Prediction', pred_mae[0, ...].float() * 50, iter_num)
                writer.add_image('train/MAE_PseudoLabel',
                                 plab_a_full[0, ...].unsqueeze(0).float() * 50, iter_num)
                writer.add_image('train/MAE_ConfMask',
                                 conf_mask_a[0, ...].unsqueeze(0), iter_num)

                # CorrMatch
                if use_corrmatch:
                    writer.add_image('train/Corr_RefinedPseudo',
                                     refined_plab_a[0, ...].unsqueeze(0).float() * 50, iter_num)
                    writer.add_image('train/Corr_RefinedConf',
                                     refined_conf_a[0, ...].unsqueeze(0), iter_num)
                    writer.add_image('train/Corr_NewRegion',
                                     new_region_a[0, ...].unsqueeze(0), iter_num)

            # ============================================================
            # 验证
            # ============================================================
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_all_metrics(
                        sampled_batch["image"], sampled_batch["label"],
                        model, classes=num_classes, patch_size=args.patch_size
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                mean_dice, mean_jaccard, mean_hd95, mean_asd = log_validation_metrics(
                    metric_list, num_classes, iter_num, writer, logging, prefix="val"
                )

                if mean_dice > best_performance:
                    best_performance = mean_dice
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(
                        snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("=> saved best model at iter {} with dice {:.4f}".format(
                        iter_num, best_performance))

                if mean_hd95 < best_hd:
                    best_hd = mean_hd95

                logging.info('BEST so far - dice: {:.4f}, hd95: {:.2f}'.format(
                    best_performance, best_hd))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    ema_handle.remove()
    writer.close()
    logging.info("=" * 60)
    logging.info("Self-training (BCP+MAE+CorrMatch) finished.")
    logging.info("Best mean_dice: {:.4f}".format(best_performance))
    logging.info("Best mean_hd95: {:.2f}".format(best_hd))
    logging.info("=" * 60)


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

    pre_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    shutil.copy(__file__, self_snapshot_path)

    # Pre-train
    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self-train
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt", level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)