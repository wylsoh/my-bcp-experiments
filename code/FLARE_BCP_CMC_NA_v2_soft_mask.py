"""
FLARE BCP + NA-CMC V2: 软权重掩码（3D 多器官版本）

基于 LA_BCP_CMC_NA_v2_soft_mask.py 适配：
  - num_classes = 14（13个腹部器官 + 背景）
  - patch_size = (64, 128, 128)
  - 数据集使用 FLAREDataSets，从 ../data_split/flare 加载
  - labelnum = 42（约 10% × 419 总样本）
  - 使用 mask_DiceLoss(nclass=14) 支持多类 Dice
  - 验证改用 var_all_case_FLARE（多类评估）
"""
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import h5py
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import losses, ramps, test_3d_patch
from dataloaders.dataset import (RandomRotFlip, RandomCrop, RandomNoise,
                                  ToTensor, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils.BCP_utils import mix_loss, update_ema_variables


# ================================================================
# FLARE 数据集
# ================================================================
class FLAREDataSets(Dataset):
    """FLARE 数据集（13个腹部器官 + 背景 = 14 类）"""

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.txt'
        test_path = self._base_dir + '/test.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.strip() for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}/2022.h5".format(image_name), 'r')
        image = h5f['image'][:]
        if "label" in h5f.keys():
            label = h5f['label'][:]
        else:
            label_shape = image.shape
            label = np.zeros((label_shape), dtype=int, order='C')
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


# ================================================================
# 参数
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/flare')
parser.add_argument('--exp', type=str,  default='BCP_CMC_NA_v2')
parser.add_argument('--model', type=str, default='VNet')
parser.add_argument('--pre_max_iteration', type=int,  default=6000)
parser.add_argument('--self_max_iteration', type=int,  default=30000)
parser.add_argument('--max_samples', type=int,  default=419)
parser.add_argument('--labeled_bs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--base_lr', type=float,  default=0.01)
parser.add_argument('--deterministic', type=int,  default=1)
parser.add_argument('--labelnum', type=int,  default=42)
parser.add_argument('--gpu', type=str,  default='1')
parser.add_argument('--seed', type=int,  default=1337)
parser.add_argument('--consistency', type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
parser.add_argument('--magnitude', type=float,  default=10.0)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--mask_ratio', type=float, default=2/3)
parser.add_argument('--u_alpha', type=float, default=2.0)
parser.add_argument('--loss_weight', type=float, default=0.5)
# ---------- CMC 参数 ----------
parser.add_argument('--cmc_patch_size',        type=int,   default=8)
parser.add_argument('--cmc_warmup_iter',       type=int,   default=2000)
parser.add_argument('--cmc_init_shared',       type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',       type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',     type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh',type=float, default=0.75)
parser.add_argument('--conf_thresh_init',      type=float, default=0.90)
parser.add_argument('--conf_thresh_final',     type=float, default=0.70)
# ---------- NA-CMC V2 新增参数 ----------
parser.add_argument('--cmc_soft_sigma', type=float, default=0.3,
    help='平滑强度，控制过渡带宽度。0=接近二值，1=最大平滑。'
         '实际核大小 = int(patch_size * sigma) | 1。')
args = parser.parse_args()

# ================================================================
# 全局设置
# ================================================================
train_data_path = args.root_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations  = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size  = (64, 128, 128)  # FLARE 3D patch
num_classes = 14  # 13器官 + 背景

# FLARE 专用的 DICE loss（多类）
DICE = losses.mask_DiceLoss(nclass=num_classes)
CE_14 = nn.CrossEntropyLoss(reduction='none')


# ================================================================
# FLARE 兼容的 context_mask（动态计算裁剪边界）
# ================================================================
def context_mask_flare(img, mask_ratio):
    """使用 tensor 实际尺寸动态生成 BCP mask（兼容任意 patch_size）"""
    batch_size = img.shape[0]
    img_x, img_y, img_z = img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x = int(img_x * mask_ratio)
    patch_pixel_y = int(img_y * mask_ratio)
    patch_pixel_z = int(img_z * mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()


# ================================================================
# FLARE 验证函数（多类评估）
# ================================================================
def var_all_case_FLARE(model, num_classes=14, patch_size=(64, 128, 128),
                       stride_xy=32, stride_z=16):
    """FLARE 数据集验证（14类，计算 Dice + HD95）"""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data_split', 'flare')
    with open(os.path.join(base_dir, 'test.txt'), 'r') as f:
        image_list = f.readlines()
    image_list = [
        os.path.join(base_dir, 'data', item.strip(), '2022.h5')
        for item in image_list
    ]
    logging.info("Validating on {} FLARE test cases".format(len(image_list)))
    loader = tqdm(image_list)
    total_metric = None
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_3d_patch.test_single_case(
            model, image, stride_xy, stride_z, patch_size,
            num_classes=num_classes)
        # 计算多类指标
        single_metric = test_3d_patch._calc_multiclass_metric(
            prediction, label, num_classes)
        if total_metric is None:
            total_metric = np.zeros_like(single_metric)
        total_metric += np.asarray(single_metric)
        loader.set_description("Dice={:.4f}".format(np.mean(single_metric[:, 0])))

    avg_metric = total_metric / len(image_list)
    mean_dice = np.mean(avg_metric[:, 0])
    mean_hd95 = np.mean(avg_metric[:, 2])
    print('Average Dice: {:.4f}, Average HD95: {:.2f}'.format(mean_dice, mean_hd95))
    return mean_dice


# ================================================================
# 函数：多类版本
# ================================================================
def get_cut_mask_multi(out, thres=0.5, nms=0):
    """多类版本：取 argmax 作为硬标签"""
    probs = F.softmax(out, 1)
    masks = probs.argmax(dim=1).long()
    return masks


def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# ================================================================
# NA-CMC V2 3D 核心函数（与 LA 版本一致，支持任意类数）
# ================================================================
def generate_cmc_masks_3d_soft(img, cmc_patch_size=8, shared_ratio=0.0, soft_sigma=0.3):
    B, C, H, W, D = img.shape
    p = cmc_patch_size
    assert H % p == 0 and W % p == 0 and D % p == 0, \
        f"H={H}, W={W}, D={D} 必须均能被 cmc_patch_size={p} 整除"
    n_h, n_w, n_d = H // p, W // p, D // p

    smooth_k = max(int(cmc_patch_size * soft_sigma) | 1, 1)
    smooth_pad = smooth_k // 2

    masks_a, masks_b = [], []
    for _ in range(B):
        base = (torch.rand(n_h, n_w, n_d) > 0.5).float()

        if shared_ratio > 0.0:
            shared = torch.rand(n_h, n_w, n_d) < shared_ratio
            pa = ((base == 0) | shared).float()
            pb = ((base == 1) | shared).float()
        else:
            pa = (base == 0).float()
            pb = (base == 1).float()

        pa_up = F.interpolate(
            pa.view(1, 1, n_h, n_w, n_d),
            size=(H, W, D), mode='trilinear', align_corners=False
        )
        pb_up = F.interpolate(
            pb.view(1, 1, n_h, n_w, n_d),
            size=(H, W, D), mode='trilinear', align_corners=False
        )

        if smooth_k > 1:
            pa_up = F.avg_pool3d(pa_up, kernel_size=smooth_k,
                                  stride=1, padding=smooth_pad)
            pb_up = F.avg_pool3d(pb_up, kernel_size=smooth_k,
                                  stride=1, padding=smooth_pad)

        masks_a.append(pa_up.squeeze(0))
        masks_b.append(pb_up.squeeze(0))

    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device))


def cmc_mutual_loss_3d_soft(out_vA, out_vB, teacher_hard, conf_mask,
                             mask_a_soft, mask_b_soft, mutual_conf_thresh,
                             mutual_weight):
    ma = mask_a_soft.squeeze(1)
    mb = mask_b_soft.squeeze(1)
    w = conf_mask

    denom_a = (w * ma).sum() + 1e-6
    denom_b = (w * mb).sum() + 1e-6
    la = CE_14(out_vA, teacher_hard)
    lb = CE_14(out_vB, teacher_hard)
    loss_anchor = ((la * ma * w).sum() / denom_a +
                   (lb * mb * w).sum() / denom_b) / 2.0

    with torch.no_grad():
        prob_a = F.softmax(out_vA, dim=1)
        prob_b = F.softmax(out_vB, dim=1)
        conf_va = prob_a.max(dim=1).values
        conf_vb = prob_b.max(dim=1).values
        plab_va = prob_a.argmax(dim=1).long()
        plab_vb = prob_b.argmax(dim=1).long()

    w_b = (1.0 - mb) * ma * (conf_va > mutual_conf_thresh).float()
    w_a = (1.0 - ma) * mb * (conf_vb > mutual_conf_thresh).float()

    denom_mut_b = w_b.sum() + 1e-6
    denom_mut_a = w_a.sum() + 1e-6
    l_b_from_a = (CE_14(out_vB, plab_va) * w_b).sum() / denom_mut_b
    l_a_from_b = (CE_14(out_vA, plab_vb) * w_a).sum() / denom_mut_a
    loss_mutual = (l_b_from_a + l_a_from_b) / 2.0

    return loss_anchor + mutual_weight * loss_mutual


def get_progressive_shared_ratio(current_iter, warmup_iter,
                                  init_ratio=0.4, final_ratio=0.0):
    if warmup_iter <= 0 or current_iter >= warmup_iter:
        return float(final_ratio)
    return init_ratio + (final_ratio - init_ratio) * float(current_iter) / float(warmup_iter)


def get_adaptive_threshold(current_iter, max_iter,
                            init_threshold=0.90, final_threshold=0.70):
    progress = min(1.0, float(current_iter) / float(max_iter))
    return init_threshold + (final_threshold - init_threshold) * progress


# ================================================================
# 多类 mix_loss（使用 CE_14）
# ================================================================
def mix_loss_multi(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    # 自动扩展 label 和 mask 的 batch 维度（sub_bs 可能与 output batch 不一致）
    if mask.size(0) != net3_output.size(0):
        mask = mask.expand(net3_output.size(0), -1, -1, -1)
    if img_l.size(0) != net3_output.size(0):
        img_l = img_l.expand(net3_output.size(0), -1, -1, -1)
    if patch_l.size(0) != net3_output.size(0):
        patch_l = patch_l.expand(net3_output.size(0), -1, -1, -1)
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE_14(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE_14(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss


# ================================================================
# Pre-train
# ================================================================
def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = FLAREDataSets(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(patch_size),
                                 ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs   = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size,
                                           args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs]
            label_batch  = sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs],  label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask_flare(img_a, args.mask_ratio)

            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch  = lab_a * img_mask + lab_b * (1 - img_mask)

            outputs, _ = model(volume_batch)
            loss_ce   = CE_14(outputs, label_batch).mean()
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce',   loss_ce,   iter_num)
            writer.add_scalar('pre/loss_all',  loss,      iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' %
                         (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = var_all_case_FLARE(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=32, stride_z=16)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice',      dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice,   iter_num)
                model.train()

        if iter_num == pre_max_iterations:
            final_best_path = os.path.join(snapshot_path,
                '{}_best_model.pth'.format(args.model))
            save_net_opt(model, optimizer, final_best_path)
            logging.info("save final model to {}".format(final_best_path))

            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# Self-train：BCP + NA-CMC V2
# ================================================================
def self_train(args, pre_snapshot_path, self_snapshot_path):
    model     = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()

    db_train = FLAREDataSets(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(patch_size),
                                 ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs   = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler  = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                           args.batch_size,
                                           args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path,
                                    '{}_best_model.pth'.format(args.model))
    load_net(model,     pretrained_model)
    load_net(ema_model, pretrained_model)

    smooth_k = max(int(args.cmc_patch_size * args.cmc_soft_sigma) | 1, 1)
    logging.info("Start self_training (BCP + NA-CMC V2 Soft Mask 3D) - FLARE")
    logging.info("soft_sigma={} → avg_pool3d 核 {}px".format(
        args.cmc_soft_sigma, smooth_k))

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num  = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a,   img_b   = volume_batch[:sub_bs],            volume_batch[sub_bs:args.labeled_bs]
            lab_a,   lab_b   = label_batch[:sub_bs],             label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = (volume_batch[args.labeled_bs:args.labeled_bs + sub_bs],
                                volume_batch[args.labeled_bs + sub_bs:])

            # ==============================================================
            # BCP 部分
            # ==============================================================
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask_multi(unoutput_a, nms=0)
                plab_b = get_cut_mask_multi(unoutput_b, nms=0)
                img_mask, loss_mask = context_mask_flare(img_a, args.mask_ratio)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a   * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b   * (1 - img_mask)
            mixl_lab = lab_a   * img_mask + plab_a  * (1 - img_mask)
            mixu_lab = plab_b  * img_mask + lab_b   * (1 - img_mask)

            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            loss_l = mix_loss_multi(outputs_l, lab_a,  plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss_multi(outputs_u, plab_b, lab_b,  loss_mask,
                                     u_weight=args.u_weight, unlab=True)
            loss_bcp = loss_l + loss_u

            # ==============================================================
            # NA-CMC V2 分支
            # ==============================================================
            shared_ratio = get_progressive_shared_ratio(
                iter_num, args.cmc_warmup_iter, args.cmc_init_shared, 0.0)
            current_conf_thresh = get_adaptive_threshold(
                iter_num, self_max_iterations,
                args.conf_thresh_init, args.conf_thresh_final)

            mask_a_ab, mask_b_ab = generate_cmc_masks_3d_soft(
                unimg_a, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)
            mask_a_cd, mask_b_cd = generate_cmc_masks_3d_soft(
                unimg_b, args.cmc_patch_size, shared_ratio, args.cmc_soft_sigma)

            uimg_a_viewA = unimg_a * mask_a_ab
            uimg_a_viewB = unimg_a * mask_b_ab
            uimg_b_viewC = unimg_b * mask_a_cd
            uimg_b_viewD = unimg_b * mask_b_cd

            out_ab_all, _ = model(torch.cat([uimg_a_viewA, uimg_a_viewB], dim=0))
            out_cd_all, _ = model(torch.cat([uimg_b_viewC, uimg_b_viewD], dim=0))
            # 使用实际样本数切分视图（sub_bs 可能 < unimg_b 的样本数）
            na = unimg_a.size(0)
            nb = unimg_b.size(0)
            out_a_viewA = out_ab_all[:na]
            out_a_viewB = out_ab_all[na:]
            out_b_viewC = out_cd_all[:nb]
            out_b_viewD = out_cd_all[nb:]

            with torch.no_grad():
                teacher_hard_a = plab_a.long()
                teacher_hard_b = plab_b.long()
                conf_a = F.softmax(unoutput_a, dim=1).max(dim=1).values
                conf_b = F.softmax(unoutput_b, dim=1).max(dim=1).values
                conf_mask_a = (conf_a > current_conf_thresh).float()
                conf_mask_b = (conf_b > current_conf_thresh).float()

            loss_cmc_a = cmc_mutual_loss_3d_soft(
                out_a_viewA, out_a_viewB,
                teacher_hard_a, conf_mask_a, mask_a_ab, mask_b_ab,
                args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
            loss_cmc_b = cmc_mutual_loss_3d_soft(
                out_b_viewC, out_b_viewD,
                teacher_hard_b, conf_mask_b, mask_a_cd, mask_b_cd,
                args.cmc_mutual_conf_thresh, args.cmc_mutual_weight)
            loss_cmc = (loss_cmc_a + loss_cmc_b) / 2.0

            cmc_rampup = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * cmc_rampup * loss_cmc

            iter_num += 1
            writer.add_scalar('Self/consistency',   consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l',        loss_l,             iter_num)
            writer.add_scalar('Self/loss_u',        loss_u,             iter_num)
            writer.add_scalar('Self/loss_bcp',      loss_bcp,           iter_num)
            writer.add_scalar('Self/loss_cmc',      loss_cmc,           iter_num)
            writer.add_scalar('Self/loss_all',      loss,               iter_num)
            writer.add_scalar('Self/cmc_rampup',    cmc_rampup,         iter_num)
            writer.add_scalar('Self/shared_ratio',  shared_ratio,       iter_num)
            writer.add_scalar('Self/conf_threshold', current_conf_thresh, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : loss: %03f, bcp: %03f, cmc: %03f (anchor=%.4f mutual=%.4f) | border=%.3f conf_t=%.2f' %
                (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                 loss_cmc_a.item(), loss_cmc_b.item(),
                 consistency_weight, current_conf_thresh))

            update_ema_variables(model, ema_model, 0.99)

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = var_all_case_FLARE(
                    model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=32, stride_z=16)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,
                        '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice',      dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice,   iter_num)
                model.train()

            if iter_num >= self_max_iterations:
                break
        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    pre_snapshot_path  = "./model/BCP/FLARE_{}_{}_labeled/pre_train".format(
        args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/FLARE_{}_{}_labeled/self_train".format(
        args.exp, args.labelnum)
    print("Starting FLARE BCP + NA-CMC V2 (Soft Mask) training.")
    print("  exp={}, labelnum={}, soft_sigma={}, cmc_patch_size={}".format(
        args.exp, args.labelnum, args.cmc_soft_sigma, args.cmc_patch_size))
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(__file__, self_snapshot_path)

    # 检查 pre-train checkpoint 是否存在，存在则跳过 pre-train
    pretrained_model = os.path.join(pre_snapshot_path,
                                    '{}_best_model.pth'.format(args.model))
    if os.path.exists(pretrained_model):
        print("Found pre-trained checkpoint: {}".format(pretrained_model))
        print("Skipping pre-train, directly starting self-train...")
    else:
        # Pre-train
        logging.basicConfig(
            filename=pre_snapshot_path + "/log.txt",
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        pre_train(args, pre_snapshot_path)

    # 释放 pre-train 的 CUDA 缓存，避免 self-train OOM
    torch.cuda.empty_cache()

    # Self-train
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)
