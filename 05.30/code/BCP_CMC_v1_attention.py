"""
BCP + CMC v1-Attention：注意力引导互补掩码

在 BCP_CMC_v1_mutual.py 基础上，仅替换掩码生成函数：
  generate_cmc_masks  →  generate_attention_guided_masks

核心思想：
  用教师模型的像素级预测熵作为注意力引导，决定每个网格块分配给哪个视图。
    高熵（模型不确定）的块 → 视图A：强迫模型关注困难区域
    低熵（模型高置信）的块 → 视图B：互补区域，提供稳定监督
  与随机分配相比，两视图的信息难度有明显差异，互教信号更有针对性。

掩码生成流程：
  1. 教师 softmax 输出 → 像素级预测熵 H = -Σ p*log(p)
  2. avg_pool2d → 块级平均熵 [n, n]
  3. 以中位数为阈值：高熵块→A，低熵块→B（确保近似50/50分配）
  4. shared_ratio 控制热身阶段的重叠（与 v1 相同逻辑）

新增参数：
  无额外超参（使用中位数自适应阈值，无需手动设定）
  复用 cmc_patch_size / cmc_warmup_iter / cmc_init_shared

依赖：复用 BCP 分支已计算的 pre_a/pre_b（EMA 教师输出）
"""
import argparse, logging, os, random, shutil, sys
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
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import BCP_net
from utils import losses, ramps, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/ACDC')
parser.add_argument('--exp', type=str, default='BCP_CMC_v1_attention')
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--pre_iterations', type=int, default=10000)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[256, 256])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--labeled_bs', type=int, default=12)
parser.add_argument('--labelnum', type=int, default=7)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=200.0)
parser.add_argument('--magnitude', type=float, default=6.0)
parser.add_argument('--s_param', type=int, default=6)
parser.add_argument('--cmc_patch_size',        type=int,   default=16)
parser.add_argument('--cmc_warmup_iter',        type=int,   default=5000)
parser.add_argument('--cmc_init_shared',        type=float, default=0.4)
parser.add_argument('--cmc_loss_weight',        type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',      type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh', type=float, default=0.75)
parser.add_argument('--conf_thresh_init',       type=float, default=0.90)
parser.add_argument('--conf_thresh_final',      type=float, default=0.70)
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)

# ================================================================
# 原始 BCP 函数（逐字复制）
# ================================================================
def load_net(net, path):
    state = torch.load(str(path)); net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net']); optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    torch.save({'net': net.state_dict(), 'opt': optimizer.state_dict()}, str(path))

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        batch_list.append(class_list[0] + class_list[1] + class_list[2])
    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs

def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict(); ema_state = ema_model.state_dict()
    ema_model.load_state_dict({k: alpha * ema_state[k] + (1 - alpha) * model_state[k]
                               for k in model_state})

def generate_mask(img):
    bs, _, img_x, img_y = img.shape
    loss_mask = torch.ones(bs, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    px, py = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w, h = np.random.randint(0, img_x - px), np.random.randint(0, img_y - py)
    mask[w:w+px, h:h+py] = 0; loss_mask[:, w:w+px, h:h+py] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    iw, pw = (u_weight, l_weight) if unlab else (l_weight, u_weight)
    pm = 1 - mask
    ld = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * iw
    ld += dice_loss(output_soft, patch_l.unsqueeze(1), pm.unsqueeze(1)) * pw
    lc = iw * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    lc += pw * (CE(output, patch_l) * pm).sum() / (pm.sum() + 1e-16)
    return ld, lc

def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        ref = {"1":32,"3":68,"7":136,"14":256,"21":396,"28":512,"35":664,"70":1312}
    elif "Prostate" in dataset:
        ref = {"2":27,"4":53,"8":120,"12":179,"16":256,"21":312,"42":623}
    else:
        print("Error"); return
    return ref[str(patiens_num)]

def get_progressive_shared_ratio(cur, warm, init=0.4, final=0.0):
    if warm <= 0 or cur >= warm: return float(final)
    return init + (final - init) * float(cur) / float(warm)

def get_adaptive_threshold(cur, mx, init=0.90, final=0.70):
    return init + (final - init) * min(1.0, float(cur) / float(mx))

# ================================================================
# 核心改动：注意力引导掩码
# ================================================================
def generate_attention_guided_masks(img, teacher_logit,
                                     cmc_patch_size=16, shared_ratio=0.0):
    """
    注意力引导互补掩码

    利用教师预测熵作为注意力图，指导网格块的视图分配：
      高熵块（模型不确定，边界/模糊区域） → 视图A
      低熵块（模型高置信，心腔内部区域） → 视图B

    直觉：视图A专注于难样本区域，视图B覆盖易学习区域，
    互教时 A→B 传递边界知识，B→A 传递全局结构知识。

    以块级熵中位数为自适应阈值，保证每个样本近似 50/50 分配，
    无需手动设定额外超参。

    Args:
        img           : [B, C, H, W]
        teacher_logit : [B, C, H, W]  EMA 教师原始输出（未经 softmax）
                        直接复用 BCP 分支的 pre_a / pre_b
        cmc_patch_size: 网格块大小
        shared_ratio  : 热身阶段共享块比例

    Returns:
        mask_a : [B, 1, H, W]  视图A（高熵区域）可见掩码
        mask_b : [B, 1, H, W]  视图B（低熵区域）可见掩码
        mean_entropy : float   本批次平均块级熵（用于 TensorBoard 监控）
    """
    B, C, H, W = img.shape
    p = cmc_patch_size
    n = H // p

    with torch.no_grad():
        prob    = F.softmax(teacher_logit, dim=1)            # [B, C, H, W]
        # 像素级预测熵 H = -Σ p*log(p) ∈ [0, log(C)]
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        # 下采样到块级 [B, 1, n, n]
        ent_blocks = F.avg_pool2d(entropy, kernel_size=p, stride=p)

    masks_a, masks_b = [], []
    mean_entropies   = []

    for b in range(B):
        eb = ent_blocks[b, 0]                   # [n, n] CUDA
        mean_entropies.append(eb.mean().item())

        # 中位数自适应阈值：高熵→A，低熵→B，近似 50/50
        thresh = eb.median()
        high_ent = (eb >= thresh).cpu()         # [n, n] bool CPU

        if shared_ratio > 0.0:
            shared = torch.rand(n, n) < shared_ratio
            pa = (high_ent  | shared).float()
            pb = (~high_ent | shared).float()
        else:
            pa = high_ent.float()
            pb = (~high_ent).float()

        pa = F.interpolate(pa.view(1,1,n,n), size=(H,W), mode='nearest').squeeze(0)
        pb = F.interpolate(pb.view(1,1,n,n), size=(H,W), mode='nearest').squeeze(0)
        masks_a.append(pa)
        masks_b.append(pb)

    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device),
            float(np.mean(mean_entropies)))

# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr, num_classes = args.base_lr, args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lbs = int(args.labeled_bs / 2)
    model = BCP_net(in_chns=1, class_num=num_classes)
    def worker_init_fn(wid): random.seed(args.seed + wid)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")
    ls = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices: {}, labeled: {}".format(len(db_train), ls))
    bs = TwoStreamBatchSampler(list(range(ls)), list(range(ls, len(db_train))),
                               args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=bs, num_workers=4,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer   = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training"); logging.info("{} iters/epoch".format(len(trainloader)))
    model.train(); iter_num = 0; best_perf = 0.0
    iterator = tqdm(range(max_iterations // len(trainloader) + 1), ncols=70)
    for _ in iterator:
        for _, sb in enumerate(trainloader):
            vb, lb = sb['image'].cuda(), sb['label'].cuda()
            ia, ib = vb[:lbs], vb[lbs:args.labeled_bs]
            la, lb2 = lb[:lbs], lb[lbs:args.labeled_bs]
            im, lm = generate_mask(ia)
            gt = la * im + lb2 * (1 - im)
            ni = ia * im + ib * (1 - im); out = model(ni)
            ld, lc = mix_loss(out, la, lb2, lm, u_weight=1.0, unlab=True)
            loss = (ld + lc) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step(); iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice',   ld,   iter_num)
            writer.add_scalar('info/mix_ce',     lc,   iter_num)
            logging.info('iter %d: loss:%f dice:%f ce:%f' % (iter_num, loss, ld, lc))
            if iter_num % 20 == 0:
                writer.add_image('pre_train/Mixed_Image', ni[1,0:1], iter_num)
                writer.add_image('pre_train/Mixed_Prediction',
                    torch.argmax(torch.softmax(out,dim=1),dim=1,keepdim=True)[1,...]*50, iter_num)
                writer.add_image('pre_train/Mixed_GroundTruth', gt[1,...].unsqueeze(0)*50, iter_num)
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ml = sum(np.array(val_2d.test_single_volume(sb2["image"], sb2["label"],
                    model, classes=num_classes)) for _, sb2 in enumerate(valloader)) / len(db_val)
                for ci in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(ci+1), ml[ci,0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(ci+1), ml[ci,1], iter_num)
                perf = np.mean(ml, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', perf, iter_num)
                if perf > best_perf:
                    best_perf = perf
                    save_net_opt(model, optimizer, os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_perf,4))))
                    save_net_opt(model, optimizer, os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model)))
                logging.info('iter %d: mean_dice:%f' % (iter_num, perf)); model.train()
            if iter_num >= max_iterations: break
        if iter_num >= max_iterations: iterator.close(); break
    writer.close()

# ================================================================
# Self-train
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr, num_classes = args.base_lr, args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lbs = int(args.labeled_bs / 2)
    ubs = int((args.batch_size - args.labeled_bs) / 2)
    model     = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)
    def worker_init_fn(wid): random.seed(args.seed + wid)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")
    ls = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices: {}, labeled: {}".format(len(db_train), ls))
    bs = TwoStreamBatchSampler(list(range(ls)), list(range(ls, len(db_train))),
                               args.batch_size, args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=bs, num_workers=4,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model)))
    load_net_opt(model, optimizer, os.path.join(pre_snapshot_path,
                                                '{}_best_model.pth'.format(args.model)))
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training (BCP + CMC Attention-guided)")
    model.train(); ema_model.train()
    iter_num = 0; best_perf = 0.0
    iterator = tqdm(range(max_iterations // len(trainloader) + 1), ncols=70)
    for _ in iterator:
        for _, sb in enumerate(trainloader):
            vb, lb = sb['image'].cuda(), sb['label'].cuda()
            ia   = vb[:lbs];                  ib   = vb[lbs:args.labeled_bs]
            ua   = vb[args.labeled_bs:args.labeled_bs+ubs]; ub = vb[args.labeled_bs+ubs:]
            ula  = lb[args.labeled_bs:args.labeled_bs+ubs]; ulb = lb[args.labeled_bs+ubs:]
            la   = lb[:lbs];                  lb2  = lb[lbs:args.labeled_bs]

            # ---- BCP（与原始完全一致）----
            with torch.no_grad():
                pre_a = ema_model(ua); pre_b = ema_model(ub)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                im, lm = generate_mask(ia)
                unl_lbl = ula * im + la * (1-im); l_lbl = lb2 * im + ulb * (1-im)
            cw = get_current_consistency_weight(iter_num // 150)
            ni_u = ua * im + ia * (1-im); ni_l = ib * im + ub * (1-im)
            ou = model(ni_u); ol = model(ni_l)
            ud, uc = mix_loss(ou, plab_a, la,   lm, u_weight=args.u_weight, unlab=True)
            ld, lc = mix_loss(ol, lb2,    plab_b, lm, u_weight=args.u_weight)
            loss_bcp = ((ud+ld) + (uc+lc)) / 2

            # ---- CMC：注意力引导掩码 ----
            sr  = get_progressive_shared_ratio(iter_num, args.cmc_warmup_iter,
                                               args.cmc_init_shared, 0.0)
            ct  = get_adaptive_threshold(iter_num, max_iterations,
                                         args.conf_thresh_init, args.conf_thresh_final)

            # 传入教师 logit（pre_a/pre_b），由熵图决定块分配
            maa, mba, ent_a = generate_attention_guided_masks(ua, pre_a, args.cmc_patch_size, sr)
            mac, mbc, ent_b = generate_attention_guided_masks(ub, pre_b, args.cmc_patch_size, sr)

            ua_vA = ua * maa; ua_vB = ua * mba
            ub_vC = ub * mac; ub_vD = ub * mbc
            oAB = model(torch.cat([ua_vA, ua_vB], dim=0))
            oCD = model(torch.cat([ub_vC, ub_vD], dim=0))
            oA, oB = oAB[:ubs], oAB[ubs:]
            oC, oD = oCD[:ubs], oCD[ubs:]

            with torch.no_grad():
                cm_a = (F.softmax(pre_a,dim=1).max(dim=1).values > ct).float()
                cm_b = (F.softmax(pre_b,dim=1).max(dim=1).values > ct).float()
                pt_a = plab_a.long(); pt_b = plab_b.long()

            def cmc_loss(ovA, ovB, pt, cm, ma, mb):
                w = cm; d = w.sum() + 1e-6
                la_ = F.cross_entropy(ovA, pt, reduction='none')
                lb_ = F.cross_entropy(ovB, pt, reduction='none')
                anc = ((la_ + lb_) * w).sum() / d / 2.0
                with torch.no_grad():
                    pa_ = F.softmax(ovA,dim=1); pb_ = F.softmax(ovB,dim=1)
                    cva = pa_.max(dim=1).values; cvb = pb_.max(dim=1).values
                    pva = pa_.argmax(dim=1).long(); pvb = pb_.argmax(dim=1).long()
                ea = ma.squeeze(1)*(1-mb.squeeze(1)); eb = mb.squeeze(1)*(1-ma.squeeze(1))
                wb = ea*(cva > args.cmc_mutual_conf_thresh).float()
                wa = eb*(cvb > args.cmc_mutual_conf_thresh).float()
                l_b = (F.cross_entropy(ovB,pva,reduction='none')*wb).sum()/(wb.sum()+1e-6)
                l_a = (F.cross_entropy(ovA,pvb,reduction='none')*wa).sum()/(wa.sum()+1e-6)
                return anc + args.cmc_mutual_weight*(l_b+l_a)/2

            lca = cmc_loss(oA, oB, pt_a, cm_a, maa, mba)
            lcb = cmc_loss(oC, oD, pt_b, cm_b, mac, mbc)
            loss_cmc = (lca + lcb) / 2

            ramp = min(1.0, float(iter_num) / max(args.cmc_warmup_iter, 1))
            loss = loss_bcp + args.cmc_loss_weight * ramp * loss_cmc
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            iter_num += 1; update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss',  loss,     iter_num)
            writer.add_scalar('info/loss_bcp',    loss_bcp, iter_num)
            writer.add_scalar('info/loss_cmc',    loss_cmc, iter_num)
            writer.add_scalar('info/cmc_rampup',  ramp,     iter_num)
            writer.add_scalar('info/shared_ratio', sr,      iter_num)
            writer.add_scalar('info/conf_threshold', ct,    iter_num)
            # 注意力专用指标：平均块级熵（越高说明教师越不确定，掩码越有指导意义）
            writer.add_scalar('cmc/mean_block_entropy', (ent_a+ent_b)/2, iter_num)
            logging.info('iter %d: loss:%f bcp:%f cmc:%f entropy:%.4f' %
                         (iter_num, loss.item(), loss_bcp.item(), loss_cmc.item(),
                          (ent_a+ent_b)/2))

            if iter_num % 20 == 0:
                writer.add_image('train/Un_Image', ni_u[1,0:1], iter_num)
                writer.add_image('train/Un_Pred',
                    torch.argmax(torch.softmax(ou,dim=1),dim=1,keepdim=True)[1,...]*50, iter_num)
                writer.add_image('cmc/Original',    ua[0,0:1],  iter_num)
                writer.add_image('cmc/ViewA_HighEnt', ua_vA[0,0:1], iter_num)
                writer.add_image('cmc/ViewB_LowEnt',  ua_vB[0,0:1], iter_num)
                writer.add_image('cmc/EntropyMap', # 可视化熵图（高亮不确定区域）
                    F.avg_pool2d(
                        -(F.softmax(pre_a[:1],dim=1)*torch.log(F.softmax(pre_a[:1],dim=1)+1e-8)
                         ).sum(dim=1,keepdim=True),
                        kernel_size=1)[0], iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ml = sum(np.array(val_2d.test_single_volume(sb2["image"], sb2["label"],
                    model, classes=num_classes)) for _, sb2 in enumerate(valloader)) / len(db_val)
                for ci in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(ci+1), ml[ci,0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(ci+1), ml[ci,1], iter_num)
                perf = np.mean(ml, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', perf, iter_num)
                if perf > best_perf:
                    best_perf = perf
                    torch.save(model.state_dict(), os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_perf,4))))
                    torch.save(model.state_dict(), os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model)))
                logging.info('iter %d: mean_dice:%f' % (iter_num, perf)); model.train()
            if iter_num >= max_iterations: break
        if iter_num >= max_iterations: iterator.close(); break
    writer.close()

if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False; cudnn.deterministic = True
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    pre_p  = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_p = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for p in [pre_p, self_p]:
        if not os.path.exists(p): os.makedirs(p)
    shutil.copy(__file__, self_p)
    logging.basicConfig(filename=pre_p+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)); pre_train(args, pre_p)
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(filename=self_p+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)); self_train(args, pre_p, self_p)
