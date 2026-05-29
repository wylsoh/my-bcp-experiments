"""
BCP + CMC v1-Freq：频域互补视图

核心改动：generate_frequency_complementary_views
  视图A = 高斯低通滤波(img)        → 低频成分：解剖结构、心腔形状
  视图B = img - 视图A（归一化）    → 高频成分：边界、纹理细节
  两视图在频率空间上互补，分别携带不同层次的语义信息

新增参数：
  --freq_kernel_size  高斯核大小（奇数，默认21）
  --freq_sigma        高斯标准差（默认7.0）
                      kernel/sigma 越大 → 截止频率越低 → 视图A越模糊 → 视图B细节越多
                      对256×256医学图像推荐：kernel_size=21, sigma=7.0

实现方式：
  使用可分离高斯卷积（F.conv2d 先横向再纵向），避免 FFT API 版本依赖
  高频视图做 min-max 归一化到 [0,1] 防止负值输入模型

注意：此方案不使用网格掩码，两视图是连续空间分布的，
与网格类方案（v1/classbal/attention）有本质区别。
无 cmc_patch_size / cmc_init_shared 参数。
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
parser.add_argument('--exp', type=str, default='BCP_CMC_v1_freq')
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
parser.add_argument('--cmc_warmup_iter',        type=int,   default=5000)
parser.add_argument('--cmc_loss_weight',        type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',      type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh', type=float, default=0.75)
parser.add_argument('--conf_thresh_init',       type=float, default=0.90)
parser.add_argument('--conf_thresh_final',      type=float, default=0.70)
# ---------- 频域专用参数 ----------
parser.add_argument('--freq_kernel_size', type=int,   default=21,
                    help='高斯低通滤波核大小（必须为奇数）。'
                         '越大截止频率越低，视图A越模糊，视图B高频细节越多。'
                         '对 256×256 推荐范围：11~31')
parser.add_argument('--freq_sigma',       type=float, default=7.0,
                    help='高斯标准差。与 kernel_size 共同控制截止频率。'
                         '推荐设为 kernel_size/3')
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
    for i in range(segmentation.shape[0]):
        class_list = []
        for c in range(1, 4):
            tp = torch.zeros_like(segmentation[i]); tp[segmentation[i]==c] = 1
            tp = tp.detach().cpu().numpy(); labs = label(tp)
            class_list.append((labs == np.argmax(np.bincount(labs.flat)[1:])+1)*c
                               if labs.max() != 0 else tp)
        batch_list.append(class_list[0]+class_list[1]+class_list[2])
    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1); _, probs = torch.max(probs, dim=1)
    return get_ACDC_2DLargestCC(probs) if nms == 1 else probs

def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    ms = model.state_dict(); es = ema_model.state_dict()
    ema_model.load_state_dict({k: alpha*es[k]+(1-alpha)*ms[k] for k in ms})

def generate_mask(img):
    bs,_,H,W = img.shape
    lm = torch.ones(bs,H,W).cuda(); m = torch.ones(H,W).cuda()
    px,py = int(H*2/3), int(W*2/3)
    w,h = np.random.randint(0,H-px), np.random.randint(0,W-py)
    m[w:w+px,h:h+py]=0; lm[:,w:w+px,h:h+py]=0
    return m.long(), lm.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    os_ = F.softmax(output, dim=1); iw,pw = (u_weight,l_weight) if unlab else (l_weight,u_weight)
    pm = 1-mask
    ld = dice_loss(os_,img_l.unsqueeze(1),mask.unsqueeze(1))*iw
    ld += dice_loss(os_,patch_l.unsqueeze(1),pm.unsqueeze(1))*pw
    lc = iw*(CE(output,img_l)*mask).sum()/(mask.sum()+1e-16)
    lc += pw*(CE(output,patch_l)*pm).sum()/(pm.sum()+1e-16)
    return ld, lc

def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        r={"1":32,"3":68,"7":136,"14":256,"21":396,"28":512,"35":664,"70":1312}
    elif "Prostate" in dataset:
        r={"2":27,"4":53,"8":120,"12":179,"16":256,"21":312,"42":623}
    else: print("Error"); return
    return r[str(patiens_num)]

def get_adaptive_threshold(cur, mx, init=0.90, final=0.70):
    return init+(final-init)*min(1.0,float(cur)/float(mx))

# ================================================================
# 核心改动：频域互补视图生成
# ================================================================
def _build_gaussian_kernel(kernel_size, sigma, device):
    """
    构建可分离高斯卷积核

    使用 1D 高斯核的外积得到 2D 核，然后通过两次 1D 卷积实现，
    等价于 2D 高斯卷积但速度更快。

    Args:
        kernel_size : 核大小（奇数）
        sigma       : 标准差
        device      : torch device

    Returns:
        kernel_h : [C, 1, 1, ks]  水平方向卷积核
        kernel_v : [C, 1, ks, 1]  垂直方向卷积核
        （C=1，使用 groups=C 实现各通道独立卷积）
    """
    ks = kernel_size
    x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g = g / g.sum()
    kernel_h = g.view(1, 1, 1, ks)   # [1, 1, 1, ks]
    kernel_v = g.view(1, 1, ks, 1)   # [1, 1, ks, 1]
    return kernel_h, kernel_v

def generate_frequency_complementary_views(img, kernel_size=21, sigma=7.0):
    """
    频域互补视图生成（高斯低通滤波实现）

    原理：
      高斯低通滤波提取低频成分（结构/形状）→ 视图A
      原图减去低频 = 高频残差（边界/纹理）→ 视图B（min-max 归一化到 [0,1]）

    与网格类掩码的区别：
      网格掩码：空间上离散分块，各块像素值为原始值或0
      频域分解：全局平滑操作，每个像素都保留，只是频率成分不同
      → 模型接收到的输入统计特性更接近真实图像（无大量零值块）

    高斯参数的选择影响：
      kernel_size=11, sigma=3.7  → 高截止频率，视图A保留较多细节，视图B差异较小
      kernel_size=21, sigma=7.0  → 中截止频率（推荐，均衡结构与细节）
      kernel_size=31, sigma=10.3 → 低截止频率，视图A极度模糊，视图B差异最大

    Args:
        img         : [B, C, H, W]，像素值通常在 [-1,1] 或 [0,1]
        kernel_size : 高斯核大小（奇数）
        sigma       : 高斯标准差

    Returns:
        view_low  : [B, C, H, W]  低频视图（视图A），值域与输入相同
        view_high : [B, C, H, W]  高频视图（视图B），归一化到 [0,1]
        high_energy : float       高频分量的平均能量（监控频域分解质量）
    """
    B, C, H, W = img.shape
    pad = kernel_size // 2

    kernel_h, kernel_v = _build_gaussian_kernel(kernel_size, sigma, img.device)
    # 扩展到 C 通道（各通道独立卷积）
    kh = kernel_h.expand(C, 1, 1, kernel_size)   # [C, 1, 1, ks]
    kv = kernel_v.expand(C, 1, kernel_size, 1)   # [C, 1, ks, 1]

    # 可分离高斯滤波：先横向再纵向
    view_low = F.conv2d(img, kh, padding=(0, pad), groups=C)   # 横向
    view_low = F.conv2d(view_low, kv, padding=(pad, 0), groups=C)  # 纵向

    # 高频残差 = 原图 - 低频
    view_high_raw = img - view_low    # 可能含负值

    # 逐样本逐通道归一化到 [0,1]（防止负值干扰模型）
    vmin = view_high_raw.view(B, C, -1).min(dim=2).values.view(B, C, 1, 1)
    vmax = view_high_raw.view(B, C, -1).max(dim=2).values.view(B, C, 1, 1)
    view_high = (view_high_raw - vmin) / (vmax - vmin + 1e-8)

    # 高频能量（监控频域分解是否有效；若接近0说明图像很平滑，高频信息少）
    high_energy = (view_high_raw ** 2).mean().item()

    return view_low, view_high, high_energy

# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr, num_classes = args.base_lr, args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lbs = int(args.labeled_bs/2)
    model = BCP_net(in_chns=1, class_num=num_classes)
    def worker_init_fn(wid): random.seed(args.seed+wid)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    ls = patients_to_slices(args.root_path, args.labelnum)
    print("Total:{}, labeled:{}".format(len(db_train), ls))
    bs = TwoStreamBatchSampler(list(range(ls)), list(range(ls,len(db_train))),
                               args.batch_size, args.batch_size-args.labeled_bs)
    tl = DataLoader(db_train,batch_sampler=bs,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    vl = DataLoader(db_val,batch_size=1,shuffle=False,num_workers=1)
    opt = optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("Start pre_training"); model.train()
    iter_num=0; best_perf=0.0
    it = tqdm(range(max_iterations//len(tl)+1), ncols=70)
    for _ in it:
        for _, sb in enumerate(tl):
            vb,lb = sb['image'].cuda(), sb['label'].cuda()
            ia,ib = vb[:lbs], vb[lbs:args.labeled_bs]
            la,lb2= lb[:lbs], lb[lbs:args.labeled_bs]
            im,lm = generate_mask(ia)
            gt = la*im+lb2*(1-im); ni=ia*im+ib*(1-im); out=model(ni)
            ld,lc=mix_loss(out,la,lb2,lm,u_weight=1.0,unlab=True)
            loss=(ld+lc)/2
            opt.zero_grad(); loss.backward(); opt.step(); iter_num+=1
            writer.add_scalar('info/total_loss',loss,iter_num)
            logging.info('iter %d: loss:%f'%(iter_num,loss))
            if iter_num%20==0:
                writer.add_image('pre_train/Mixed_Image',ni[1,0:1],iter_num)
            if iter_num>0 and iter_num%200==0:
                model.eval()
                ml=sum(np.array(val_2d.test_single_volume(s["image"],s["label"],model,classes=num_classes))
                       for _,s in enumerate(vl))/len(db_val)
                perf=np.mean(ml,axis=0)[0]; writer.add_scalar('info/val_mean_dice',perf,iter_num)
                if perf>best_perf:
                    best_perf=perf
                    save_net_opt(model,opt,os.path.join(snapshot_path,'iter_{}_dice_{}.pth'.format(iter_num,round(best_perf,4))))
                    save_net_opt(model,opt,os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model)))
                logging.info('iter %d: mean_dice:%f'%(iter_num,perf)); model.train()
            if iter_num>=max_iterations: break
        if iter_num>=max_iterations: it.close(); break
    writer.close()

# ================================================================
# Self-train
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr,num_classes = args.base_lr,args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lbs=int(args.labeled_bs/2); ubs=int((args.batch_size-args.labeled_bs)/2)
    model     = BCP_net(in_chns=1,class_num=num_classes)
    ema_model = BCP_net(in_chns=1,class_num=num_classes,ema=True)
    def worker_init_fn(wid): random.seed(args.seed+wid)
    db_train = BaseDataSets(base_dir=args.root_path,split="train",num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path,split="val")
    ls = patients_to_slices(args.root_path,args.labelnum)
    bs = TwoStreamBatchSampler(list(range(ls)),list(range(ls,len(db_train))),
                               args.batch_size,args.batch_size-args.labeled_bs)
    tl = DataLoader(db_train,batch_sampler=bs,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    vl = DataLoader(db_val,batch_size=1,shuffle=False,num_workers=1)
    opt = optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)
    load_net(ema_model,os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model)))
    load_net_opt(model,opt,os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model)))
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("Start self_training (BCP + CMC Frequency Complementary)")
    logging.info("Gaussian kernel_size={}, sigma={}".format(args.freq_kernel_size,args.freq_sigma))
    model.train(); ema_model.train()
    iter_num=0; best_perf=0.0
    it = tqdm(range(max_iterations//len(tl)+1),ncols=70)

    for _ in it:
        for _, sb in enumerate(tl):
            vb,lb = sb['image'].cuda(), sb['label'].cuda()
            ia=vb[:lbs]; ib=vb[lbs:args.labeled_bs]
            ua=vb[args.labeled_bs:args.labeled_bs+ubs]; ub=vb[args.labeled_bs+ubs:]
            ula=lb[args.labeled_bs:args.labeled_bs+ubs]; ulb=lb[args.labeled_bs+ubs:]
            la=lb[:lbs]; lb2=lb[lbs:args.labeled_bs]

            # ---- BCP（与原始完全一致）----
            with torch.no_grad():
                pre_a=ema_model(ua); pre_b=ema_model(ub)
                plab_a=get_ACDC_masks(pre_a,nms=1); plab_b=get_ACDC_masks(pre_b,nms=1)
                im,lm=generate_mask(ia)
                unl_lbl=ula*im+la*(1-im); l_lbl=lb2*im+ulb*(1-im)
            cw=get_current_consistency_weight(iter_num//150)
            ni_u=ua*im+ia*(1-im); ni_l=ib*im+ub*(1-im)
            ou=model(ni_u); ol=model(ni_l)
            ud,uc=mix_loss(ou,plab_a,la, lm,u_weight=args.u_weight,unlab=True)
            ld,lc=mix_loss(ol,lb2,plab_b,lm,u_weight=args.u_weight)
            loss_bcp=((ud+ld)+(uc+lc))/2

            # ---- CMC：频域互补视图 ----
            ct = get_adaptive_threshold(iter_num,max_iterations,
                                        args.conf_thresh_init,args.conf_thresh_final)

            # 生成频域互补视图（对 uimg_a 和 uimg_b 分别处理）
            ua_low, ua_high, eng_a = generate_frequency_complementary_views(
                ua, args.freq_kernel_size, args.freq_sigma)
            ub_low, ub_high, eng_b = generate_frequency_complementary_views(
                ub, args.freq_kernel_size, args.freq_sigma)

            # 4次学生 forward（低频+高频 各两批）
            # 通过 concat 减少为 2 次
            out_a_all = model(torch.cat([ua_low, ua_high], dim=0))
            out_b_all = model(torch.cat([ub_low, ub_high], dim=0))
            out_a_low, out_a_high = out_a_all[:ubs], out_a_all[ubs:]
            out_b_low, out_b_high = out_b_all[:ubs], out_b_all[ubs:]

            with torch.no_grad():
                cm_a=(F.softmax(pre_a,dim=1).max(dim=1).values>ct).float()
                cm_b=(F.softmax(pre_b,dim=1).max(dim=1).values>ct).float()
                pt_a=plab_a.long(); pt_b=plab_b.long()

            def cmc_freq_loss(out_low, out_high, pt, cm):
                """
                频域互补一致性损失

                L_anchor : 低频视图和高频视图均对齐教师硬标签（置信度加权）
                L_mutual : 低频视图和高频视图的预测相互对齐（软一致性）
                           两者应对同一图像的分割结果达成一致

                注意：与网格掩码不同，频域视图没有"独有区域"的概念，
                因此互教损失改为全局软一致性（KL 散度）
                """
                w=cm; d=w.sum()+1e-6
                # L_anchor
                la_=F.cross_entropy(out_low, pt,reduction='none')
                lh_=F.cross_entropy(out_high,pt,reduction='none')
                loss_anchor=((la_+lh_)*w).sum()/d/2.0
                # L_mutual：低频视图 ↔ 高频视图预测软一致性（双向KL加权）
                with torch.no_grad():
                    p_low  = F.softmax(out_low, dim=1)   # [B,C,H,W]
                    p_high = F.softmax(out_high,dim=1)
                # KL(p_low || p_high)：高频预测对低频视图的监督
                kl_lh=(p_low*(torch.log(p_low+1e-8)-torch.log(p_high+1e-8))).sum(dim=1)
                # KL(p_high || p_low)：低频预测对高频视图的监督
                kl_hl=(p_high*(torch.log(p_high+1e-8)-torch.log(p_low+1e-8))).sum(dim=1)
                # 教师置信度加权
                loss_mutual=((kl_lh+kl_hl)*w).sum()/d/2.0
                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            lca=cmc_freq_loss(out_a_low,out_a_high,pt_a,cm_a)
            lcb=cmc_freq_loss(out_b_low,out_b_high,pt_b,cm_b)
            loss_cmc=(lca+lcb)/2

            ramp=min(1.0,float(iter_num)/max(args.cmc_warmup_iter,1))
            loss=loss_bcp+args.cmc_loss_weight*ramp*loss_cmc
            opt.zero_grad(); loss.backward(); opt.step()
            iter_num+=1; update_model_ema(model,ema_model,0.99)

            writer.add_scalar('info/total_loss',  loss,     iter_num)
            writer.add_scalar('info/loss_bcp',    loss_bcp, iter_num)
            writer.add_scalar('info/loss_cmc',    loss_cmc, iter_num)
            writer.add_scalar('info/cmc_rampup',  ramp,     iter_num)
            writer.add_scalar('info/conf_threshold',ct,     iter_num)
            # 频域专用：高频分量能量（越大说明图像细节越丰富，频域分解越有意义）
            writer.add_scalar('cmc/high_freq_energy',(eng_a+eng_b)/2,iter_num)
            logging.info('iter %d: loss:%f bcp:%f cmc:%f hf_energy:%.6f'%
                         (iter_num,loss.item(),loss_bcp.item(),loss_cmc.item(),(eng_a+eng_b)/2))

            if iter_num%20==0:
                writer.add_image('train/Un_Image',ni_u[1,0:1],iter_num)
                # 频域视图对比可视化
                writer.add_image('cmc/Original',   ua[0,0:1],      iter_num)
                writer.add_image('cmc/ViewA_Low',  ua_low[0,0:1],  iter_num)  # 低频（结构）
                writer.add_image('cmc/ViewB_High', ua_high[0,0:1], iter_num)  # 高频（边缘）
                pred_low =torch.argmax(torch.softmax(out_a_low, dim=1),dim=1,keepdim=True)
                pred_high=torch.argmax(torch.softmax(out_a_high,dim=1),dim=1,keepdim=True)
                writer.add_image('cmc/PredLow',  pred_low[0].float()*50,  iter_num)
                writer.add_image('cmc/PredHigh', pred_high[0].float()*50, iter_num)

            if iter_num>0 and iter_num%200==0:
                model.eval()
                ml=sum(np.array(val_2d.test_single_volume(s["image"],s["label"],model,
                    classes=num_classes)) for _,s in enumerate(vl))/len(db_val)
                for ci in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(ci+1),ml[ci,0],iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(ci+1),ml[ci,1],iter_num)
                perf=np.mean(ml,axis=0)[0]; writer.add_scalar('info/val_mean_dice',perf,iter_num)
                if perf>best_perf:
                    best_perf=perf
                    torch.save(model.state_dict(),os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num,round(best_perf,4))))
                    torch.save(model.state_dict(),os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model)))
                logging.info('iter %d: mean_dice:%f'%(iter_num,perf)); model.train()
            if iter_num>=max_iterations: break
        if iter_num>=max_iterations: it.close(); break
    writer.close()

if __name__=="__main__":
    if args.deterministic:
        cudnn.benchmark=False; cudnn.deterministic=True
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    pre_p ="./model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp,args.labelnum)
    self_p="./model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp,args.labelnum)
    for p in [pre_p,self_p]:
        if not os.path.exists(p): os.makedirs(p)
    shutil.copy(__file__,self_p)
    logging.basicConfig(filename=pre_p+"/log.txt",level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)); pre_train(args,pre_p)
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(filename=self_p+"/log.txt",level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)); self_train(args,pre_p,self_p)
