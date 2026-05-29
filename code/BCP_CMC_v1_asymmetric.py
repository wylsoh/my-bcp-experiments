"""
BCP + CMC v1-Asymmetric：不对称掩码比例

核心改动：generate_asymmetric_masks + 单向互教（A→B）

设计思想：
  对称互补（v1）：视图A和B各看50%，两者信息量相同，双向互教
  不对称互补（本方案）：
    视图A（rich）← 70% 的块，信息量更多，预测质量更高
    视图B（sparse）← 30% 的块，信息量少，主要依赖从A学习
    互教方向：仅 A→B（单向），A的高质量预测监督B的盲区
              B→A 的信号因B信息量不足而噪声较大，故去掉

  这是介于 MAE（1个视图 vs 教师完整图）和 CMC 对称方案之间的设计：
    MAE   : 学生 ← 遮挡图（~35%），教师 ← 完整图
    CMC v1: 视图A ← 50%，视图B ← 50%（对称）
    本方案 : 视图A ← 70%，视图B ← 30%（不对称，A向B单向教）

新增参数：
  --cmc_view_a_ratio  视图A的块占比（默认0.7，即70%）
  --cmc_teach_dir     互教方向：'A2B'（仅A教B）或 'both'（双向，但A权重更大）
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
parser.add_argument('--exp', type=str, default='BCP_CMC_v1_asymmetric')
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
parser.add_argument('--cmc_loss_weight',        type=float, default=1.0)
parser.add_argument('--cmc_mutual_weight',      type=float, default=0.5)
parser.add_argument('--cmc_mutual_conf_thresh', type=float, default=0.75)
parser.add_argument('--conf_thresh_init',       type=float, default=0.90)
parser.add_argument('--conf_thresh_final',      type=float, default=0.70)
# ---------- 不对称专用参数 ----------
parser.add_argument('--cmc_view_a_ratio', type=float, default=0.7,
                    help='视图A的块占比（0~1）。'
                         '0.5 = 对称（退化为 v1），0.7 = 推荐，0.9 = 接近 MAE。'
                         '视图B占比 = 1 - cmc_view_a_ratio（两视图互补，无重叠）。')
parser.add_argument('--cmc_teach_dir', type=str, default='A2B',
                    choices=['A2B', 'both'],
                    help='互教方向。'
                         'A2B：仅A（rich视图）教B（sparse视图），推荐默认。'
                         'both：双向，但A→B损失权重为1.0，B→A损失权重为0.3'
                         '（B信息少，其预测质量低，需降权）。')
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
            tp = torch.zeros_like(segmentation[i]); tp[segmentation[i]==c]=1
            tp = tp.detach().cpu().numpy(); labs = label(tp)
            class_list.append((labs==np.argmax(np.bincount(labs.flat)[1:])+1)*c
                               if labs.max()!=0 else tp)
        batch_list.append(class_list[0]+class_list[1]+class_list[2])
    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1); _, probs = torch.max(probs, dim=1)
    return get_ACDC_2DLargestCC(probs) if nms==1 else probs

def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    ms=model.state_dict(); es=ema_model.state_dict()
    ema_model.load_state_dict({k:alpha*es[k]+(1-alpha)*ms[k] for k in ms})

def generate_mask(img):
    bs,_,H,W=img.shape; lm=torch.ones(bs,H,W).cuda(); m=torch.ones(H,W).cuda()
    px,py=int(H*2/3),int(W*2/3); w,h=np.random.randint(0,H-px),np.random.randint(0,W-py)
    m[w:w+px,h:h+py]=0; lm[:,w:w+px,h:h+py]=0; return m.long(),lm.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE=nn.CrossEntropyLoss(reduction='none')
    img_l,patch_l=img_l.type(torch.int64),patch_l.type(torch.int64)
    os_=F.softmax(output,dim=1); iw,pw=(u_weight,l_weight) if unlab else (l_weight,u_weight)
    pm=1-mask
    ld=dice_loss(os_,img_l.unsqueeze(1),mask.unsqueeze(1))*iw
    ld+=dice_loss(os_,patch_l.unsqueeze(1),pm.unsqueeze(1))*pw
    lc=iw*(CE(output,img_l)*mask).sum()/(mask.sum()+1e-16)
    lc+=pw*(CE(output,patch_l)*pm).sum()/(pm.sum()+1e-16)
    return ld,lc

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
# 核心改动：不对称互补掩码生成
# ================================================================
def generate_asymmetric_masks(img, view_a_ratio=0.7, cmc_patch_size=16):
    """
    不对称互补掩码生成器

    P(block → 视图A) = view_a_ratio（默认0.7，视图A是"rich"视图）
    P(block → 视图B) = 1 - view_a_ratio（视图B是"sparse"视图）
    两视图严格互补：mask_a + mask_b = 1（每个块仅属于一个视图）

    与 v1（对称 50/50）的区别：
      v1    : P(A) = P(B) = 0.5，信息量相等，双向互教公平
      本方案 : P(A) = 0.7, P(B) = 0.3，A信息量更多，适合单向A→B互教

    Args:
        img          : [B, C, H, W]
        view_a_ratio : 视图A的块占比（0.5=对称，0.7=推荐，0.9=接近MAE）
        cmc_patch_size: 网格块大小

    Returns:
        mask_a : [B, 1, H, W]  视图A可见掩码（约占 view_a_ratio 的块）
        mask_b : [B, 1, H, W]  视图B可见掩码（约占 1-view_a_ratio 的块）
        actual_ratio_a : float 本批次视图A实际平均块占比（监控分配是否符合预期）
    """
    B, C, H, W = img.shape
    p = cmc_patch_size
    n = H // p
    masks_a, masks_b, ratios = [], [], []

    for _ in range(B):
        # P(block→A) = view_a_ratio，P(block→B) = 1 - view_a_ratio
        # rand < view_a_ratio → A，否则 → B
        base = (torch.rand(n, n) < view_a_ratio).float()  # 1=A, 0=B
        actual_ratio = base.mean().item()
        ratios.append(actual_ratio)

        pa = base                           # A可见：base==1
        pb = 1.0 - base                    # B可见：base==0，严格互补

        pa = F.interpolate(pa.view(1,1,n,n), size=(H,W), mode='nearest').squeeze(0)
        pb = F.interpolate(pb.view(1,1,n,n), size=(H,W), mode='nearest').squeeze(0)
        masks_a.append(pa); masks_b.append(pb)

    return (torch.stack(masks_a).to(img.device),
            torch.stack(masks_b).to(img.device),
            float(np.mean(ratios)))

# ================================================================
# Pre-train（与原始 BCP 完全一致）
# ================================================================
def pre_train(args, snapshot_path):
    base_lr,num_classes=args.base_lr,args.num_classes
    max_iterations=args.pre_iterations; os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    lbs=int(args.labeled_bs/2); model=BCP_net(in_chns=1,class_num=num_classes)
    def worker_init_fn(wid): random.seed(args.seed+wid)
    db_train=BaseDataSets(base_dir=args.root_path,split="train",num=None,
                          transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val=BaseDataSets(base_dir=args.root_path,split="val")
    ls=patients_to_slices(args.root_path,args.labelnum)
    print("Total:{}, labeled:{}".format(len(db_train),ls))
    bs=TwoStreamBatchSampler(list(range(ls)),list(range(ls,len(db_train))),
                             args.batch_size,args.batch_size-args.labeled_bs)
    tl=DataLoader(db_train,batch_sampler=bs,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    vl=DataLoader(db_val,batch_size=1,shuffle=False,num_workers=1)
    opt=optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)
    writer=SummaryWriter(snapshot_path+'/log'); logging.info("Start pre_training")
    model.train(); iter_num=0; best_perf=0.0
    it=tqdm(range(max_iterations//len(tl)+1),ncols=70)
    for _ in it:
        for _,sb in enumerate(tl):
            vb,lb=sb['image'].cuda(),sb['label'].cuda()
            ia,ib=vb[:lbs],vb[lbs:args.labeled_bs]
            la,lb2=lb[:lbs],lb[lbs:args.labeled_bs]
            im,lm=generate_mask(ia); gt=la*im+lb2*(1-im); ni=ia*im+ib*(1-im); out=model(ni)
            ld,lc=mix_loss(out,la,lb2,lm,u_weight=1.0,unlab=True); loss=(ld+lc)/2
            opt.zero_grad(); loss.backward(); opt.step(); iter_num+=1
            writer.add_scalar('info/total_loss',loss,iter_num)
            logging.info('iter %d: loss:%f'%(iter_num,loss))
            if iter_num%20==0:
                writer.add_image('pre_train/Mixed_Image',ni[1,0:1],iter_num)
            if iter_num>0 and iter_num%200==0:
                model.eval()
                ml=sum(np.array(val_2d.test_single_volume(s["image"],s["label"],model,
                    classes=num_classes)) for _,s in enumerate(vl))/len(db_val)
                perf=np.mean(ml,axis=0)[0]; writer.add_scalar('info/val_mean_dice',perf,iter_num)
                if perf>best_perf:
                    best_perf=perf
                    save_net_opt(model,opt,os.path.join(snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num,round(best_perf,4))))
                    save_net_opt(model,opt,os.path.join(snapshot_path,
                        '{}_best_model.pth'.format(args.model)))
                logging.info('iter %d: mean_dice:%f'%(iter_num,perf)); model.train()
            if iter_num>=max_iterations: break
        if iter_num>=max_iterations: it.close(); break
    writer.close()

# ================================================================
# Self-train
# ================================================================
def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr,num_classes=args.base_lr,args.num_classes
    max_iterations=args.max_iterations; os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    lbs=int(args.labeled_bs/2); ubs=int((args.batch_size-args.labeled_bs)/2)
    model=BCP_net(in_chns=1,class_num=num_classes)
    ema_model=BCP_net(in_chns=1,class_num=num_classes,ema=True)
    def worker_init_fn(wid): random.seed(args.seed+wid)
    db_train=BaseDataSets(base_dir=args.root_path,split="train",num=None,
                          transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val=BaseDataSets(base_dir=args.root_path,split="val")
    ls=patients_to_slices(args.root_path,args.labelnum)
    bs=TwoStreamBatchSampler(list(range(ls)),list(range(ls,len(db_train))),
                             args.batch_size,args.batch_size-args.labeled_bs)
    tl=DataLoader(db_train,batch_sampler=bs,num_workers=4,pin_memory=True,worker_init_fn=worker_init_fn)
    vl=DataLoader(db_val,batch_size=1,shuffle=False,num_workers=1)
    opt=optim.SGD(model.parameters(),lr=base_lr,momentum=0.9,weight_decay=0.0001)
    load_net(ema_model,os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model)))
    load_net_opt(model,opt,os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model)))
    writer=SummaryWriter(snapshot_path+'/log')
    logging.info("Start self_training (BCP + CMC Asymmetric)")
    logging.info("view_a_ratio={}, teach_dir={}".format(args.cmc_view_a_ratio,args.cmc_teach_dir))
    model.train(); ema_model.train(); iter_num=0; best_perf=0.0
    it=tqdm(range(max_iterations//len(tl)+1),ncols=70)

    for _ in it:
        for _,sb in enumerate(tl):
            vb,lb=sb['image'].cuda(),sb['label'].cuda()
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

            # ---- CMC：不对称掩码 ----
            ct=get_adaptive_threshold(iter_num,max_iterations,
                                      args.conf_thresh_init,args.conf_thresh_final)

            # 生成不对称互补掩码（A:view_a_ratio，B:1-view_a_ratio）
            maa,mba,ra = generate_asymmetric_masks(ua, args.cmc_view_a_ratio, args.cmc_patch_size)
            mac,mbc,rb = generate_asymmetric_masks(ub, args.cmc_view_a_ratio, args.cmc_patch_size)

            # 视图A（rich）：看到更多区域
            # 视图B（sparse）：看到更少区域，主要从A学习
            ua_vA=ua*maa; ua_vB=ua*mba
            ub_vC=ub*mac; ub_vD=ub*mbc

            oAB=model(torch.cat([ua_vA,ua_vB],dim=0))
            oCD=model(torch.cat([ub_vC,ub_vD],dim=0))
            oA,oB=oAB[:ubs],oAB[ubs:]
            oC,oD=oCD[:ubs],oCD[ubs:]

            with torch.no_grad():
                cm_a=(F.softmax(pre_a,dim=1).max(dim=1).values>ct).float()
                cm_b=(F.softmax(pre_b,dim=1).max(dim=1).values>ct).float()
                pt_a=plab_a.long(); pt_b=plab_b.long()

            def cmc_asym_loss(out_rich, out_sparse, pt, cm, m_rich, m_sparse, teach_dir):
                """
                不对称互补一致性损失

                L_anchor : rich 和 sparse 两视图均对齐教师硬标签
                L_A2B   : rich视图(A)的高置信预测 → 监督sparse视图(B)的盲区
                           B盲区 = m_rich 独有区域（A看得见B看不见）
                L_B2A   : （仅 teach_dir='both' 时）sparse视图(B)→监督rich视图(A)的盲区
                           权重降为0.3，因B信息量少，预测质量较低

                不对称设计逻辑：
                  A看70%，B看30%。A的盲区（30%）= B的独有区域。
                  A从B学这30%（B独有），但B预测质量低，故降权。
                  B的盲区（70%）= A的独有区域。
                  B从A学这70%（A独有），A预测质量高，全权重。
                  → A2B方向信号更可靠，优先使用。
                """
                w=cm; d=w.sum()+1e-6
                # L_anchor：两视图均对齐教师
                la_=F.cross_entropy(out_rich,  pt,reduction='none')
                lb_=F.cross_entropy(out_sparse,pt,reduction='none')
                loss_anchor=((la_+lb_)*w).sum()/d/2.0

                # B的盲区：rich独有（A看到，B看不到）
                excl_rich  = m_rich.squeeze(1)*(1-m_sparse.squeeze(1))   # A独有 [B,H,W]
                # A的盲区：sparse独有（B看到，A看不到）
                excl_sparse= m_sparse.squeeze(1)*(1-m_rich.squeeze(1))   # B独有

                with torch.no_grad():
                    prob_r=F.softmax(out_rich,dim=1);   conf_r=prob_r.max(dim=1).values
                    prob_s=F.softmax(out_sparse,dim=1); conf_s=prob_s.max(dim=1).values
                    plab_r=prob_r.argmax(dim=1).long(); plab_s=prob_s.argmax(dim=1).long()

                # A→B：A的高置信预测监督B在A独有区域（B盲区）
                w_b=excl_rich*(conf_r>args.cmc_mutual_conf_thresh).float()
                l_b_from_a=(F.cross_entropy(out_sparse,plab_r,reduction='none')*w_b
                           ).sum()/(w_b.sum()+1e-6)

                if teach_dir == 'A2B':
                    loss_mutual = l_b_from_a
                else:  # 'both'：双向，但B→A降权0.3
                    w_a=excl_sparse*(conf_s>args.cmc_mutual_conf_thresh).float()
                    l_a_from_b=(F.cross_entropy(out_rich,plab_s,reduction='none')*w_a
                               ).sum()/(w_a.sum()+1e-6)
                    loss_mutual = l_b_from_a + 0.3 * l_a_from_b

                return loss_anchor + args.cmc_mutual_weight * loss_mutual

            lca=cmc_asym_loss(oA,oB,pt_a,cm_a,maa,mba,args.cmc_teach_dir)
            lcb=cmc_asym_loss(oC,oD,pt_b,cm_b,mac,mbc,args.cmc_teach_dir)
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
            # 不对称专用：视图A实际块占比（期望收敛到 view_a_ratio）
            writer.add_scalar('cmc/actual_ratio_a',(ra+rb)/2,iter_num)
            # 视图A和视图B的信息量比（rich:sparse = ratio_a:(1-ratio_a)）
            writer.add_scalar('cmc/info_ratio_AB',
                              (ra+rb)/2 / (1-(ra+rb)/2+1e-6), iter_num)
            logging.info('iter %d: loss:%f bcp:%f cmc:%f ratio_a:%.2f dir:%s'%
                         (iter_num,loss.item(),loss_bcp.item(),loss_cmc.item(),
                          (ra+rb)/2,args.cmc_teach_dir))

            if iter_num%20==0:
                writer.add_image('train/Un_Image',ni_u[1,0:1],iter_num)
                writer.add_image('cmc/Original',    ua[0,0:1],  iter_num)
                writer.add_image('cmc/ViewA_Rich',  ua_vA[0,0:1],iter_num)  # 70%，信息丰富
                writer.add_image('cmc/ViewB_Sparse',ua_vB[0,0:1],iter_num)  # 30%，信息稀疏
                writer.add_image('cmc/MaskA', maa[0],           iter_num)
                writer.add_image('cmc/MaskB', mba[0],           iter_num)
                pA=torch.argmax(torch.softmax(oA,dim=1),dim=1,keepdim=True)
                pB=torch.argmax(torch.softmax(oB,dim=1),dim=1,keepdim=True)
                writer.add_image('cmc/PredA_Rich',  pA[0].float()*50, iter_num)
                writer.add_image('cmc/PredB_Sparse',pB[0].float()*50, iter_num)

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
