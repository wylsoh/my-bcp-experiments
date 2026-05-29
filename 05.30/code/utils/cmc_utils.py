"""
utils/cmc_utils.py

互补掩码一致性 (CMC, Complementary Mask Consistency) 工具模块

核心组件：
  - CMCGridMaskGenerator : 生成互补网格掩码对，支持渐进式共享比例热身
  - cmc_consistency_loss : 三重损失（教师锚点 + 双向互补互教）

设计原则：
  生成掩码 M_a 和 M_b，满足 M_a + M_b = 1（纯互补，每个像素归属唯一视图）
  视图A = img ⊙ M_a （仅保留 A 区域）
  视图B = img ⊙ M_b = img ⊙ (1 - M_a)（保留互补区域）

  损失构成：
  1. L_anchor_A  : pred_A vs 教师伪标签（教师置信度加权，全局像素）
  2. L_anchor_B  : pred_B vs 教师伪标签
  3. L_mutual_AB : A 的高置信预测 → 监督 B 在 A 独有区域（B 看不见）
  4. L_mutual_BA : B 的高置信预测 → 监督 A 在 B 独有区域（A 看不见）

  L_CMC = L_anchor + λ_mutual * L_mutual

渐进式训练策略：
  训练初期: 两视图有较多重叠（shared_ratio 较高），任务宽松，梯度稳定
  训练中后期: 逐步收敛到纯互补（shared_ratio→0），任务难度最高
"""

import torch
import torch.nn.functional as F


# ============================================================
# CMCGridMaskGenerator
# ============================================================
class CMCGridMaskGenerator:
    """
    互补网格掩码生成器

    将图像划分为 patch_size × patch_size 的均匀网格，随机将每个网格块
    分配给视图A或视图B。支持通过 shared_ratio 控制两视图的重叠程度。

    Attributes:
        img_size    (int)   : 图像边长（仅支持正方形图像）
        patch_size  (int)   : 每个网格块的边长（像素数），需整除 img_size
        shared_ratio (float): 同时出现在两个视图中的块比例，0=纯互补

    使用方式:
        gen = CMCGridMaskGenerator(img_size=256, patch_size=16)
        # 渐进热身阶段动态更新 shared_ratio
        gen.set_shared_ratio(CMCGridMaskGenerator.get_progressive_shared_ratio(
            current_iter, warmup_iter, init_ratio=0.4, final_ratio=0.0
        ))
        mask_a, mask_b = gen.generate(batch_size=6, device=device)
    """

    def __init__(self, img_size: int = 256, patch_size: int = 16):
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) 必须能被 patch_size ({patch_size}) 整除"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.n = img_size // patch_size  # 每行/列的块数，如 256/16=16
        self.shared_ratio = 0.0

    # ----------------------------------------------------------
    # 核心生成函数
    # ----------------------------------------------------------
    def generate(self, batch_size: int, device: torch.device):
        """
        生成一对互补掩码（每次调用均重新随机采样）

        Args:
            batch_size : 当前批次大小
            device     : 目标设备

        Returns:
            mask_a : Tensor [B, 1, H, W]，视图A可见区域掩码，值域 {0, 1}
            mask_b : Tensor [B, 1, H, W]，视图B可见区域掩码，值域 {0, 1}

        当 shared_ratio == 0 时（训练后期），mask_a + mask_b == 1（逐像素）
        当 shared_ratio > 0  时（热身阶段），部分块在两视图均可见
        """
        n = self.n
        H = W = self.img_size
        masks_a, masks_b = [], []

        for _ in range(batch_size):
            # 随机分配块：0 → 仅A，1 → 仅B（各约50%）
            base = (torch.rand(n, n) > 0.5).float()  # [n, n]

            if self.shared_ratio > 0.0:
                # 额外随机选一批块作为"共享块"（两视图都可见）
                shared = torch.rand(n, n) < self.shared_ratio
                pa = ((base == 0) | shared).float()  # A独有 + 共享
                pb = ((base == 1) | shared).float()  # B独有 + 共享
            else:
                # 严格互补：每块只属于一个视图
                pa = (base == 0).float()
                pb = (base == 1).float()

            # 上采样：[1,1,n,n] → [1,1,H,W]（最近邻，保持块形状）
            pa = F.interpolate(pa.view(1, 1, n, n),
                               size=(H, W), mode='nearest').squeeze(0)  # [1,H,W]
            pb = F.interpolate(pb.view(1, 1, n, n),
                               size=(H, W), mode='nearest').squeeze(0)

            masks_a.append(pa)
            masks_b.append(pb)

        return (torch.stack(masks_a).to(device),   # [B,1,H,W]
                torch.stack(masks_b).to(device))

    # ----------------------------------------------------------
    # 动态调整接口
    # ----------------------------------------------------------
    def set_shared_ratio(self, ratio: float):
        """动态设置共享块比例（供训练循环调用）"""
        self.shared_ratio = float(max(0.0, min(1.0, ratio)))

    @staticmethod
    def get_progressive_shared_ratio(current_iter: int,
                                     warmup_iter: int,
                                     init_ratio: float = 0.4,
                                     final_ratio: float = 0.0) -> float:
        """
        计算当前迭代步对应的共享比例（线性退火）

        训练初期 shared_ratio = init_ratio（两视图信息重叠多，梯度稳定）
        热身结束后  shared_ratio = final_ratio = 0（纯互补，最大挑战）

        Args:
            current_iter : 当前自训练迭代步（从 0 开始）
            warmup_iter  : 热身结束步数
            init_ratio   : 初始共享比例（推荐 0.3~0.5）
            final_ratio  : 最终共享比例（推荐 0.0）

        Returns:
            当前步应使用的 shared_ratio（float）
        """
        if warmup_iter <= 0 or current_iter >= warmup_iter:
            return float(final_ratio)
        progress = float(current_iter) / float(warmup_iter)
        return init_ratio + (final_ratio - init_ratio) * progress


# ============================================================
# cmc_consistency_loss
# ============================================================
def cmc_consistency_loss(pred_a: torch.Tensor,
                          pred_b: torch.Tensor,
                          teacher_plab: torch.Tensor,
                          teacher_conf_mask: torch.Tensor,
                          mask_a: torch.Tensor,
                          mask_b: torch.Tensor,
                          n_classes: int,
                          mutual_conf_thresh: float = 0.75):
    """
    互补掩码一致性损失（CMC Loss）

    三重监督：
      Part A — 教师锚点损失 (L_anchor)
        pred_A 和 pred_B 均向 EMA 教师的伪标签对齐，以教师置信度加权。
        覆盖全局所有像素（不局限于遮挡区域）。

      Part B — 互补互教损失 (L_mutual)
        L_mutual_AB：在 B 独有区域（A 看不见）中，
                     用 B 的高置信预测监督 A → 迫使 A 从局部推断全局
        L_mutual_BA：在 A 独有区域（B 看不见）中，
                     用 A 的高置信预测监督 B → 同理

        "互教"仅在对方高置信度时才施加，避免早期低质量伪标签污染。

    损失关系：
        L_CMC = L_anchor + λ_mutual * L_mutual

    Args:
        pred_a             : 学生处理视图A的输出   [B, C, H, W]，未经 softmax
        pred_b             : 学生处理视图B的输出   [B, C, H, W]，未经 softmax
        teacher_plab       : EMA 教师硬伪标签      [B, H, W]，long 类型
        teacher_conf_mask  : EMA 教师置信度权重    [B, H, W]，float [0,1]
        mask_a             : 视图A可见掩码         [B, 1, H, W]，值域 {0,1}
        mask_b             : 视图B可见掩码         [B, 1, H, W]，值域 {0,1}
        n_classes          : 分割类别数
        mutual_conf_thresh : 互教最低置信度门限（高于此才作为软教师）

    Returns:
        loss_anchor  : 教师锚点损失（标量 Tensor，含梯度）
        loss_mutual  : 互补互教损失（标量 Tensor，含梯度）
        stats        : dict，包含各区域有效像素数和平均置信度，用于 TensorBoard

    典型用法::
        loss_anchor, loss_mutual, stats = cmc_consistency_loss(
            pred_a=out_a_viewA,
            pred_b=out_a_viewB,
            teacher_plab=plab_teacher,
            teacher_conf_mask=conf_mask,
            mask_a=mask_a_ab,
            mask_b=mask_b_ab,
            n_classes=4,
            mutual_conf_thresh=args.cmc_mutual_conf_thresh
        )
        loss_cmc = loss_anchor + args.cmc_mutual_weight * loss_mutual
    """
    # ---- Step 1: 从学生输出推导置信度和硬标签（无梯度）----
    with torch.no_grad():
        prob_a = F.softmax(pred_a, dim=1)              # [B, C, H, W]
        prob_b = F.softmax(pred_b, dim=1)
        conf_a = prob_a.max(dim=1).values              # [B, H, W]  最大类别概率
        conf_b = prob_b.max(dim=1).values
        plab_a = prob_a.argmax(dim=1).long()           # [B, H, W]  硬标签
        plab_b = prob_b.argmax(dim=1).long()

    # ---- Step 2: 教师锚点损失 ----
    # 两视图的预测都要与教师伪标签对齐，教师置信度 w 作为像素级权重。
    # 覆盖全局像素：无论该像素是否被当前视图遮挡，都需要正确预测。
    teacher_plab = teacher_plab.long()   # 确保 target 为 Long 类型
    loss_a_teacher = F.cross_entropy(pred_a, teacher_plab, reduction='none')  # [B,H,W]
    loss_b_teacher = F.cross_entropy(pred_b, teacher_plab, reduction='none')

    w_teacher = teacher_conf_mask                          # [B, H, W]
    denom_t = w_teacher.sum() + 1e-6
    # 注意：损失在全局归一化（而非仅在置信区域）以避免有效像素稀少时的尺度爆炸
    loss_anchor = ((loss_a_teacher + loss_b_teacher) * w_teacher).sum() / denom_t / 2.0

    # ---- Step 3: 互补互教损失 ----
    #
    # 核心直觉：
    #   mask_a 区域（A 可见）= B 的盲区 → B 应该从 A 的预测中学习这里
    #   mask_b 区域（B 可见）= A 的盲区 → A 应该从 B 的预测中学习这里
    #
    # 在纯互补模式（shared_ratio=0）下：
    #   excl_a = mask_a （A 的所有可见区域均为 B 的独有盲区）
    #   excl_b = mask_b （同理）
    #
    # 在部分重叠时：
    #   excl_a = mask_a * (1 - mask_b)（仅 A 能看到，B 看不到）
    #   excl_b = mask_b * (1 - mask_a)
    excl_a = mask_a.squeeze(1) * (1.0 - mask_b.squeeze(1))  # [B, H, W] 仅A可见
    excl_b = mask_b.squeeze(1) * (1.0 - mask_a.squeeze(1))  # [B, H, W] 仅B可见

    # A → B：用 A 的高置信预测监督 B，仅在 A 独有区域施加
    #   B 在这些像素上完全没有输入信号，只能从 A 的教学中学习
    high_conf_a = (conf_a > mutual_conf_thresh).float()      # [B, H, W]
    w_b_learn = excl_a * high_conf_a                         # A独有 且 A高置信
    loss_b_from_a = F.cross_entropy(pred_b, plab_a.long(), reduction='none')
    denom_ba = w_b_learn.sum() + 1e-6
    loss_b_from_a = (loss_b_from_a * w_b_learn).sum() / denom_ba

    # B → A：用 B 的高置信预测监督 A，仅在 B 独有区域施加
    high_conf_b = (conf_b > mutual_conf_thresh).float()      # [B, H, W]
    w_a_learn = excl_b * high_conf_b                         # B独有 且 B高置信
    loss_a_from_b = F.cross_entropy(pred_a, plab_b.long(), reduction='none')
    denom_ab = w_a_learn.sum() + 1e-6
    loss_a_from_b = (loss_a_from_b * w_a_learn).sum() / denom_ab

    loss_mutual = (loss_b_from_a + loss_a_from_b) / 2.0

    # ---- 统计信息（仅用于日志，不参与反向传播）----
    stats = {
        'valid_teacher_px'  : denom_t.item(),
        'valid_a_learn_px'  : denom_ab.item(),   # B独有区域中B高置信像素数（A从B学）
        'valid_b_learn_px'  : denom_ba.item(),   # A独有区域中A高置信像素数（B从A学）
        'mean_conf_a'       : conf_a.mean().item(),
        'mean_conf_b'       : conf_b.mean().item(),
        'high_conf_ratio_a' : high_conf_a.mean().item(),
        'high_conf_ratio_b' : high_conf_b.mean().item(),
    }

    return loss_anchor, loss_mutual, stats


    # ============================================================
# cmc_fusion_consistency_loss（v2 核心，替换 v1 的 cmc_consistency_loss）
# ============================================================
def cmc_fusion_consistency_loss(pred_a: torch.Tensor,
                                 pred_b: torch.Tensor,
                                 teacher_logit: torch.Tensor,
                                 teacher_conf_mask: torch.Tensor,
                                 n_classes: int):
    """
    互补掩码预测融合一致性损失
 
    两部分损失：
    ┌─────────────────────────────────────────────────────────┐
    │  L_anchor                                               │
    │    每个视图独立对齐教师硬伪标签                            │
    │    CE(pred_A, argmax(teacher)) + CE(pred_B, argmax(teacher))│
    │    以教师置信度 w 加权                                    │
    ├─────────────────────────────────────────────────────────┤
    │  L_fusion                                               │
    │    两视图概率均值对齐教师软概率                            │
    │    H(teacher_soft, avg(softmax_A, softmax_B))            │
    │    = -Σ_c p_teacher(c) * log((p_A(c)+p_B(c))/2)        │
    │    以教师置信度 w 加权                                    │
    └─────────────────────────────────────────────────────────┘
 
    L_fusion 的数学动机：
      互补掩码满足 E[view] = full_image（期望意义下）
      → 理想模型应满足：avg(f(viewA), f(viewB)) ≈ f(full)
      → 用教师 f(full) 作为目标，约束两视图预测的算术平均
      → 梯度同时流向 pred_A 和 pred_B，强迫两者协调而非独立对齐
 
    L_fusion vs L_anchor 的互补关系：
      L_anchor：每个视图独立对齐教师（pointwise 约束）
      L_fusion：两视图联合对齐教师（pairwise 约束）
      由 Jensen 不等式：H(p, avg(q1,q2)) ≤ (H(p,q1)+H(p,q2))/2
      即联合约束比单独约束更严格，两者不冗余
 
    Args:
        pred_a           : 学生处理视图A的输出 [B, C, H, W]，未经 softmax
        pred_b           : 学生处理视图B的输出 [B, C, H, W]，未经 softmax
        teacher_logit    : EMA 教师的原始输出  [B, C, H, W]，未经 softmax
                           ！必须传入原始 logit，函数内部做 softmax 以获得软标签
        teacher_conf_mask: 教师置信度权重      [B, H, W]，float [0,1]
                           通常由 get_confidence_mask() 生成
        n_classes        : 分割类别数
 
    Returns:
        loss_anchor : 锚点损失（标量 Tensor，含梯度）
        loss_fusion : 融合一致性损失（标量 Tensor，含梯度）
        stats       : dict，记录关键统计量，用于 TensorBoard 分析
 
    典型用法::
        loss_anchor, loss_fusion, stats = cmc_fusion_consistency_loss(
            pred_a=out_a_viewA,
            pred_b=out_a_viewB,
            teacher_logit=pre_a,          # EMA 教师对完整图像的原始输出
            teacher_conf_mask=conf_mask_a,
            n_classes=4
        )
        loss_cmc = loss_anchor + args.cmc_fusion_weight * loss_fusion
    """
    B, C, H, W = pred_a.shape
 
    # ---- Step 1: 教师软标签和硬标签（stop gradient）----
    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logit, dim=1)         # [B, C, H, W] 软概率
        teacher_hard = teacher_logit.argmax(dim=1).long()      # [B, H, W]   硬标签
 
    # ---- Step 2: 学生两视图的 softmax 概率 ----
    prob_a = F.softmax(pred_a, dim=1)   # [B, C, H, W]
    prob_b = F.softmax(pred_b, dim=1)   # [B, C, H, W]
 
    # ---- Step 3: L_anchor —— 每视图独立对齐教师硬标签 ----
    # 使用硬标签（与 BCP 伪标签监督风格一致），教师置信度加权
    w = teacher_conf_mask                                       # [B, H, W]
    denom = w.sum() + 1e-6
 
    loss_anchor_a = F.cross_entropy(pred_a, teacher_hard, reduction='none')  # [B,H,W]
    loss_anchor_b = F.cross_entropy(pred_b, teacher_hard, reduction='none')
    loss_anchor = ((loss_anchor_a + loss_anchor_b) * w).sum() / denom / 2.0
 
    # ---- Step 4: L_fusion —— 两视图均值对齐教师软标签 ----
    #
    # 公式：H(p_teacher, avg(p_A, p_B))
    #     = -Σ_c p_teacher(c) * log( (p_A(c)+p_B(c))/2 )
    #
    # 实现细节：
    #   1. 先求概率均值，再取 log（避免先 log 再平均的 Jensen 不等式偏差）
    #   2. 加 eps 防止 log(0)
    #   3. 对类别维度 sum，得到像素级标量损失
    #
    prob_fused = (prob_a + prob_b) / 2.0                        # [B, C, H, W]
    log_prob_fused = torch.log(prob_fused + 1e-8)               # [B, C, H, W]
 
    # 像素级软交叉熵：-Σ_c p_teacher * log(p_fused)，结果 [B, H, W]
    loss_fusion_per_px = -(teacher_prob * log_prob_fused).sum(dim=1)
 
    # 教师置信度加权，高质量像素权重高
    loss_fusion = (loss_fusion_per_px * w).sum() / denom
 
    # ---- Step 5: 统计信息 ----
    with torch.no_grad():
        # 融合预测的最大概率（衡量两视图是否形成一致的高置信预测）
        fused_conf   = prob_fused.max(dim=1).values             # [B, H, W]
        # 两视图最大概率均值（衡量各视图本身的置信度）
        conf_a_mean  = prob_a.max(dim=1).values.mean().item()
        conf_b_mean  = prob_b.max(dim=1).values.mean().item()
        fused_conf_mean = fused_conf.mean().item()
 
        # 两视图预测是否一致（argmax 相同）
        agree_mask = (prob_a.argmax(dim=1) == prob_b.argmax(dim=1)).float()
        agree_ratio = agree_mask.mean().item()
 
    stats = {
        'valid_px'         : denom.item(),
        'conf_a_mean'      : conf_a_mean,
        'conf_b_mean'      : conf_b_mean,
        'fused_conf_mean'  : fused_conf_mean,
        'agree_ratio'      : agree_ratio,       # 两视图预测一致比例，期望随训练提升
        'teacher_conf_mean': w.mean().item(),
    }
 
    return loss_anchor, loss_fusion, stats


# ============================================================
# cmc_uncertainty_reweighted_loss（v3 核心）
# ============================================================
def cmc_uncertainty_reweighted_loss(pred_a: torch.Tensor,
                                     pred_b: torch.Tensor,
                                     teacher_logit: torch.Tensor,
                                     teacher_conf_mask: torch.Tensor,
                                     n_classes: int,
                                     disagree_beta: float = 5.0):
    """
    互补视图不确定性重加权锚点损失
 
    核心机制：用两视图预测的 JS 散度作为像素级不确定性，
    与教师置信度相乘，得到更精准的像素级监督权重。
 
    权重计算：
        JSD(p_A, p_B) = 0.5*KL(p_A||M) + 0.5*KL(p_B||M)  ∈ [0, ln2]
        agree_weight  = exp(-β * JSD)                       ∈ [exp(-β*ln2), 1]
        w_combined    = teacher_conf * agree_weight
 
    锚点损失：
        L = Σ_i w_i * [CE(pred_A_i, y_i) + CE(pred_B_i, y_i)] / 2
            / Σ_i w_i
 
    参数 disagree_beta（β）的作用：
        β 越大，分歧像素被压制得越狠：
          β=1 ： JSD=ln2 时 weight=0.50（较温和，仅减半）
          β=5 ： JSD=ln2 时 weight=0.03（中等强度，推荐默认值）
          β=10：JSD=ln2 时 weight=0.001（极激进，仅保留高置信像素）
        实际效果取决于 JSD 的分布，建议先看 TensorBoard 中
        jsd_mean 和 agree_weight_mean 的曲线再调整。
 
    Args:
        pred_a            : 视图A的学生输出  [B, C, H, W]，未经 softmax
        pred_b            : 视图B的学生输出  [B, C, H, W]，未经 softmax
        teacher_logit     : EMA 教师原始输出 [B, C, H, W]，未经 softmax
        teacher_conf_mask : 教师置信度权重   [B, H, W]，float [0,1]
        n_classes         : 分割类别数
        disagree_beta     : JSD 衰减系数 β（默认 5.0）
 
    Returns:
        loss_anchor  : 重加权锚点损失（标量 Tensor，含梯度）
        stats        : dict，包含 JSD / 权重 / 置信度等统计量
 
    典型用法::
        loss_anchor, stats = cmc_uncertainty_reweighted_loss(
            pred_a=out_a_viewA,
            pred_b=out_a_viewB,
            teacher_logit=pre_a,
            teacher_conf_mask=conf_mask_a,
            n_classes=4,
            disagree_beta=args.cmc_disagree_beta
        )
        loss_cmc = loss_anchor    # 无额外损失项，即 L_CMC = L_anchor_reweighted
    """
 
    # ---- Step 1: 计算两视图的 JS 散度（stop gradient，仅用于加权）----
    #
    # 梯度不通过 JSD 传播：JSD 仅作为权重系数，不作为优化目标。
    # 若允许 JSD 梯度，模型会倾向于"让两视图预测相同"而非"让预测更准确"，
    # 产生 trivial solution（所有像素预测背景类即可使 JSD=0）。
    with torch.no_grad():
        prob_a = F.softmax(pred_a, dim=1)          # [B, C, H, W]
        prob_b = F.softmax(pred_b, dim=1)
 
        prob_m = (prob_a + prob_b) / 2.0           # 混合分布 M
 
        # KL(p_A || M) = Σ_c p_A(c) * [log p_A(c) - log M(c)]
        log_pa = torch.log(prob_a + 1e-8)
        log_pb = torch.log(prob_b + 1e-8)
        log_pm = torch.log(prob_m + 1e-8)
 
        kl_a = (prob_a * (log_pa - log_pm)).sum(dim=1)   # [B, H, W]
        kl_b = (prob_b * (log_pb - log_pm)).sum(dim=1)
 
        # JSD ∈ [0, ln2 ≈ 0.693]，clamp 避免数值误差导致负值
        jsd = ((kl_a + kl_b) / 2.0).clamp(min=0.0)      # [B, H, W]
 
        # 一致性权重：JSD=0 → 1.0，JSD=ln2 → exp(-β*ln2)
        agree_weight = torch.exp(-disagree_beta * jsd)   # [B, H, W]
 
        # 组合权重：同时屏蔽"教师不确定"和"两视图分歧大"两类噪声
        w_combined = teacher_conf_mask * agree_weight    # [B, H, W]
 
    # ---- Step 2: 教师硬伪标签 ----
    with torch.no_grad():
        teacher_hard = teacher_logit.argmax(dim=1).long()  # [B, H, W]
 
    # ---- Step 3: 重加权锚点损失 ----
    denom = w_combined.sum() + 1e-6
 
    loss_a = F.cross_entropy(pred_a, teacher_hard, reduction='none')  # [B,H,W]
    loss_b = F.cross_entropy(pred_b, teacher_hard, reduction='none')
    loss_anchor = ((loss_a + loss_b) * w_combined).sum() / denom / 2.0
 
    # ---- Step 4: 统计信息（供 TensorBoard 诊断）----
    with torch.no_grad():
        # JSD 分布统计
        jsd_mean   = jsd.mean().item()
        jsd_median = jsd.median().item()
        # 高分歧区域占比（JSD > ln2/2 即超过最大值一半）
        high_disagree_ratio = (jsd > 0.347).float().mean().item()
 
        # 权重统计
        agree_w_mean  = agree_weight.mean().item()
        combined_w_mean = w_combined.mean().item()
 
        # 两视图 argmax 一致比例（离散版 agreement，与 JSD 互为印证）
        argmax_agree = (
            prob_a.argmax(dim=1) == prob_b.argmax(dim=1)
        ).float().mean().item()
 
    stats = {
        'jsd_mean'           : jsd_mean,
        'jsd_median'         : jsd_median,
        'high_disagree_ratio': high_disagree_ratio,   # 期望随训练下降
        'agree_weight_mean'  : agree_w_mean,          # 期望随训练上升
        'combined_weight_mean': combined_w_mean,
        'argmax_agree_ratio' : argmax_agree,          # 离散一致性，期望随训练上升
        'teacher_conf_mean'  : teacher_conf_mask.mean().item(),
        'valid_px'           : denom.item(),
    }
 
    return loss_anchor, stats
 