"""
通用训练工具函数
包含：模型保存加载、EMA更新、学习率调度、损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import losses, ramps

dice_loss = losses.DiceLoss(n_classes=4)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def update_model_ema(model, ema_model, alpha):
    """EMA 模型参数更新"""
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=200.0):
    """一致性权重 ramp-up"""
    return 5 * consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def poly_lr(optimizer, base_lr, current_iter, max_iter, power=0.9):
    """Poly 学习率衰减"""
    lr = base_lr * (1.0 - current_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    """
    BCP 混合损失

    Args:
        output: 网络输出 [B, C, H, W]
        img_l: 图像区域的标签 [B, H, W]
        patch_l: 粘贴区域的标签 [B, H, W]
        mask: 掩码 [B, H, W], 1=图像区域, 0=粘贴区域
        l_weight: 有标签权重
        u_weight: 无标签权重
        unlab: 是否为无标签分支
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
    MAE 一致性损失：
    学生在遮挡输入上的预测 ↔ 教师在完整输入上的伪标签
    仅在 [被遮挡区域 ∩ 高置信度区域] 计算损失

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


def patients_to_slices(dataset, patiens_num):
    """数据集患者数到切片数的映射"""
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError("Unsupported dataset")
    return ref_dict[str(patiens_num)]
