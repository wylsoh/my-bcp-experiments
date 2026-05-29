"""
伪标签生成与过滤工具
包含：最大连通域后处理、置信度过滤、伪标签生成
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label


def get_ACDC_2DLargestCC(segmentation):
    """
    对 batch 中每个 2D slice 的每个类别取最大连通域

    Args:
        segmentation: [B, H, W] 整数标签
    Returns:
        cleaned: [B, H, W] 清理后的标签
    """
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
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    """
    从网络输出生成伪标签

    Args:
        output: [B, C, H, W] 网络 logits
        nms: 是否进行最大连通域后处理
    Returns:
        pseudo_labels: [B, H, W]
    """
    probs = F.softmax(output, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    if nms == 1:
        pseudo_labels = get_ACDC_2DLargestCC(pseudo_labels)
    return pseudo_labels


def get_confidence_mask(output, threshold=0.85):
    """
    像素级置信度过滤

    Args:
        output: [B, C, H, W] 网络 logits
        threshold: 置信度阈值
    Returns:
        pseudo_label: [B, H, W] 伪标签
        conf_mask: [B, H, W] 置信度掩码, 1=可信
        mean_conf: float 平均置信度
    """
    probs = F.softmax(output, dim=1)
    max_probs, pseudo_label = torch.max(probs, dim=1)
    conf_mask = (max_probs >= threshold).float()
    mean_conf = max_probs.mean().item()
    return pseudo_label, conf_mask, mean_conf


def get_adaptive_threshold(current_iter, max_iter,
                           init_threshold=0.90, final_threshold=0.70):
    """
    自适应置信度阈值：初期严格，后期逐步放宽
    """
    progress = min(current_iter / max_iter, 1.0)
    threshold = init_threshold - (init_threshold - final_threshold) * progress
    return threshold
