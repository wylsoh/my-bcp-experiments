"""
CorrMatch 相关性匹配伪标签传播模块

修复说明:
  - 明确所有 tensor 的空间分辨率
  - 区分特征分辨率 (H_feat, W_feat) 和输入图像分辨率 (H_img, W_img)
  - 传播在下采样空间进行, 结果上采样回输入图像分辨率

参考文献:
  CorrMatch: Label Propagation via Correlation Matching
  for Semi-Supervised Semantic Segmentation (CVPR 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationPropagator:
    """
    特征相关性伪标签传播器

    Args:
        num_classes: 分割类别数
        temperature: 相关性 softmax 温度
        propagation_threshold: 传播后的置信度保留阈值
        downsample_factor: 在特征图基础上进一步下采样的倍数
    """

    def __init__(self, num_classes=4, temperature=0.1,
                 propagation_threshold=0.6, downsample_factor=4):
        self.num_classes = num_classes
        self.temperature = temperature
        self.propagation_threshold = propagation_threshold
        self.downsample_factor = downsample_factor

    @torch.no_grad()
    def propagate(self, features, pseudo_label, confidence_mask, teacher_probs):
        """
        执行相关性传播

        Args:
            features:        [B, C, H_feat, W_feat]  教师中间特征 (例: 64×64)
            pseudo_label:    [B, H_img, W_img]       初始伪标签   (例: 256×256)
            confidence_mask: [B, H_img, W_img]       初始置信度掩码
            teacher_probs:   [B, K, H_img, W_img]    教师 softmax 概率

        Returns:
            refined_pseudo_label: [B, H_img, W_img]  精炼伪标签
            refined_conf_mask:    [B, H_img, W_img]  扩展置信度掩码
            propagation_ratio:    float              传播扩展比率
        """
        B, C, H_feat, W_feat = features.shape
        H_img, W_img = pseudo_label.shape[1], pseudo_label.shape[2]

        # ---- 设置传播工作分辨率 ----
        # 在特征空间基础上再下采样 downsample_factor 倍, 控制显存
        ds = max(1, self.downsample_factor)
        H_ds = max(8, H_feat // ds)
        W_ds = max(8, W_feat // ds)

        # ---- 将所有输入对齐到工作分辨率 (H_ds, W_ds) ----
        feat_ds = F.interpolate(features, size=(H_ds, W_ds),
                                 mode='bilinear', align_corners=False)
        prob_ds = F.interpolate(teacher_probs, size=(H_ds, W_ds),
                                 mode='bilinear', align_corners=False)
        conf_ds = F.interpolate(
            confidence_mask.unsqueeze(1).float(),
            size=(H_ds, W_ds), mode='nearest'
        ).squeeze(1)  # [B, H_ds, W_ds]

        # ---- 特征 L2 归一化 ----
        feat_flat = feat_ds.reshape(B, C, -1)  # [B, C, N]
        feat_norm = F.normalize(feat_flat, dim=1)

        # ---- 计算相关性矩阵 ----
        N = H_ds * W_ds
        corr = torch.bmm(feat_norm.transpose(1, 2), feat_norm)  # [B, N, N]
        corr = corr / self.temperature

        # 只让高置信度像素作为传播源
        conf_flat = conf_ds.reshape(B, -1)  # [B, N]
        source_mask = conf_flat.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]
        # 屏蔽低置信度源 (置信度低的像素不参与传播)
        corr = corr.masked_fill(source_mask < 0.5, -1e4)

        # Softmax 归一化
        corr = F.softmax(corr, dim=-1)  # [B, N, N]

        # ---- 传播概率分布 ----
        prob_flat = prob_ds.reshape(B, self.num_classes, -1)  # [B, K, N]
        # propagated[b, k, j] = Σ_i prob[b, k, i] * corr[b, j, i]
        propagated = torch.bmm(prob_flat, corr.transpose(1, 2))  # [B, K, N]
        propagated = propagated.reshape(B, self.num_classes, H_ds, W_ds)

        # ---- 融合原始预测与传播结果 ----
        # 高置信度区域保持原始, 低置信度区域使用传播结果
        alpha = conf_ds.unsqueeze(1)  # [B, 1, H_ds, W_ds]
        fused = alpha * prob_ds + (1 - alpha) * propagated  # [B, K, H_ds, W_ds]

        # ---- 上采样回输入图像分辨率 ----
        fused_full = F.interpolate(fused, size=(H_img, W_img),
                                    mode='bilinear', align_corners=False)
        # 现在 fused_full: [B, K, H_img, W_img]

        # ---- 生成精炼后的伪标签与置信度掩码 ----
        refined_max_probs, refined_pseudo_label = torch.max(fused_full, dim=1)
        # refined_pseudo_label: [B, H_img, W_img]
        # refined_max_probs:    [B, H_img, W_img]

        refined_conf_mask = (refined_max_probs >= self.propagation_threshold).float()

        # ---- 保留原始高置信度区域的伪标签 ----
        original_high_conf = confidence_mask.float()  # [B, H_img, W_img]

        # 确保两个张量类型一致再做 where
        refined_pseudo_label = torch.where(
            original_high_conf.bool(),
            pseudo_label.long(),
            refined_pseudo_label.long()
        )
        refined_conf_mask = torch.clamp(refined_conf_mask + original_high_conf, 0, 1)

        # ---- 统计 ----
        original_count = original_high_conf.sum().item()
        refined_count = refined_conf_mask.sum().item()
        propagation_ratio = (refined_count - original_count) / max(original_count, 1.0)

        return refined_pseudo_label, refined_conf_mask, propagation_ratio


class FeatureExtractorHook:
    """通过 forward hook 提取网络中间特征"""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        self.features = output.detach()

    def clear(self):
        self.features = None


def register_feature_hooks(model):
    """
    为 BCP_net 注册特征提取 hook
    返回 hook 对象、句柄、被 hook 的层名
    """
    hook = FeatureExtractorHook()
    target_layer = None
    logging_name = None

    # 优先选择 decoder 的中间层
    for name, module in model.named_modules():
        if 'up2' in name or 'decoder2' in name or 'up_conv2' in name:
            target_layer = module
            logging_name = name
            break

    if target_layer is None:
        for name, module in model.named_modules():
            if 'up3' in name or 'decoder3' in name or 'up_conv3' in name:
                target_layer = module
                logging_name = name
                break

    if target_layer is None:
        for name, module in model.named_modules():
            if 'bottleneck' in name or 'bridge' in name or 'center' in name:
                target_layer = module
                logging_name = name
                break

    if target_layer is None:
        modules_list = list(model.named_modules())
        logging_name, target_layer = modules_list[-5] if len(modules_list) > 5 else modules_list[-2]

    handle = target_layer.register_forward_hook(hook)
    return hook, handle, logging_name


def get_features_from_model(model, input_tensor, hook, handle):
    """前向传播并获取中间特征"""
    hook.clear()
    output = model(input_tensor)
    features = hook.features
    return output, features
