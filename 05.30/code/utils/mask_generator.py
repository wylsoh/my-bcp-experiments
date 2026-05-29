"""
MAE 风格的网格遮挡掩码生成器
用于半监督分割中的学生端上下文推断训练
"""
import torch
import torch.nn.functional as F
import numpy as np
import random


class MAEGridMaskGenerator:
    """
    MAE 网格遮挡掩码生成器

    将图像划分为 patch_size × patch_size 的网格块，
    随机遮挡 mask_ratio 比例的网格块。

    Args:
        img_size: 输入图像尺寸 (假设正方形)
        patch_size: 网格块大小
        mask_ratio: 遮挡比例 (0~1)
    """

    def __init__(self, img_size=256, patch_size=16, mask_ratio=0.65):
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

    def generate(self, batch_size, device):
        """
        生成批次遮挡掩码

        Returns:
            visible_mask: [B, 1, H, W], 1=可见, 0=被遮挡
        """
        num_mask = int(self.num_patches * self.mask_ratio)
        masks = []

        for _ in range(batch_size):
            # 随机打乱 patch 索引
            noise = torch.rand(self.num_patches, device=device)
            ids_shuffle = torch.argsort(noise)

            # 前 num_mask 个被遮挡
            mask_flat = torch.ones(self.num_patches, device=device)
            mask_flat[ids_shuffle[:num_mask]] = 0

            # 上采样到原始分辨率
            mask_2d = mask_flat.reshape(1, 1, self.grid_size, self.grid_size)
            mask_full = F.interpolate(
                mask_2d,
                size=(self.img_size, self.img_size),
                mode='nearest'
            )
            masks.append(mask_full)

        return torch.cat(masks, dim=0)  # [B, 1, H, W]

    def get_progressive_ratio(self, current_iter, warmup_iter,
                              min_ratio=0.25, max_ratio=0.65):
        """
        渐进式遮挡：训练初期遮挡少，后期逐步增大
        """
        if current_iter < warmup_iter:
            progress = current_iter / warmup_iter
            return min_ratio + (max_ratio - min_ratio) * progress
        return max_ratio


class BCPMaskGenerator:
    """
    BCP 原始 copy-paste 掩码生成器（复用封装）
    """

    @staticmethod
    def generate(img):
        """
        生成 BCP 原始的大块矩形遮挡掩码

        Args:
            img: [B, C, H, W]
        Returns:
            mask: [H, W] long tensor, 1=保留原图, 0=粘贴区域
            loss_mask: [B, H, W] long tensor
        """
        batch_size, channel, img_x, img_y = img.shape
        loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
        mask = torch.ones(img_x, img_y).cuda()
        patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
        w = np.random.randint(0, img_x - patch_x)
        h = np.random.randint(0, img_y - patch_y)
        mask[w:w + patch_x, h:h + patch_y] = 0
        loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
        return mask.long(), loss_mask.long()

    @staticmethod
    def generate_random(img, shrink_param=3):
        """多块分散遮挡"""
        batch_size, channel, img_x, img_y = img.shape
        loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
        x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
        patch_x = int(img_x * 2 / (3 * shrink_param))
        patch_y = int(img_y * 2 / (3 * shrink_param))
        mask = torch.ones(img_x, img_y).cuda()
        for x_s in range(shrink_param):
            for y_s in range(shrink_param):
                w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
                h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
                mask[w:w + patch_x, h:h + patch_y] = 0
                loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
        return mask.long(), loss_mask.long()
