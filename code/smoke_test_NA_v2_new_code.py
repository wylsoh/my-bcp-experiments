#!/usr/bin/env python3
"""冒烟测试：验证新 MM_BCP_train.py 能正常导入并运行自训练"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 测试能否导入
from MM_BCP_train import generate_cmc_masks_v2_soft, cmc_mutual_loss_soft, get_progressive_shared_ratio, get_adaptive_threshold
print("✅ 导入成功")

# 测试函数能否运行
import torch
x = torch.randn(2, 1, 256, 256).cuda()
ma, mb = generate_cmc_masks_v2_soft(x, cmc_patch_size=8, soft_sigma=0.5)
print(f"✅ generate_cmc_masks_v2_soft: mask_a shape={ma.shape}, range=[{ma.min().item():.3f}, {ma.max().item():.3f}]")

# 测试 boundary ratio
border = ((ma.squeeze(1) > 0.1) & (ma.squeeze(1) < 0.9)).float().mean().item()
print(f"✅ boundary ratio = {border:.3f} (应为 ~0.03-0.10)")

# 测试损失函数
out = torch.randn(2, 8, 256, 256).cuda()
plab = torch.randint(0, 8, (2, 256, 256)).cuda()
conf = torch.ones(2, 256, 256).cuda()
loss, anchor, mutual = cmc_mutual_loss_soft(out, out, plab, conf, ma, mb, 0.75, 0.5)
print(f"✅ cmc_mutual_loss_soft: loss={loss.item():.4f}, anchor={anchor.item():.4f}, mutual={mutual.item():.4f}")

print("🎉 所有测试通过！")
