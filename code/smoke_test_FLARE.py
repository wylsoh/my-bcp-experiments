"""
FLARE 冒烟测试 — CPU 模式，验证全部函数正确性
"""
import os, sys, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import torch.nn.functional as F
from networks.net_factory import net_factory
from utils import losses

num_classes = 14
device = torch.device('cpu')
print(f"Device: {device}, Torch: {torch.__version__}")

from FLARE_BCP_CMC_NA_v2_soft_mask import FLAREDataSets
from FLARE_BCP_CMC_NA_v2_soft_mask import context_mask_flare, get_cut_mask_multi
from FLARE_BCP_CMC_NA_v2_soft_mask import mix_loss_multi
from FLARE_BCP_CMC_NA_v2_soft_mask import generate_cmc_masks_3d_soft, cmc_mutual_loss_3d_soft
from FLARE_BCP_CMC_NA_v2_soft_mask import get_progressive_shared_ratio, get_adaptive_threshold

errors = []

def check(name, cond, detail=''):
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}: {detail}")
        errors.append(name)

# ===== Test 1: 数据加载 =====
print("\n[Test 1] FLAREDataSets data loading")
from torchvision import transforms
from dataloaders.dataset import RandomRotFlip, RandomCrop, ToTensor
db = FLAREDataSets(base_dir='../data_split/flare', split='train')
check("train samples >0", len(db) > 0, f"got {len(db)}")
img, lab = db[0]['image'], db[0]['label']
check("image 3D", img.ndim == 3, f"shape={img.shape}")
check("label same shape", lab.shape == img.shape)
check("unique labels 0-13", set(np.unique(lab)) <= set(range(14)))
print(f"    total {len(db)} samples, image {img.shape}, label values {np.unique(lab)[:5]}...")

# ===== Test 2: 数据增强 + DataLoader =====
print("\n[Test 2] Data augmentation + TwoStreamBatchSampler")
from dataloaders.dataset import TwoStreamBatchSampler
from torch.utils.data import DataLoader
patch_size = (64, 128, 128)
db_aug = FLAREDataSets(base_dir='../data_split/flare', split='train',
    transform=transforms.Compose([RandomRotFlip(), RandomCrop(patch_size), ToTensor()]))
labeled_idxs = list(range(42))
unlabeled_idxs = list(range(42, 419))
sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 8, 6)
loader = DataLoader(db_aug, batch_sampler=sampler, num_workers=0)
batch = next(iter(loader))
check("batch image shape", batch['image'].shape == (8, 1, 64, 128, 128), f"got {batch['image'].shape}")
check("batch label shape", batch['label'].shape == (8, 64, 128, 128), f"got {batch['label'].shape}")
check("label range 0-13", batch['label'].min() >= 0 and batch['label'].max() <= 13)

# ===== Test 3: context_mask_flare =====
print("\n[Test 3] context_mask_flare")
img_t = torch.rand(2, 1, 64, 128, 128)
m, lm = context_mask_flare(img_t, 2/3)
check("mask shape", m.shape == (64, 128, 128))
check("loss_mask shape", lm.shape == (2, 64, 128, 128))
check("mask binary", set(m.unique().tolist()) == {0, 1})
# 3D 中 mask_ratio=2/3 → 立方体占 (2/3)^3 ≈ 29.6%
ratio = (m == 0).float().mean().item()
check(f"mask ratio ~(2/3)^3", 0.25 < ratio < 0.35, f"got {ratio:.3f}")

# ===== Test 4: VNet 前向传播 (14类) =====
print("\n[Test 4] VNet forward (14 classes)")
net = net_factory('VNet', 1, num_classes, 'test').to(device)
x = torch.rand(2, 1, 64, 128, 128).to(device)
out, _ = net(x)
check("output shape", out.shape == (2, 14, 64, 128, 128), f"got {out.shape}")

# ===== Test 5: get_cut_mask_multi =====
print("\n[Test 5] get_cut_mask_multi")
logits = torch.rand(2, 14, 64, 128, 128)
mask = get_cut_mask_multi(logits)
check("output shape", mask.shape == (2, 64, 128, 128))
check("long dtype", mask.dtype == torch.int64)
check("label range 0-13", mask.min() >= 0 and mask.max() <= 13)

# ===== Test 6: mix_loss_multi =====
print("\n[Test 6] mix_loss_multi")
logits = torch.rand(2, 14, 64, 128, 128)
lab = torch.randint(0, 14, (2, 64, 128, 128))
patch_lab = torch.randint(0, 14, (2, 64, 128, 128))
loss_mask = torch.randint(0, 2, (2, 64, 128, 128)).float()
loss = mix_loss_multi(logits, lab, patch_lab, loss_mask, l_weight=1.0, u_weight=0.5)
check("loss scalar", loss.ndim == 0, f"shape={loss.shape}")
check("loss > 0", loss.item() > 0, f"got {loss.item()}")
loss_u = mix_loss_multi(logits, lab, patch_lab, loss_mask, l_weight=1.0, u_weight=0.5, unlab=True)
check("unlab loss > 0", loss_u.item() > 0)

# ===== Test 7: generate_cmc_masks_3d_soft =====
print("\n[Test 7] generate_cmc_masks_3d_soft")
img = torch.rand(2, 1, 64, 128, 128)
ma, mb = generate_cmc_masks_3d_soft(img, cmc_patch_size=8, shared_ratio=0.0, soft_sigma=0.3)
check("mask_a shape", ma.shape == (2, 1, 64, 128, 128))
check("mask_b shape", mb.shape == (2, 1, 64, 128, 128))
# 由于 trilinear + avg_pool, a+b 应接近 1
sum_ok = torch.allclose(ma + mb, torch.ones_like(ma), atol=1e-2)
check("a + b ≈ 1", sum_ok, f"max diff={(ma+mb-1).abs().max().item():.4f}")

# ===== Test 8: cmc_mutual_loss_3d_soft =====
print("\n[Test 8] cmc_mutual_loss_3d_soft")
out_a = torch.rand(2, 14, 64, 128, 128)
out_b = torch.rand(2, 14, 64, 128, 128)
teacher = torch.randint(0, 14, (2, 64, 128, 128)).long()
conf_mask = torch.ones(2, 64, 128, 128).float()
loss_cmc = cmc_mutual_loss_3d_soft(out_a, out_b, teacher, conf_mask, ma, mb, 0.75, 0.5)
check("cmc loss scalar", loss_cmc.ndim == 0)
check("cmc loss > 0", loss_cmc.item() > 0)

# ===== Test 9: get_progressive_shared_ratio =====
print("\n[Test 9] Progressive shared ratio")
check("at start", get_progressive_shared_ratio(0, 2000, 0.4, 0.0) == 0.4)
check("midway", abs(get_progressive_shared_ratio(1000, 2000, 0.4, 0.0) - 0.2) < 1e-6)
check("after warmup", get_progressive_shared_ratio(3000, 2000, 0.4, 0.0) == 0.0)

# ===== Test 10: get_adaptive_threshold =====
print("\n[Test 10] Adaptive threshold")
check("at start", get_adaptive_threshold(0, 30000, 0.90, 0.70) == 0.90)
check("halfway", abs(get_adaptive_threshold(15000, 30000, 0.90, 0.70) - 0.80) < 0.01)
check("end", get_adaptive_threshold(30000, 30000, 0.90, 0.70) == 0.70)

# ===== Summary =====
print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED! ✓")
    print("FLARE training script is ready for production use.")
print("=" * 50)
