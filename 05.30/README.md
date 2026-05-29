# 🚀 BCP 融合优化项目 v05.30

> 基于 [BCP (CVPR 2023)](https://github.com/DeepMed-Lab-ECNU/BCP) 半监督医学图像分割框架，融合自研 CMC/MAE 实验系列，适配 **4×NVIDIA L20 (46GB)** 服务器的最优配置版本。

---

## 📋 版本亮点

| 特性 | 原始代码 | v05.30 优化版 |
|------|---------|---------------|
| GPU 利用 | 单 GPU（浪费 75%） | **4 GPU DDP** |
| Batch size | 24 | **96（24/卡）** |
| 混合精度 | ❌ FP32 | **✅ AMP (FP16/BF16)** |
| cuDNN 优化 | benchmark=False | **benchmark=True** |
| BN 同步 | 单卡 BN | **SyncBN** |
| 预估值速度 | 1× | **~4-6×** |

---

## 🏗 项目结构

```
05.30/
├── README.md                      # 本文件
├── run_exp.sh                     # 智能实验运行器 v3.0 (支持 torchrun)
├── code/
│   ├── ──────────── 基础设施 ────────────
│   ├── train_template.py           # DDP+AMP 模板（新实验从这里开始）
│   ├── ddp_train_adapter.py        # DDP+AMP 适配器（旧实验最小改造）
│   ├── utils/
│   │   ├── train_utils.py          # ★ 增强版: DDP init, AMP, GradScaler
│   │   ├── loss函数集 (losses.py, cmc_utils.py, corrmatch_utils.py)
│   │   ├── 伪标签工具 (pseudo_label_utils.py)
│   │   ├── 掩码生成器 (mask_generator.py)
│   │   ├── 度量工具 (metrics.py, metric_utils.py)
│   │   └── ... 其他工具
│   ├── dataloaders/
│   │   └── dataset.py              # ★ 增强版: 新增 DistributedTwoStreamSampler
│   ├── networks/
│   │   ├── net_factory.py          # ★ 修复: 移除 .cuda() 硬编码
│   │   ├── unet.py / VNet.py / Unet3D.py / unetr.py
│   │   └── ...
│   ├── ──────────── DDP+AMP 实验脚本 ────────────
│   ├── BCP_CMC_fusion_ddp.py       # CMC-Fusion DDP 版
│   ├── ACDC_BCP_MAE_train_ddp.py   # BCP+MAE DDP 版
│   ├── ──────────── 原始兼容脚本 ────────────
│   ├── BCP_CMC_fusion.py           # 原始脚本（可单 GPU 运行）
│   ├── ACDC_BCP_MAE_train.py
│   └── ... 其他原始实验脚本
└── data_split/                     # 数据划分文件
```

---

## 🔧 环境搭建

### 方式 1: 使用已有 conda 环境（推荐）

```bash
conda activate ssl
pip install -r code/requirements.txt
```

### 方式 2: 新建 conda 环境

```bash
conda create -n bcp python=3.9 -y
conda activate bcp
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r code/requirements.txt
```

---

## 🎯 运行实验

### 单 GPU 模式（兼容原始脚本）

```bash
# 方式 1: 使用 run_exp.sh 自动选 GPU
./run_exp.sh code/BCP_CMC_fusion.py --exp fusion_test --labelnum 7

# 方式 2: 直接指定 GPU
python code/BCP_CMC_fusion.py --gpu 0 --batch_size 24 --labeled_bs 12
```

### DDP 4 GPU 模式（推荐性能模式）

```bash
# 方式 1: run_exp.sh 自动选 GPU + 自动 CUDA_VISIBLE_DEVICES
./run_exp.sh --torchrun --nproc 4 code/BCP_CMC_fusion_ddp.py \
    --batch_size 96 --labeled_bs 24 --amp --ddp

# 方式 2: 手动 torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    code/BCP_CMC_fusion_ddp.py \
    --root_path ../data_split/ACDC \
    --batch_size 96 --labeled_bs 24 \
    --amp --ddp

# 方式 3: 后台运行（推荐长时间实验）
./run_exp.sh -b --torchrun --nproc 4 code/BCP_CMC_fusion_ddp.py \
    --batch_size 96 --labeled_bs 24 --amp --ddp
# 日志: logs/BCP_CMC_fusion_student_ddp_20250529_143000.log
```

### 调试模式（2 GPU + 小迭代）

```bash
torchrun --nproc_per_node=2 code/BCP_CMC_fusion_ddp.py \
    --batch_size 48 --labeled_bs 12 \
    --amp --ddp \
    --pre_iterations 500 --max_iterations 1000
```

---

## ⚙️ 参数配置指南

### Batch Size 配置（4×L20 46GB）

| 模型 | 全局 batch | 每卡 batch | 预期显存 |
|------|-----------|-----------|---------|
| UNet-2d (256²) | 96 | 24 | ~40 GB |
| UNet-2d (256²) | 128 | 32 | ~45 GB |
| VNet-3D (112×112×80) | 32 | 8 | ~35 GB |
| VNet-3D (112×112×80) | 48 | 12 | ~42 GB |

### 关键参数说明

```
--ddp             启用分布式训练（与 torchrun 配合）
--amp             启用 AMP 混合精度（推荐开启，提速 1.5-2×）
--batch_size N    全局 batch size（跨所有 GPU，必须能被 world_size 整除）
--labeled_bs M    全局有标签 batch size
--labelnum N      有标签患者数（7, 14, 21, 28, 35, 70）
--deterministic 0 关闭确定性模式（开启 cudnn.benchmark 加速）
--sync_bn         启用 SyncBatchNorm（多卡时推荐开启）
```

---

## 📊 实验系列

### CMC 系列 (Complementary Mask Consistency)

| 脚本 | 描述 |
|------|------|
| `BCP_CMC_fusion.py` | CMC-Fusion: 互补掩码预测融合（v2） |
| `BCP_CMC_fusion_ddp.py` | ★ CMC-Fusion DDP+AMP 版 |
| `BCP_CMC_student.py` | CMC 互教版（v1） |
| `BCP_CMC_Uncertainly.py` | CMC 不确定性加权版（v3） |

### MAE 系列 (Masked Autoencoder)

| 脚本 | 描述 |
|------|------|
| `ACDC_BCP_MAE_train.py` | BCP + MAE 一致性 |
| `ACDC_BCP_MAE_train_ddp.py` | ★ BCP+MAE DDP+AMP 版 |
| `ACDC_BCP_MAE_CorrMatch_train.py` | BCP + MAE + CorrMatch |
| `ACDC_MAE_train.py` | 纯 MAE 无 BCP |

### MultiPatch 系列

| 脚本 | 描述 |
|------|------|
| `train_bcp_multiPatch.py` | 多 patch BCP |
| `train_bcp_multiPatch_dynamic.py` | 动态多 patch |
| `train_bcp_multiPatch_LA.py` | 多 patch LA 心脏数据集 |

---

## 🛠 创建新实验脚本

### 方式 1: 使用模板（推荐）

```python
# 从 train_template.py 复制，只需填充 pre_train 和 self_train 中的算法逻辑
# 模板已内置: DDP init, AMP, SyncBN, GradScaler, DDP DataLoader, EMA
```

### 方式 2: 从现有 ddp 脚本修改

```python
# 从 BCP_CMC_fusion_ddp.py 复制，修改损失函数和网络逻辑
```

### 方式 3: 为旧脚本添加 DDP+AMP（最小改造）

```python
# 在旧脚本顶部添加:
from ddp_train_adapter import ddp_run, ddp_main_wrapper, add_ddp_args

# 在参数解析处:
parser = add_ddp_args(parser)

# 在主入口:
if __name__ == "__main__":
    ddp_run(
        main_func=lambda args: your_main_func(args),
        args=args,
        use_amp=args.amp,
        use_ddp=args.ddp,
    )
```

---

## 🔍 与原版的差异

### 1. DDP 数据并行（核心改动）
- `torchrun --nproc_per_node=4` 启动
- 模型 `DistributedDataParallel` 包装
- `DistributedTwoStreamBatchSampler` 替代 `TwoStreamBatchSampler`

### 2. AMP 混合精度（核心改动）
- `torch.cuda.amp.autocast()` 包裹 forward
- `GradScaler` 处理 FP16 梯度缩放
- `AMPTrainer` 类封装整套流程

### 3. 模型管理（核心改动）
- `net_factory.py` 去除了 `.cuda()` 调用
- 模型在 CPU 创建 → `ddp_wrap_model()` 统一管理
- EMA 仅在 rank 0 更新
- 保存/日志仅在 rank 0

### 4. 网络架构（未改动）
- `unet.py` / `VNet.py` / `Unet3D.py` / `unetr.py` — 与原版完全一致
- `BCP_utils.py` / `losses.py` / `ramps.py` — 与原版完全一致

---

## 📚 引用

```bibtex
@inproceedings{BCP2023,
  title={Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation},
  author={Bai, Yalong and Chen, Duo and Li, Qingli and Shen, Wei and Wang, Yan},
  booktitle={CVPR},
  year={2023}
}
```

## 📄 许可

本项目基于 [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) 的 Apache 2.0 许可。
