# 🧪 BCP Experiments

基于 [**BCP: Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation**](https://github.com/DeepMed-Lab-ECNU/BCP) (CVPR 2023) 的自研实验脚本集合。

## 📋 实验列表

### CMC 系列 (Complementary Mask Consistency)

核心思路：通过互补掩码 (Complementary Mask) 将输入图像分为两个视图，利用视图之间的一致性进行半监督学习。

| 实验 | 文件 | 描述 |
|------|------|------|
| CMC-Fusion (v2) | [`code/BCP_CMC_fusion.py`](code/BCP_CMC_fusion.py) | 互补掩码预测融合一致性。两视图概率均值对齐教师软概率(pairwise)，信号质量依赖 EMA 教师，梯度同时流向两视图 |
| CMC-Student (v1) | [`code/BCP_CMC_student.py`](code/BCP_CMC_student.py) | 互补掩码互教一致性。A 预测监督 B 盲区，B 预测监督 A 盲区 |
| CMC-Uncertainty (v3) | [`code/BCP_CMC_Uncertainly.py`](code/BCP_CMC_Uncertainly.py) | 在 CMC 基础上引入不确定性建模 |

旧版/存档：
- [`code/BCP_CMC_v1_mutual.py`](code/BCP_CMC_v1_mutual.py) — 早期互教实现
- [`code/BCP_CMC_v2_fusion.py`](code/BCP_CMC_v2_fusion.py) — 早期融合实现
- [`code/BCP_CMC_v3_uncertainty.py`](code/BCP_CMC_v3_uncertainty.py) — 早期不确定性实现

### MAE 系列 (Masked Autoencoder)

| 实验 | 文件 | 描述 |
|------|------|------|
| BCP + MAE | [`code/ACDC_BCP_MAE_train.py`](code/ACDC_BCP_MAE_train.py) | BCP 双向 copy-paste + MAE 遮挡重建分支，像素级置信度过滤 |
| BCP + MAE + CorrMatch | [`code/ACDC_BCP_MAE_CorrMatch_train.py`](code/ACDC_BCP_MAE_CorrMatch_train.py) | MAE 基础上引入 CorrMatch 传播策略 |
| MAE Only | [`code/ACDC_MAE_train.py`](code/ACDC_MAE_train.py) | 纯 MAE 遮挡重建的半监督方法 |

### MultiPatch 系列

| 实验 | 文件 | 描述 |
|------|------|------|
| MultiPatch Baseline | [`code/train_bcp_multiPatch.py`](code/train_bcp_multiPatch.py) | 多块 copy-paste 策略 |
| MultiPatch Dynamic | [`code/train_bcp_multiPatch_dynamic.py`](code/train_bcp_multiPatch_dynamic.py) | 动态多块策略 (ACDC) |
| MultiPatch Dynamic (LA) | [`code/train_bcp_multiPatch_dynamic_LA.py`](code/train_bcp_multiPatch_dynamic_LA.py) | 动态多块策略 (LA 3D) |
| MultiPatch Random | [`code/train_bcp_multiPatch_ran.py`](code/train_bcp_multiPatch_ran.py) | 随机多块策略 |
| MultiPatch TrueDynamic | [`code/train_bcp_multiPatch_TrueDynamic.py`](code/train_bcp_multiPatch_TrueDynamic.py) | 真正的动态多块策略 |

### 其他

| 文件 | 描述 |
|------|------|
| [`code/train_bcp_random_patch.py`](code/train_bcp_random_patch.py) | 随机单块 copy-paste |
| [`code/train_bcp_true.py`](code/train_bcp_true.py) | 原始 BCP 的重构版本 |
| [`code/train_fgbg_random_patch.py`](code/train_fgbg_random_patch.py) | 前景/背景引导的随机 patch |
| [`code/test_MAE_create.py`](code/test_MAE_create.py) | MAE 可视化测试 |
| [`code/test_show.py`](code/test_show.py) | 分割结果可视化 |
| [`code/generate_list.py`](code/generate_list.py) | 数据列表生成工具 |
| [`code/KDE_demo.py`](code/KDE_demo.py) | KDE (核密度估计) 演示 |

### 辅助工具

| 文件 | 描述 |
|------|------|
| [`code/utils/cmc_utils.py`](code/utils/cmc_utils.py) | CMC 核心工具：互补网格掩码生成、CMC 一致性损失 |
| [`code/utils/corrmatch_utils.py`](code/utils/corrmatch_utils.py) | CorrMatch 传播策略工具 |
| [`code/utils/mask_generator.py`](code/utils/mask_generator.py) | MAE 网格掩码生成器 (`MAEGridMaskGenerator`) |
| [`code/utils/metric_utils.py`](code/utils/metric_utils.py) | 自定义评估指标工具 |
| [`code/utils/pseudo_label_utils.py`](code/utils/pseudo_label_utils.py) | 伪标签质量过滤与增强工具 |
| [`code/utils/train_utils.py`](code/utils/train_utils.py) | 训练辅助函数 (poly_lr, EMA 更新等) |
| [`code/original_code/`](code/original_code/) | 原始参考代码存档 |

### 运行脚本

| 文件 | 描述 |
|------|------|
| [`run_exp.sh`](run_exp.sh) | 智能实验运行脚本 — 自动选择空闲显存最多的 GPU，遇 OOM 自动切换 |
| [`run_test.sh`](run_test.sh) | 优化版测试运行与结果记录脚本 — 自动解析指标、生成 CSV/Markdown 汇总 |

## ⚙️ 依赖

### 基础框架 (不包含在本仓库中)

本仓库**只包含实验脚本**，运行需要依赖 [BCP 原始代码库](https://github.com/DeepMed-Lab-ECNU/BCP) 的以下基础设施：

```
BCP/
├── code/
│   ├── dataloaders/          # 数据集加载 (dataset.py, acdc_data_processing.py, la_heart_processing.py)
│   ├── networks/             # 网络定义 (unet.py, VNet.py, net_factory.py, Unet3D.py, unetr.py)
│   ├── utils/
│   │   ├── BCP_utils.py      # context_mask(), mix_loss(), random_mask()
│   │   ├── losses.py         # DiceLoss, CrossEntropyLoss
│   │   ├── metrics.py        # cal_dice, calculate_metric_percase
│   │   ├── ramps.py          # sigmoid_rampup, linear_rampup
│   │   ├── val_2d.py         # 2D 验证
│   │   ├── test_3d_patch.py  # 3D 滑动窗口测试
│   │   ├── feature_memory.py
│   │   └── contrastive_losses.py
│   ├── test_ACDC.py          # ACDC 测试脚本
│   ├── test_LA.py            # LA 测试脚本
│   ├── ACDC_BCP_train.py     # BCP ACDC 训练脚本 (原始)
│   └── LA_BCP_train.py       # BCP LA 训练脚本 (原始)
└── data_split/               # 数据拆分列表 (h5 数据文件需自行准备)
```

### Python 依赖

关键依赖见 [`code/requirements.txt`](code/requirements.txt)，主要包括：
- PyTorch 1.8+ (CUDA)
- torchvision, tensorboardX
- numpy, scipy, SimpleITK, nibabel
- MedPy, h5py, tqdm

## 🛠️ 使用方式

所有脚本设计为在 BCP 项目的 `code/` 目录结构下运行。推荐的工作目录布局：

```
BCP/                          # 原始 BCP 项目根目录
├── code/
│   ├── (原始文件)              # dataloaders/, networks/, utils/ 等
│   └── (本仓库文件)            # 复制本仓库的 code/ 内容到此处
├── data_split/
│   └── ...
└── model/
    └── ...
```

### 运行实验

```bash
# 使用智能运行脚本（自动选择 GPU）
./run_exp.sh code/BCP_CMC_fusion.py --exp BCP_CMC_fusion --labelnum 7

# 后台运行
./run_exp.sh -b code/train_bcp_multiPatch.py --exp MultiPatch --labelnum 7

# 直接运行
cd code && python BCP_CMC_fusion.py --gpu 0 --exp BCP_CMC_fusion --labelnum 7
```

### 运行测试

```bash
# 自动选择 GPU 测试
./run_test.sh --exp BCP_CMC_fusion_student --labelnum 7

# 指定 GPU
./run_test.sh --gpu 0 --exp BCP_CMC_fusion_student --labelnum 7
```

## 📊 结果汇总

测试结果自动汇总到 [`test_results/`](test_results/) 目录：
- `master_metrics.csv` — 全部实验的指标汇总表格
- `SUMMARY.md` — 自动生成的 Markdown 汇总报告
- `{exp}_{labelnum}_{stage}_{timestamp}.txt` — 单次实验的详细结果

## 📄 许可

- 本仓库的实验脚本：MIT License
- 原始 BCP 代码：[MIT License](https://github.com/DeepMed-Lab-ECNU/BCP/blob/master/LICENSE) (Copyright 2023 DeepMed Lab @ECNU)

## 📚 引用

```bibtex
@inproceedings{BCP2023,
  title={Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation},
  author={Bai, Yunhao and Chen, Duo and Li, Qingli and Shen, Wei and Wang, Yan},
  booktitle={CVPR},
  year={2023}
}
```
