# NA_v2 (NA-CMC V2) 参数配置调查报告

## 1. MM 数据集实验结果汇总

| 实验 | 标注量 | Mean Dice | Mean HD95 | 最佳迭代 | 最终迭代 |
|------|:-----:|:---------:|:---------:|:--------:|:--------:|
| **NA_v2 10%** | 382 slices (10%) | **0.6376** | 108.92 | iter 4600 | iter 30000 |
| **NA_v2 20%** | 764 slices (20%) | **0.8108** | 60.04 | iter 30000 | iter 30000 |
| **BCP 10%** | 5 patients (~33%) | **0.6769** | 120.55 | iter 40000 | iter 40000 |
| **BCP 20%** | 10 patients (~67%) | **0.8458** | 58.84 | iter 40000 | iter 40000 |

> **⚠️ BCP 与 NA_v2 的标注方式完全不同**：
> - NA_v2: `labelnum=10` → 从全部 3823 slices 中随机抽 10% = **382 slices 有标签**
> - BCP: `labelnum=5` → 从 15 个病人中选 **5 个病人的全部 slice ≈ 1275 slices 有标签**
> 
> 所以 BCP 并非真正的 "10%标注"，而是 33%病人全标注。BCP 实际上用了 **3.3 倍以上** 的标注数据！

---

## 2. NA_v2 参数配置诊断

### 2.1 pre_train vs self_train 参数不一致 ❌

| 参数 | pre_train (run_mm_pre_train.py) | self_train (实际运行的) | 问题 |
|------|:-------------------------------:|:----------------------:|:----:|
| `batch_size` | **48** | **24** | 减半 |
| `labeled_bs` | **24** | **12** | 减半 |
| `max_iterations` | 30000 (但只跑 pre_iterations=10000) | **30000** | 比 BCP 少 25% |
| `cmc_patch_size` | 8 | 8 | ✅ 一致 |
| `cmc_soft_sigma` | 0.3 | 0.3 | ✅ 一致 |

**影响分析**：
- pre_train batch_size=48 加速了预训练。但 self_train 只用 batch_size=24，**每 batch 的 labeled 样本从 24 降到 12**，有效监督信号减半
- 而 BCP 全程 batch_size=24, labeled_bs=12，所以这方面与 BCP 一致，问题不大

### 2.2 max_iterations 比 BCP 少 25% ❌

| 实验 | max_iterations | 出 best 的迭代 | 性能退化 |
|------|:--------------:|:--------------:|:--------:|
| NA_v2 10% | 30000 | iter 4600 (Dice=0.6376) | iter 30000 → Dice=0.5649 🔻|
| BCP 10% | **40000** | iter 40000 (Dice=0.6769) | 无明显退化 |
| NA_v2 20% | 30000 | iter 30000 (Dice=0.8108) | 无明显退化 |

NA_v2 的 `max_iterations` 默认值 = 30000，而 BCP 默认为 40000。
如果 NA_v2 也跑 40000 iter，10% 的结果有可能继续提升。

### 2.3 NA_v2 10% 的严重性能退化 ⚠️

```
iter 4600  → Dice=0.6376  ★ 最佳
iter 30000 → Dice=0.5649  ☆ 最终 (比最佳差 0.0727!)
```

这说明 **NA_v2 10% 在训练后期严重退化**。可能原因：
1. **soft_sigma=0.3 → avg_pool 核 3px**：soft mask 过渡带太窄，导致 CMC mask 提供的丰富信息不够，后期模型学会了"偷懒"
2. **conf_thresh 从 0.90 退火到 0.70**：伪标签质量随阈值降低而下降
3. **过拟合**：382 个 labeled slices，但训练了 30000 个 iteration（约 968 epochs），严重过拟合

### 2.4 `restart_NA_v2_10pct.py` 的 cmc_patch_size 不一致 ❌

该文件设置了 `cmc_patch_size=16`，但：
- pre_train 实际用了 `cmc_patch_size=8`
- self_train 实际从 pre_train 的 snapshot 加载，所以应该用的是 8
- **这个文件没有被实际使用**（self_train 是用别的脚本启动的），所以没有造成问题

### 2.5 generate_cmc_masks_v2_soft 函数默认参数不一致

```
函数定义: def generate_cmc_masks_v2_soft(img, cmc_patch_size=16, ...)
参数解析: parser.add_argument('--cmc_patch_size', type=int, default=8)
```

函数默认 cmc_patch_size=16，但 argparse 默认是 8。好在本次实验通过 argparse 显式传了 cmc_patch_size=8，所以实际用的是 8。

---

## 3. 各类别 Dice 详细对比 (10%)

| 类别 | NA_v2 10% | BCP 10% | 差距 | 含义 |
|:----:|:---------:|:-------:|:----:|:----|
| BG | 0.5602 | 0.7288 | -0.1686 | 🔻 NA_v2 背景差 |
| AA (升主动脉) | 0.9021 | 0.9172 | -0.0151 | ≈ 接近 |
| LA (左心房) | **0.2380** | **0.3531** | **-0.1151** | 🔻 最难类别的差距 |
| LV (左心室) | 0.5526 | 0.5236 | +0.0290 | ✅ NA_v2 更好 |
| RA (右心房) | 0.7890 | 0.7905 | -0.0015 | ≈ 几乎一样 |
| RV (右心室) | 0.6552 | 0.6811 | -0.0259 | ≈ 接近 |
| PA (肺动脉) | 0.7664 | 0.7443 | +0.0221 | ✅ NA_v2 更好 |

**关键发现**：NA_v2 在 LA（左心房）上特别差 (0.2380 vs 0.3531)，而 LA 通常较小且形态复杂，可能是 soft mask 过渡带太窄导致。

---

## 4. 建议

### 方案 A：与 BCP 公平对比
- 统一标注方式：要么都用"病人全标注"，要么都用"slice 随机标注"
- 统一 max_iterations：NA_v2 应该用 40000 而不是 30000

### 方案 B：优化 NA_v2 配置
1. **增大 soft_sigma**：0.3 → 0.5，让 CMC mask 过渡更平滑
2. **增大 cmc_patch_size**：8 → 16，让互补区域更大
3. **增加 max_iterations**：30000 → 40000，与 BCP 持平
4. **调高 conf_thresh**：0.90→0.70 的退火太快，可以 0.95→0.85

### 方案 C：解决退化问题
- **early stopping**：NA_v2 10% 在 iter 4600 就达到最佳，之后退化
- **降低 u_weight**：0.5 → 0.3，减少无标签数据的权重
- **增大 cmc_loss_weight**：1.0 → 1.5，让 CMC 在后期贡献更多
