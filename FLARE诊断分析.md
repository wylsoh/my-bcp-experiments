# FLARE 数据集训练问题诊断分析

## 概述

对比 `GuidedNet` 的 FLARE 代码 (`czs/GuidedNet-main/GuidedNet-main/predict_organ_flare.py`) 与 `my-bcp-experiments` 的 FLARE 代码 (`code/flare_BCP_CMC.py`, `code/utils/test_3d_patch.py`, `code/test_flare.py`)，发现以下 **1 个严重 Bug** 和 **3 个重要问题**。

---

## 🚨 BUG #1（致命）: `test_3d_patch.py` 多分类评估完全错误

### 文件
[`code/utils/test_3d_patch.py:195-206`](code/utils/test_3d_patch.py:195)

### 问题描述

[`test_single_case()`](code/utils/test_3d_patch.py:163) 中的代码：

```python
# 第 196-198 行：只取 channel index 1（第一个器官类别）！
y1, _ = model(test_patch)           
y = F.softmax(y1, dim=1)            # [1, 14, D, H, W]
y = y.cpu().data.numpy()[0, **1**, :, :, :]  # ← 只取通道 1！shape=[D,H,W]

# 第 199-204 行：y（shape=[D,H,W]）被 broadcast 到 score_map 的 14 个通道
# 导致所有 14 个通道的值完全相同！
score_map[:, xs:..., ys:..., zs:...] += y

# 第 206 行：二值阈值，不是 argmax！
label_map = (score_map[0] > 0.5).astype(np.int)  # ← 二值阈值，完全错误！
```

### 为什么是致命错误

- **FLARE 是 14 类分割**（背景0 + 13个器官），但 `test_single_case` **只取 channel 1（spleen）** 的 softmax 输出
- 由于 NumPy broadcasting，这个单通道的值被 broadcast 到所有 14 个通道，导致所有通道概率完全相同
- 最终预测是 `score_map[0] > 0.5` 的 **二值阈值**，而非 `argmax` 多分类
- [`calculate_metric_percase()`](code/utils/test_3d_patch.py:266) 使用 `metric.binary.dc()` 计算二值 Dice

**后果**：
1. 每 200 个 iter 的验证 Dice 值完全错误（实际测量的是 spleen 通道的概率 vs 所有器官前景的二值重叠）
2. **模型保存基于错误的指标**，可能保存了实际较差的 checkpoint
3. 最终测试报告的 Dice 毫无意义

### 对比 GuidedNet 的正确实现

[`czs/GuidedNet-main/GuidedNet-main/predict_organ_flare.py:131`](czs/GuidedNet-main/GuidedNet-main/predict_organ_flare.py:131)：

```python
# 保留所有通道
y = y[0, :, :, :, :]  # shape = [15, D, H, W] ← 保存所有 15 个通道

# 用 argmax 进行多分类决策
label_map = np.argmax(score_map, axis=0)  # ← 正确！
```

---

## 🚨 BUG #2: test_flare.py 测试数据路径错误

### 文件
[`code/test_flare.py:9`](code/test_flare.py:9)

### 问题描述

```python
parser.add_argument('--root_path', type=str, default='../data_split/flare', ...)
# ...
image_list = [os.path.join(FLAGS.root_path, item.strip(), '2022.h5') for item in image_list]
```

`root_path = '../data_split/flare'`，则测试数据路径为 `../data_split/flare/train_000/2022.h5`。

但实际数据在 `../data_split/flare/data/train_000/2022.h5`。

### 修复

应将 `--root_path` 默认值改为 `'../data_split/flare/data'`，与原训练脚本 [`flare_BCP_CMC.py:40`](code/flare_BCP_CMC.py:40) 保持一致：

```python
parser.add_argument('--root_path', type=str, default='../data_split/flare/data', ...)
```

---

## ⚠️ ISSUE #3: 验证直接在 test set 上进行，可能导致过拟合

### 文件
[`code/flare_BCP_CMC.py:331-351`](code/flare_BCP_CMC.py:331) 和 [`code/flare_BCP_CMC.py:493-513`](code/flare_BCP_CMC.py:493)

### 问题描述

`pre_train` 和 `self_train` 中每隔 200 个 iter 都在 **test set**（14 个测试 case）上做验证并据此保存最佳模型。这意味着：

1. **测试集被反复用于模型选择** → 模型间接"看到"了测试集
2. 实际性能应该用独立的验证集或最终一次性测试

### 建议

- 创建单独的验证集 `val.txt`（数据目录下已有 [`data_split/flare/data/val.txt`](data_split/flare/data/val.txt)）
- 训练时用 `val.txt` 做验证，最后用 `test.txt` 做最终评估

---

## ⚠️ ISSUE #4: 训练数据仅 378 例但有 420 条训练列表

### 文件
[`data_split/flare/train.txt`](data_split/flare/train.txt) 和 [`data_split/flare/data/train.txt`](data_split/flare/data/train.txt)

### 问题描述

- [`flare_BCP_CMC.py:47`](code/flare_BCP_CMC.py:47)：`parser.add_argument('--data_num', type=int, default=378)`
- 但 `train.txt` 中有 420 个 case，而实际数据目录只有约 290 个 `train_xxx` 文件夹

### 原因

FLARE22 官方训练集有 2200+ 例，420 是一个子集。但数据复制可能不完整，导致部分 case 没有对应的 H5 文件。

### 建议

确认 `data_num=378` 与训练列表中实际存在的 H5 文件数一致。

---

## ✅ 训练代码中正确的地方

1. **`dice_loss_3d` 使用 one-hot encoding + scatter_ 正确**：支持 14 类多分类 Dice 损失
2. **`F.cross_entropy` 正确**：native 支持多分类
3. **`generate_cmc_masks_3d` 逻辑正确**：互补掩码生成
4. **`TwoStreamBatchSampler` 正确**：有标签/无标签分离
5. **数据路径在训练脚本中正确**：`root_path='../data_split/flare/data'`

---

## 🔧 修复优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **P0** | BUG #1: 测试评估多分类错误 | 验证指标完全无效，模型选择错误 |
| **P1** | BUG #2: test_flare.py 路径错误 | 无法正确加载测试数据 |
| **P2** | ISSUE #3: 验证集过拟合风险 | 长期影响 |
| **P3** | ISSUE #4: data_num 与实际不符 | 潜在数据加载失败 |
