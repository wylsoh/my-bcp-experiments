"""
统一指标计算模块
支持 Dice, Jaccard(IoU), HD95, ASD 四个指标
"""
import numpy as np
from medpy import metric


def calculate_metric_percase(pred, gt):
    """
    计算单个样本的四个指标：Dice, Jaccard, HD95, ASD

    Args:
        pred: numpy array, 二值预测
        gt: numpy array, 二值标签
    Returns:
        dice, jaccard, hd95, asd
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jaccard, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0.0, 0.0, 100.0, 100.0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 0.0, 100.0, 100.0
    else:
        # 都为空，视为完美匹配
        return 1.0, 1.0, 0.0, 0.0


def test_single_volume_all_metrics(image, label, net, classes, patch_size=[256, 256]):
    """
    测试单个 volume，返回每个类别的 [dice, jaccard, hd95, asd]

    Args:
        image: [1, 1, slices, H, W] 或 [slices, 1, H, W]
        label: [1, slices, H, W] 或 [slices, H, W]
        net: 分割模型
        classes: 类别数
        patch_size: 输入尺寸
    Returns:
        metric_list: numpy array, shape [classes-1, 4]
                     每行为 [dice, jaccard, hd95, asd]
    """
    import torch
    import torch.nn.functional as F

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_input = image[ind, :, :]
        x, y = slice_input.shape[0], slice_input.shape[1]

        slice_input = torch.from_numpy(slice_input).unsqueeze(0).unsqueeze(0).float().cuda()

        # 如果输入尺寸与 patch_size 不同，进行 resize
        if x != patch_size[0] or y != patch_size[1]:
            slice_input = F.interpolate(slice_input, size=patch_size, mode='bilinear', align_corners=True)

        net.eval()
        with torch.no_grad():
            out = net(slice_input)
            if isinstance(out, tuple):
                out = out[0]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if x != patch_size[0] or y != patch_size[1]:
                pred = F.interpolate(
                    torch.from_numpy(out).unsqueeze(0).unsqueeze(0).float(),
                    size=(x, y), mode='nearest'
                ).squeeze().numpy()
            else:
                pred = out

        prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(
                prediction == i,
                label == i
            )
        )
    return metric_list  # shape: [classes-1, 4], 每行=[dice, jaccard, hd95, asd]


def log_validation_metrics(metric_list, num_classes, iter_num, writer, logging, prefix="val"):
    """
    统一格式记录验证指标到 TensorBoard 和 日志

    Args:
        metric_list: numpy array [num_classes-1, 4], 列=[dice, jaccard, hd95, asd]
        num_classes: 类别总数
        iter_num: 当前迭代
        writer: TensorBoard writer
        logging: logging 模块
        prefix: 日志前缀
    """
    metric_names = ['dice', 'jaccard', 'hd95', 'asd']

    for class_i in range(num_classes - 1):
        for m_idx, m_name in enumerate(metric_names):
            writer.add_scalar(
                f'{prefix}/class{class_i + 1}_{m_name}',
                metric_list[class_i, m_idx],
                iter_num
            )

    # 计算各指标的均值
    mean_dice = np.mean(metric_list[:, 0])
    mean_jaccard = np.mean(metric_list[:, 1])
    mean_hd95 = np.mean(metric_list[:, 2])
    mean_asd = np.mean(metric_list[:, 3])

    writer.add_scalar(f'{prefix}/mean_dice', mean_dice, iter_num)
    writer.add_scalar(f'{prefix}/mean_jaccard', mean_jaccard, iter_num)
    writer.add_scalar(f'{prefix}/mean_hd95', mean_hd95, iter_num)
    writer.add_scalar(f'{prefix}/mean_asd', mean_asd, iter_num)

    logging.info(
        f'iteration {iter_num} : '
        f'mean_dice={mean_dice:.4f}, '
        f'mean_jaccard={mean_jaccard:.4f}, '
        f'mean_hd95={mean_hd95:.2f}, '
        f'mean_asd={mean_asd:.2f}'
    )

    # 逐类别详细输出
    for class_i in range(num_classes - 1):
        logging.info(
            f'  class {class_i + 1}: '
            f'dice={metric_list[class_i, 0]:.4f}, '
            f'jaccard={metric_list[class_i, 1]:.4f}, '
            f'hd95={metric_list[class_i, 2]:.2f}, '
            f'asd={metric_list[class_i, 3]:.2f}'
        )

    return mean_dice, mean_jaccard, mean_hd95, mean_asd
