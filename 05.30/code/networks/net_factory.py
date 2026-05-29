from networks.unet import UNet, UNet_2d
from networks.VNet import VNet
import torch.nn as nn


def net_factory(net_type="unet", in_chns=1, class_num=2, mode="train", tsne=0):
    """
    创建网络模型（创建在 CPU 上，由调用方决定是否 .cuda() 或 to(device)）。
    
    在 DDP 模式下，模型将在创建后统一被 DDP 包装，不应在这里 .cuda()。
    
    Args:
        net_type: "unet" 或 "VNet"
        in_chns: 输入通道数
        class_num: 输出类别数
        mode: "train" 或 "test"
        tsne: t-SNE 模式标志
    Returns:
        nn.Module 模型（在 CPU 上）
    """
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "VNet":
        if mode == "train":
            net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True)
        else:
            net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False)
    else:
        raise ValueError(f"Unsupported net_type: {net_type}")
    return net


def BCP_net(in_chns=1, class_num=2, ema=False):
    """
    创建 BCP 模型（UNet_2d），创建在 CPU 上。
    
    Args:
        in_chns: 输入通道数
        class_num: 输出类别数
        ema: 是否为 EMA 教师模型（冻结参数）
    Returns:
        nn.Module 模型（在 CPU 上）
    """
    net = UNet_2d(in_chns=in_chns, class_num=class_num)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

