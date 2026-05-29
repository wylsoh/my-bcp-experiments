from networks.unet_3D import unet_3D
from networks.VNet import VNet
from networks.unet_3D_gmm import unet_gmm
from networks.Vnet_ss import VNet
from networks.UNet3D_contrastive import UNet3D

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "guidedNet":
        net = unet_gmm(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "VNet_Train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_Test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    else:
        net = None
    return net


def net_factory_3d_dycon(net_type="unet_3D", in_chns=1, class_num=2, scaler=4, use_aspp=False):
    if net_type == "unet_3D":
        net = UNet3D(in_channels=in_chns, n_classes=class_num, scale_factor=scaler, use_aspp=use_aspp).cuda() # .cuda()
    # elif net_type == "vnet":
    #     net = VNet(n_channels=in_chns, n_classes=class_num, scale_factor=scaler, has_dropout=True, use_aspp=use_aspp) # .cuda()
    else:
        net = None
    return net
#
# net1 = net_factory_3d(net_type='VNet_Train', in_chns=1, class_num=14).cuda()
# print(net1)