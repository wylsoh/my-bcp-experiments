import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',   type=str,  default='../data_split/flare', help='数据集根目录')
parser.add_argument('--exp',         type=str,  default='BCP_CMC_v1_mutual',   help='实验名')
parser.add_argument('--model',       type=str,  default='VNet',                help='模型名')
parser.add_argument('--gpu',         type=str,  default='0',                   help='GPU编号')
parser.add_argument('--detail',      type=int,  default=1,                     help='是否打印每个样本的指标')
parser.add_argument('--nms',         type=int,  default=0,                     help='是否使用NMS后处理')
parser.add_argument('--labeled_num', type=int,  default=21,                    help='有标签样本数')
parser.add_argument('--stage_name',  type=str,  default='self_train',          help='self_train 或 pre_train')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path  = "./model/BCP/flare_{}_{}_labeled/{}".format(
    FLAGS.exp, FLAGS.labeled_num, FLAGS.stage_name)
test_save_path = "./model/BCP/flare_{}_{}_labeled/{}_predictions/".format(
    FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
num_classes = 14

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print("test_save_path:", test_save_path)

# flare 用 test.txt，每行是 case 目录名，数据格式 {root}/{case}/2022.h5
with open(os.path.join(FLAGS.root_path, 'test.txt'), 'r') as f:
    image_list = f.readlines()
image_list = [
    os.path.join(FLAGS.root_path, item.strip(), '2022.h5')
    for item in image_list
]


def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1,
                        class_num=num_classes, mode="test")
    save_model_path = os.path.join(snapshot_path,
                                   '{}_best_model.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))
    model.eval()

    avg_metric = test_all_case(
        model, image_list, num_classes=num_classes,
        patch_size=(64, 128, 128), stride_xy=32, stride_z=16,
        save_result=False, test_save_path=test_save_path,
        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_flare.py --exp BCP_CMC_v1_mutual_flare --labeled_num 21 --gpu 0
