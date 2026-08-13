"""
MM (MMWHS) 测试脚本
=====================
基于 2D UNet 逐切片推理 + 3D 体积级评估（8类分割）

使用方法：
    # 测试 self_train 阶段
    python test_MM.py --gpu 1 --exp MM_BCP_CMC_NA_v2_s0.3_5_labeled --labelnum 5 --num_classes 8 --stage_name self_train
    
    # 测试 pre_train 阶段
    python test_MM.py --gpu 1 --exp MM_BCP_CMC_NA_v2_s0.3_5_labeled --labelnum 5 --num_classes 8 --stage_name pre_train
"""
import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--root_path', type=str, default='../data_split/MM', help='data root path')
parser.add_argument('--exp', type=str, default='MM_BCP_CMC_NA_v2_s0.3_5_labeled', help='experiment name')
parser.add_argument('--model', type=str, default='unet', help='model name')
parser.add_argument('--num_classes', type=int, default=8, help='output channels')
parser.add_argument('--labelnum', type=int, default=5, help='labeled data count')
parser.add_argument('--stage_name', type=str, default='self_train', help='pre_train or self_train')
FLAGS = parser.parse_args()

CLASS_NAMES = [
    "1-BG(blood pool)", "2-AA(ascending aorta)", "3-LA(left atrium)", "4-LV(left ventricle)",
    "5-RA(right atrium)", "6-RV(right ventricle)", "7-PA(pulmonary artery)", "8-TS(trunk septum)"
]


def calculate_metric_percase(pred, gt):
    """Calculate Dice/Jaccard/HD95/ASD for a single class"""
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 0.0, 0.0  # both empty → perfect
    else:
        return 0.0, 0.0, 0.0, 0.0  # FN or FP


def test_single_volume(case, net, test_save_path, FLAGS):
    """Slice-by-slice inference on a 3D MM volume"""
    print("\n  Loading {}...".format(case), flush=True)
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data", "{}.h5".format(case)), 'r')
    image = h5f['image'][:]   # (D, 512, 512)
    label = h5f['label'][:]   # (D, 512, 512)  values: 0-7
    print("  image shape: {}, label unique: {}".format(image.shape, np.unique(label)), flush=True)
    prediction = np.zeros_like(label)

    total_slices = image.shape[0]
    for ind in range(total_slices):
        if ind % 50 == 0:
            print("  slice {}/{}".format(ind, total_slices), flush=True)
        slice_2d = image[ind, :, :]
        x, y = slice_2d.shape
        # resize to 256x256 for network input
        slice_resized = zoom(slice_2d, (256.0 / x, 256.0 / y), order=0)
        input_t = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input_t)
            if isinstance(out_main, (list, tuple)):
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256.0, y / 256.0), order=0)
            prediction[ind] = pred

    # Per-class metrics
    metrics = []
    for c in range(1, FLAGS.num_classes):
        metrics.append(calculate_metric_percase(prediction == c, label == c))

    # Save predictions as NIfTI
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1.22, 1.22, 2.5))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
    prd_itk.SetSpacing((1.22, 1.22, 2.5))
    lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
    lab_itk.SetSpacing((1.22, 1.22, 2.5))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))

    return metrics


def Inference(FLAGS):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Read test list
    test_list_path = os.path.join(FLAGS.root_path, 'test.list')
    with open(test_list_path, 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    print("Found {} test volumes".format(len(image_list)))
    for v in image_list:
        print("  - {}".format(v))

    # Build model path: try multiple possible naming conventions
    possible_names = [
        "MM_{}_{}_labeled".format(FLAGS.exp, FLAGS.labelnum),  # MM_MM_BCP_..._5_labeled
        "{}_{}_labeled".format(FLAGS.exp, FLAGS.labelnum),     # MM_BCP_..._5_labeled
        "{}_{}_labeled".format("MM_" + FLAGS.exp, FLAGS.labelnum),  # MM_MM_BCP_..._5_labeled
    ]

    snapshot_path = None
    for name in possible_names:
        candidate = "./model/BCP/{}/{}".format(name, FLAGS.stage_name)
        if os.path.exists(os.path.join(candidate, '{}_best_model.pth'.format(FLAGS.model))):
            snapshot_path = candidate
            save_name = name
            print("Found model at: {}".format(candidate))
            break

    if snapshot_path is None:
        # Try listing available models
        print("ERROR: Could not find model checkpoint!")
        print("Tried paths in ./model/BCP/")
        if os.path.exists("./model/BCP/"):
            for d in os.listdir("./model/BCP/"):
                print("  - ./model/BCP/{}/".format(d))
        return

    test_save_path = "./model/BCP/{}/{}_predictions/".format(save_name, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    # Load network
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    checkpoint = torch.load(save_model_path)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print("Model loaded from {}".format(save_model_path))

    # Inference per volume
    total_metric = np.zeros((FLAGS.num_classes - 1, 4))
    for case in tqdm(image_list, desc="Testing MM volumes"):
        metric_list = test_single_volume(case, net, test_save_path, FLAGS)
        metric_array = np.array([list(m) for m in metric_list])
        total_metric += metric_array
        print("Case: {}, Dice={:.4f}±{:.4f}, HD95={:.2f}±{:.2f}".format(
            case,
            metric_array[:, 0].mean(), metric_array[:, 0].std(),
            metric_array[:, 2].mean(), metric_array[:, 2].std()))

    # Aggregate results
    avg_metric = total_metric / len(image_list)
    mean_dice = avg_metric[:, 0].mean()
    mean_hd95 = avg_metric[:, 2].mean()

    print("\n" + "=" * 70)
    print("MM Test Results ({} volumes)".format(len(image_list)))
    print("Model: {}".format(save_model_path))
    print("=" * 70)
    for c in range(FLAGS.num_classes - 1):
        print("Class {:d} ({:30s}): Dice={:.4f}, Jaccard={:.4f}, HD95={:.2f}, ASD={:.2f}".format(
            c + 1, CLASS_NAMES[c] if c < len(CLASS_NAMES) else "Class_{}".format(c + 1),
            avg_metric[c, 0], avg_metric[c, 1], avg_metric[c, 2], avg_metric[c, 3]))
    print("-" * 70)
    print("Mean Dice: {:.4f}".format(mean_dice))
    print("Mean HD95: {:.2f}".format(mean_hd95))
    print("=" * 70)

    # Save results
    result_path = os.path.join(test_save_path, "summary.txt")
    with open(result_path, 'w') as f:
        f.write("MM Test Results ({} volumes)\n".format(len(image_list)))
        f.write("Model: {}\n\n".format(save_model_path))
        for c in range(FLAGS.num_classes - 1):
            f.write("Class {}: Dice={:.4f}, Jaccard={:.4f}, HD95={:.2f}, ASD={:.2f}\n".format(
                c + 1, avg_metric[c, 0], avg_metric[c, 1], avg_metric[c, 2], avg_metric[c, 3]))
        f.write("\nMean Dice: {:.4f}\n".format(mean_dice))
        f.write("Mean HD95: {:.2f}\n".format(mean_hd95))
    print("Results saved to {}".format(result_path))


if __name__ == "__main__":
    Inference(FLAGS)
