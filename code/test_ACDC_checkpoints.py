"""
批量测试指定实验的多个中间 checkpoint，记录每个 checkpoint 的测试指标。
用法：
    python test_ACDC_checkpoints.py --exp BCP_CMC_NA_v2_s0.3_label3_ps8_v2 --labelnum 3 --gpu 0
    python test_ACDC_checkpoints.py --exp BCP_CMC_NA_v2_s0.3_label3_ps16_v2 --labelnum 3 --gpu 0
"""
import argparse
import os
import re
import shutil

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--root_path', type=str, default='../data_split/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, required=True, help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--selected', type=str, default=None,
                    help='Comma-separated list of checkpoint iteration numbers (e.g. 16000,18000,23400). '
                         'If not set, all discovered checkpoints are tested.')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main) > 1:
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
    if np.sum(prediction == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
    return first_metric, second_metric, third_metric


def load_checkpoint(net, path):
    checkpoint = torch.load(path)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        print("  Loading checkpoint with 'net' key (pre_train format)")
        net.load_state_dict(checkpoint['net'])
    else:
        print("  Loading raw state_dict (self_train format)")
        net.load_state_dict(checkpoint)
    return net


def test_checkpoint(net, image_list, FLAGS):
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list, desc="  Testing", leave=False):
        first_metric, second_metric, third_metric = test_single_volume(case, net, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric


def parse_ckpt_name(filename):
    # iter_{iter}_dice_{dice}.pth
    m = re.match(r'iter_(\d+)_dice_([\d.]+)\.pth', filename)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def main():
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # 1) Get image list
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    print("Found {} test cases".format(len(image_list)))

    # 2) Get checkpoint directory
    snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    if not os.path.exists(snapshot_path):
        print("ERROR: snapshot path not found:", snapshot_path)
        return

    # 3) Discover checkpoint files
    ckpt_files = [f for f in os.listdir(snapshot_path) if f.startswith('iter_') and f.endswith('.pth')]
    ckpt_entries = []
    for f in sorted(ckpt_files):
        it, dice = parse_ckpt_name(f)
        if it is not None:
            ckpt_entries.append((it, dice, f))
    ckpt_entries.sort(key=lambda x: x[0])

    if not ckpt_entries:
        print("No checkpoint files found!")
        return

    # 4) Filter by --selected if provided
    if FLAGS.selected:
        selected_iters = [int(x.strip()) for x in FLAGS.selected.split(',')]
        ckpt_entries = [e for e in ckpt_entries if e[0] in selected_iters]

    print("Will test {} checkpoints in exp '{}'".format(len(ckpt_entries), FLAGS.exp))
    for it, dice, fname in ckpt_entries:
        print("  iter_{:05d}  (val_dice={:.4f})  -> {}".format(it, dice, fname))
    print()

    # 5) Initialize network once (reuse across checkpoints)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    # 6) Run tests
    results = []
    for it, val_dice, fname in ckpt_entries:
        ckpt_path = os.path.join(snapshot_path, fname)
        print("[iter_{:05d}] val_dice={:.4f}".format(it, val_dice))
        net = load_checkpoint(net, ckpt_path)
        net.eval()
        avg_metric = test_checkpoint(net, image_list, FLAGS)

        # avg_metric: [class1_metrics, class2_metrics, class3_metrics]
        # each is [dice, jc, hd95, asd]
        c1 = avg_metric[0]  # RV
        c2 = avg_metric[1]  # MYO
        c3 = avg_metric[2]  # LV
        mean_dice = (c1[0] + c2[0] + c3[0]) / 3
        mean_hd95 = (c1[2] + c2[2] + c3[2]) / 3
        mean_asd = (c1[3] + c2[3] + c3[3]) / 3

        print("  RV:   Dice={:.4f}, JC={:.4f}, HD95={:.4f}, ASD={:.4f}".format(*c1))
        print("  MYO:  Dice={:.4f}, JC={:.4f}, HD95={:.4f}, ASD={:.4f}".format(*c2))
        print("  LV:   Dice={:.4f}, JC={:.4f}, HD95={:.4f}, ASD={:.4f}".format(*c3))
        print("  MEAN: Dice={:.4f}, HD95={:.4f}, ASD={:.4f}".format(mean_dice, mean_hd95, mean_asd))
        print()

        results.append({
            'iter': it,
            'val_dice': val_dice,
            'rv_dice': c1[0], 'rv_jc': c1[1], 'rv_hd95': c1[2], 'rv_asd': c1[3],
            'myo_dice': c2[0], 'myo_jc': c2[1], 'myo_hd95': c2[2], 'myo_asd': c2[3],
            'lv_dice': c3[0], 'lv_jc': c3[1], 'lv_hd95': c3[2], 'lv_asd': c3[3],
            'mean_dice': mean_dice, 'mean_hd95': mean_hd95, 'mean_asd': mean_asd,
        })

    # 7) Summary table
    print("=" * 100)
    print("SUMMARY: {}".format(FLAGS.exp))
    print("=" * 100)
    print("{:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(
        'iter', 'val_dice', 'test_dice', 'RV', 'MYO', 'LV', 'HD95', 'ASD', 'RV_HD95', 'LV_HD95'))
    print("-" * 100)
    for r in results:
        print("{:>6} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f}".format(
            r['iter'], r['val_dice'], r['mean_dice'],
            r['rv_dice'], r['myo_dice'], r['lv_dice'],
            r['mean_hd95'], r['mean_asd'],
            r['rv_hd95'], r['lv_hd95']))
    print("=" * 100)

    # Also find best test dice
    best = max(results, key=lambda r: r['mean_dice'])
    print("\nBest checkpoint: iter_{} (val_dice={:.4f}) -> test_mean_dice={:.4f}".format(
        best['iter'], best['val_dice'], best['mean_dice']))

    # Save to file
    save_path = os.path.join(snapshot_path, 'checkpoint_test_results.txt')
    with open(save_path, 'w') as f:
        f.write("Results for {}\n\n".format(FLAGS.exp))
        header = "{:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}\n".format(
            'iter', 'val_dice', 'test_dice', 'RV', 'MYO', 'LV', 'HD95', 'ASD')
        f.write(header)
        f.write("-" * 80 + "\n")
        for r in results:
            line = "{:>6} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f} | {:>8.4f}\n".format(
                r['iter'], r['val_dice'], r['mean_dice'],
                r['rv_dice'], r['myo_dice'], r['lv_dice'],
                r['mean_hd95'], r['mean_asd'])
            f.write(line)
        f.write("\nBest checkpoint: iter_{} (val_dice={:.4f}) -> test_mean_dice={:.4f}\n".format(
            best['iter'], best['val_dice'], best['mean_dice']))
    print("Results saved to:", save_path)


if __name__ == '__main__':
    main()
