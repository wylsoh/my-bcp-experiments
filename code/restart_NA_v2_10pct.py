"""
直接启动 NA_v2 10% self_train，跳过 pre_train
"""
import logging, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 手动构造 args
class Args:
    root_path = '../data_split/MM'
    exp = 'MM_BCP_CMC_NA_v2_s0.3'
    model = 'unet'
    pre_iterations = 10000
    max_iterations = 30000
    batch_size = 24
    deterministic = 1
    base_lr = 0.01
    patch_size = [256, 256]
    seed = 1337
    num_classes = 8
    labeled_bs = 12
    labelnum = 10
    u_weight = 0.5
    gpu = '1'
    consistency = 0.1
    consistency_rampup = 200.0
    magnitude = 6.0
    s_param = 6
    cmc_patch_size = 16
    cmc_warmup_iter = 5000
    cmc_init_shared = 0.4
    cmc_loss_weight = 1.0
    cmc_mutual_weight = 0.5
    cmc_mutual_conf_thresh = 0.75
    conf_thresh_init = 0.90
    conf_thresh_final = 0.70
    cmc_soft_sigma = 0.3

args = Args()
dataset_name = 'MM'
pre_snapshot_path = f"./model/BCP/{dataset_name}_{args.exp}_{args.labelnum}_labeled/pre_train"
self_snapshot_path = f"./model/BCP/{dataset_name}_{args.exp}_{args.labelnum}_labeled/self_train"

for p in [pre_snapshot_path, self_snapshot_path]:
    os.makedirs(p, exist_ok=True)

# 初始化 logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from BCP_CMC_v1_NA_v2_soft_mask import self_train
logging.info("=== Direct restart: skipping pre_train ===")
logging.info(str(args.__dict__))
self_train(args, pre_snapshot_path, self_snapshot_path)
