import os
import math
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# ============================================================================
# DDP (Distributed Data Parallel) 兼容的 Batch Sampler
# ============================================================================

def _is_dist_avail_and_initialized():
    """检查分布式环境是否可用且已初始化"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def _get_rank():
    if not _is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def _get_world_size():
    if not _is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


class DistributedTwoStreamBatchSampler(Sampler):
    """
    分布式 Two-Stream Batch Sampler。
    
    TwoStreamBatchSampler 的 DDP 版本。
    每个 epoch 中：
    - primary_indices (有标签) 完整遍历一次 + shuffle
    - secondary_indices (无标签) 无限循环 + shuffle
    - 生成 batch = [primary_batch_size 个有标签] + [secondary_batch_size 个无标签]
    
    与 TwoStreamBatchSampler 的区别:
    1. 内部使用 DistributedSampler-like 的索引分片，每个 rank 只拿 part of primary
    2. secondary_indices 在各 rank 间独立 shuffle（不影响无标签数据的多样性）
    3. 确保总 batch_size 在 DDP 下被 world_size 分割 → 每卡实际 batch = total_batch / world_size
    
    Args:
        primary_indices: 有标签数据索引列表
        secondary_indices: 无标签数据索引列表
        batch_size: 全局 batch size（跨所有 GPU）
        secondary_batch_size: 全局无标签 batch size
        world_size: GPU 数量
        rank: 当前 GPU rank
    """
    def __init__(self, primary_indices, secondary_indices, batch_size,
                 secondary_batch_size, world_size=None, rank=None):
        self.world_size = world_size if world_size is not None else _get_world_size()
        self.rank = rank if rank is not None else _get_rank()

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices

        # 全局 batch size → 每卡的 batch size
        self.global_batch_size = batch_size
        self.global_secondary_batch_size = secondary_batch_size
        self.global_primary_batch_size = batch_size - secondary_batch_size

        # 每卡实际 batch size（必须能被 world_size 整除）
        assert batch_size % self.world_size == 0, \
            f"batch_size={batch_size} 必须能被 world_size={self.world_size} 整除"
        assert secondary_batch_size % self.world_size == 0, \
            f"secondary_batch_size={secondary_batch_size} 必须能被 world_size={self.world_size} 整除"

        self.per_gpu_batch_size = batch_size // self.world_size
        self.per_gpu_primary_batch_size = self.global_primary_batch_size // self.world_size
        self.per_gpu_secondary_batch_size = self.global_secondary_batch_size // self.world_size

        assert len(self.primary_indices) >= self.global_primary_batch_size > 0
        assert len(self.secondary_indices) >= self.global_secondary_batch_size > 0

        # 每个 rank 分片的 primary 索引数
        self.num_primary_per_rank = len(primary_indices) // self.world_size
        self.primary_start = self.rank * self.num_primary_per_rank
        self.primary_end = self.primary_start + self.num_primary_per_rank
        if self.rank == self.world_size - 1:
            # 最后一个 rank 拿剩余所有（处理不可整除的情况）
            self.primary_end = len(primary_indices)

        self.rank_primary_indices = primary_indices[self.primary_start:self.primary_end]
        print(f"[DDP-Sampler] rank={self.rank}, "
              f"primary: global={len(primary_indices)}, local={len(self.rank_primary_indices)}, "
              f"range=[{self.primary_start}:{self.primary_end}]")

    def __iter__(self):
        # 每个 rank 独立 shuffle primary indices
        primary_iter = iterate_once(self.rank_primary_indices)
        # secondary 无限循环（全量，各 rank 独立 shuffle）
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.per_gpu_primary_batch_size),
                    grouper(secondary_iter, self.per_gpu_secondary_batch_size))
        )

    def __len__(self):
        return len(self.rank_primary_indices) // self.per_gpu_primary_batch_size


class DistributedThreeStreamBatchSampler(Sampler):
    """
    分布式 Three-Stream Batch Sampler。
    
    ThreeStreamBatchSampler 的 DDP 版本。
    生成 batch = [primary_batch] + [secondary_batch] + [primary_batch]
    
    用于需要独立无标签支路和有标签支路的实验（如多 patch BCP）。
    """
    def __init__(self, primary_indices, secondary_indices, batch_size,
                 secondary_batch_size, world_size=None, rank=None):
        self.world_size = world_size if world_size is not None else _get_world_size()
        self.rank = rank if rank is not None else _get_rank()

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices

        self.global_batch_size = batch_size
        self.global_secondary_batch_size = secondary_batch_size
        self.global_primary_batch_size = batch_size - secondary_batch_size

        assert batch_size % self.world_size == 0, \
            f"batch_size={batch_size} 必须能被 world_size={self.world_size} 整除"
        assert secondary_batch_size % self.world_size == 0, \
            f"secondary_batch_size={secondary_batch_size} 必须能被 world_size={self.world_size} 整除"

        self.per_gpu_batch_size = batch_size // self.world_size
        self.per_gpu_primary_batch_size = self.global_primary_batch_size // self.world_size
        self.per_gpu_secondary_batch_size = self.global_secondary_batch_size // self.world_size

        assert len(self.primary_indices) >= self.global_primary_batch_size > 0
        assert len(self.secondary_indices) >= self.global_secondary_batch_size > 0

        self.num_primary_per_rank = len(primary_indices) // self.world_size
        self.primary_start = self.rank * self.num_primary_per_rank
        self.primary_end = self.primary_start + self.num_primary_per_rank
        if self.rank == self.world_size - 1:
            self.primary_end = len(primary_indices)

        self.rank_primary_indices = primary_indices[self.primary_start:self.primary_end]

        # ThreeStream 需要每个 batch 取两次 primary，确保够用
        assert len(self.rank_primary_indices) >= 2 * self.per_gpu_primary_batch_size, \
            f"rank={self.rank}: primary 索引数 {len(self.rank_primary_indices)} 不足，需要至少 {2 * self.per_gpu_primary_batch_size}"

    def __iter__(self):
        primary_iter = iterate_once(self.rank_primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch1 + secondary_batch + primary_batch2
            for (primary_batch1, secondary_batch, primary_batch2)
            in zip(grouper(primary_iter, self.per_gpu_primary_batch_size),
                    grouper(secondary_iter, self.per_gpu_secondary_batch_size),
                    grouper(primary_iter, self.per_gpu_primary_batch_size))
        )

    def __len__(self):
        return len(self.rank_primary_indices) // (2 * self.per_gpu_primary_batch_size)


def get_distributed_sampler(dataset, shuffle=True):
    """
    创建分布式 DataLoader 的 DistributedSampler。
    
    用法:
        sampler = get_distributed_sampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, ...)
    
    Args:
        dataset: PyTorch Dataset
        shuffle: 是否 shuffle
    
    Returns:
        DistributedSampler 实例（单 GPU 时返回 None）
    """
    if not _is_dist_avail_and_initialized() or _get_world_size() <= 1:
        return None
    return DistributedSampler(dataset, shuffle=shuffle)
