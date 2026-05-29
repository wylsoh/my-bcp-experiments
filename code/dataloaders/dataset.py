import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
import h5py
from scipy.ndimage import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision.transforms import *
from PIL.ImageEnhance import *
from PIL import Image
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter
import warnings
import math
from torch.distributions.beta import Beta
from torchvision.transforms import functional as F
from skimage import transform as sk_trans
import pdb
import augmentations
from augmentations.ctaugment import OPS


# =========================================================================
# 工具函数
# =========================================================================

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return image


def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
                ratio_2=1 / 0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)
        img_h, img_w = img.shape
        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)
            if x + erase_w <= img_w and y + erase_h <= img_h:
                break
        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)
        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0
    return img, mask


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _compute_intersection(box1, box2):
    i1, j1, h1, w1 = box1
    i2, j2, h2, w2 = box2
    x_overlap = max(0, min(j1 + w1, j2 + w2) - max(j1, j2))
    y_overlap = max(0, min(i1 + h1, i2 + h2) - max(i1, i2))
    return x_overlap * y_overlap


# =========================================================================
# 数据增强辅助函数
# =========================================================================

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


def random_crop(image, label):
    output_size = [256, 256]
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
    (w, h) = image.shape
    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    return image, label


# =========================================================================
# Dataset 类
# =========================================================================

class BaseDataSetsWithIndex(Dataset):
    """支持 ACDC 和 MM 数据集，按 index 划分有标签/无标签"""

    def __init__(self, base_dir=None, split='train', num=None, transform=None,
                 index=16, label_type=0):
        self._base_dir = base_dir
        self.index = index
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == 'train' and 'ACDC' in base_dir:
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5', '').replace('\n', '').strip() for item in self.sample_list]
            if label_type == 1:
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'train' and 'MM' in base_dir:
            with open(self._base_dir + '/train_slices.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5\n', '') for item in self.sample_list]
            if label_type == 1:
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('.h5', '').replace('\n', '').strip() for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num - index]
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
        if self.split == "train" and self.transform is not None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class BaseDataSets(Dataset):
    """主训练数据集，支持 ACDC 和 MM，支持 CTAugment"""

    def __init__(self, base_dir=None, split="train", num=None, transform=None,
                 ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(ops_strong), \
            "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == 'train' and 'ACDC' in base_dir:
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5', '').replace('\n', '').strip() for item in self.sample_list]

        elif self.split == 'train' and 'MM' in base_dir:
            with open(self._base_dir + '/train_slice.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5\n', '') for item in self.sample_list]

        elif self.split == 'val' and 'ACDC' in base_dir:
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('.h5', '').replace('\n', '').strip() for item in self.sample_list]

        elif self.split == 'val' and 'MM' in base_dir:
            with open(self._base_dir + '/test.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('.h5', '').replace('\n', '').strip() for item in self.sample_list]

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
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample['case'] = case
        return sample


# =========================================================================
# LA 数据集（第二个文件原有）
# =========================================================================

class LAHeart(Dataset):
    """LA Dataset"""

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(
            self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =========================================================================
# Transform / Augmentation 类
# =========================================================================

class WeakStrongAugment(object):
    """返回弱增强和强增强的图像对"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_strong, label_strong = cutout_gray(image, label, p=0.5)
        image_strong = color_jitter(image_strong).type("torch.FloatTensor")
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        label_strong = torch.from_numpy(label_strong.astype(np.uint8))
        sample = {
            "image": image,
            "image_strong": image_strong,
            "label": label,
            "label_strong": label_strong
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class CTATransform(object):
    """CTA 数据增强 Transform，用于半监督训练"""

    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image": image,
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
            "label": label
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            image, label = random_crop(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        (w, h) = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        return {'image': image, 'label': label}


class RandomCrop(object):
    """随机裁剪，支持 2D（ACDC/MM）和 3D（LA）"""

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # 3D 情况（LA）
        if len(image.shape) == 3:
            if (label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]
                    or label.shape[2] <= self.output_size[2]):
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
            return {'image': image, 'label': label}

        # 2D 情况（ACDC / MM）
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        return {'image': image, 'label': label}


class RandomCropBatch(object):
    """对一个 batch 内每个样本分别随机裁剪"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        new_image = []
        new_label = []
        for i in range(image.shape[0]):
            cur_image = image[i]
            cur_label = label[i]
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                cur_image = np.pad(cur_image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
                cur_label = np.pad(cur_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            (w, h) = image[0].shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            cur_label = cur_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            cur_image = cur_image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            new_image.append(cur_image)
            new_label.append(cur_label)
        new_image = torch.FloatTensor(np.array(new_image))
        new_label = torch.FloatTensor(np.array(new_label))
        return {'image': new_image, 'label': new_label}


class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 支持 batch（4D）和单样本（2D/3D）
        if isinstance(image, (torch.Tensor, np.ndarray)) and len(image.shape) == 4:
            for i in range(image.shape[0]):
                cur_img = image[i]
                cur_label = label[i]
                k = np.random.randint(0, 4)
                cur_img = np.rot90(cur_img, k)
                cur_label = np.rot90(cur_label, k)
                axis = np.random.randint(0, 2)
                cur_img = np.flip(cur_img, axis=axis).copy()
                cur_label = np.flip(cur_label, axis=axis).copy()
                image[i] = torch.FloatTensor(cur_img)
                label[i] = torch.FloatTensor(cur_label)
        else:
            image, label = random_rot_flip(image, label)
        return {'image': image, 'label': label}


class RandomRot(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1, p=0.5):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 3D 情况（LA）
        if len(image.shape) == 3:
            if np.random.uniform(0, 1) > self.p:
                return sample
            noise = np.clip(self.sigma * np.random.randn(*image.shape),
                            -2 * self.sigma, 2 * self.sigma) + self.mu
            image = image + noise
            return {'image': image, 'label': label}
        # 2D batch 情况
        if np.random.uniform(0, 1) > self.p:
            return sample
        new_image = []
        sigma = random.uniform(0.15, 1.15)
        for i in range(image.shape[0]):
            image_i = ToPILImage()(image[i, 0, :, :]).filter(ImageFilter.GaussianBlur(radius=sigma))
            new_image.append(np.array(image_i) / 255)
        image = torch.tensor(np.array(new_image), dtype=torch.float64)
        return {'image': image, 'label': label}


class RandomColorJitter(object):
    def __init__(self, color=(0.4, 0.4, 0.4, 0.1), p=0.1):
        self.color = color
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.p:
            return sample
        image, label = sample['image'], sample['label']
        for j in range(image.shape[0]):
            image[j, :, :, :] = ColorJitter(
                brightness=self.color[0],
                contrast=self.color[1],
                saturation=self.color[2],
                hue=self.color[3])(image[j, :, :, :])
        return {'image': image, 'label': label}


class BrightnessTransform(object):
    def __init__(self, p=0.5, mu=0.8, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0, 1) > self.p:
            return sample
        image, label = sample['image'], sample['label']
        for j in range(image.shape[0]):
            image[j, :, :, :] = torch.clamp(self.mu * image[j, :, :, :] + self.sigma, min=0.0, max=1.0)
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # 3D
        if len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # 2D
        else:
            image = image.reshape(1, *image.shape).astype(np.float64)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(sample['label']).long()}


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 3D（LA）
        if len(image.shape) == 3:
            label = label.astype(np.bool_)
            image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
            label = sk_trans.resize(label, self.output_size, order=0)
            return {'image': image, 'label': label}
        # 2D
        pose = transforms.Compose([transforms.Resize((1, 256, 256))])
        image = pose(image)
        return {'image': image, 'label': label}


# =========================================================================
# CustomMultiCropping（来自第一个文件）
# =========================================================================

class CustomMultiCropping(object):
    """自定义多尺度裁剪策略，生成大裁剪和小裁剪"""

    def __init__(self, size_large=160, scale_large=(0.2, 1.0),
                 size_small=96, scale_small=(0.05, 0.14), N_large=2, N_small=4,
                 ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR,
                 condition_small_crops_on_key=True):
        self.size_large = (size_large, size_large) if not isinstance(size_large, (tuple, list)) else size_large
        self.size_small = (size_small, size_small) if not isinstance(size_small, (tuple, list)) else size_small
        if (scale_large[0] > scale_large[1]) or (scale_small[0] > scale_small[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        self.interpolation = interpolation
        self.scale_large = scale_large
        self.scale_small = scale_small
        self.N_large = N_large
        self.N_small = N_small
        self.ratio = ratio
        self.condition_small_crops_on_key = condition_small_crops_on_key

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = _get_image_size(img)
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def get_params_conditioned(self, img, scale, ratio, constraint):
        width, height = _get_image_size(img)
        area = height * width
        for _ in range(10):
            rand_scale = random.uniform(*scale)
            target_area = rand_scale * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                intersection = _compute_intersection((i, j, h, w), constraint)
                if intersection >= 0.1 * target_area:
                    return i, j, h, w
        return self.get_params(img, scale, ratio)

    def __call__(self, img):
        multi_crop = []
        multi_crop_params = []
        for ii in range(self.N_large):
            i, j, h, w = self.get_params(img, self.scale_large, self.ratio)
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_large, self.interpolation))
        for ii in range(self.N_small):
            if not self.condition_small_crops_on_key:
                i, j, h, w = self.get_params(img, self.scale_small, self.ratio)
            else:
                i, j, h, w = self.get_params_conditioned(
                    img, self.scale_small, self.ratio, multi_crop_params[self.N_large - 1])
            multi_crop_params.append((i, j, h, w))
            multi_crop.append(F.resized_crop(img, i, j, h, w, self.size_small, self.interpolation))
        return multi_crop, multi_crop_params

    def __repr__(self):
        return (f"{self.__class__.__name__}(size_large={self.size_large}, "
                f"scale_large={self.scale_large}, size_small={self.size_small}, "
                f"scale_small={self.scale_small}, ratio={self.ratio}, "
                f"condition_small_crops_on_key={self.condition_small_crops_on_key})")


# =========================================================================
# Sampler 工具类
# =========================================================================

class TwoStreamBatchSampler(Sampler):
    """同时迭代有标签和无标签两组索引"""

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
    """三流采样器（第二个文件原有）"""

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
    """Collect data into fixed-length chunks or blocks"""
    args = [iter(iterable)] * n
    return zip(*args)


def worker_init_fn(worker_id):
    random.seed(100 + worker_id)