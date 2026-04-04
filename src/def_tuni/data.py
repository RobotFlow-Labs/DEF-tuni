"""Dataset loaders for FMB, PST900, and CART RGB-T datasets."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random
import numbers
import collections.abc


# ---------------------------------------------------------------------------
# Augmentations (from reference toolbox/datasets/augmentations.py)
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        for key in sample:
            if key in ['image']:
                sample[key] = TF.resize(sample[key], self.size, interpolation=InterpolationMode.BILINEAR)
            else:
                sample[key] = TF.resize(sample[key], self.size, interpolation=InterpolationMode.NEAREST)
        return sample


class RandomCrop:
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        img = sample['image']
        if self.pad_if_needed and img.size[0] < self.size[1]:
            for key in sample:
                sample[key] = TF.pad(sample[key], (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < self.size[0]:
            for key in sample:
                sample[key] = TF.pad(sample[key], (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        w, h = sample['image'].size
        th, tw = self.size
        if w == tw and h == th:
            return sample
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        for key in sample:
            sample[key] = TF.crop(sample[key], i, j, th, tw)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample:
                sample[key] = TF.hflip(sample[key])
        return sample


class RandomScale:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        w, h = sample['image'].size
        scale = random.uniform(self.scale[0], self.scale[1])
        size = (int(round(h * scale)), int(round(w * scale)))
        for key in sample:
            if key in ['image']:
                sample[key] = TF.resize(sample[key], size, interpolation=InterpolationMode.BILINEAR)
            else:
                sample[key] = TF.resize(sample[key], size, interpolation=InterpolationMode.NEAREST)
        return sample


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample):
        img = sample['image']
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = TF.adjust_brightness(img, factor)
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = TF.adjust_contrast(img, factor)
        if self.saturation > 0:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = TF.adjust_saturation(img, factor)
        sample['image'] = img
        return sample


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

RGB_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

THERMAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
])


# ---------------------------------------------------------------------------
# FMB Dataset
# ---------------------------------------------------------------------------

FMB_CLASS_WEIGHT = np.array([
    10.2249, 9.6609, 32.8497, 6.0635, 48.1396, 44.9108, 4.4491, 3.1748,
    43.9271, 15.9236, 43.1266, 44.8469, 48.6038, 50.4826, 27.1057
])


class FMB(data.Dataset):
    def __init__(self, root: str, mode: str = 'train', do_aug: bool = True,
                 crop_size=(600, 800), scale_range=(0.5, 2.0)):
        assert mode in ['train', 'val', 'test', 'test_day', 'test_night']
        self.root = root
        self.mode = mode
        self.do_aug = do_aug
        self.n_classes = 15
        self.class_weight = torch.from_numpy(FMB_CLASS_WEIGHT).float()

        self.aug = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomHorizontalFlip(0.5),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True),
        ])

        with open(os.path.join(root, f'{mode}.txt'), 'r') as f:
            self.infos = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        name = self.infos[index]
        image = Image.open(os.path.join(self.root, self.mode, 'Visible', name))
        thermal = Image.open(os.path.join(self.root, self.mode, 'Infrared', name)).convert('RGB')
        label = Image.open(os.path.join(self.root, self.mode, 'Label', name))

        sample = {'image': image, 'thermal': thermal, 'label': label}

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)

        sample['image'] = RGB_TRANSFORM(sample['image'])
        sample['thermal'] = THERMAL_TRANSFORM(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = name.split('/')[-1]
        return sample


# ---------------------------------------------------------------------------
# PST900 Dataset
# ---------------------------------------------------------------------------

PST900_CLASS_WEIGHT = np.array([1.45369372, 44.2457428, 31.66502391, 46.40709901, 30.13909209])


class PST900(data.Dataset):
    def __init__(self, root: str, mode: str = 'train', do_aug: bool = True,
                 crop_size=(640, 1280), scale_range=(0.5, 2.0)):
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        self.do_aug = do_aug
        self.n_classes = 5
        self.class_weight = torch.from_numpy(PST900_CLASS_WEIGHT).float()

        self.resize = Resize(crop_size)
        self.aug = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomHorizontalFlip(0.5),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True),
        ])

        with open(os.path.join(root, f'{mode}.txt'), 'r') as f:
            self.infos = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        name = self.infos[index]
        image = Image.open(os.path.join(self.root, self.mode, 'rgb', name + '.png'))
        thermal = Image.open(os.path.join(self.root, self.mode, 'thermal', name + '.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, self.mode, 'labels', name + '.png'))

        sample = {'image': image, 'thermal': thermal, 'label': label}
        sample = self.resize(sample)

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)

        sample['image'] = RGB_TRANSFORM(sample['image'])
        sample['thermal'] = THERMAL_TRANSFORM(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = name.split('/')[-1]
        return sample


# ---------------------------------------------------------------------------
# CART Dataset
# ---------------------------------------------------------------------------

CART_CLASS_WEIGHT = np.array([
    50.2527, 50.4935, 4.8389, 6.3680, 24.0135, 26.3811,
    9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928
])


def _preprocess_thermal_16bit(image_array):
    p1, p99 = np.percentile(image_array, (1, 99))
    rescaled = np.clip((image_array - p1) / (p99 - p1 + 1e-6), 0, 1)
    uint8 = (rescaled * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(uint8))


class CART(data.Dataset):
    def __init__(self, root: str, mode: str = 'train', do_aug: bool = True,
                 crop_size=(512, 512), scale_range=(1.0, 1.5)):
        assert mode in ['train', 'val', 'test', 'trainval']
        self.root = root
        self.mode = mode
        self.do_aug = do_aug
        self.n_classes = 12
        self.class_weight = torch.from_numpy(CART_CLASS_WEIGHT).float()

        self.aug = Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomHorizontalFlip(0.5),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True),
        ])

        self.rgb_infos = []
        self.t_infos = []
        self.label_infos = []

        with open(os.path.join(root, 'rgbt_splits', f'rgb_{mode}.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                self.rgb_infos.append(parts[0].strip())
                self.label_infos.append(parts[1].strip())

        with open(os.path.join(root, 'rgbt_splits', f'thermal16_{mode}.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                self.t_infos.append(parts[0].strip())

    def __len__(self):
        return len(self.rgb_infos)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.rgb_infos[index]))
        thermal_raw = cv2.imread(os.path.join(self.root, self.t_infos[index]), cv2.IMREAD_UNCHANGED)
        thermal = _preprocess_thermal_16bit(thermal_raw).convert('RGB')
        label = Image.open(os.path.join(self.root, self.label_infos[index]))

        sample = {'image': image, 'thermal': thermal, 'label': label}

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)

        sample['image'] = RGB_TRANSFORM(sample['image'])
        sample['thermal'] = THERMAL_TRANSFORM(sample['thermal'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = self.label_infos[index].split('/')[-1]
        return sample


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "fmb": FMB,
    "pst900": PST900,
    "cart": CART,
}

DATASET_DEFAULTS = {
    "fmb": dict(n_classes=15, crop_size=(600, 800), scale_range=(0.5, 2.0)),
    "pst900": dict(n_classes=5, crop_size=(640, 1280), scale_range=(0.5, 2.0)),
    "cart": dict(n_classes=12, crop_size=(512, 512), scale_range=(1.0, 1.5)),
}


def get_dataset(name: str, root: str, mode: str = 'train', **kwargs):
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key](root=root, mode=mode, **kwargs)
