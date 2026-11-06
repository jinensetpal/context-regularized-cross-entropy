#!/usr/bin/env python3

"""
The code for the Hard-ImageNet dataset was adapted from the Hard-ImageNet codebase:
    https://github.com/mmoayeri/HardImageNet/blob/a39e3f996f70eb538ff71829b80781bac617a2d4/datasets/hard_imagenet.py
and
    https://github.com/mmoayeri/HardImageNet/blob/a39e3f996f70eb538ff71829b80781bac617a2d4/augmentations.py
"""

import torchvision.transforms.v2 as T
from PIL import Image
import pickle
import torch
import glob
import os

from crce.utils import DataLoader
from crce import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='val', device=const.DEVICE, eval_mode=False):
        self.split = split
        self.device = device
        self.num_classes = 15

        self._IMAGENET_ROOT = const.DATA_DIR / 'imagenet' / 'images'
        self._MASK_ROOT = const.DATA_DIR / 'hardimagenet'

        with open(self._MASK_ROOT/'meta/idx_to_wnid.pkl', 'rb') as f:
            idx_to_wnid = pickle.load(f)
        self.wnid_to_idx = dict({v: k for k, v in idx_to_wnid.items()})

        with open(self._MASK_ROOT/'meta/hard_imagenet_idx.pkl', 'rb') as f:
            self.inet_idx = pickle.load(f)

        self.collect_mask_paths()
        self.eval_mode = eval_mode

    def standard_resize_center_crop(self, img, mask, resize_shape=const.IMAGE_SIZE):
        t = T.Compose([T.Resize(256), T.CenterCrop(resize_shape), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        img, mask = [t(x) for x in [img, mask]]
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], axis=0)
        if mask.shape[0] == 1:
            mask = torch.cat([mask, mask, mask], axis=0)
        return img, mask

    def map_wnid_to_label(self, wnid):
        ind = self.wnid_to_idx[wnid]
        return self.inet_idx.index(ind)

    def collect_mask_paths(self):
        self.mask_paths = glob.glob((self._MASK_ROOT / self.split / '*').as_posix())

    def __getitem__(self, ind):
        mask_path = self.mask_paths[ind]
        mask_path_suffix = mask_path.split('/')[-1]
        wnid = mask_path_suffix.split('_')[0]
        fname = mask_path_suffix[len(wnid)+1:]

        img_path = os.path.join(self._IMAGENET_ROOT, self.split, wnid, fname)
        img, mask = [Image.open(p) for p in [img_path, mask_path]]

        img, mask = self.standard_resize_center_crop(img, mask)

        if img.shape[0] > 3:  # weird bug
            img, mask = [x[:3] for x in [img, mask]]

        class_ind = self.map_wnid_to_label(wnid)
        mask[mask > 0] = 1

        if self.eval_mode:
            return img.to(self.device), mask.to(self.device), class_ind

        mask = T.functional.resize(mask, const.CAM_SIZE, interpolation=T.InterpolationMode.NEAREST_EXACT)[0]
        y = torch.zeros(self.num_classes, device=self.device)
        y[class_ind] = 1

        return img.to(self.device), (mask.to(self.device), y.to(self.device))

    def __len__(self):
        return len(self.mask_paths)


def get_generators():
    const.SPLITS[1] = 'val'

    datasets = [Dataset(split=split, device='cpu') for split in const.SPLITS[:2]]
    dataloaders = *[DataLoader(dataset, shuffle=True, num_workers=const.N_WORKERS,
                               pin_memory=True, batch_size=const.BATCH_SIZE) for dataset in datasets], None

    const.SPLITS[1] = 'valid'
    return dataloaders
