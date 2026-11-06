#!/usr/bin/env python3

import torchvision.transforms.v2 as T
from PIL import Image
from crce import const
import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mix_object_contexts, add_mask_noise, split=None, device=const.DEVICE):
        df = pd.read_csv(const.DATA_DIR / 'imagenet' / 'index.csv')
        self.device = device

        if split:
            self.df = df[df['split'] == split].reset_index()
            self.split = split
        else:
            self.df = df
            self.split = 'all'

        self.include_mask_classes = [int(x) for x in open(const.DATA_DIR / 'imagenet' / 'haqa_label_indices.txt', 'r').read().split(',')]

        self.transforms = self.get_transforms(self.split)
        self.cam_resize = T.Resize(const.CAM_SIZE, interpolation=T.InterpolationMode.BILINEAR)

        self.add_mask_noise = add_mask_noise
        if self.add_mask_noise:
            self.mask_noise_transform = T.RandomAffine(5, translate=(0.075, 0.075), scale=(0.7, 1.3))

        self.mix_object_contexts = mix_object_contexts and self.split == 'train'
        if self.mix_object_contexts:
            for y in self.df.label_idx.unique():
                differing_context_mask = self.df.label_idx != y
                self.df.loc[~differing_context_mask, 'alternate_context'] = self.df[differing_context_mask].sample(n=len(self.df) - differing_context_mask.sum(), replace=True)['path'].tolist()

    @classmethod
    def get_transforms(self, split):
        return T.Compose([T.PILToTensor(),
                          T.RandomResizedCrop(size=const.IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR, scale=(.7, 1.), ratio=(.75, 1.33)) if split == 'train' else T.Resize((232, 232), interpolation=T.InterpolationMode.BILINEAR),
                          T.RandomHorizontalFlip(p=0.5) if split == 'train' else T.CenterCrop(const.IMAGE_SIZE),
                          T.ToDtype(torch.float32, scale=True)]), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return (1 + int(self.mix_object_contexts)) * len(self.df)

    def __getitem__(self, idx):
        idx_mix_object_contexts = False
        if idx >= len(self.df):  # load the context-shifted image
            idx_mix_object_contexts = True
            assert self.mix_object_contexts
            idx -= len(self.df)

        img_path = const.DATA_DIR / 'imagenet' / self.df['path'].iloc[idx]
        mask_path = const.DATA_DIR / 'imagenet' / 'masks' / (self.df['path'].iloc[idx][7:-4] + 'png')
        label_idx = int(self.df['label_idx'][idx])

        X = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.add_mask_noise:
            mask = self.mask_noise_transform(mask)

        state = torch.get_rng_state()
        X = self.transforms[1](self.transforms[0](X)).to(self.device)

        torch.set_rng_state(state)
        mask = self.transforms[0](mask).to(self.device)

        if idx_mix_object_contexts:
            torch.set_rng_state(state)
            alternate_context = Image.open(const.DATA_DIR / 'imagenet' / self.df['alternate_context'].iloc[idx]).convert('RGB')
            alternate_context = self.transforms[1](self.transforms[0](alternate_context))
            X = X * mask + alternate_context * (1 - mask)

        mask = self.cam_resize(mask)[0]
        if mask.sum() == 0: mask[:] = 1.
        mask = (mask >= mask.mean()).to(torch.float)
        if len(self.include_mask_classes) > 0 and label_idx not in self.include_mask_classes:
            mask[:] = 1.

        y = torch.zeros(const.N_CLASSES, device=self.device)
        y[label_idx] = 1.

        return X, (mask, y)


def get_generators():
    const.N_CLASSES = 1000
    const.SPLITS[1] = 'val'

    datasets = [Dataset(split=split, mix_object_contexts=const.MIX_OBJECT_CONTEXTS, add_mask_noise=const.ADD_MASK_NOISE, device='cpu') for split in const.SPLITS[:2]]
    dataloaders = *[torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=const.N_WORKERS, batch_size=const.BATCH_SIZE, prefetch_factor=4) for dataset in datasets], None

    const.SPLITS[1] = 'valid'
    return dataloaders
