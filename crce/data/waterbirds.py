#!/usr/bin/env python3

from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as T
from crce.utils import DataLoader
from crce import const
import pandas as pd
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split=None, state='training', device='cpu', transforms=None, add_mask_noise=False, mix_object_contexts=False):
        df = pd.read_csv(const.DATA_DIR / 'waterbirds' / 'metadata.csv')
        self.encodings = {'place': ['land', 'water'],
                          'label': ['landbird', 'waterbird'],
                          'split': ['train', 'valid', 'test']}

        if split:
            self.df = df[df['split'] == self.encodings['split'].index(split)].reset_index()
            self.split = split
        else:
            self.df = df
            self.split = 'all'

        if self.split in ['train', 'all']:  # Waterbirds-100; ignore spurious groups
            self.df = self.df[self.df['y'] == self.df['place']].reset_index()

        self.cam_resize = T.Resize(const.CAM_SIZE, interpolation=T.InterpolationMode.BILINEAR)
        self.image_resize = T.Resize(const.IMAGE_SIZE, interpolation=T.InterpolationMode.NEAREST_EXACT)
        self.transforms = transforms
        self.mask_folder = 'segmentations'
        self.add_mask_noise = add_mask_noise
        self.device = device
        self.state = state

        if self.add_mask_noise:
            self.mask_noise_transform = T.RandomAffine(5, translate=(0.075, 0.075), scale=(0.7, 1.3))

        self.mix_object_contexts = mix_object_contexts and self.split == 'train'
        if self.mix_object_contexts:
            for y in self.df.y.unique():
                differing_context_mask = self.df.y != y
                self.df.loc[~differing_context_mask, 'alternate_context'] = self.df[differing_context_mask].sample(n=len(self.df) - differing_context_mask.sum(), replace=True)['img_filename'].tolist()

    def __len__(self):
        return (1 + int(self.mix_object_contexts)) * len(self.df)

    def __getitem__(self, idx):
        if self.transforms is not None:

            idx_mix_object_contexts = False
            if idx >= len(self.df):  # load the context-shifted image
                idx_mix_object_contexts = True
                assert self.mix_object_contexts
                idx -= len(self.df)

            X = torchvision.io.read_image((const.DATA_DIR / 'waterbirds' / 'images' / self.df['img_filename'].iloc[idx]).as_posix())
            heatmap = torchvision.io.read_image((const.DATA_DIR / 'waterbirds' / self.mask_folder / f'{self.df["img_filename"].iloc[idx][:-3]}png').as_posix())

            if self.add_mask_noise:
                heatmap = self.mask_noise_transform(heatmap)

            state = torch.get_rng_state()
            X = self.transforms[1](self.transforms[0](X))

            torch.set_rng_state(state)
            heatmap = (self.transforms[0](heatmap)[0] > .2).to(torch.float)

            if idx_mix_object_contexts:
                torch.set_rng_state(state)
                alternate_context = torchvision.io.read_image((const.DATA_DIR / 'waterbirds' / 'images' / self.df['alternate_context'].iloc[idx]).as_posix())
                alternate_context = self.transforms[1](self.transforms[0](alternate_context))
                X = X * heatmap + alternate_context * (1 - heatmap)
            heatmap = (self.cam_resize(heatmap[None,])[0] > .2).to(torch.float)
        else:
            X = self.image_resize(torchvision.io.read_image((const.DATA_DIR / 'waterbirds' / 'images' / self.df['img_filename'].iloc[idx]).as_posix())).to(self.device)
            X = X / 255  # normalization

            if self.add_mask_noise:
                heatmap = self.mask_noise_transform(heatmap)

            heatmap = self.cam_resize(torchvision.io.read_image((const.DATA_DIR / 'waterbirds' / 'segmentations' / f'{self.df["img_filename"].iloc[idx][:-3]}png').as_posix()).to(self.device) > (255 / 2)).to(torch.float)[0]

        if heatmap.sum() == 0: heatmap[:, :] = 1.

        y = torch.tensor([self.df['y'][idx],], dtype=torch.float32)

        if self.state == 'evaluation': return X, (heatmap, y), torch.tensor(self.df['place'][idx], device=self.device)
        return X, (heatmap, y)


def get_transforms(split):
    resize_resolution = (256, 256)
    return T.Compose([T.RandomResizedCrop(size=const.IMAGE_SIZE, interpolation=InterpolationMode.NEAREST_EXACT, scale=(.7, 1.), ratio=(.75, 1.33)) if split == 'train' else T.Resize(resize_resolution, interpolation=InterpolationMode.NEAREST_EXACT),
                      T.RandomHorizontalFlip(p=0.5) if split == 'train' else T.CenterCrop(const.IMAGE_SIZE),
                      T.ToDtype(torch.float32, scale=True)]), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_generators(state='training'):
    datasets = [Dataset(split=split, state=state, transforms=get_transforms(split), add_mask_noise=const.ADD_MASK_NOISE, mix_object_contexts=const.MIX_OBJECT_CONTEXTS, device='cpu') for split in const.SPLITS]
    return [DataLoader(dataset, num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE, shuffle=True if dataset.split == 'train' else False) for dataset in datasets]
