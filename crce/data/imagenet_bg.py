#!/usr/bin/env python3

import torchvision.transforms.v2 as T
from crce.utils import DataLoader
from PIL import Image
from crce import const
import torch
import json
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_file, transform=None, return_masks=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_masks = return_masks
        self.samples = []
        self.targets = []

        if return_masks:
            self.mask_transform = T.Compose([T.PILToTensor(),
                                             T.Resize(size=[232, 232], antialias=True),
                                             T.CenterCrop(size=(224, 224)),
                                             T.Resize(const.CAM_SIZE, interpolation=T.InterpolationMode.NEAREST_EXACT)])

        with open(labels_file, "r") as f:
            label_mappings = json.load(f)

        # quality > type > background
        syn_to_class = {v[0]: k for k, v in label_mappings.items()}
        for quality in ['good', 'v_good'] if return_masks else ['bad', 'good', 'ok', 'questionable', 'v_good']:
            quality_path = os.path.join(self.root_dir, quality)
            for type in os.listdir(quality_path):
                type_path = os.path.join(quality_path, type)
                for background in os.listdir(type_path):
                    background_path = os.path.join(type_path, background)
                    for image_class in os.listdir(background_path):
                        class_path = os.path.join(background_path, image_class)
                        for image_name in os.listdir(class_path):
                            image_path = os.path.join(class_path, image_name)
                            if image_class in syn_to_class:
                                if not image_name.lower().endswith('.png') and not image_name.lower().endswith('.jpeg'):
                                    continue

                                self.samples.append(image_path)
                                self.targets.append(int(syn_to_class[image_class]))

    def extract_info_from_path(self, path):
        quality, type, background = path.split('/')[-5:-2]
        background_real_type = ""
        if "-" in background:
            background_real_type = background.split("-")[-1]

        return (quality, type, background, background_real_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]

        info = self.extract_info_from_path(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.return_masks:
            mask_path = os.path.join(const.DATA_DIR / 'imagenet-s' / 'validation-segmentation', *img_path.split('/')[-2:]).replace('JPEG', 'png')
            mask = self.mask_transform(Image.open(mask_path))[0] != 0

            ohe = torch.zeros(const.N_CLASSES, device=image.device)
            ohe[target] = 1.  # choice of index does not matter
            return image, (mask.to(torch.float), ohe)
        return image, target, info


def get_generators(return_masks=False):
    transform = T.Compose([T.PILToTensor(),
                           T.Resize((232, 232), interpolation=T.InterpolationMode.BILINEAR),
                           T.CenterCrop(const.IMAGE_SIZE),
                           T.ToDtype(torch.float32, scale=True),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = Dataset(
        root_dir=const.DATA_DIR / 'imagenet-bg' / 'images',
        labels_file=const.DATA_DIR / 'imagenet-bg' / 'imagenet_class_index.json',
        transform=transform,
        return_masks=return_masks
    )
    return None, None, DataLoader(dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=const.N_WORKERS, pin_memory=True)  # ImageNet-BG is a test-set only


if __name__ == '__main__':
    _, _, dataloader = get_generators()
    print(next(iter(dataloader)))
    del dataloader
