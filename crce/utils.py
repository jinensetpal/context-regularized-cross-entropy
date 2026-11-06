#!/usr/bin/env python3

import mlflow
import pprint
import random
import torch


class _RepeatSampler(object):
    """
    The `_RepeatSampler` class is taken from:
        https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class DataLoader(torch.utils.data.dataloader.DataLoader):
    """
    The `DataLoader` class is taken from:
        https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


# overrides by model name; schedule multiple jobs w/o reconfiguration overhead
def configure(model_name, overrides={}, eval_mode=False):
    from . import const

    const.MODEL_NAME = model_name

    if 'crce' in const.MODEL_NAME:
        const.LOSS_FN = 'crce'
    elif 'ce' in const.MODEL_NAME:
        const.LOSS_FN = 'ce'
    elif 'energy' in const.MODEL_NAME:
        const.LOSS_FN = 'energy'
    elif 'gals' in const.MODEL_NAME:
        const.LOSS_FN = 'gals'
    elif 'abn' in const.MODEL_NAME:
        const.LOSS_FN = 'abn'

    if 'hardimagenet' in const.MODEL_NAME: const.DATASET = 'hardimagenet'
    elif 'waterbirds' in const.MODEL_NAME: const.DATASET = 'waterbirds'
    elif 'spawrious' in const.MODEL_NAME: const.DATASET = 'spawrious'
    elif 'imagenetbg' in const.MODEL_NAME: const.DATASET = 'imagenet_bg'
    elif 'imagenet' in const.MODEL_NAME: const.DATASET = 'imagenet'

    const.ADD_MASK_NOISE = 'noisy_masks' in const.MODEL_NAME
    if const.ADD_MASK_NOISE: assert const.LOSS_FN == 'crce' and const.DATASET in ['waterbirds', 'spawrious', 'imagenet']

    const.MIX_OBJECT_CONTEXTS = 'mixobjcont' in const.MODEL_NAME
    if const.MIX_OBJECT_CONTEXTS:
        assert const.DATASET in ['waterbirds', 'spawrious', 'imagenet']

    if 'resnet18' in const.MODEL_NAME:
        const.BACKBONE_TYPE = 'resnet18'
    elif 'resnet50abn' in const.MODEL_NAME:
        const.BACKBONE_TYPE = 'resnet50abn'
    elif 'resnet50' in const.MODEL_NAME:
        const.BACKBONE_TYPE = 'resnet50'
    elif 'efficientnetv2' in const.MODEL_NAME:
        const.BACKBONE_TYPE = 'efficientnetv2'

    if const.MODEL_NAME.endswith('2'): const.SEED += 1
    elif const.MODEL_NAME.endswith('3'): const.SEED += 2

    if '7x7cam' in const.MODEL_NAME:
        const.UPSAMPLE_CAMS = False
        const.CAM_SIZE = (7, 7)
    elif '14x14cam' in const.MODEL_NAME:
        const.UPSAMPLE_CAMS = True
        const.CAM_SIZE = (14, 14)
    else:
        raise ValueError('CAM resolution must be explicit.')

    if 'contextmasking' in const.MODEL_NAME: const.CAM_SIZE = const.IMAGE_SIZE

    if const.DATASET == 'imagenet':
        const.N_CLASSES = 1000

        const.EPOCHS = 20
        const.WEIGHT_DECAY = 1e-5

        if const.BACKBONE_TYPE == 'efficientnetv2':
            if const.CAM_SIZE == (14, 14): const.GRAD_ACCUMULATION_STEPS = 16
            else: const.GRAD_ACCUMULATION_STEPS = 8

            const.BATCH_SIZE = 4096 // const.GRAD_ACCUMULATION_STEPS
            const.LR = 1e-2
        else:
            const.BATCH_SIZE = 512 // const.GRAD_ACCUMULATION_STEPS
            const.LR = 5e-4
    elif const.DATASET == 'hardimagenet':
        const.N_CLASSES = 15

        const.EPOCHS = 100
        const.BATCH_SIZE = 256
        const.LR = 5e-3
        const.WEIGHT_DECAY = 1e-4
    elif const.DATASET == 'spawrious':
        const.N_CLASSES = 4

        const.EPOCHS = 150
        const.BATCH_SIZE = 128
        const.LR = 1e-3
        const.WEIGHT_DECAY = 1e-4
    elif const.DATASET == 'waterbirds':
        const.N_CLASSES = 2

        const.EPOCHS = 150
        const.BATCH_SIZE = 64
        const.LR = 1e-3
        const.WEIGHT_DECAY = 1e-4
    else: raise ValueError('Invalid dataset.')

    for k, v in overrides.items():
        setattr(const, k, v)

    pprint.pprint({k: v for k, v in const.__dict__.items() if k == k.upper()})
    torch.manual_seed(const.SEED)
    random.seed(const.SEED)

    path = const.MODELS_DIR / const.MODEL_NAME

    if not eval_mode:
        path.mkdir(exist_ok=False, parents=True)
        mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)

    return path


def get_dataloaders():
    from .data.hard_imagenet import get_generators as hardimagenet
    from .data.imagenet_bg import get_generators as imagenet_bg
    from .data.waterbirds import get_generators as waterbirds
    from .data.spawrious import get_generators as spawrious
    from .data.imagenet import get_generators as imagenet
    from . import const

    if const.DATASET == 'imagenet': return imagenet()
    elif const.DATASET == 'imagenet_bg': return imagenet_bg(return_masks=const.INETBG_RETURN_MASKS)
    elif const.DATASET == 'hardimagenet': return hardimagenet()
    elif const.DATASET == 'waterbirds': return waterbirds()
    elif const.DATASET == 'spawrious':
        tasks = ["o2o_easy", "o2o_medium", "o2o_hard", "m2m_hard", "m2m_easy", "m2m_medium"]
        task_index = [task in const.MODEL_NAME for task in tasks].index(True)
        return spawrious(tasks[task_index])
