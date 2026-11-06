#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EVALS_DIR = BASE_DIR / 'evals'
MODELS_DIR = BASE_DIR / 'models'

# dataset
SEED = 1
N_WORKERS = 16
CAM_SIZE = (14, 14)
DATASET = 'waterbirds'
IMAGE_SIZE = (224, 224)
SPLITS = ['train', 'valid', 'test']

INETBG_RETURN_MASKS = False

ADD_MASK_NOISE = False
MIX_OBJECT_CONTEXTS = False

if DATASET == 'hardimagenet': N_CLASSES = 15
elif DATASET == 'imagenet': N_CLASSES = 1000
elif DATASET == 'waterbirds': N_CLASSES = 2
elif DATASET == 'spawrious': N_CLASSES = 4
else: raise ValueError('No such dataset')

# model
BACKBONE_TYPE = 'resnet50'
UPSAMPLE_CAMS = False

# training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GRAD_ACCUMULATION_STEPS = 1
BATCH_SIZE = 64 // GRAD_ACCUMULATION_STEPS
WEIGHT_DECAY = 1e-4
LOSS_FN = 'crce'
EPOCHS = 150
LR = 1e-3

# logging
MLFLOW_TRACKING_URI = 'http://localhost:5000/'
