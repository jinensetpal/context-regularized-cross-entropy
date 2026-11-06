"""
The code for Hard-ImageNet evals was taken from the Hard-ImageNet codebase:
    https://github.com/mmoayeri/HardImageNet/blob/a39e3f996f70eb538ff71829b80781bac617a2d4/supplemental_model_eval.py
"""

import sys

from crce.data.hard_imagenet import Dataset
from crce.model.arch import ConvNet
from crce.utils import configure
from crce import const

from .ablations import *
from .rfs import *


# model loading
def load_supplemental_model():
    '''
    First, one will need to load the model they wish to evalute. In addition to the model, you will need
    to specify the target_layer used for GradCAM in order to do saliency analysis. Check the following repo
    for more info on picking the target_layer: https://github.com/jacobgil/pytorch-grad-cam

    We demonstrate loading a few different models, who's results we present in the appendix. Check utils.py
    for some useful auxiliary model loading functions.

    To repeat experiments in main text, it suffices to use the 'get_model(mkey, ft)' command from utils.py

    We also show the general procedure for loading a finetuned model, triggered when ft=False
    '''
    name = sys.argv[1]

    configure(name)
    model = ConvNet(logits_only=True, return_logits=True)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    return model, model.backbone.layer4[-1].relu if const.BACKBONE_TYPE == 'resnet18' else model.backbone[-3]


# Noise based analysis: RFS
def eval_rfs(model, dset_name='hard_imagenet', sigma=0.25, l2=False, apply_norm=False):
    rfs, noisy_fg_acc, noisy_bg_acc, noisy_fg_acc_by_class, noisy_bg_acc_by_class = \
                                noisy_fg_bg_accs(model, dset_name, sigma, apply_norm, l2, True)
    print(dset_name, rfs, noisy_fg_acc, noisy_bg_acc, sigma)
    return rfs


# Ablation analysis
def eval_ablations(model, dset_name='hard_imagenet'):
    dset = Dataset(split='val', device='cpu', eval_mode=True)
    ablation_names_fns_batchsizes = [('tile', tile, 1), ('replace_with_gray', replace_with_gray, 16),
                                     ('replace_bbox_with_gray', lambda x, y: replace_with_gray(x, y, False), 16),
                                     ('none', lambda x, y: x, 16)]
    results_dict = dict()
    for ab_name, ab, bs in ablation_names_fns_batchsizes:
        acc, confidences = eval_under_ablation(model, ab, dset, bs)
        results_dict[ab_name] = dict({'acc': acc, 'confidences': confidences})
        print(dset_name, acc, np.average(confidences), ab_name)
    return results_dict


# analysis for appendix/rebutttal
def eval_supplementary_models():
    results = {}
    model, target_layer = load_supplemental_model()
    mkey = 'resnet50_context_shift_robustness'
    if mkey not in results:
        results[mkey] = dict()
    for dset_name in ['hard_imagenet']:
        results[mkey][dset_name] = dict()
        # noise based analysis
        results[mkey][dset_name]['rfs'] = eval_rfs(model, dset_name=dset_name, apply_norm=False)
        results[mkey][dset_name]['rfs'] = eval_rfs(model, dset_name=dset_name, sigma=0.5, apply_norm=False)
        # ablation analysis
        results[mkey][dset_name]['ablation'] = eval_ablations(model, dset_name=dset_name)


if __name__ == '__main__':
    eval_supplementary_models()
