#!/usr/bin/env python3

import torch
import tqdm
import sys

from crce.model.arch import ConvNet
from crce import utils
from crce import const


@torch.no_grad
def context_contribution_ratio(model, gen):
    fg_contribs = torch.tensor((), device=model.device)
    bg_contribs = torch.tensor((), device=model.device)

    for X, (h, y) in tqdm.tqdm(gen, total=len(gen)):
        y_pred, cams = model(X.to(model.device))
        y = y.to(model.device)

        if const.DATASET != 'waterbirds' and const.BACKBONE_TYPE != 'resnet50_abn':
            cams = model.get_contrastive_cams(y, cams)
        h = h.to(model.device).unsqueeze(1)

        fg_contribs = torch.cat([fg_contribs, (cams.abs() * h).sum(dim=(1, 2, 3))])
        bg_contribs = torch.cat([bg_contribs, (cams.abs() * (1 - h)).sum(dim=(1, 2, 3))])

    return (bg_contribs / (fg_contribs + bg_contribs)).mean().item()


if __name__ == '__main__':
    name = sys.argv[1]
    utils.configure(name, eval_mode=True, overrides={'INETBG_RETURN_MASKS': True})
    if const.DATASET == 'imagenet':
        const.DATASET = 'imagenet_bg'
    if 'contextmasking' in const.MODEL_NAME:
        const.CAM_SIZE = (14, 14) if const.UPSAMPLE_CAMS else (7, 7)

    model = ConvNet(return_logits=True)
    if not (const.DATASET == 'imagenet_bg' and const.CAM_SIZE == (7, 7) and const.LOSS_FN == 'ce'):
        model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    _, val, test = utils.get_dataloaders()

    print(context_contribution_ratio(model, test or val))
    del _, _, test
