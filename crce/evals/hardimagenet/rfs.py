"""
The code for Hard-ImageNet evals was taken from the Hard-ImageNet codebase:
    https://github.com/mmoayeri/HardImageNet/blob/a39e3f996f70eb538ff71829b80781bac617a2d4/rfs.py
"""

from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch

from crce.data.hard_imagenet import Dataset
from crce.utils import DataLoader


def l2_normalize(X):
    flat_X = X.flatten(1)
    denom = flat_X.pow(2).sum(keepdim=True, dim=1).sqrt()
    out = flat_X / denom
    return out.reshape(X.shape)


def compute_rfs(noisy_fg_acc, noisy_bg_acc):
    avg = 0.5*(noisy_fg_acc + noisy_bg_acc)
    return 0 if (avg == 1 or avg == 0) else (noisy_bg_acc - noisy_fg_acc) / (2*min(avg, 1-avg))


def noisy_fg_bg_accs(model, dset_name='hard_imagenet', noise_sigma=0.25, apply_norm=True, l2=False, trials=3):
    dset = Dataset(device='cpu', eval_mode=True)
    loader = DataLoader(dset, batch_size=36, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cnt_by_class, noisy_fg_cc_by_class, noisy_bg_cc_by_class = dict(), dict(), dict()

    for imgs, masks, labels in tqdm(loader):
        if noise_sigma > 0:
            masks = masks.cuda()
        imgs = imgs.cuda()
        labels = labels.cuda()

        for trial in range(trials):

            if noise_sigma > 0:
                noise = torch.randn_like(imgs, device=imgs.device)
                if l2:
                    noisy_fg, noisy_bg = [torch.clamp(imgs + noise_sigma*l2_normalize(x * noise), 0, 1) for x in [masks, 1-masks]]
                else:
                    noisy_fg, noisy_bg = [torch.clamp(imgs + (x * noise * noise_sigma), 0, 1) for x in [masks, 1-masks]]

                if apply_norm:
                    noisy_fg, noisy_bg = [normalize(x) for x in [noisy_fg, noisy_bg]]
                noisy_fg_preds, noisy_bg_preds = [model(x).argmax(1) for x in [noisy_fg, noisy_bg]]
            else:
                if apply_norm:
                    imgs = normalize(imgs)
                preds = model(imgs).argmax(1)

            for y in np.unique(labels.cpu().numpy()):
                if y not in noisy_fg_cc_by_class:
                    noisy_fg_cc_by_class[y], noisy_bg_cc_by_class[y], cnt_by_class[y] = 0, 0, 0

                if noise_sigma > 0:
                    noisy_fg_cc_by_class[y] += (noisy_fg_preds[labels == y] == y).sum().item()
                    noisy_bg_cc_by_class[y] += (noisy_bg_preds[labels == y] == y).sum().item()
                else:
                    noisy_fg_cc_by_class[y] += (preds[labels == y] == y).sum().item()
                cnt_by_class[y] += (labels == y).sum().item()

    noisy_fg_acc_by_class = dict({c: noisy_fg_cc_by_class[c] / cnt_by_class[c] for c in cnt_by_class})
    noisy_bg_acc_by_class = dict({c: noisy_bg_cc_by_class[c] / cnt_by_class[c] for c in cnt_by_class})
    noisy_fg_acc, noisy_bg_acc = [np.average(list(x.values())) for x in [noisy_fg_acc_by_class, noisy_bg_acc_by_class]]

    rfs = compute_rfs(noisy_fg_acc, noisy_bg_acc)
    return rfs, noisy_fg_acc, noisy_bg_acc, noisy_fg_acc_by_class, noisy_bg_acc_by_class
