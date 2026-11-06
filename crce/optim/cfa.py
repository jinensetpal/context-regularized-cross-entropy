#!/usr/bin/env python3

from ..model.arch import ConvNet
from crce import const, utils
from .loss import CFALoss

from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from contextlib import nullcontext
from copy import deepcopy
from torch import optim
from tqdm import tqdm
import numpy as np
import pathlib
import mlflow
import torch
import sys


def _save_output(module, i, o):
    module.prev_output = o


def fit(model, model_erm, optimizer, scheduler, criterion, train, val, path):
    selected = {'last': None,
                'epoch': 0,
                'acc': 0.0,
                'loss': torch.inf}

    # insert hooks here to extract features
    assert model.backbone_type == 'resnet50' and model_erm.backbone_type == 'resnet50'
    model.backbone.avgpool.register_forward_hook(_save_output)
    model_erm.backbone.avgpool.register_forward_hook(_save_output)

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH', 'EPOCHS', 'DEVICE'])})

        for epoch in range(const.EPOCHS+1):
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['cosine_loss', 'ce_loss', 'joint_loss'] for split in const.SPLITS[:2]]}
            for split in const.SPLITS[:2]:
                if model.n_classes == 2: metrics[f'{split}_acc'] = BinaryAccuracy(threshold=0, device=torch.device(model.device))
                else: metrics[f'{split}_acc'] = MulticlassAccuracy(device=torch.device(model.device))

            try:
                for split, dataloader in zip(const.SPLITS[:2], (train, val)):
                    update_weights = split == 'train' and epoch > 0
                    if update_weights: model.train()
                    else: model.eval()

                    for batch_idx, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=split):
                        if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS: optimizer.zero_grad()
                        X = X.to(model.device)
                        y = [y_i.to(model.device) for y_i in y]

                        with torch.no_grad() if not update_weights else nullcontext():
                            model(X)

                        with torch.no_grad():
                            y_pred = model_erm(X)
                            model_erm(X * y[0].unsqueeze(1))

                        batch_loss = criterion(model_erm.backbone.avgpool.prev_output, model.backbone.avgpool.prev_output, y_pred, y[1]) / const.GRAD_ACCUMULATION_STEPS

                        metrics[f'{split}_ce_loss'].append(criterion.prev_ce)
                        metrics[f'{split}_cosine_loss'].append(criterion.prev_cosine)
                        metrics[f'{split}_joint_loss'].append(batch_loss.item())

                        if model.n_classes == 2:
                            pred_logits = y_pred.detach().ravel()
                            labels = y[1].ravel()

                            metrics[f'{split}_acc'].update(pred_logits, labels)
                        else:
                            metrics[f'{split}_acc'].update(y_pred.detach(), y[1].argmax(1))

                        if update_weights:  # epoch 0 is for evaluating performance on initialization
                            batch_loss.backward(inputs=optimizer.param_groups[0]['params'])

                            if not (batch_idx+1) % const.GRAD_ACCUMULATION_STEPS:
                                optimizer.step()
            except (KeyboardInterrupt, torch.OutOfMemoryError) as exception:
                print(exception)
                break

            for split in const.SPLITS[:2]: metrics[f'{split}_acc'] = metrics[f'{split}_acc'].compute().item()
            metrics = {metric: np.mean(metrics[metric]) for metric in metrics}

            metrics['lr'] = scheduler.get_last_lr()[-1]
            mlflow.log_metrics(metrics, synchronous=False, step=epoch-1)
            if const.DATASET == 'imagenet' and epoch: scheduler.step()

            if metrics['valid_acc'] > selected['acc']:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['acc'] = metrics['valid_acc']

            print(f'epoch\t\t\t: {epoch}')
            for key in metrics: print(f'{key}\t\t: {metrics[key]}')
            print('-' * 10)

            torch.save(optimizer.state_dict(), path / 'optim.pt')
            torch.save(scheduler.state_dict(), path / 'scheduler.pt')
            torch.save(model.state_dict(), path / 'last.pt')

        selected['last'] = deepcopy(model.state_dict())

        if 'best' not in selected:
            selected['best'] = deepcopy(model.state_dict())
            selected['epoch'] = epoch
            selected['acc'] = metrics['valid_acc']

        mlflow.log_metrics({'selected_epoch': selected['epoch'],
                            'selected_valid_acc': selected['acc']}, synchronous=False, step=epoch)
        model.load_state_dict(selected['best'])

        return epoch, selected


def main(model_name, overrides={}):
    path = utils.configure(sys.argv[1])

    erm_path = pathlib.Path(str(path).replace('cfa', 'ce'))
    const.CAM_SIZE = const.IMAGE_SIZE

    train, val, _ = utils.get_dataloaders()

    model = ConvNet(return_logits=True, logits_only=True)
    model.load_state_dict(torch.load(erm_path / 'best.pt'))

    model_erm = ConvNet(return_logits=True, logits_only=True)
    model_erm.load_state_dict(torch.load(erm_path / 'best.pt'))
    model_erm.eval()

    criterion = CFALoss(use_bce=model.n_classes == 2)
    optimizer = optim.SGD(model.backbone.parameters(), lr=const.LR, momentum=0.9, weight_decay=const.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=const.EPOCHS)

    print(criterion, scheduler, optimizer, sep='\n')
    completed_epochs, selected = fit(model, model_erm, optimizer, scheduler, criterion, train, val, path)

    torch.save(selected['last'], path / 'last.pt')
    torch.save(selected['best'], path / 'best.pt')

    torch.save(optimizer.state_dict(), path / 'optim.pt')
    torch.save(scheduler.state_dict(), path / 'scheduler.pt')

    return selected


if __name__ == '__main__':
    main(sys.argv[1])
