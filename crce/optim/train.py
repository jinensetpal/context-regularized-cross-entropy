#!/usr/bin/env python3

from .loss import ContextRegularizedCrossEntropy, EnergyLoss, GALSLoss, ABNLoss
from ..model.arch import ConvNet
from crce import const, utils

from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from contextlib import nullcontext
import torch.nn.functional as F
from torch import nn, optim
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import mlflow
import torch
import sys


def fit(model, optimizer, scheduler, criterion, train, val, path):
    selected = {'last': None,
                'epoch': 0,
                'score': -torch.inf}

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH', 'EPOCHS', 'DEVICE'])})

        for epoch in range(const.EPOCHS+1):
            metrics = {metric: [] for metric in [f'{split}_{report}' for report in ['net_loss', 'crce_loss', 'ce_loss'] for split in const.SPLITS[:2]]}
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
                            if update_weights and 'contextmasking' in const.MODEL_NAME: y_pred = model(X * y[0].unsqueeze(1))
                            else: y_pred = model(X)

                        if criterion._get_name() == 'ContextRegularizedCrossEntropy':
                            batch_loss = criterion(y_pred, y, model.linear.bias)
                        elif criterion._get_name() in ['EnergyLoss', 'GALSLoss']:
                            batch_loss = criterion(y_pred, y)
                        elif criterion._get_name() == 'ABNLoss':
                            batch_loss = criterion(y_pred, y, model.backbone.ab_logits)
                        else:
                            batch_loss = criterion(y_pred[0], y[1])

                        if model.n_classes == 2:
                            pred_logits = y_pred[0].detach().ravel()
                            labels = y[1].ravel()

                            metrics[f'{split}_acc'].update(pred_logits, labels)
                            metrics[f'{split}_ce_loss'].append(F.binary_cross_entropy_with_logits(pred_logits, labels).item())
                        else:
                            metrics[f'{split}_acc'].update(y_pred[0].detach(), y[1].argmax(1))
                            batch_ce = F.cross_entropy(y_pred[0], y[1]).item()
                            metrics[f'{split}_ce_loss'].append(batch_ce)
                        metrics[f'{split}_net_loss'].append(batch_loss.item() * const.GRAD_ACCUMULATION_STEPS)

                        if criterion._get_name() == 'ContextRegularizedCrossEntropy':
                            metrics[f'{split}_crce_loss'].append(criterion.prev)

                        if update_weights:  # epoch 0 is for evaluating performance on initialization
                            batch_loss = batch_loss / const.GRAD_ACCUMULATION_STEPS
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

            if epoch > 0 and -metrics['valid_crce_loss'] > selected['score']:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['score'] = metrics['valid_crce_loss']
            elif epoch > 0 and metrics['valid_acc'] > selected['score']:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['score'] = metrics['valid_acc']

            print(f'epoch\t\t\t: {epoch}')
            for key in metrics: print(f'{key}\t\t: {metrics[key]}')
            print('-' * 10)

            if const.MIX_OBJECT_CONTEXTS and epoch == const.EPOCHS // 2:
                torch.save(optimizer.state_dict(), path / 'optim_halfway.pt')
                torch.save(scheduler.state_dict(), path / 'scheduler_halfway.pt')
                torch.save(model.state_dict(), path / 'last_halfway.pt')
                torch.save(selected['best'], path / 'best_halfway.pt')

            torch.save(optimizer.state_dict(), path / 'optim.pt')
            torch.save(scheduler.state_dict(), path / 'scheduler.pt')
            torch.save(model.state_dict(), path / 'last.pt')

        selected['last'] = deepcopy(model.state_dict())

        if 'best' not in selected:
            selected['best'] = deepcopy(model.state_dict())
            selected['epoch'] = epoch
            selected['score'] = metrics['valid_acc']

        mlflow.log_metrics({'selected_epoch': selected['epoch'],
                            'selected_score': selected['score']}, synchronous=False, step=epoch)
        model.load_state_dict(selected['best'])

        return epoch, selected


def main(model_name, overrides={}):
    path = utils.configure(model_name, overrides)

    train, val, _ = utils.get_dataloaders()
    model = ConvNet(return_logits=True)

    is_binary_cls = model.n_classes == 2
    if const.LOSS_FN == 'crce': criterion = ContextRegularizedCrossEntropy(model.get_contrastive_cams, binary_cls=is_binary_cls)
    elif const.LOSS_FN == 'energy': criterion = EnergyLoss(binary_cls=is_binary_cls)
    elif const.LOSS_FN == 'gals': criterion = GALSLoss(binary_cls=is_binary_cls)
    elif const.LOSS_FN == 'abn': criterion = ABNLoss(binary_cls=is_binary_cls)
    elif is_binary_cls: criterion = nn.BCEWithLogitsLoss()
    else: criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=const.LR, momentum=0.9, weight_decay=const.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=const.EPOCHS)

    print(criterion, scheduler, optimizer, sep='\n')
    completed_epochs, selected = fit(model, optimizer, scheduler, criterion, train, val, path)

    torch.save(selected['last'], path / 'last.pt')
    torch.save(selected['best'], path / 'best.pt')

    torch.save(optimizer.state_dict(), path / 'optim.pt')
    torch.save(scheduler.state_dict(), path / 'scheduler.pt')

    return selected


if __name__ == '__main__':
    main(sys.argv[1])
