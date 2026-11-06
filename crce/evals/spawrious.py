#!/usr/bin/env python3

from ..data.spawrious import get_generators as spawrious
from torcheval.metrics import MulticlassAccuracy
import torch
import sys

from crce.model.arch import ConvNet
from crce.utils import configure
from crce import const


@torch.no_grad
def accuracy(model, metric, gen):
    for X, (_, y) in gen: metric.update(model(X.to(model.device))[0], y.argmax(1).to(model.device))
    return metric.compute()


if __name__ == '__main__':
    name = sys.argv[1]
    configure(name.split('/')[-2], eval_mode=True)

    tasks = ["o2o_easy", "o2o_medium", "o2o_hard", "m2m_hard", "m2m_easy", "m2m_medium"]
    task_index = [task in const.MODEL_NAME for task in tasks].index(True)
    _, _, test = spawrious(tasks[task_index])

    model = ConvNet(return_logits=True)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=model.device, weights_only=True))
    model.name = name
    model.eval()

    print(accuracy(model, MulticlassAccuracy(device=torch.device(model.device)), test).item() * 100)

    del test
