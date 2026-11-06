#!/usr/bin/env python3

from ..data.waterbirds import get_generators
import torch
import sys

from crce.model.arch import ConvNet
from crce.utils import configure
from crce import const


@torch.no_grad()
def group_accuracy(model, gen):
    grp = {}
    overall_acc = torch.tensor([], device=model.device)

    for (X, (h, y), p) in gen:
        y = y.to(model.device)
        p = p.to(model.device)
        y_pred = (model(X.to(model.device))[0] > 0).to(torch.int)

        overall_acc = torch.cat([overall_acc, y == y_pred])

        for label in [0, 1]:
            for place in [0, 1]:
                if label not in grp: grp[label] = {}
                if place not in grp[label]: grp[label][place] = torch.tensor([], device='cpu')
                grp[label][place] = torch.cat([grp[label][place], y_pred[(label == y).ravel() & (place == p)].ravel().to('cpu')], dim=0)

    for label in [0, 1]:
        grp[gen.dataset.encodings['label'][label]] = {}
        for place in [0, 1]:
            grp[gen.dataset.encodings['label'][label]][gen.dataset.encodings['place'][place]] = (grp[label][place].to(model.device) == label).to(torch.float).mean().item()
        del grp[label]

    return grp


if __name__ == '__main__':
    name = sys.argv[1]
    configure(name, eval_mode=True)

    model = ConvNet(return_logits=True)
    model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    _, _, test = get_generators(state='evaluation')
    print(test.dataset.split + ': ' + str(group_accuracy(model, test)))

    del test
