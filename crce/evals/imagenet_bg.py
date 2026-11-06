#!/usr/bin/env python3

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import sys

from crce.model.arch import ConvNet
from crce import utils
from crce import const


def calculate_model_df(model, data_loader):
    all_out = []
    data_id = 0
    for i_batch, (inputs, targets, infos) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            logits = model(inputs.cuda())
            for i in range(len(inputs)):
                out = {}
                out["id"] = data_id
                out["label"] = targets[i].item()
                out["quality"] = infos[0][i]
                out["type"] = infos[1][i]
                out["background"] = infos[2][i]
                out["background_real_type"] = infos[3][i]
                out["logits"] = np.array(logits[i].detach().cpu())
                all_out.append(out)
                data_id += 1

    df = pd.DataFrame(all_out)
    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")

    return df


if __name__ == '__main__':
    name = sys.argv[1]
    utils.configure(name, overrides={'DATASET': 'imagenet_bg'}, eval_mode=True)

    path = const.EVALS_DIR / 'imagenet_bg_results'
    path.mkdir(parents=True, exist_ok=True)
    path /= f'{name.replace("/", "_")}.pickle'

    if path.exists():
        df = pd.read_pickle(path)
    else:
        _, _, test = utils.get_dataloaders()

        model = ConvNet(return_logits=True, logits_only=True)
        if not (const.CAM_SIZE == (7, 7) and const.LOSS_FN == 'ce' and not const.MIX_OBJECT_CONTEXTS):
            model.load_state_dict(torch.load(const.MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
        model.name = name
        model.eval()

        df = calculate_model_df(model, test)
        df.to_pickle(path)

    df["prediction"] = df["logits"].apply(np.argmax)
    df = df[df['quality'].isin(['good', 'v_good'])]

    df.loc[df['type'] == 'abstract', 'type'] = 'altered'
    df.loc[df['type'] == 'real', 'type'] = 'altered'

    print(name)
    for grp, subset in df.groupby('type'):
        print(grp, f"{(subset['label'] == subset['prediction']).mean():.7f}")
