# Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy

[![NeurIPS26 Poster](https://img.shields.io/badge/Poster-NeurIPS26-blue)](https://neurips.cc/virtual/2026/loc/atlanta/poster/150848)
[![arXiv](https://img.shields.io/badge/arXiv-2611.08157-b31b1b.svg)](https://arxiv.org/abs/2611.08157)

![overview](https://jinen.setpal.net/img/crce_overview.png)

## Introduction

Convolutional neural networks can be highly sensitive to contexts when making predictions, making them vulnerable to misleading or adversarial context.
We introduce *Context-Regularized Cross-Entropy* (CR-CE), which encourages models to suppress irrelevant context and focus on the target object when making predictions.

## Setup
Install the `crce` library:
```console
$ pip install git+https://codeberg.org/jinensetpal/context-regularized-cross-entropy.git
```

> [!TIP]
> To use CRCE for your own applications, use the `get_contrastive_cams` function from `crce.model.arch` in conjunction with `crce.ContextRegularizedCrossEntropy`.

The following instructions aid in the reproducibility of the empirical experiments in the paper.

### Datasets

Please install the following datasets to their associated directories:
1. [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) to `$PROJECT_ROOT/data/waterbirds`.
2. [Spawrious](https://www.aenguslynch.com/spawrious.github.io/) to `$PROJECT_ROOT/data/spawrious224`.
3. [ImageNet-1k](https://www.image-net.org/) to `$PROJECT_ROOT/data/imagenet`.
4. [ImageNet-BG](https://github.com/TrustAIRiders/ICCVW-2025-ImageNet-BG/) to `$PROJECT_ROOT/data/imagenet-bg`.
5. [ImageNet-S](https://github.com/LUSSeg/ImageNet-S/) to `$PROJECT_ROOT/data/imagenet-s`.
6. [Hard-ImageNet](https://mmoayeri.github.io/HardImageNet/) to `$PROJECT_ROOT/data/hardimagenet`.

### Segmentation Masks

#### ImageNet-1k

1. Setup LUSS's [Model Zoo repository](https://github.com/LUSSeg/ImageNetSegModel), and apply the patch in `$PROJECT_ROOT/imagenetsegmodel.patch`.
2. Download the [model checkpoint](https://github.com/LUSSeg/ImageNetSegModel/releases/download/vit/imagenets_ssl-sup_mae_vit_base.pth).
3. Run the following command:
```console
$ python inference.py --model vit_base_patch16 \
        --nb_classes 920 \
        --output_dir /path/to/project/root/data/imagenet/masks/train \
        --data_path /path/to/project/root/data/imagenet \
        --finetune /path/to/model/checkpoint.pth \
        --mode train
```
4. Similarly, for the validation set:
```console
$ python inference.py --model vit_base_patch16 \
        --nb_classes 920 \
        --output_dir /path/to/project/root/data/imagenet/masks/val \
        --data_path /path/to/project/root/data/imagenet \
        --finetune /path/to/model/checkpoint.pth \
        --mode val
```

#### Spawrious

To generate object-context segmentation masks for the Spawrious dataset, run:
```console
$ python -m crce.data.spawrious
```

### Pretrained Weights

Pretrained weights are downloaded automatically by the `torchvision` library, which supports all backbones but ABNs. To train ABNs, please store [model_best.pth.tar](https://drive.google.com/drive/folders/1SRtzbnE-IpB5talp7PLNK1mzMV3UPQNV) to `$PROJECT_ROOT/models/resnet50_abn_imagenet.pth.tar`.


## Training

To train models, ensure that `MLFLOW_TRACKING_URI` points to an active MLFlow instance, and then run:
```console
$ python -m crce.optim.train $MODEL_NAME
```
where `$MODEL_NAME` includes the following substrings (in any order):
1. `ce` / `crce` / `energy` / `gals` / `abn` / `contextmasking` - which method to use.
2. `waterbirds` / `spawrious_{o2o|m2m}_{easy|medium|hard}`/ `imagenet` / `hardimagenet` - which dataset to train on.
3. `14x14cam` / `7x7cam` - determine the resolution of the convolutional backbone.
4. `resnet50` / `resnet50abn` / `resnet18` / `efficientnetv2` - which backbone to use.

I like to use `_` separation, using which an example command would be:
```console
$ python -m crce.optim.train imagenet_crce_14x14cam_resnet50
```

For [CFA](https://openaccess.thecvf.com/content/WACV2024/papers/Venkataramani_Causal_Feature_Alignment_Learning_To_Ignore_Spurious_Background_Features_WACV_2024_paper.pdf), first train the corresponding cross-entropy model, then run:
```console
$ python -m crce.optim.cfa $MODEL_NAME
```

## Evaluation

To run any of the eval modules, use the following template:
```console
$ python -m crce.evals.path.to.file $MODEL_NAME/$CHECKPOINT
```
When running from `crce`, with `$CHECKPOINT` being either 'best' (w.r.t. validation metric) or 'last' (final weight update).

## Citation

```bib
@misc{setpal2026improvingcontextshiftrobustnessconvolutional,
      title={Improving Context-Shift Robustness of Convolutional Networks via Context-Regularized Cross-Entropy},
      author={Jinen Setpal and Chaoyue Liu},
      year={2026},
      eprint={2611.08157},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.08157}
}
```

## Acknowledgement

This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.
