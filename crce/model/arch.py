#!/usr/bin/env python3

from .abn import resnet50_abn
from crce import const

from torch import nn
import torchvision
import torch


class ConvNet(nn.Module):
    def __init__(self, return_logits=False, logits_only=False, backbone_type=None,
                 n_classes=None, upsample_cams=None, device=None):
        super().__init__()
        self.populate_with_defaults(locals())

        if self.backbone_type == 'resnet18': self.penultimate_latent_dim = 512
        elif self.backbone_type in ['resnet50', 'resnet50abn']: self.penultimate_latent_dim = 2048
        elif self.backbone_type == 'efficientnetv2': self.penultimate_latent_dim = 1280
        else: raise ValueError('Unrecognized backbone type.')

        if self.backbone_type == 'resnet18':
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.layer4[-1].relu.register_forward_hook(self._hook)

            if self.upsample_cams:
                self.backbone.layer4[0].conv1.stride = (1, 1)
                self.backbone.layer4[0].downsample[0].stride = (1, 1)

        elif self.backbone_type == 'resnet50':
            self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone.layer4[-1].relu.register_forward_hook(self._hook)

            if self.upsample_cams:
                self.backbone.layer4[0].conv2.stride = (1, 1)
                self.backbone.layer4[0].downsample[0].stride = (1, 1)

        elif self.backbone_type == 'resnet50abn':
            assert self.upsample_cams  # ABN intrinsically upsamples CAMS
            self.backbone = resnet50_abn(num_classes=self.n_classes if self.n_classes > 2 else 1)

            """
            The code for loading the state dictionary was adapted from the GALS codebase:
                https://github.com/spetryk/GALS/blob/0d33e2598b3f749477584ebfe143e38dc5f6c61f/approaches/abn.py#L40-L51
            """
            state_dict = torch.load(const.MODELS_DIR / 'resnet50_abn_imagenet.pth.tar', weights_only=False)['state_dict']
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
            if self.n_classes != 1000:
                # Remove imagenet-specific weights from state dict
                # present in the attention branch.
                new_dict = {}
                for k, v in state_dict.items():
                    if 'att_conv' in k or 'bn_att2' in k or 'fc' in k:
                        print('Removing {} from pretained weights, with weight shape {}'.format(k, v.shape))
                        continue
                    new_dict[k] = v
                state_dict = new_dict

            self.backbone.load_state_dict(state_dict, strict=False)

        elif self.backbone_type == 'efficientnetv2':
            self.backbone = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.backbone.features[-1][-1].register_forward_hook(self._hook)

            if self.upsample_cams:
                self.backbone.features[6][0].block[1][0].stride = (1, 1)

        self.linear = nn.Linear(self.penultimate_latent_dim, self.n_classes if self.n_classes > 2 else 1, bias=True)
        self.probabilities = nn.Identity() if self.return_logits else nn.Sigmoid() if self.n_classes == 2 else nn.Softmax(dim=1)

        if const.DATASET in ['imagenet', 'imagenet_bg']:
            if self.backbone_type == 'efficientnetv2':
                self.linear.weight = self.backbone.classifier[1].weight
                self.linear.bias = self.backbone.classifier[1].bias
            else:
                self.linear.weight = self.backbone.fc.weight
                self.linear.bias = self.backbone.fc.bias

        if self.backbone_type == 'efficientnetv2': self.backbone.classifier[1] = nn.Identity()
        else: self.backbone.fc = nn.Identity()

        print(self)
        self.to(self.device)
        if not self.logits_only and self.backbone_type != 'resnet50abn': self.initialize_and_verify()

    def populate_with_defaults(self, kwargs):
        for key in kwargs:
            if kwargs[key] is None: setattr(self, key, getattr(const, key.upper()))
            elif key not in ['self', '__class__']: setattr(self, key, kwargs[key])

    def initialize_and_verify(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.randn(100, 3, *const.IMAGE_SIZE, device=self.device)

            logits, cam = self(x)
            cam_logits = cam.view(*cam.shape[:2], -1).sum(2)

            if self.linear.bias is not None: cam_logits += self.linear.bias
            print('Approx. cam logit err bound:', (logits - cam_logits).abs().max().item())

            assert torch.allclose(logits, cam_logits, atol=1e-5)
        self.train(was_training)

    def _hook(self, model, i, o):
        self.feature_rect = o

    def forward(self, x):
        x = self.backbone(x)
        logits = self.linear(x)

        if self.logits_only: return logits
        return logits if self.training else self.probabilities(logits), self._bp_free_hi_res_cams() if self.backbone_type != 'resnet50abn' else self.backbone.att

    def get_recon_cams(self, cams):
        return cams - cams.mean(1, keepdim=True)

    def get_contrastive_cams(self, y, cams):
        return torch.index_select(cams.view(-1, *cams.shape[2:]), 0, y.argmax(1) + (torch.arange(cams.size(0), device=self.device) * cams.size(1))).repeat(1, cams.size(1), 1).view(*cams.shape) - cams

    def _bp_free_hi_res_cams(self):
        return (self.linear.weight @ self.feature_rect.flatten(2)).unflatten(2, self.feature_rect.shape[2:]) / self.feature_rect.shape[-1]**2
