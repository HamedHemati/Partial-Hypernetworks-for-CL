import torch
import torch.nn as nn
import os
from hydra.utils import get_original_cwd


def init_weights_identity(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.eye_(module.weight)
        module.bias.data.fill_(0.0)


def init_weights_kaiming_normal(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.0)

    elif isinstance(module, nn.Conv2d):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0.0)


def load_ckpt_partially(model, config):
    ckpt_path = os.path.join(get_original_cwd(), config.pretrained_ckpt_path)
    print("Loading checkpoint from ", ckpt_path)

    ckpt = torch.load(ckpt_path)
    for n, p in model.named_parameters():
        if n not in ckpt:
            print(f"Skipping parameter {n}, it does not exist.")
            continue

        if p.shape != ckpt[n].shape:
            print(f"Skipping parameter {n}: shape not matching ...")
        else:
            p.data.copy_(ckpt[n].data)
