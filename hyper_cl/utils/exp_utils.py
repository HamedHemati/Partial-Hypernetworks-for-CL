import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import tqdm
from itertools import chain


def get_module_optimizer(module, config):
    if config.optimizer in ["SGD", "Adam"]:
        if config.optimizer == "SGD":
            optimizer = torch.optim.SGD(module.parameters(), **config.optim_params)

        elif config.optimizer == "Adam":
            optimizer = torch.optim.Adam(module.parameters(), **config.optim_params)
    else:
        raise NotImplementedError()

    return optimizer


def get_exp_config(exp_name, outputs_path="./out/outputs/", param_file_name="params.yml"):
    exp_path = os.path.join(outputs_path, exp_name)

    # OmegaConf
    config = OmegaConf.load(os.path.join(exp_path, param_file_name))
    config.exp_path = exp_path

    return config


def get_ckpt_path(config, ckpt_id):
    ckpt_path = os.path.join(config.exp_path, f"checkpoints/{ckpt_id}")
    return ckpt_path


def load_checkpoint(model, ckpt_path):
    # Checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

    for n, p in model.named_parameters():
        if n not in ckpt:
            print(f"Skipping parameter {n}, it does not exist.")
            continue

        if p.shape != ckpt[n].shape:
            print(f"Skipping parameter {n}: shape not matching ...")
        else:
            p.data.copy_(ckpt[n].data)


def extract_features(feature_extractor, dataset, device):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    features = []

    dataloader_progess = tqdm.tqdm(dataloader)
    with torch.no_grad():
        for batch in dataloader_progess:
            x = batch[0].to(device)
            features.append(feature_extractor(x))

    features = torch.cat(features)

    return features
