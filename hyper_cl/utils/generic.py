import yaml
import random
import numpy as np
import collections
import torch
import os
import wandb
import time
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd


def init_paths(config, exp_name):
    results_path = os.path.join(config.outputs_dir, exp_name)
    checkpoints_path = os.path.join(results_path, "checkpoints")
    paths = {"results": results_path,
             "checkpoints": checkpoints_path
             }

    if config.save_results:
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)

        # Save a copy of params to the results folder
        output_params_yml_path = os.path.join(results_path, "params.yml")
        print(dict(config))
        with open(output_params_yml_path, 'w') as outfile:
            yaml.dump(OmegaConf.to_container(config), outfile,
                      default_flow_style=False)

    return paths


def flatten_omegaconf_for_wandb(config):
    config = OmegaConf.to_container(config)

    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened_config = flatten(config)

    return flattened_config


def get_device(config):
    device = torch.device(config.device)
    print("Device: ", device)

    return device


def set_random_seed(seed):
    print("Setting random seed: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def get_exp_name(config):
    exp_name = f'{config.strategy}'
    exp_name += f'_{config.benchmark}'
    exp_name += f'_s{config.seed}'
    t_suff = time.strftime("%m%d%H%M%S")
    exp_name += f'_{t_suff}'

    return exp_name


def init_config(config):
    OmegaConf.set_struct(config, False)

    # Initialization
    exp_suffix = "_" + config.benchmark + "_" + time.strftime("%Y%m%d%H%M%S")
    config.exp_name = config.exp_type + exp_suffix
    config.dataset_root = os.path.join(get_original_cwd(), config.dataset_root)
    config.outputs_dir = os.path.join(get_original_cwd(), config.outputs_dir)

    config.paths = init_paths(config, config.exp_name)

    set_random_seed(config.seed)

    config.log_to_wandb = config.wandb_proj != ""


def save_model(config, model, itr):
    if config.save_results:
        ckpt_path = os.path.join(config.paths["results"],
                                 f"checkpoints/ckpt_{itr}.pt")
        torch.save(model.state_dict(), ckpt_path)


def finalize_experiment(config, model):
    # Save results and checkpoints
    if config.save_results:
        save_model(config, model, "final")

    if config.log_to_wandb:
        wandb.finish()
