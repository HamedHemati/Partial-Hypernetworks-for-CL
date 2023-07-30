from hyper_cl.utils.limit_threads import *

import torch.nn as nn
import hydra
import wandb

from avalanche.evaluation.metrics import (
    loss_metrics,
    accuracy_metrics,
    forgetting_metrics
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from hyper_cl.utils.generic import get_device, init_config, finalize_experiment
from hyper_cl.utils.exp_utils import get_module_optimizer
from hyper_cl.benchmarks import get_benchmark
from hyper_cl.models import get_model
from hyper_cl.strategies import get_strategy


@hydra.main(config_path="conf", config_name="train.yaml")
def main(config):
    # Device
    device = get_device(config)

    # Initialization
    config.exp_type = "mt_"
    init_config(config)

    # Benchmark
    benchmark = get_benchmark(config)

    # Loggers
    loggers = [InteractiveLogger()]
    if config.log_to_wandb:
        wandb_logger = WandBLogger(
            project_name=config.wandb_proj,
            run_name=config.exp_name,
            dir=config.paths["results"]
            config=flatten_omegaconf_for_wandb(config),)
        loggers += [wandb_logger]

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=True, epoch=True,
                         experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers
    )

    # Strategy
    model = get_model(config)
    optimizer = get_module_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    strategy = get_strategy(config, model, optimizer,
                            criterion, eval_plugin, device)

    # Training and evaluation
    strategy.train(benchmark.train_stream, num_workers=config.num_workers)
    strategy.eval(benchmark.test_stream, num_workers=config.num_workers)

    # Accuracy metrics
    if config.log_to_wandb:
        retained_acc, learning_acc = get_accuracy_metrics(
            strategy, len(benchmark.train_stream), config.multi_head)
        wandb.log({"Retained Accuracy": retained_acc,
                   "Learning Accuracy": learning_acc})

    # Finalize logs and finish results
    finalize_experiment(config, model)


def get_accuracy_metrics(strategy, n_exp, return_task_id):
    all_plugins = [isinstance(p, EvaluationPlugin) for p in strategy.plugins]
    eval_plugin = strategy.plugins[all_plugins.index(True)]
    eval_results = eval_plugin.all_metric_results
    ra_list = []
    la_list = []
    for exp_id in range(0, n_exp):
        if return_task_id:
            metric_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{exp_id:03d}/Exp{exp_id:03d}"
        else:
            metric_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{exp_id:03d}"

        value = eval_results[metric_key][1]
        if isinstance(value, list):
            value = value[0]
        ra_list.append(value)
        la_list.append(value)

    retained_acc = sum(ra_list) / len(ra_list)
    learning_acc = sum(la_list) / len(la_list)

    return retained_acc, learning_acc


if __name__ == "__main__":
    main()
