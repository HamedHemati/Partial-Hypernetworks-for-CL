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

from hyper_cl.utils.generic import (
    get_device,
    init_config,
    flatten_omegaconf_for_wandb,
    finalize_experiment
)
from hyper_cl.utils.exp_utils import get_module_optimizer
from hyper_cl.benchmarks import get_benchmark
from hyper_cl.models import get_model
from hyper_cl.strategies import get_strategy


@hydra.main(config_path="conf", config_name="train.yaml")
def main(config):
    # Device
    device = get_device(config)

    # Initialization
    config.exp_type = "inc"
    init_config(config)

    # Benchmark
    benchmark = get_benchmark(config)

    # Loggers
    loggers = [InteractiveLogger()]
    if config.log_to_wandb:
        wandb_logger = WandBLogger(
            project_name=config.wandb_proj,
            run_name=config.exp_name,
            dir=config.paths["results"],
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
    for i, exp in enumerate(benchmark.train_stream, start=1):
        strategy.train(exp, num_workers=config.num_workers)
        strategy.eval(benchmark.test_stream[:i],
                      num_workers=config.num_workers)

        # Break? for quick experimental tests
        end_exp = config.end_after_n_exps
        if end_exp is not None and end_exp == i:
            break

    # Accuracy metrics
    if config.log_to_wandb:
        retained_acc, learning_acc = get_accuracy_metrics(strategy)
        wandb.log({"Retained Accuracy": retained_acc,
                  "Learning Accuracy": learning_acc})

    # Finalize logs and finish results
    finalize_experiment(config, model)


def get_accuracy_metrics(strategy):
    all_plugins = [isinstance(p, EvaluationPlugin) for p in strategy.plugins]
    eval_plugin = strategy.plugins[all_plugins.index(True)]
    eval_results = eval_plugin.all_metric_results
    last_exp = strategy.clock.train_exp_counter
    ra_list = []
    la_list = []
    for exp_id in range(0, last_exp):
        metric_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{exp_id:03d}/Exp{exp_id:03d}"
        value_list = eval_results[metric_key][1]
        ra_list.append(value_list[-1])
        la_list.append(value_list[0])

    retained_acc = sum(ra_list) / len(ra_list)
    learning_acc = sum(la_list) / len(la_list)

    return retained_acc, learning_acc


if __name__ == "__main__":
    main()
