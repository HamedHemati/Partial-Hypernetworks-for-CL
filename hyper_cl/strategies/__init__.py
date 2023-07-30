from avalanche.training.supervised import Naive, EWC, JointTraining
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer,\
    ClassBalancedBuffer

from .hyper_naive import HyperNaive
from .hyper_alg_reg_NM import HyperAlgRegNM
from .latent_replay1 import LatentReplay1
from .freezing_naive import FreezingNaive


def get_strategy(
        config,
        model,
        optimizer,
        criterion,
        eval_plugin,
        device
):
    if config.strategy == "Naive":
        strategy = Naive(model, optimizer, criterion, evaluator=eval_plugin,
                         device=device, **config.strategy_params)

    elif config.strategy == "EWC":
        strategy = EWC(model, optimizer, criterion, evaluator=eval_plugin,
                       device=device, **config.strategy_params)

    elif config.strategy == "ER-RS":
        buffer = ReservoirSamplingBuffer(max_size=config.buffer_size)
        batch_size = config.strategy_params.train_mb_size

        replay_plugin = ReplayPlugin(mem_size=config.buffer_size,
                                     batch_size=batch_size,
                                     batch_size_mem=batch_size//2,
                                     storage_policy=buffer)
        strategy = Naive(
            model, optimizer, criterion, evaluator=eval_plugin, device=device,
            plugins=[replay_plugin], **config.strategy_params)

    elif config.strategy == "ER-CB":
        buffer = ClassBalancedBuffer(max_size=config.buffer_size)
        batch_size = config.strategy_params.train_mb_size

        replay_plugin = ReplayPlugin(mem_size=config.buffer_size,
                                     batch_size=batch_size,
                                     batch_size_mem=batch_size//2,
                                     storage_policy=buffer)
        strategy = Naive(
            model, optimizer, criterion, evaluator=eval_plugin, device=device,
            plugins=[replay_plugin], **config.strategy_params)

    elif config.strategy == "Hyper-Naive":
        strategy = HyperNaive(model, optimizer, criterion,
                              evaluator=eval_plugin,
                              device=device, **config.strategy_params)

    elif config.strategy == "Freezing-Naive":
        strategy = FreezingNaive(model, optimizer, criterion,
                                 evaluator=eval_plugin,
                                 device=device, **config.strategy_params)

    elif config.strategy == "JointTraining":
        strategy = JointTraining(model, optimizer, criterion,
                                 evaluator=eval_plugin,
                                 device=device, **config.strategy_params)

    elif config.strategy == "Hyper-ER-RS":
        buffer = ReservoirSamplingBuffer(max_size=config.buffer_size)
        replay_plugin = ReplayPlugin(mem_size=config.buffer_size,
                                     storage_policy=buffer,
                                     batch_size_mem=config.batch_size_mem)
        strategy = HyperNaive(model, optimizer, criterion,
                              evaluator=eval_plugin,
                              device=device,  plugins=[replay_plugin],
                              **config.strategy_params)

    elif config.strategy == "Hyper-ER-CB":
        buffer = ClassBalancedBuffer(max_size=config.buffer_size)
        replay_plugin = ReplayPlugin(mem_size=config.buffer_size,
                                     storage_policy=buffer,
                                     batch_size_mem=config.batch_size_mem)
        strategy = HyperNaive(model, optimizer, criterion,
                              evaluator=eval_plugin,
                              device=device,  plugins=[replay_plugin],
                              **config.strategy_params)

    elif config.strategy == "Hyper-Alg-Reg-NM":
        strategy = HyperAlgRegNM(model, optimizer, criterion,
                                 evaluator=eval_plugin,
                                 device=device, **config.strategy_params)

    elif config.strategy == "LatentReplay1":
        strategy = LatentReplay1(model, optimizer, criterion,
                                 evaluator=eval_plugin,
                                 device=device, **config.strategy_params)

    else:
        raise NotImplementedError()

    return strategy
