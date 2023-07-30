from avalanche.models import SimpleCNN as AVLSimpleCNN

from .resnet import ResNet18, MTResNet18
from .hyper_networks.hyper_resnet18_SH import HyperResNet18SH
from .hyper_networks.hyper_resnet18_sp_v1_SH import HyperResNet18SPv1SH
from .hyper_networks.hyper_resnet18_sp_v2_SH import HyperResNet18SPv2SH
from .hyper_networks.hyper_resnet18_sp_v3_SH import HyperResNet18SPv3SH
from .hyper_networks.hyper_resnet18_sp_v4_SH import HyperResNet18SPv4SH
from .latent_replay.resnet_latentrep import (
    ResNet18LatentReplay,
    MTResNet18LatentReplay
)
from .utils import *


def get_model(config):
    n_classes = config.n_classes

    if config.model == "ResNet18":
        if config.multi_head:
            if config.model_params is None:
                model = MTResNet18()
            else:
                model = MTResNet18(**config.model_params)
        else:
            if config.model_params is None:
                model = ResNet18(num_classes=n_classes)
            else:
                model = ResNet18(num_classes=n_classes, **config.model_params)

    elif config.model == "ResNet18LatentReplay":
        if config.multi_head:
            model = MTResNet18LatentReplay(latent_depth=config.latent_depth)
        else:
            model = ResNet18LatentReplay(num_classes=n_classes,
                                         latent_depth=config.latent_depth)

    elif config.model == "HyperResNet18SH":
        model = HyperResNet18SH(num_tasks=config.bnch_params.n_experiences,
                                num_classes=config.n_classes,
                                **config.model_params)

    elif config.model == "HyperResNet18SPv1SH":
        model = HyperResNet18SPv1SH(num_tasks=config.bnch_params.n_experiences,
                                    num_classes=config.n_classes,
                                    **config.model_params)

    elif config.model == "HyperResNet18SPv2SH":
        model = HyperResNet18SPv2SH(num_tasks=config.bnch_params.n_experiences,
                                    num_classes=config.n_classes,
                                    **config.model_params)

    elif config.model == "HyperResNet18SPv3SH":
        model = HyperResNet18SPv3SH(num_tasks=config.bnch_params.n_experiences,
                                    num_classes=config.n_classes,
                                    **config.model_params)

    elif config.model == "HyperResNet18SPv4SH":
        model = HyperResNet18SPv4SH(num_tasks=config.bnch_params.n_experiences,
                                    num_classes=config.n_classes,
                                    **config.model_params)

    else:
        raise NotImplementedError()

    return model
