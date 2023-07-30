import torchvision.transforms as transforms

from avalanche.benchmarks.classic import (
    SplitCIFAR100,
    SplitTinyImageNet,
)


def get_benchmark(config):
    if config.benchmark == "scifar100":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

        benchmark = SplitCIFAR100(**config.bnch_params,
                                  train_transform=train_transform,
                                  eval_transform=eval_transform)

    elif config.benchmark == "stinyimagenet":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Resize(config.input_size)
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Resize(config.input_size)
            ]
        )

        benchmark = SplitTinyImageNet(**config.bnch_params,
                                      train_transform=train_transform,
                                      eval_transform=eval_transform)

    else:
        raise NotImplementedError()

    return benchmark
