import random

import torch
from torch.utils.data import Subset
import copy
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from .datasets import get_dataset
from avalanche.benchmarks.scenarios.generic_benchmark_creation import \
    GenericCLScenario
from typing import (
    Sequence,
    Any,
    Tuple,
    Dict,
)
from avalanche.benchmarks.utils import (
    SupportedDataset,
    make_classification_dataset,
)


def create_multi_dataset_generic_benchmark(
    train_datasets: Sequence[SupportedDataset],
    test_datasets: Sequence[SupportedDataset],
    *,
    other_streams_datasets: Dict[str, Sequence[SupportedDataset]] = None,
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Dict[str, Tuple[Any, Any]] = None
) -> GenericCLScenario:
    transform_groups = dict(
        train=(train_transform, train_target_transform),
        eval=(eval_transform, eval_target_transform),
    )

    if other_streams_transforms is not None:
        for stream_name, stream_transforms in other_streams_transforms.items():
            if isinstance(stream_transforms, Sequence):
                if len(stream_transforms) == 1:
                    # Suppose we got only the transformation for X values
                    stream_transforms = (stream_transforms[0], None)
            else:
                # Suppose it's the transformation for X values
                stream_transforms = (stream_transforms, None)

            transform_groups[stream_name] = stream_transforms

    input_streams = dict(train=train_datasets, test=test_datasets)

    if other_streams_datasets is not None:
        input_streams = {**input_streams, **other_streams_datasets}

    if complete_test_set_only:
        if len(input_streams["test"]) != 1:
            raise ValueError(
                "Test stream must contain one experience when"
                "complete_test_set_only is True"
            )

    stream_definitions = dict()

    for stream_name, dataset_list in input_streams.items():
        initial_transform_group = "train"
        if stream_name in transform_groups:
            initial_transform_group = stream_name

        stream_datasets = []
        for dataset_idx in range(len(dataset_list)):
            dataset = dataset_list[dataset_idx]
            stream_datasets.append(
                make_classification_dataset(
                    dataset,
                    transform_groups=transform_groups,
                    initial_transform_group=initial_transform_group,
                    task_labels=dataset_idx
                )
            )
        stream_definitions[stream_name] = (stream_datasets,)

    return GenericCLScenario(
        stream_definitions=stream_definitions,
        complete_test_set_only=complete_test_set_only,
    )


def get_transformation_set(idx, config):
    if idx == 0:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

    elif idx == 1:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomSolarize(0.5, p=1.0),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomSolarize(0.5, p=1.0),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

    elif idx == 2:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.9, 0.9)),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.9, 0.9)),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

    elif idx == 3:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomAutocontrast(p=1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.6, 0.7)),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomAutocontrast(p=1.0),
                transforms.Grayscale(num_output_channels=3),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.6, 0.7)),
                transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762)),
                transforms.Resize(config.input_size)
            ]
        )

    else:
        raise NotImplementedError()

    return train_transform, eval_transform


def get_dataset_subsets(train_set, test_set, classes_i):
    train_targets = torch.LongTensor(train_set.targets)
    test_targets = torch.LongTensor(test_set.targets)

    def get_class_ind(c):
        ind_c_train = torch.where(train_targets == c)[0]
        ind_c_train = ind_c_train[torch.randperm(len(ind_c_train))]

        ind_c_test = torch.where(test_targets == c)[0]
        ind_c_test = ind_c_test[torch.randperm(len(ind_c_test))]

        return ind_c_train, ind_c_test

    map_targets = {c: i for (i, c) in enumerate(classes_i)}
    def target_mapper(c): return map_targets[c]

    train_idx = torch.cat([get_class_ind(c)[0] for c in classes_i])
    subset_train = Subset(train_set, train_idx)
    transformed_target = [target_mapper(train_targets[train_idx][i].item()) for
                          i in range(len(train_targets[train_idx]))]

    subset_train = make_classification_dataset(subset_train,
                                               target_transform=target_mapper,
                                               targets=transformed_target)

    test_idx = torch.cat([get_class_ind(c)[1] for c in classes_i])
    subset_test = Subset(test_set, test_idx)
    transformed_target = [target_mapper(test_targets[test_idx][i].item()) for
                          i in range(len(test_targets[test_idx]))]
    subset_test = make_classification_dataset(subset_test,
                                              target_transform=target_mapper,
                                              targets=transformed_target)

    subset_train = copy.deepcopy(subset_train)
    subset_test = copy.deepcopy(subset_test)

    return subset_train, subset_test


def get_noisy_cifar_benchmark(config):
    add_noise = config.bnch_params.add_noise

    all_classes = list(range(100))
    random.Random(config.seed).shuffle(all_classes)
    n_exp_cls = config.bnch_params.n_exp_cls

    train_sets, test_sets = [], []
    for i in range(0, config.bnch_params.n_experiences):
        if add_noise:
            tr_train, tr_test = get_transformation_set(i, config)
        else:
            tr_train, tr_test = get_transformation_set(0, config)

        train_set = CIFAR100(root=config.dataset_root, train=True,
                             transform=tr_train, download=True)
        test_set = CIFAR100(root=config.dataset_root, train=False,
                            transform=tr_test, download=True)
        classes_i = all_classes[i * n_exp_cls: (i + 1) * n_exp_cls]

        train_set_i, test_set_i = get_dataset_subsets(
            train_set, test_set, classes_i)
        train_sets.append(train_set_i)
        test_sets.append(test_set_i)

    benchmark = create_multi_dataset_generic_benchmark(train_sets, test_sets)

    config.n_classes = config.bnch_params.n_exp_cls

    return benchmark
