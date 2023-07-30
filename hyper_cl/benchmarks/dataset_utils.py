import torch
from torch.utils.data import Subset
from avalanche.benchmarks.utils.classification_dataset import \
    make_classification_dataset


def get_dataset_subset(dataset, max_n_samples_per_class):
    if isinstance(dataset.targets, torch.Tensor):
        all_classes = set(list(dataset.targets.numpy()))
    else:
        all_classes = set(dataset.targets)

    targets = torch.LongTensor(dataset.targets)

    def get_class_ind(c):
        ind_c = torch.where(targets == c)[0]
        ind_c = ind_c[torch.randperm(len(ind_c))][:max_n_samples_per_class]

        return ind_c

    all_indices = [get_class_ind(c) for c in all_classes]
    all_indices = torch.cat(all_indices, dim=0)

    subset = Subset(dataset, all_indices)

    subset = make_classification_dataset(subset)

    return subset
