from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.transforms as transforms


def get_dataset(dataset_name, config, no_transform=False):
    if dataset_name == "CIFAR100":
        if no_transform:
            train_transform = None

            eval_transform = None
        else:
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

        dataset_train = CIFAR100(
            root=config.dataset_root, train=True,
            transform=train_transform, download=True)
        dataset_test = CIFAR100(
            root=config.dataset_root, train=False,
            transform=eval_transform, download=True)

        return dataset_train, dataset_test

    elif dataset_name == "CIFAR10":

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                transforms.Resize(config.input_size)
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                transforms.Resize(config.input_size)
            ]
        )

        dataset_train = CIFAR10(
            root=config.dataset_root, train=True,
            transform=train_transform, download=True)
        dataset_test = CIFAR10(
            root=config.dataset_root, train=False,
            transform=eval_transform, download=True)

        return dataset_train, dataset_test

    elif dataset_name == "MNIST":
        train_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
             transforms.Resize(config.input_size)]
        )

        eval_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
             transforms.Resize(config.input_size)]
        )

        dataset_train = MNIST(
            root=config.dataset_root, train=True,
            transform=train_transform, download=True)
        dataset_test = MNIST(root=config.dataset_root,
                             train=True,
                             transform=eval_transform, download=True)

        return dataset_train, dataset_test

    else:
        raise NotImplementedError()
