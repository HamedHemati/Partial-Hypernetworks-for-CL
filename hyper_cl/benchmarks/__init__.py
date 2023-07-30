from .avl_benchmarks import get_benchmark as get_benchmark_avl
from .noisy_cifar_benchmark import get_noisy_cifar_benchmark


def get_benchmark(config):
    if config.benchmark.startswith("noisy_cifar"):
        return get_noisy_cifar_benchmark(config)

    else:
        return get_benchmark_avl(config)
