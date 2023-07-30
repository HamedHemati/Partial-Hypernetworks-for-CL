from typing import Optional, TYPE_CHECKING

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class ReplayBuffer(SupervisedPlugin):
    def __init__(
        self,
        mem_size: int = 200,
        batch_size: int = None,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        update_mode="epoch"
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )
        self.update_mode = update_mode

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def after_training_iteration(
        self, strategy, *args, **kwargs
    ):
        if self.update_mode == "iteration":
            # Update buffer only in the first epoch
            if strategy.clock.train_exp_epochs == 0:
                self.storage_policy.update(strategy, **kwargs)

    def after_training_epoch(
            self,
            strategy: "SupervisedTemplate",
            **kwargs
    ):
        if self.update_mode == "epoch":
            # Update buffer only in the first epoch
            if strategy.clock.train_exp_epochs == 1:
                self.storage_policy.update(strategy, **kwargs)
