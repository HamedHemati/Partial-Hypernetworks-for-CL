from typing import Optional, List

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates import SupervisedTemplate


class HyperNaive(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        freeze_after_first_exp=True,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.freeze_after_first_exp = freeze_after_first_exp

    def _after_training_exp(self, **kwargs):
        if self.freeze_after_first_exp:
            for p in self.model.feat_extractor_sf.parameters():
                p.requires_grad = False

        super()._after_training_exp()
