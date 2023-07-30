from typing import Optional, List

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from unittest.mock import Mock

from avalanche.models import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.storage_policy import ReservoirSamplingBuffer, ClassBalancedBuffer
from avalanche.benchmarks.utils.classification_dataset import make_classification_dataset


class LatentReplay1(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        max_buffer_size: int = 200,
        buffer_mb_size: int = 32,
        buffer_update_mode: str = "ClassBalanced",
        coef_exemplar_replay: float = 1.0,
        model_checkpoint_path: str = None,
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

        self.buffer = Buffer(max_buffer_size=max_buffer_size, buffer_mb_size=buffer_mb_size,
                             buffer_update_mode=buffer_update_mode)
        self.coef_exemplar_replay = coef_exemplar_replay

        # Load checkpoint and copy parameters for feature-ext sf and sl
        if model_checkpoint_path is not None:
            print("Copying checkpoint parameters.")
            ckpt = torch.load(model_checkpoint_path)
            for n, p in model.named_parameters():
                if not n.startswith("classifier."):
                    p.data.copy_(ckpt["feat_extractor_sf." + n].data)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0
            self.mb_output = avalanche_forward(self.model, self.mb_x, self.mb_task_id)
            self.loss += self._criterion(self.mb_output, self.mb_y)

            current_task_id = self.experience.task_label
            if current_task_id > 0:
                # Gradients from buffer exemplar regularization
                if self.coef_exemplar_replay > 0.0:
                    b_x, b_y, b_t = self.buffer.get_batch()
                    out_exp = self.model.forward_feat(b_x.to(self.device), b_t.to(self.device))
                    loss_exp = self._criterion(out_exp, b_y.to(self.device))
                    loss_exp = self.coef_exemplar_replay * loss_exp
                    self.loss += loss_exp

            self.loss.backward()
            # Optimization step
            self.optimizer.step()

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        # Update buffer
        self.buffer.update(self)
        self.buffer.reset_loader()

        # Freeze feature layers
        for param in self.model.get_feat_params():
            param.requires_grad = False

        super()._after_training_exp()


class Buffer:
    def __init__(self, max_buffer_size=100, buffer_mb_size=10, buffer_update_mode="ReservoirSampling"):
        if buffer_update_mode == "ReservoirSampling":
            self.storage_policy = ReservoirSamplingBuffer(max_size=max_buffer_size)
        elif buffer_update_mode == "ClassBalanced":
            self.storage_policy = ClassBalancedBuffer(max_size=max_buffer_size)
        else:
            raise NotImplementedError()

        self.buffer_mb_size = buffer_mb_size
        self.buffer_update_mode = buffer_update_mode

    def update(self, strategy):
        # Wrapper class for dataset features extracted by the model's feature extractor
        class FeatureDS(Dataset):
            def __init__(self, origin_ds):
                super().__init__()
                ds_feats = []
                ds_y = []
                ds_t = []
                strategy.model.eval()
                dataloader = DataLoader(origin_ds, batch_size=256)
                with torch.no_grad():
                    for i, (x, y, t) in enumerate(dataloader):
                        feats = strategy.model.extract_feat(
                            x.to(strategy.device)).cpu()
                        ds_feats.append(feats)
                        ds_y.append(y)
                        ds_t.append(t)
                strategy.model.train()

                # New memory samples
                x = torch.concat(ds_feats, dim=0)
                y = torch.concat(ds_y, dim=0)
                t = torch.concat(ds_t, dim=0)

                self.x = x
                self.y = y
                self.t = t
                self.targets = list(y.numpy())

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                x = self.x[idx]
                y = self.y[idx]
                t = self.t[idx]

                return (x, y, t)

        ds = strategy.experience.dataset
        feat_ds = FeatureDS(ds)
        feat_ds = make_classification_dataset(feat_ds)
        mock_strategy = Mock()
        mock_strategy.experience = Mock()
        mock_strategy.experience.dataset = feat_ds
        self.storage_policy.update(mock_strategy)

    def __len__(self):
        return len(self.storage_policy.buffer)

    def reset_loader(self):
        dataloader = DataLoader(self.storage_policy.buffer, batch_size=self.buffer_mb_size,
                                drop_last=True, shuffle=True)
        self.loader = iter(dataloader)

    def get_batch(self):
        try:
            data = next(self.loader)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            dataloader = DataLoader(self.storage_policy.buffer, batch_size=self.buffer_mb_size,
                                    drop_last=True, shuffle=True)
            self.loader = iter(dataloader)
            data = next(self.loader)

        buff_x, buff_y, buff_t, _ = data

        return buff_x, buff_y, buff_t
