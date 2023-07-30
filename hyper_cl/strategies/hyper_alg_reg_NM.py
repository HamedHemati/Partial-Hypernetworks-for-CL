from typing import Optional, List

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector
import functorch
from copy import deepcopy

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate


class HyperAlgRegNM(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        hnet_reg_ntasks: int = 10,
        coef_hnet_replay: float = 0.1,
        second_order: bool = True,
        freeze_after_first_exp: bool = False,
        wg_embedding_mode: str = "copy_prev",
        model_checkpoint_path: bool = None,
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
        self.hnet_reg_ntasks = hnet_reg_ntasks
        self.coef_hnet_replay = coef_hnet_replay
        self.freeze_after_first_exp = freeze_after_first_exp
        self.second_order = second_order
        self.wg_embedding_mode = wg_embedding_mode

        # Load checkpoint and copy parameters for feature-ext sf and sl
        if model_checkpoint_path is not None:
            print("Copying checkpoint parameters.")
            ckpt = torch.load(model_checkpoint_path)
            for n, p in model.feat_extractor_sf.named_parameters():
                p.data.copy_(ckpt["feat_extractor_sf." + n].data)
                p.requires_grad = False

    def training_epoch(self, **kwargs):
        self.cos_sim_list = []
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            current_task_id = self.experience.task_label
            if current_task_id == 0:
                self.mb_output = self.model(self.mb_x, self.mb_task_id)
                self.loss += self.criterion()
                self.loss.backward()

            else:
                # Look-head -----------> take one inner SGD step
                out, func_wg, params_wg_1, buff_wg_1, params_featsf_1 = self.fast_update()
                self.mb_output = out

                # Inner loss - 1
                loss_inner_1 = self._criterion(out, self.mb_y)
                self.loss += loss_inner_1

                # Look-head -----------> compute hnet's regularization
                if self.coef_hnet_replay > 0.0:
                    # Compute gradients for the inner SGD step
                    lr = self.optimizer.param_groups[0]["lr"]
                    g_wg_1 = torch.autograd.grad(loss_inner_1, params_wg_1,
                                                 create_graph=self.second_order,
                                                 retain_graph=True)
                    params_wg_2 = [p - lr * g for (p, g) in zip(params_wg_1, g_wg_1)]

                    rnd_tasks = torch.randperm(current_task_id)
                    rnd_tasks = rnd_tasks[:self.hnet_reg_ntasks].to(self.device)
                    reg_hnet_losses = [self.hnet_task_diff_norm(task_id, func_wg, params_wg_2, buff_wg_1)
                                       for task_id in rnd_tasks]
                    loss_reg_hnet = sum(reg_hnet_losses) / len(reg_hnet_losses)
                    loss_reg_hnet = self.coef_hnet_replay * loss_reg_hnet
                    self.loss += loss_reg_hnet

                    # Gradient conflict
                    fl_1 = parameters_to_vector(g_wg_1)
                    g_wg_2 = torch.autograd.grad(loss_reg_hnet, params_wg_1,retain_graph=True)
                    fl_2 = parameters_to_vector(g_wg_2)
                    cos_sim = torch.nn.functional.cosine_similarity(fl_1, fl_2, dim=0)
                    self.cos_sim_list.append(cos_sim.item())

                self.loss.backward()

                # Gradient for the stateful part
                if not self.freeze_after_first_exp:
                    self.apply_grad(self.model.feat_extractor_sf, params_featsf_1)

                self.apply_grad(self.model.weight_generator, params_wg_1)

            # Optimization step
            self.optimizer.step()

            self._after_training_iteration(**kwargs)

    def fast_update(self):
        # Functionals for model segments
        func_wg, params_wg_1, buff_wg_1 = functorch.make_functional_with_buffers(
            self.model.weight_generator)
        func_featsf, params_featsf_1 = functorch.make_functional(
            self.model.feat_extractor_sf)
        func_featsl, params_featsl_1 = functorch.make_functional(
            self.model.feat_extractor_sl)

        # Weight generator
        gen_params = func_wg(params_wg_1, buff_wg_1, self.mb_task_id)
        x = func_featsf(params_featsf_1, self.mb_x)
        x = x.unsqueeze(1)
        out = functorch.vmap(func_featsl)(gen_params, x).squeeze(1)

        return out, func_wg, params_wg_1, buff_wg_1, params_featsf_1

    def hnet_task_diff_norm(self, task_id, func_wg, params_wg_2, wg_buff_1):
        task_id = torch.LongTensor([task_id]).to(self.device)
        prev_w = parameters_to_vector(
            self.prev_model.weight_generator(task_id)).detach()
        cur_w = parameters_to_vector(func_wg(params_wg_2, wg_buff_1, task_id))
        # loss_reg_hnet = torch.norm(cur_w - prev_w)
        loss_reg_hnet = (prev_w - cur_w).pow(2).sum()
        return loss_reg_hnet

    def apply_grad(self, module, ref):
        for (p1, p2) in zip(module.parameters(), ref):
            if p2.grad is not None:
                p1.grad = p2.grad
            else:
                p1.grad = None

    def _after_training_exp(self, **kwargs):
        # Save current model as `prev_model`
        self.prev_model = deepcopy(self.model).to(self.device)

        if self.wg_embedding_mode == "copy_prev":
            current_task_id = self.experience.task_label
            with torch.no_grad():
                w_prev = self.model.weight_generator.embedding.weight[current_task_id].detach().data
                if current_task_id + 1 < self.model.weight_generator.embedding.weight.shape[0]:
                    self.model.weight_generator.embedding.weight[current_task_id + 1].copy_(w_prev)

        super()._after_training_exp()
