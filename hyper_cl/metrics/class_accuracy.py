from collections import defaultdict
import torch
from torch import Tensor
from typing import Union, Dict
from typing import TypeVar, Optional, TYPE_CHECKING
import numpy as np

from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_definitions import get_metric_name
from avalanche.evaluation.metric_results import MetricValue
if TYPE_CHECKING:
    from avalanche.evaluation.metric_results import MetricResult
    from avalanche.training.templates.supervised import SupervisedTemplate


class ClassAccuracy(Metric[float]):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset_mean_accuracy()

    def reset_mean_accuracy(self):
        self._mean_accuracy = {i: Mean() for i in range(self.n_classes)}

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor
    ) -> None:
        if len(true_y) != len(predicted_y):
            raise ValueError(
                "Size mismatch for true_y and predicted_y tensors")

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        def calc_class_acc(predicted_y, true_y, class_id):
            idx_c = true_y == class_id
            predicted_y_i = predicted_y[idx_c]
            true_y_i = true_y[idx_c]

            true_positives = float(
                torch.sum(torch.eq(predicted_y_i, true_y_i)))
            total_patterns = len(true_y_i)
            self._mean_accuracy[class_id].update(
                true_positives / total_patterns, total_patterns
            )

        _ = [calc_class_acc(predicted_y, true_y, int(class_id))
             for class_id in torch.unique(true_y)]

    def result(self, task_label=None) -> Dict[int, float]:
        return {k: v.result() for k, v in self._mean_accuracy.items()}

    def reset(self, task_label=None) -> None:
        self.reset_mean_accuracy()


class ClassAccuracyPluginMetric(GenericPluginMetric[float]):
    def __init__(self, reset_at, emit_at, mode, n_classes,
                 present_classes=None):
        self._accuracy = ClassAccuracy(n_classes)
        super(ClassAccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.n_classes = n_classes
        self.present_classes = present_classes

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        self._accuracy.update(strategy.mb_output, strategy.mb_y)

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        metric_value = self.result(strategy)
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations

        metrics = []

        # Log for all classes
        for k, v in metric_value.items():
            if self.present_classes is None or k in self.present_classes:
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=None
                )
                metric_name += f"/class_{k}"
                metrics.append(
                    MetricValue(self, metric_name, v, plot_x_position)
                )

        return metrics


##################################################################
#                           Wrappers                             #
##################################################################

class MinibatchClassAccuracy(ClassAccuracyPluginMetric):
    def __init__(self, n_classes):
        super(MinibatchClassAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train",
            n_classes=n_classes
        )

    def __str__(self):
        return "Top1_ClassAcc_MB"


class EpochClassAccuracy(ClassAccuracyPluginMetric):
    def __init__(self, n_classes):
        super(EpochClassAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train",
            n_classes=n_classes
        )

    def __str__(self):
        return "Top1_ClassAcc_Epoch"


class ExperienceClassAccuracy(ClassAccuracyPluginMetric):
    def __init__(self, n_classes, present_classes=None):
        super(ExperienceClassAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval",
            n_classes=n_classes,
            present_classes=present_classes
        )

    def __str__(self):
        return "Top1_ClassAcc_Experience"
