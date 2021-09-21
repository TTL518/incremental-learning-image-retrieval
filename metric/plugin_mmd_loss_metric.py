from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metrics.loss import Loss
from avalanche.evaluation.metric_utils import phase_and_task


class MMDLossPluginMetric(GenericPluginMetric[float]):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = Loss()
        super(MMDLossPluginMetric, self).__init__(
            self._loss, reset_at, emit_at, mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        if(strategy.plugins[0].prev_model_frozen != None):
            # task labels defined for each experience
            task_labels = strategy.experience.task_labels
            if len(task_labels) > 1:
                # task labels defined for each pattern
                # fall back to single task case
                task_label = 0
            else:
                task_label = task_labels[0]
            self._loss.update(strategy.plugins[0].mmdLoss,
                            patterns=len(strategy.mb_y), task_label=task_label)

class MinibatchMMDLoss(MMDLossPluginMetric):
    """
    The minibatch loss metric.
    This plugin metric only works at training time.

    This metric computes the average loss over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochLoss` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchLoss metric.
        """
        super(MinibatchMMDLoss, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "MMDLoss_MB"

class EpochMMDLoss(MMDLossPluginMetric):
    """
    The average loss over a single training epoch.
    This plugin metric only works at training time.

    The loss will be logged after each training epoch by computing
    the loss on the predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochLoss metric.
        """

        super(EpochMMDLoss, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "MMD_Loss_Epoch"