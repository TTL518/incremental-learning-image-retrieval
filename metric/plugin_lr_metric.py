import math 
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue

class LRPlugin(PluginMetric[float]):
    """
    This plugin will return the value of the learning rate after each epoch for Adam optimizer
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self.lr = 0.0
        self.x_coord = 0
        self.group_idx = 0
        self.param_idx = 0

    def reset(self) -> None:
        """
        Reset the metric
        """
        self.lr = 0.0

    def result(self) -> float:
        """
        Emit the result
        """
        return self.lr

    def get_adam_current_lr(self, optimizer, group_idx, parameter_idx):
        "This function return the step size of a particular network parameter for Adam optimizer without weight decay and amsgrad set to False"
        group = optimizer.param_groups[group_idx]
        p = group['params'][parameter_idx]

        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']

        state = optimizer.state[p]
        step = state['step']
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        return step_size

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        self.lr = self.get_adam_current_lr(strategy.optimizer, self.group_idx,self.param_idx)
        self.x_coord += 1 # increment x value
        name = "Learning_rate_x_epoch"
        return [MetricValue(self, name, self.lr, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Learning_rate_x_epoch"
        return name