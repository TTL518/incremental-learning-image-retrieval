from typing import Optional, Sequence, List, Union
from avalanche.training.strategies.strategy_wrappers import Naive
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from trainer.plugins import ILFGIR_plugin

from avalanche.training import default_logger
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies import BaseStrategy, Naive

from trainer.plugins import ILFGIR_plugin

class ILFGIR_strategy(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):

        ILFGIR = ILFGIR_plugin()
        if plugins is None:
            plugins = [ILFGIR]
        else:
            plugins.append(ILFGIR)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)