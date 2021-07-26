from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
import numpy as np

from metric.recall_at_k import RecallAtK
from retrieval.nearest_neighbor import NearestNeighborDistanceMetric


class RecallAtKPlugin(PluginMetric[float]):
    """
    This metric plugin will return a `float` value after
    each evaluation experience
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._NN = NearestNeighborDistanceMetric("cosine")
        self._recall_AT_1= RecallAtK(k=1)
        self._recallK_values_list = []
        # current x values for the metric curve
        self.x_coord = 0


    def reset(self) -> None:
        """
        Reset the metric
        """
        self._recallK_values_list = []
        self._recall_AT_1.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._recall_AT_1.result()

    
    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        print("mb_x shape: ", strategy.mb_x.shape)
        print("mb_y shape: ", strategy.mb_y.shape)
        print("mb_output shape: ", strategy.mb_output.shape)
        print("out_flattened_feature shape: ", strategy.out_flattened_features.shape)

        if(strategy.out_flattened_features.is_cuda or strategy.mb_y.is_cuda):
                out_flattened_features = strategy.out_flattened_features.detach().cpu()
                mb_y = strategy.mb_y.detach().cpu()

        distances_matrix = self._NN.distance(out_flattened_features, out_flattened_features)
        self._recall_AT_1.update(distances_matrix, mb_y, mb_y, True )
        self._recallK_values_list.append(self.result())

        print("Recall@1: ",self.result())
        

    
    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the next experience begins
        """
        self.reset()


    def after_eval_exp(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result, if the eval batch size is smallest than the entire eval set, the metric value is averaged between all batches
        """
        
        value = np.array(self._recallK_values_list).mean()
        self.x_coord += 1 # increment x value
        name = "Recall@"+"1"+"_Exp"
        return [MetricValue(self, name, value, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Recall@"+"1"+"_Exp"
        return name