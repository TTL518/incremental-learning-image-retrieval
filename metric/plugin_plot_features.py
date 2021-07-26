from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
import numpy as np
from utils import create_features_plot


class PlotFeaturesPlugin(PluginMetric[float]):
    """
    This plugin will return a list of Image after
    each evaluation experience
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self.features_list=[]
        self.labels = []
        self.x_coord = 0


    def reset(self) -> None:
        """
        Reset the metric
        """
        self.features_list = []
        self.labels_list = []

    def result(self) -> float:
        """
        Emit the result
        """
        return None

    
    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the list of features with current
        batch features
        """

        print("out_flattened_feature shape: ", strategy.out_flattened_features.shape)
        
        if(strategy.out_flattened_features.is_cuda):
                out_flattened_features = strategy.out_flattened_features.detach().cpu().numpy()
                labels = strategy.mb_y.detach().cpu().numpy()
                self.features_list.append(out_flattened_features)
                self.labels_list.append(labels)

        

    def before_eval_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the next experience begins
        """
        self.reset()

    def after_eval_epoch(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result
        """
        features = np.vstack(self.features_list)
        labels = np.vstack(self.labels_list)

        image = create_features_plot(features,labels)
        self.x_coord += 1 # increment x value
        name = "Features_plot_epoch"
        return [MetricValue(self, name, image, self.x_coord)]

    '''
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
    '''

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Recall@"+"1"+"_Exp"
        return name