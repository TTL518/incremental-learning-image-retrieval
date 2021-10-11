from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
import numpy as np
from utils import create_features_plot_matplot, create_features_plot_plotly


class PlotFeaturesPlugin(PluginMetric[float]):
    """
    This plugin will return a list of Image after
    each evaluation experience
    """

    def __init__(self, class_names):
        """
        Initialize the metric
        """
        super().__init__()

        self.features_list=[]
        self.labels_list = []
        self.x_coord = 0
        self.class_names = class_names
        self.finalEval = False


    def reset(self) -> None:
        """
        Reset the metric
        """
        #print("RESET plugin PLOT")
        self.features_list = []
        self.labels_list = []
        self.finalEval = False

    def result(self) -> float:
        """
        Emit the result
        """
        return None

    '''
    def before_training_epoch(self, strategy: 'BaseStrategy') -> None:
        """
        Reset the accuracy before the next epoch begins
        """
        print("Befor train epoch")
        print(strategy.out_flattened_feature)

    
    def after_training_epoch(self, strategy: 'BaseStrategy') -> 'MetricResult':
        """
        Emit the result after 5 epoch
        """

        print("After training epoch")
        features = np.vstack(self.features_list)
        labels = np.vstack(self.labels_list)
        print("Creating features plot")
        image = create_features_plot(features,labels)
        self.x_coord += 1 # increment x value
        name = "Train_Features_plot_x_epoch"
        return [MetricValue(self, name, image, self.x_coord)]
    
    def after_training_iteration(self, strategy: 'BaseStrategy') -> None:
        """
        Update the list of features with current
        batch features
        """

        print("out_flattened_feature shape: ", strategy.out_flattened_features)
        
        if(strategy.out_flattened_features.is_cuda):
                out_flattened_features = strategy.out_flattened_features.detach().cpu().numpy()
                labels = strategy.mb_y.detach().cpu().numpy()
                self.features_list.append(out_flattened_features)
                self.labels_list.append(labels)
    '''

    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the list of features with current
        batch features
        """
        if self.finalEval:
            if(strategy.plugins[0].mb_out_flattened_features.is_cuda):
                    mb_out_flattened_features = strategy.plugins[0].mb_out_flattened_features.detach().cpu().numpy()
                    labels = strategy.mb_y.detach().cpu().numpy().reshape((-1,1))
                    #print("----SHAPES----")
                    #print(mb_out_flattened_features.shape)
                    #print(labels.shape)
                    self.features_list.append(mb_out_flattened_features)
                    self.labels_list.append(labels)
            #else:

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the next experience begins
        """
        #print("Befor eval exp")
        self.reset()
        if strategy.epoch == strategy.train_epochs-1:
            self.finalEval = True

    def after_eval(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result
        """
        if self.finalEval:
            #print("After eval exp")
            features = np.vstack(self.features_list)
            labels = np.vstack(self.labels_list)
            #print("Creating features plot")
            image = create_features_plot_plotly(features,labels, self.class_names)
            #image = create_features_plot_matplot(features,labels, self.class_names)
            self.x_coord += 1 # increment x value
            name = "Eval_Features_plot_x_exp"+str(self.x_coord)
            return [MetricValue(self, name, image, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Eval_Features_plot_x_exp"
        return name