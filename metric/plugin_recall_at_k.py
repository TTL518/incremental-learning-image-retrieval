import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
import numpy as np
from avalanche.evaluation import Metric
from torch.nn.functional import normalize
from retrieval.nearest_neighbor import NearestNeighborDistanceMetric

# a standalone metric implementation
class RecallAtK(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self, k=1):
        """
        Initialize the metric
        """
        self.k = k
        self.recall_at_k = 0

    def update(self, distances_matrix, query_labels, gallery_labels, is_equal_query=True ):
        """
        Update metric value like recall@k definition in "Product quantization for nearest neighbor search"
        """
        l = len(distances_matrix)
        match_counter = 0

        for i in range(l):
            pos_sim = distances_matrix[i][gallery_labels == query_labels[i]]
            neg_sim = distances_matrix[i][gallery_labels != query_labels[i]]

            thresh = np.sort(pos_sim)[1] if is_equal_query else np.max(pos_sim)
            #print("Thresh: ", thresh)
            #print("Sum: ", np.sum(neg_sim < thresh))
            if np.sum(neg_sim < thresh) >= self.k:
                match_counter = match_counter
            else:
                match_counter += 1
        #print("Match_counter: ", match_counter)
        self.recall_at_k = float(match_counter) / l
        #print(self.recall_at_k)

    def update_II(self, distances_matrix, query_labels, gallery_labels, is_equal_query=True ):
        """
        Update metric value like classic definitio of Recall n_relevant_in_first_k/n_relevant_tot
        """
        l = len(distances_matrix)
        tot_relevant = 0
        relevant_in_top_k = 0

        for i in range(l):
            tot_relevant += (np.sum(gallery_labels == query_labels[i]) - 1) if is_equal_query else np.sum(gallery_labels == query_labels[i])
            print("Tot relevant ", tot_relevant)
            top_k_indices = np.argsort(distances_matrix[i])[1:self.k+1] if is_equal_query else np.argsort(distances_matrix[i])[:self.k]
            print("Top k indices ",top_k_indices)
            print(gallery_labels[top_k_indices])
            relevant_in_top_k += np.sum(gallery_labels[top_k_indices] == query_labels[i])
            print("#Relevant in top k ",relevant_in_top_k)

        self.precision_at_k = relevant_in_top_k/(self.k*l)
        self.recall_at_k = relevant_in_top_k/tot_relevant

    def result(self) -> float:
        """
        Emit the metric result
        """
        return self.recall_at_k 

    def reset(self):
        """
        Reset the metric value
        """
        self.recall_at_k = 0


class RecallAtKPluginSingleTask(PluginMetric[float]):
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
        self.out_flattened_features = []
        self.y = []
        # current x values for the metric curve
        self.x_coord = 0
        self.InitialFinalEval = False

    def reset(self) -> None:
        """
        Reset the metric
        """
        self.out_flattened_features = []
        self.y = []
        self._recall_AT_1.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._recall_AT_1.result()

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        if strategy.epoch == strategy.train_epochs or strategy.epoch == 0:
            self.InitialFinalEval = True
        else:
            self.InitialFinalEval = False

    def after_eval(self, strategy: 'PluggableStrategy') -> None:
        self.InitialFinalEval = False
    
    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Get features of batch test images
        """
        if self.InitialFinalEval:
            if(strategy.plugins[0].mb_out_flattened_features.is_cuda or strategy.mb_y.is_cuda):
                mb_out_flattened_features = strategy.plugins[0].mb_out_flattened_features.detach().cpu()
                labels = strategy.mb_y.detach().cpu()
                self.out_flattened_features.append(mb_out_flattened_features)
                self.y.append(labels)
        
    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the recall@k before the next experience begins
        """
        self.reset()


    def after_eval_exp(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result at the end of experience
        """
        if self.InitialFinalEval:
            features = torch.cat(self.out_flattened_features)
            y = torch.cat(self.y)
            features = torch.squeeze(features)
            y = torch.squeeze(y)

            distances_matrix = self._NN.distance(features, features)
            self._recall_AT_1.update(distances_matrix, y, y, True )
            value = self.result()
            
            self.x_coord += 1 # increment x value
            name = "Recall@"+"1"+"_Single_Exp"+str(strategy.experience.current_experience)
            
            return [MetricValue(self, name, value, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Recall@"+"1"+"_Single_Exp"
        return name

class RecallAtKPluginAllTasks(PluginMetric[float]):
    """
    This metric plugin will return a `float` value at the end of the last experience evaluation, let to calculate recall@k on all tasks all toghether (passed+actual)
    """
    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()
        self._NN = NearestNeighborDistanceMetric("cosine")
        self._recall_AT_1= RecallAtK(k=1)
        self.out_flattened_features = []
        self.y = []
        # current x values for the metric curve
        self.x_coord = 0
        self.finalEval = False

    def reset(self) -> None:
        """
        Reset the metric
        """
        self.out_flattened_features = []
        self.y = []
        self._recall_AT_1.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._recall_AT_1.result()

    def before_eval(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the recall@k before the evaluation phase
        """
        self.reset()
        if strategy.epoch == strategy.train_epochs:
            self.finalEval = True
    
    def after_eval_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Get features of batch test images
        """
        #if self.finalEval:
        
        if(strategy.plugins[0].mb_out_flattened_features.is_cuda or strategy.mb_y.is_cuda):
            mb_out_flattened_features = strategy.plugins[0].mb_out_flattened_features.detach().cpu()
            labels = strategy.mb_y.detach().cpu()
            self.out_flattened_features.append(mb_out_flattened_features)
            self.y.append(labels) 

    def after_eval(self, strategy: 'PluggableStrategy') -> 'MetricResult':
        """
        Emit the result at the end of all experiences
        """
        if self.finalEval:
            self.finalEval = False
            
        features = torch.cat(self.out_flattened_features)
        y = torch.cat(self.y)
        features = torch.squeeze(features)
        y = torch.squeeze(y)

        features = normalize(features)

        distances_matrix = self._NN.distance(features, features)
        self._recall_AT_1.update(distances_matrix, y, y, True )
        value = self.result()
        
        self.x_coord += 1 #strategy.epoch # increment x value
        name = "Recall@"+"1"+"_All_Exp"
        
        return [MetricValue(self, name, value, self.x_coord)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        name = "Recall@"+"1"+"_Exp"
        return name


if __name__ == "__main__":
    metric = RecallAtK()

    distances = np.array([
                            [0.0, 0.1, 0.2, 0.5, 0.3, 1.1, 1.2, 1.1, 1.2, 1.4],
                            [0.1, 0.0, 0.3, 0.5, 0.2, 1.1, 1.2, 1.1, 1.2, 1.4],
                            [0.2, 0.1, 0.0, 0.5, 0.3, 1.1, 1.2, 1.1, 1.2, 1.4],
                            [0.5, 0.1, 0.2, 0.0, 0.3, 1.1, 1.2, 1.1, 1.2, 1.4],
                            [0.2, 0.1, 0.3, 0.5, 0.0, 1.1, 1.2, 1.1, 1.2, 1.4],
                            [1.0, 1.1, 1.2, 1.5, 1.2, 0.0, 0.2, 0.1, 0.2, 0.4],
                            [1.0, 1.1, 1.2, 1.5, 1.2, 0.2, 0.0, 0.1, 0.2, 0.4],
                            [1.0, 1.1, 1.2, 1.5, 1.2, 0.1, 0.2, 0.0, 0.2, 0.4],
                            [1.0, 1.1, 1.2, 1.5, 1.2, 0.2, 0.2, 0.1, 0.0, 0.4],
                            [1.0, 1.1, 1.2, 1.5, 1.2, 0.4, 0.2, 0.1, 0.2, 0.0],
                        ])
    labels = np.array([2,1,2,1,2,1,2,1,2,2])

    print(metric.update(distances, labels, labels, True))
