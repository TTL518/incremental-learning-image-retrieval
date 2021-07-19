import numpy as np
import torch
from torch import Tensor
from avalanche.evaluation import Metric


# a standalone metric implementation
class RecallAtK(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize the metric
        """
        self.recall_at_k = 0

    def update(self, distances_matrix, query_labels, gallery_labels, k=1, is_equal_query=True ):
        """
        Update metric value
        """
        l = len(distances_matrix)

        """
        if(query_labels.is_cuda or gallery_labels.is_cuda):
                query_labels = query_labels.detach().cpu()
                gallery_labels = gallery_labels.detach().cpu()
        """
        match_counter = 0

        for i in range(l):
            pos_sim = distances_matrix[i][gallery_labels == query_labels[i]]
            neg_sim = distances_matrix[i][gallery_labels != query_labels[i]]

            thresh = np.sort(pos_sim)[1] if is_equal_query else np.max(pos_sim)
            print("Thresh: ", thresh)
            print("Sum: ", np.sum(neg_sim < thresh))
            if np.sum(neg_sim < thresh) >= k:
                match_counter = match_counter
            else:
                match_counter += 1
        print(match_counter)
        self.recall_at_k = float(match_counter) / l

    def update_II(self, distances_matrix, query_labels, gallery_labels, k=1, is_equal_query=True ):
        """
        Update metric value
        """
        l = len(distances_matrix)

        """
        if(query_labels.is_cuda or gallery_labels.is_cuda):
                query_labels = query_labels.detach().cpu()
                gallery_labels = gallery_labels.detach().cpu()
        """
        tot_relevant = 0
        relevant_in_top_k = 0

        for i in range(l):
            tot_relevant += (np.sum(gallery_labels == query_labels[i]) - 1) if is_equal_query else np.sum(gallery_labels == query_labels[i])
            print("Tot relevant ", tot_relevant)
            top_k_indices = np.argsort(distances_matrix[i])[1:k+1] if is_equal_query else np.argsort(distances_matrix[i])[:k]
            print("Top k indices ",top_k_indices)
            print(gallery_labels[top_k_indices])
            relevant_in_top_k += np.sum(gallery_labels[top_k_indices] == query_labels[i])
            print("#Relevant in top k ",relevant_in_top_k)

        self.precision_at_k = relevant_in_top_k/(k*l)
        self.recall_at_k = relevant_in_top_k/tot_relevant

    def result(self) -> float:
        """
        Emit the metric result
        """
        return self.recall_at_k 

    def reset(self):
        """
        Reset the metric
        """
        self.recall_at_k = 0

if __name__ == "__main__":
    
    
    #metric = RecallAtK()

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
    labels = np.array([1,1,1,2,2,1,1,2,2,2])
    print(update(distances, labels, labels, 6, True))