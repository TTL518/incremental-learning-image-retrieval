import numpy as np
import time 
from retrieval.distances import pair_wise_cosine_dist, pair_wise_squared_dist

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    distances = pair_wise_squared_dist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of M row-vectors (query points).
    y : ndarray
        A matrix of N row-vectors (gallery points).
    Returns
    -------
    # ndarray
    #     A vector of length M that contains for each entry in `y` the
    #     smallest cosine distance to a sample in `x`.
    """
    distances = pair_wise_cosine_dist(x, y)
    return distances
    
class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    """

    def __init__(self, metric, matching_threshold=None, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget    # Gating threshold for cosine distance
        self.samples = {}

    def distance(self, queries, galleries):
        """Compute distance between galleries and queries.
        Parameters
        ----------
        queries : ndarray
            An LxM matrix of L features of dimensionality M to match the given `galleries` against.
        galleries : ndarray
            An NxM matrix of N features of dimensionality M.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape LxN
        """
        """
        if(queries.is_cuda or galleries.is_cuda):
                queries = queries.detach().cpu()
                galleries = galleries.detach().cpu()
        """
        return self._metric(queries, galleries)

def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r,c]])
        distances.append(col)

    return distances


def match_k(top_k, galleries, queries):
    
    metric = NearestNeighborDistanceMetric("cosine")
    start = time.time()
    distance_matrix = metric.distance(queries, galleries)
    end = time.time()
    print("distance measure time: {}".format(end-start))

    top_k_indice = np.argsort(distance_matrix, axis=1)[:, :top_k]
    top_k_distance = _print_distances(distance_matrix, top_k_indice)

    return top_k_indice, top_k_distance


if __name__ == "__main__":
    a = [[1,2,6,4],[2,2,2,2],[2,3,1,1],[11,23,221,1]]
    b = [[2,2,2,2],[2,31,2,3],[1,33,44,33]]

    print(match_k(2,a,a))