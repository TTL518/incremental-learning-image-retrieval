import numpy as np

def pair_wise_squared_dist(a, b):
    """Compute pair-wise squared distance between points a and b
    Parameters
    ----------
    a : array
        NxM matrix of N samples of dimensionality M.
    b : array
        LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2= np.square(a).sum(axis=1) 
    b2 = np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def pair_wise_cosine_dist(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points a and b.
    Parameters
    ----------
    a : array
        An NxM matrix of N samples of dimensionality M.
    b : array
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if not data_is_normalized:
        a_normed = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.asarray(a) / np.where(a_normed==0, 1, a_normed)
        b_normed = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.asarray(b) / np.where(b_normed==0, 1, b_normed)

    return 1. - np.dot(a, b.T)

if __name__ == "__main__":
    a = [[1,2,3,4],[2,2,2,2]]
    b = [[2,2,2,2]]

    print(pair_wise_squared_dist(a,b))
    print(pair_wise_cosine_dist(a,b,False))