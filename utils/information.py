import numpy as np
from typing import Optional, List

from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import gamma, digamma
from scipy.stats import binned_statistic

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

def _get_knn_entropy(x: np.array, 
                 num_neighbors: Optional[int]=3,
                 metric: Optional[str]='chebyshev', 
                 min_dist: Optional[float]=0.) -> float:
    """
    Estimates the entropy H of a random variable x (in nats) based on
    the kth-nearest neighbour distances between point samples.
    
    Parameters
    ----------
    x       :   (num_samples, num_dimensions) continuous multivariate distribution
    k       :   kth nearest neighbour to use in density estimate;
                imposes smoothness on the underlying probability distribution
    metric  :   'chebyshev' (max-norm) or 'minkowski' (euclidean)
                distance metric used when computing k-nearest neighbour distances
    min_dist:   minimum distance between data points;
                smaller distances will be capped using this value
    
    Returns
    -------
    h       :   entropy H(X)
    
    References
    ----------
    .. [1] Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. 
           Problemy Peredachi Informatsii, 23(2), 9–16.
    """
    
    log = np.log   # i.e. information measures are in nats
    n, d = x.shape

    if metric == 'chebyshev': # max norm:
        # p = np.inf
        log_c_d = 0 # volume of the d-dimensional unit ball
    elif metric == 'minkowski': # euclidean norm
        # p = 2
        log_c_d = (d/2.) * log(np.pi) - log(gamma(d/2. +1))
    else:
        raise NotImplementedError("Variable 'metric' either 'chebyshev' or 'minkowski'")
    
    kdtree = KDTree(x, metric=metric)

    # query all points -- k+1 as query point also in initial set
    distances, _ = kdtree.query(x, num_neighbors+1)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(log(2*distances)) # radius -> diameter
    h = -digamma(num_neighbors) + digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h

def _get_binned_entropy(x: np.array, num_bins: Optional[int]=10, 
                        bin_edges: Optional[np.array]=None) -> float:
    """
    Compute entropy by discretizing the given continuous random variables.
    Discretization here is done by binning the continuous values into 
    equally-spaced descrete clusters. 

    Parameters
    ----------
    x           :   (n_samples, d) the continuous random variable
                    where, d \in {1, 2}. 
    num_bins    :   number of bins into which to cluster the values.
                    if, d==2: we would use num_bins**2 bins.
    bin_edges   :   pre-computed bin edges for discretization of reps;
                    this is usually used when global binning is employed.
    
    Returns
    -------
    h           :   entropy H(X)

    """
    if x.shape[1] == 2:
        x_1, x_2 = x[:, 0], x[:, 1]
        if bin_edges is None:
            _, bin_edges = np.histogram2d(x_1, x_2, bins=num_bins)
        bin_edges_1, bin_edges_2 = bin_edges
        binned_x_1, binned_x_2 = np.digitize(x_1, bin_edges_1), np.digitize(x_1, bin_edges_2)
        clusters, num_c = {}, 0
        for i in np.unique(binned_x_1):
            for j in np.unique(binned_x_2):
                if((i, j) not in clusters):
                    clusters[(i, j)] = num_c
                    num_c += 1
        binned_x = np.array([clusters[(binned_x_1[i], binned_x_2[i])] for i in range(len(x))])
    else:
        x = x.reshape(-1)
        if bin_edges is None:
            _, bin_edges = np.histogram(x, bins=num_bins)
        binned_x = np.digitize(x, bin_edges)

    def _get_discrete_entropy(X: np.array) -> float:
        _, counts = np.unique(X, return_counts=True)
        probs = counts / len(X)
        if np.count_nonzero(probs) <= 1:
            return 0

        ent = 0.
        for i in probs:
            ent -= i * np.log(i)

        return ent

    return _get_discrete_entropy(binned_x)

def get_all_entropies(reps: np.array,
                      method: Optional[str] = "binned",
                      num_neighbors: Optional[int] = 10, 
                      to_tqdm: Optional[bool] = True,
                      global_binning: Optional[bool] = True) -> np.array:
    """
    Get num_neurons (N) sized vector, each value representing the 
    **unnormalized** entropy for that neuron.

    Parameters
    ----------
    reps            :   (n_samples, n_neurons) flattened model activations.
    method          :   technique to employ for computing entropies.
    num_neighbors   :   number of neighbors to consider for KNN-based entropy computation,
                        when using binning to compute entropy, this is used as ```num_bins```.
    to_tdqm         :   whether to show the progress of MI computation or not.
    global_binning  :   whether to employ global binning for binning-based computation or not.

    Returns
    -------
    entropies       :   (n_neurons, n_neurons) unnormalized MI values between each 
                        pairwise neuron pair from the given set.
    """
    num_vars = reps.shape[-1]
    entropies = np.zeros(num_vars)
    if global_binning:
        _, bin_edges = np.histogram(reps.reshape(-1), bins=num_neighbors)
    if to_tqdm:
        for i in tqdm(range(num_vars)):
            if method=="binned":
                h = _get_binned_entropy(reps[:, i].reshape(-1, 1),
                                        num_bins=num_neighbors,
                                        bin_edges=bin_edges if global_binning else None)
            else:
                h = _get_knn_entropy(reps[:, i].reshape(-1, 1), num_neighbors)
            # h = h.astype('float64')
            entropies[i] = h
    else:
        for i in range(num_vars):
            if method=="binned":
                h = _get_binned_entropy(reps[:, i].reshape(-1, 1), 
                                        num_bins=num_neighbors,
                                        bin_edges=bin_edges if global_binning else None)
            else:
                h = _get_knn_entropy(reps[:, i].reshape(-1, 1), num_neighbors)
            # h = h.astype('float64')
            entropies[i] = h
    
    return entropies

def _get_knn_mi(x: np.array, y: np.array, n_neighbors: int, clip_negative: Optional[bool] = False) -> float:
    """
    Compute mutual information between two continuous variables.
    Parameters
    ----------
    x, y            :   (n_samples,) Samples of two continuous random variables, 
                        must have an identical shape.
    n_neighbors     :   Number of nearest neighbors to search for each point, see [1].
    clip_negative   :   Whether to clip negative values to zero.
    
    Returns
    -------
    mi          :   Estimated mutual information. 
                    If it turned out to be negative it is replace by 0.
    
    NOTE
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.
    
    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = x.size

    x = x.reshape((-1, 1))      # (n_samples, 1)
    y = y.reshape((-1, 1))      # (n_samples, 1)
    xy = np.hstack((x, y))      # z = (x, y) -> (n_samples, 2)

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    # ^ distance metric used is the max-norm (https://en.wikipedia.org/wiki/Chebyshev_distance)
    # dist(z2, z1) = max(||x2-x1||, ||y2-y1||)
    # the same metric is used for x and y, i.e., simply
    # dist(x2, x1) = (x2 - x1), when x is 1-dimensional
    
    nn.fit(xy)                              # fitting KNN on joint z = (x, y)
    radius = nn.kneighbors()[0]             # kneighbors() returns (distances, neighbors) 
                                            # for all samples -> (2, n_samples, n_neighbors)
                                            # we only use the distances as the query radii 
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, 
                         count_only=True, 
                         return_distance=False) # number of points in x that are within the query radius
                                                # -> (n_samples)
    nx = np.array(nx) - 1.0                     # (nx-1)

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, 
                         count_only=True, 
                         return_distance=False) # number of points in y that are within the query radius
                                                # -> (n_samples)
    ny = np.array(ny) - 1.0                     # (ny -1)

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )                               # I(X; Y) = ψ(S) + ψ(k) - 1/N*sum(ψ(nx) + ψ(ny))
    
    if clip_negative:
        return max(0, mi)
    return mi

def _get_binned_mi(x: np.array, y: np.array, num_bins: int, 
                   clip_negative: Optional[bool] = False) -> float:
    n_samples = x.size

    x = x.reshape((-1, 1))      # (n_samples, 1)
    y = y.reshape((-1, 1))      # (n_samples, 1)
    xy = np.hstack((x, y))      # z = (x, y) -> (n_samples, 2)

    mi = _get_binned_entropy(x, num_bins) \
        + _get_binned_entropy(y, num_bins) \
        - _get_binned_entropy(xy, num_bins)
    
    if clip_negative:
        return max(0, mi)
    return mi

def _get_mi(X: np.array, y: np.array, method: Optional[str] = "knn", num_neighbors: Optional[int] = 3) -> np.array:
    """
    Obtain MI for a feature matrix with a vector by iterating over the features.
    
    Parameters
    ----------
    X               :   (n_samples, n_features) Source continuous random variable.
    y               :   (n_samples, ) Target continuous random variable.
    method          :   MI estimation to use
    n_neighbours    :   Number of nearest neighbors to search for each point.

    Returns
    -------
    np.array(mis)   :   (n_features) MI value for each feature in X with the vector y.
    """
    def _iterate_columns(X, columns=None):
        """
        Iterate over columns of a matrix.
        """
        if columns is None:
            columns = range(X.shape[1])

        for i in columns:
            yield X[:, i]
    
    assert method in ["knn", "binned"]
    mis = [
        _get_knn_mi(x, y, num_neighbors) if method=="knn" else _get_binned_mi(x, y, num_neighbors)
        for x in _iterate_columns(X)
    ]

    return np.array(mis)

def get_knn_intra_mi(reps: np.array, 
                     norm: Optional[bool] = True,
                     to_tqdm: Optional[bool] = False) -> List[List[int]]:
    """
    Get num_neurons*num_neurons (N*N) sized matrix, each value representing the 
    mutual information between neuron-pairs for the given variable.

    Parameters
    ----------
    reps    :       (n_samples, n_neurons) flattened model activations.
    norm    :       whether to normalize the obtained MI values.
    to_tdqm :       whether to show the progress of MI computation or not.

    Returns
    -------
    mis     :       (n_neurons, n_neurons) unnormalized MI values between each 
                    pairwise neuron pair from the given set.
    """
    mis = []
    if to_tqdm:
        for i in tqdm(range(reps.shape[1])):
            mi = _get_mi(reps[:, :], reps[:, i].reshape(-1,))
            mi = mi.astype('float64')
            if norm:
                mi /= (np.max(mi)+1e-4)
            mis.append(mi)
    else:
        for i in range(reps.shape[1]):
            mi = _get_mi(reps[:, :], reps[:, i].reshape(-1,))
            mi = mi.astype('float64')
            if norm:
                mi /= (np.max(mi)+1e-4)
            mis.append(mi)
            
    return mis

def get_knn_inter_mi(reps1: np.array, reps2: np.array, 
                     norm: Optional[bool] = True, 
                     to_tqdm: Optional[bool] = False) -> List[List[float]]:
    """
    Get num_neurons*num_neurons (N*N) sized matrix, each value representing the 
    mutual information between neuron-pairs between the two given variables.

    Parameters
    ----------
    reps1   :       (n_samples, n_neurons_1) flattened model activations.
    reps2   :       (n_samples, n_neurons_2) flattened model activations.
    norm    :       whether to normalize the obtained MI values.
    to_tdqm :       whether to show the progress of MI computation or not.

    Returns
    -------
    mis     :       (n_neurons_2, n_neurons_1) unnormalized MI values between each 
                    pairwise neuron pair from the given set.
    """
    mis = []
    if to_tqdm:
        for i in tqdm(range(reps2.shape[1])):
            mi = _get_mi(reps1[:, :], reps2[:, i].reshape(-1,))
            if norm:
                mi /= (np.max(mi)+1e-4)
            mis.append(mi)
    else:
        for i in range(reps2.shape[1]):
            mi = _get_mi(reps1[:, :], reps2[:, i].reshape(-1,))
            mi = mi.astype('float64')
            if norm:
                mi /= (np.max(mi)+1e-4)
            mis.append(mi)
    
    return mis

def get_square_mi(reps: np.array, 
                  num_neighbors: Optional[int]=3,
                  to_tqdm: Optional[bool] = True,
                  method: Optional[str] = "knn") -> np.array:
    """
    Get num_neurons*num_neurons (N*N) sized matrix, each value representing the 
    **unnormalized** mutual information between the neuron-pair.

    Parameters
    ----------
    reps    :       (n_samples, n_neurons) flattened model activations.
    to_tdqm :       whether to show the progress of MI computation or not.

    Returns
    -------
    mis     :       (n_neurons, n_neurons) unnormalized MI values between each 
                    pairwise neuron pair from the given set.
    """
    num_vars = reps.shape[-1]
    mis = np.zeros((num_vars, num_vars))
    if to_tqdm:
        for i in tqdm(range(num_vars)):
            mi = _get_mi(reps[:, i:], reps[:, i].reshape(-1,), 
                         method, num_neighbors)
            mi = mi.astype('float64')
            mis[i, i:] = mi
            mis[i:, i] = mi
    else:
        for i in range(num_vars):
            mi = _get_mi(reps[:, i:], reps[:, i].reshape(-1,), 
                         method, num_neighbors)
            mi = mi.astype('float64')
            mis[i, i:] = mi
            mis[i:, i] = mi
    # for i in range(num_vars):
    #     mis[i] /= np.max(mis[i])

    return mis