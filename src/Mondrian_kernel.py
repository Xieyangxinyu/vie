import heapq
import numpy as np
import scipy.sparse
from sklearn import linear_model
import sys
import time
from sklearn.metrics import mean_squared_error

def sample_discrete(weights):
    cumsums = np.cumsum(weights)
    cut = cumsums[-1] * np.random.rand()
    return np.searchsorted(cumsums, cut)


def sample_cut(lX, uX, birth_time):
    rate = np.sum(uX - lX)
    if rate > 0:
        E = np.random.exponential(scale=1.0/rate)
        cut_time = birth_time + E
        dim = sample_discrete(uX - lX)
        loc = lX[dim] + (uX[dim] - lX[dim]) * np.random.rand()
        return cut_time, dim, loc
    else:
        return np.Infinity, None, None


def evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta):
    """
    Sweeps through Mondrian kernels with all lifetime in [0, lifetime_max]. This can be used to (1) construct a Mondrian
    feature map with lifetime lifetime_max, to (2) find a suitable lifetime (inverse kernel width), or to (3) compare
    Mondrian kernel to Mondrian forest across lifetimes.
    :param X:                       training inputs
    :param y:                       training regression targets
    :param X_test:                  test inputs
    :param y_test:                  test regression targets
    :param M:                       number of Mondrian trees
    :param lifetime_max:            terminal lifetime
    :param delta:                   ridge regression regularization hyperparameter
    :param validation:              flag indicating whether a validation set should be created by halving the test set
    :param mondrian_kernel:         flag indicating whether mondrian kernel should be evaluated
    :param mondrian_forest:         flag indicating whether mondrian forest should be evaluated
    :param weights_from_lifetime:   lifetime at which forest and kernel learned weights should be saved
    :return: dictionary res containing all results
    """

    N, D = np.shape(X)
    N_test = np.shape(X_test)[0]
    X_all = np.array(np.r_[X, X_test])
    N_all = N + N_test

    
    y = np.squeeze(y)
    y_test = np.squeeze(y_test)

    # subtract target means
    y_mean = np.mean(y)
    y_train = y - y_mean

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    C = M

    # bounding box for all datapoints used to sample first cut in each tree
    feature_data = [np.array(range(N_all)) for _ in range(M)]
    lX = np.min(X_all, 0)
    uX = np.max(X_all, 0)

    # event = tuple (time, tree, feature, dim, loc), where feature is the index of feature being split
    events = []
    active_features = []
    active_features_in_tree = [[] for _ in range(M)]
    for m in range(M):
        cut_time, dim, loc = sample_cut(lX, uX, 0.0)
        if cut_time < lifetime_max:
            heapq.heappush(events, (cut_time, m, m, dim, loc))
        active_features.append(m)
        active_features_in_tree[m].append(m)

    # iterate through birth times in increasing order
    list_times = []

    w_kernel = np.zeros(M)
    list_kernel_error_test = []

    while len(events) > 0:
        (birth_time, m, c, dim, loc) = heapq.heappop(events)
        list_times.append(birth_time)

        # construct new feature
        Xd = X_all[feature_data[c], dim]
        feature_l = (feature_data[c])[Xd <= loc]
        feature_r = (feature_data[c])[Xd  > loc]
        feature_data.append(feature_l)
        feature_data.append(feature_r)

        active_features.remove(c)
        active_features_in_tree[m].remove(c)
        active_features.append(C + 0)
        active_features.append(C + 1)
        active_features_in_tree[m].append(C + 0)
        active_features_in_tree[m].append(C + 1)

        # move datapoints from split feature to child features
        Z_all.indices[feature_l * M + m] = C + 0
        Z_all.indices[feature_r * M + m] = C + 1
        Z_all = scipy.sparse.csr_matrix((Z_all.data, Z_all.indices, Z_all.indptr), shape=(N_all, C + 2), copy=False)

        # sample the cut for each child
        lX_l = np.min(X_all[feature_l, :], axis=0)
        uX_l = np.max(X_all[feature_l, :], axis=0)
        cut_time_l, dim_l, loc_l = sample_cut(lX_l, uX_l, birth_time)
        lX_r = np.min(X_all[feature_r, :], axis=0)
        uX_r = np.max(X_all[feature_r, :], axis=0)
        cut_time_r, dim_r, loc_r = sample_cut(lX_r, uX_r, birth_time)

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r))
        
        C += 2

        w_kernel = np.append(w_kernel, [w_kernel[c], w_kernel[c]])
        w_kernel[c] = 0
        Z_train = Z_all[:N]
        Z_test = Z_all[N:]

        clf = linear_model.SGDRegressor(alpha=delta, fit_intercept=False)
        clf.fit(Z_train, y_train, coef_init=w_kernel)
        w_kernel = clf.coef_

        y_hat_test = y_mean + Z_test.dot(w_kernel)
        error_test = mean_squared_error(y_test, y_hat_test)
        list_kernel_error_test.append(error_test)


    # this function returns a dictionary with all values of interest stored in it
    results = {'times': list_times, 'mse': list_kernel_error_test}
    return results