import heapq
import numpy as np
import scipy.sparse
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm


# SAMPLING
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

def populate_importance(subset_vector, importance):
    importance = list(importance)
    m = len(subset_vector)
    res = np.zeros(m)
    for i in range(m):
        if subset_vector[i] != 0:
            res[i] = importance.pop()
    return res

def two_one_norm(H):
    return np.sum(np.apply_along_axis(norm, 0, H)) / H.shape[1]


def transform_data(H, x_train, x_test):
    x_train_transformed = np.matmul(x_train, H)
    x_test_transformed = np.matmul(x_test, H)
    return x_train_transformed, x_test_transformed

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
    lX = np.min(X, 0)
    uX = np.max(X, 0)

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
    trees_y_hat_test = np.zeros((N_test, M))
    list_forest_error_test = []

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
        feature_l = feature_l[feature_l < N]
        feature_r = feature_r[feature_r < N]
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
        Z_train = Z_all[:N, active_features_in_tree[m]]
        Z_test = Z_all[N:, active_features_in_tree[m]]
        w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                            np.transpose(Z_train).dot(y_train))
        trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))

        y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)
        list_forest_error_test.append(mean_squared_error(y_test, y_hat_test))

    # this function returns a dictionary with all values of interest stored in it
    results = {'times': list_times, 'y_hat_test': y_hat_test, 'mse': list_forest_error_test}
    
    return results, y_hat_test



def evaluate(y, X_test, M, history, w_trees):
    """
    Sweeps through Mondrian kernels with all lifetime in [0, lifetime_max]. This can be used to (1) construct a Mondrian
    feature map with lifetime lifetime_max, to (2) find a suitable lifetime (inverse kernel width), or to (3) compare
    Mondrian kernel to Mondrian forest across lifetimes.
    :param X:                       training inputs
    :param y:                       training regression targets
    :param X_test:                  test inputs
    :param M:                       number of Mondrian trees
    :param w_trees:                 
    :return: dictionary res containing all results
    """
    N_test = np.shape(X_test)[0]
    X_all = np.array(np.r_[X_test])
    N_all = N_test
    history = deepcopy(history)

    # subtract target means
    y_mean = np.mean(y)

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    C = M

    trees_y_hat_test = np.zeros((N_test, M))
    
    feature_data = [np.array(range(N_all)) for _ in range(M)]
    active_features = []
    active_features_in_tree = [[] for _ in range(M)]
    for m in range(M):
        active_features.append(m)
        active_features_in_tree[m].append(m)

    while len(history) > 0:
        (birth_time, m, c, dim, loc) = history.pop(0)

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
        C += 2

    for m in range(M):
        Z_test = Z_all[:, active_features_in_tree[m]]
        w_tree = w_trees[m]
        trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))

    y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)

    return y_hat_test

import random
def estimate_H_ind(x_train, y_train, X_bd_all, M, X_test, history, w_trees, y_hat_test):
    dim_in = X_test.shape[1]
    N_test = X_test.shape[0]
    importance = []
    for dim in range(dim_in):
        x_eval = None
        y_eval = []
        x_diff = []
        subset_all = []
        #sample_size = min(M, 15)
        #random_trees = random.sample(range(M), sample_size)
        for tree in range(M):
            temp = X_bd_all[tree,dim] - X_test
            subset = temp[:,dim] != 0
            subset_all = subset_all + list(subset)
            
            if sum(subset) > 0:
                if x_eval is None:
                    x_eval = X_bd_all[tree,dim][subset]
                else:
                    x_eval = np.vstack((x_eval, X_bd_all[tree,dim][subset]))
                y_eval = y_eval + y_hat_test[subset].tolist()
                x_diff = x_diff + list(temp[:,dim][subset])
        
        y_hat_eval = evaluate(y_train, x_eval, M, history, w_trees)
        y_eval = np.array(y_eval)
        x_diff = np.array(x_diff)
        importance_temp = populate_importance(subset_all, ((y_eval - y_hat_eval)/x_diff))
        importance_temp = np.reshape(importance_temp, (N_test, M))
        importance_temp = np.median(importance_temp, axis = 1)
        importance.append(importance_temp)
    importance = np.vstack(importance)

    H = np.matmul(importance, np.transpose(importance))/N_test
    return H


def one_run(X, y, X_test, M, lifetime_max, delta):

    N, D = np.shape(X)
    N_test = np.shape(X_test)[0]
    X_all = np.array(np.r_[X, X_test])
    N_all = N + N_test

    # subtract target means
    y_mean = np.mean(y)
    y_train = y - y_mean

    history = []

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    C = M

    X_bd_all = np.tile(X_test, (M*D,1)).reshape(M,D,N_test,D)

    # bounding box for all datapoints used to sample first cut in each tree
    feature_data = [np.array(range(N_all)) for _ in range(M)]
    lX = np.min(X, 0)
    uX = np.max(X, 0)

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
    trees_y_hat_test = np.zeros((N_test, M))

    while len(events) > 0:
        (birth_time, m, c, dim, loc) = heapq.heappop(events)

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
        feature_l_train = feature_l[feature_l < N]
        feature_r_train = feature_r[feature_r < N]
        lX_l = np.min(X_all[feature_l_train, :], axis=0)
        uX_l = np.max(X_all[feature_l_train, :], axis=0)
        cut_time_l, dim_l, loc_l = sample_cut(lX_l, uX_l, birth_time)
        lX_r = np.min(X_all[feature_r_train, :], axis=0)
        uX_r = np.max(X_all[feature_r_train, :], axis=0)
        cut_time_r, dim_r, loc_r = sample_cut(lX_r, uX_r, birth_time)

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r))

        feature_l_test = feature_l[feature_l >= N]
        feature_r_test = feature_r[feature_r >= N]
        if (loc_l is not None) and (loc_r is not None):
            X_bd_all[m, dim, feature_l_test - N, dim] = lX_r[dim]
            X_bd_all[m, dim, feature_r_test - N, dim] = uX_l[dim]

        C += 2
        history.append((birth_time, m, c, dim, loc))

    w_trees = []
    for m in range(M):
        Z_train = Z_all[:N, active_features_in_tree[m]]
        Z_test = Z_all[N:, active_features_in_tree[m]]
        w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                                np.transpose(Z_train).dot(y_train))
        trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))
        w_trees.append(w_tree)
    
    y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)

    return y_hat_test, history, X_bd_all, w_trees


def estimate_H_ind_(x_train, y_train, X_bd_all, M, X_test, history, w_trees, y_hat_test):
    dim_in = X_test.shape[1]
    N_test = X_test.shape[0]
    importance = []
    for dim in range(dim_in):
        x_eval = None
        y_eval = []
        x_diff = []
        subset_all = []
        for tree in range(M):
            temp = X_bd_all[tree,dim] - X_test
            subset = temp[:,dim] != 0
            subset_all = subset_all + list(subset)
            
            if sum(subset) > 0:
                if x_eval is None:
                    x_eval = X_bd_all[tree,dim][subset]
                else:
                    x_eval = np.vstack((x_eval, X_bd_all[tree,dim][subset]))
                y_eval = y_eval + y_hat_test[subset].tolist()
                x_diff = x_diff + list(temp[:,dim][subset])
        
        y_hat_eval = evaluate(y_train, x_eval, M, history, w_trees)
        y_eval = np.array(y_eval)
        x_diff = np.abs(x_diff)
        y_diff = populate_importance(subset_all, y_eval - y_hat_eval)
        x_diff = populate_importance(subset_all, x_diff)
        y_diff = np.reshape(y_diff, (N_test, M))
        x_diff = np.reshape(x_diff, (N_test, M))
        y_diff = np.mean(y_diff, axis = 1)
        x_diff = np.mean(x_diff, axis = 1)
        y_diff[x_diff == 0] = 0
        x_diff[x_diff == 0] = 1
        importance_temp = y_diff/x_diff
        importance.append(importance_temp)
    importance = np.vstack(importance)

    H = np.matmul(importance, np.transpose(importance))/N_test
    return H
