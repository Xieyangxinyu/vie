import heapq
import numpy as np
import scipy.sparse
from sklearn import linear_model
import sys
import os
from sklearn import preprocessing
import pandas as pd
from copy import deepcopy

from utils_mondrian import sample_cut, errors_regression


def train(X, y, M, lifetime_max, delta,
          mondrian_kernel=False, mondrian_forest=False, weights_from_lifetime=None):
    """
    Sweeps through Mondrian kernels with all lifetime in [0, lifetime_max]. This can be used to (1) construct a Mondrian
    feature map with lifetime lifetime_max, to (2) find a suitable lifetime (inverse kernel width), or to (3) compare
    Mondrian kernel to Mondrian forest across lifetimes.
    :param X:                       training inputs
    :param y:                       training regression targets
    :param M:                       number of Mondrian trees
    :param lifetime_max:            terminal lifetime
    :param delta:                   ridge regression regularization hyperparameter
    :param mondrian_kernel:         flag indicating whether mondrian kernel should be evaluated
    :param mondrian_forest:         flag indicating whether mondrian forest should be evaluated
    :param weights_from_lifetime:   lifetime at which forest and kernel learned weights should be saved
    :return: dictionary res containing all results
    """
    
    N, D = np.shape(X)
    history = []

    if mondrian_forest or mondrian_kernel:
        y = np.squeeze(y)

        # subtract target means
        y_mean = np.mean(y)
        y_train = y - y_mean

    # initialize sparse feature matrix
    indptr = range(0, M * N + 1, M)
    indices = list(range(M)) * N
    data = np.ones(N * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, M))
    feature_from_repetition = list(range(M))
    C = M
    X_bd_all = np.tile(X, (M*D,1)).reshape(M,D,N,D)

    # bounding box for all datapoints used to sample first cut in each tree
    feature_data = [np.array(range(N)) for _ in range(M)]
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
    if mondrian_forest:
        w_trees = [np.zeros(1) for _ in range(M)]
        trees_y_hat_train = np.zeros((N, M))        # initialize Mondrian tree predictions and squared errors
    if mondrian_kernel:
        w_kernel = np.zeros(M)

    while len(events) > 0:
        (birth_time, m, c, dim, loc) = heapq.heappop(events)
        history.append((birth_time, m, c, dim, loc))

        # construct new feature
        Xd = X[feature_data[c], dim]
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
        Z_all = scipy.sparse.csr_matrix((Z_all.data, Z_all.indices, Z_all.indptr), shape=(N, C + 2), copy=False)

        # sample the cut for each child
        lX_l = np.min(X[feature_l, :], axis=0)
        uX_l = np.max(X[feature_l, :], axis=0)
        cut_time_l, dim_l, loc_l = sample_cut(lX_l, uX_l, birth_time)
        lX_r = np.min(X[feature_r, :], axis=0)
        uX_r = np.max(X[feature_r, :], axis=0)
        cut_time_r, dim_r, loc_r = sample_cut(lX_r, uX_r, birth_time)

        if (loc_l is not None) and (loc_r is not None):
            X_bd_all[m, dim, feature_l, dim] = lX_r[dim]
            X_bd_all[m, dim, feature_r, dim] = uX_l[dim]

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r))

        feature_from_repetition.append(m)
        feature_from_repetition.append(m)
        C += 2

        if mondrian_forest:
            # update Mondrian forest predictions in tree m
            Z_train = Z_all[:N, active_features_in_tree[m]]
            w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                                np.transpose(Z_train).dot(y_train))
            if weights_from_lifetime is not None and birth_time <= weights_from_lifetime:
                w_trees[m] = w_tree / np.sqrt(M)
            trees_y_hat_train[:, m] = np.squeeze(Z_train.dot(w_tree))

        # update Mondrian kernel predictions
        if mondrian_kernel:
            w_kernel = np.append(w_kernel, [w_kernel[c], w_kernel[c]])
            w_kernel[c] = 0

            clf = linear_model.SGDRegressor(alpha=delta, fit_intercept=False)
            clf.fit(Z_all, y_train, coef_init=w_kernel)
            w_kernel = clf.coef_

    # this function returns a dictionary with all values of interest stored in it
    
    results = {'Z': Z_all, 'feature_from_repetition': np.array(feature_from_repetition)}
    if mondrian_kernel:
        y_hat_train = y_mean + Z_all.dot(w_kernel)
    return results, X_bd_all, X, history, w_kernel, y_hat_train


def prepare_training_data(data, n_obs):
    df_train = data.head(n_obs)
    df_test = data.tail(40)

    x_train, y_train, f_train = df_train, df_train.pop("y"), df_train.pop("f")
    x_test, y_test, f_test = df_test, df_test.pop("y"), df_test.pop("f")

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    y_train = y_train.to_numpy().reshape(-1, 1).ravel()
    y_test = y_test.to_numpy().reshape(-1, 1).ravel()
    #y_test = np.hstack([y_test, y_test])

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test

def evaluate(X, y, X_test, M, delta, history, w_kernel,
             mondrian_kernel=False, mondrian_forest=False, weights_from_lifetime=None,):
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
    history = deepcopy(history)

    if mondrian_forest or mondrian_kernel:
        y = np.squeeze(y)

        # subtract target means
        y_mean = np.mean(y)
        y_train = y - y_mean

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    C = M

    # iterate through birth times in increasing order
    if mondrian_forest:
        w_trees = [np.zeros(1) for _ in range(M)]
        trees_y_hat_train = np.zeros((N, M))        # initialize Mondrian tree predictions and squared errors
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

        if mondrian_forest:
            # update Mondrian forest predictions in tree m
            Z_train = Z_all[:N, active_features_in_tree[m]]
            Z_test = Z_all[N:, active_features_in_tree[m]]
            w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                                np.transpose(Z_train).dot(y_train))
            if weights_from_lifetime is not None and birth_time <= weights_from_lifetime:
                w_trees[m] = w_tree / np.sqrt(M)
            trees_y_hat_train[:, m] = np.squeeze(Z_train.dot(w_tree))
            trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))

            # update Mondrian forest error
            y_hat_train = y_mean + np.mean(trees_y_hat_train, 1)
            y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)

        # update Mondrian kernel predictions
    
    if mondrian_kernel:
        Z_train = Z_all[:N]
        Z_test = Z_all[N:]
        y_hat_train = y_mean + Z_train.dot(w_kernel)
        y_hat_test = y_mean + Z_test.dot(w_kernel)
    return y_hat_train, y_hat_test
    


def simulate_y(x):
    y = x[:, 0]**4 + x[:, 1]**4 + x[:, 2]**4 + x[:, 3]**4 + x[:, 4]**4
    return y

import matplotlib.pyplot as plt
def draw(dim_in, psi_est, labels = None):
    x = np.linspace(0, dim_in-1, dim_in)
    fig, ax = plt.subplots()

    ax.plot(x, psi_est, linewidth=1.0, label = "diag")
    if labels is not None:
        plt.xticks(x, labels)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    '''
    data_path = os.path.join("./datasets/")
    
    dataset_name = 'cont' # @param ['cat', 'cont', 'adult', 'heart', 'mi'] 
    outcome_type = 'rbf' # @param ['linear', 'rbf', 'matern32', 'complex']
    n_obs = 1000 # @param [100, 200, 500, 1000]
    dim_in = 25 # @param [25, 50, 100, 200]
    rep =  1# @param 

    data_file = f"{outcome_type}_n{n_obs}_d{dim_in}_i{rep}.csv"
    data_file_path = os.path.join(data_path, dataset_name, data_file)
    print(f"Data '{data_file}'", end='\t', flush=True)

    data = pd.read_csv(data_file_path, index_col=0)
    x_train, y_train, x_test, y_test = prepare_training_data(data, n_obs)
    '''
    n = 1000
    dim_in = 10
    x_train = np.random.rand(n,dim_in)*2 - 1
    y_train = simulate_y(x_train)


    M = 200                      # number of Mondrian trees to use
    lifetime_max = 0.2          # terminal lifetime
    weights_lifetime = 2*1e-6   # lifetime for which weights should be plotted
    delta = 0.1              # ridge regression delta
    result, X_bd_all, X, history, w_kernel, y_hat_train = train(x_train, y_train, M, lifetime_max, delta, mondrian_kernel = True,
                                weights_from_lifetime=weights_lifetime)
    
    #y_hat_train, y_hat_test = evaluate(x_train, y_train, x_test, M, delta, history, w_kernel, mondrian_kernel = True, 
    #         weights_from_lifetime=weights_lifetime)
    
    importance = []
    for dim in range(dim_in):
        x_eval = None
        y_eval = []
        x_diff = []
        for tree in range(M):
            if sum(sum(abs(X_bd_all[tree,dim] - X))) > 0:
                temp = X_bd_all[tree,dim] - X
                subset = temp[:,dim] != 0
                if x_eval is None:
                    x_eval = X_bd_all[tree,dim][subset]
                else:
                    x_eval = np.vstack((x_eval, X_bd_all[tree,dim][subset]))
                y_eval = y_eval + list(y_hat_train[subset])
                x_diff = x_diff + list(temp[:,dim][subset])
        _, y_hat_test = evaluate(x_train, y_train, x_eval, M, delta, history, w_kernel, mondrian_kernel = True, 
                                weights_from_lifetime=weights_lifetime)
        
        importance.append(np.mean(((y_eval - y_hat_test)/x_diff)**2))
    draw(dim_in, importance)

        