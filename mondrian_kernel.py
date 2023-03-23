import heapq
import numpy as np
import scipy.sparse
from sklearn import linear_model
import jax
import jax.numpy as jnp
import sys, os
import time
import matplotlib.pyplot as plt

from utils import sample_cut, errors_regression

def plot_spectrum(y, y_diag, title):
    x = np.linspace(0, dim_in-1, dim_in)

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2.0, label = "eig")
    ax.plot(x, y_diag, linewidth=2.0, label = "diagonal")

    plt.legend()
    plt.title(title)
    plt.show()

def compute_derivative(q_tilde, d_q_tilde):
    stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))
    q = 2 * stable_sigmoid(q_tilde)
    derivative = 2 * np.matmul(np.diag(q * (1-q)), d_q_tilde)
    return derivative

def evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta,
                           validation=False, weights_from_lifetime=None):
    """
    Sweeps through Mondrian kernels with all lifetime in [0, lifetime_max]. This can be used to (1) construct a Mondrian
    feature map with lifetime lifetime_max, to (2) find a suitable lifetime (inverse kernel width).
    :param X:                       training inputs
    :param y:                       training regression targets
    :param X_test:                  test inputs
    :param y_test:                  test regression targets
    :param M:                       number of Mondrian trees
    :param lifetime_max:            terminal lifetime
    :param delta:                   ridge regression regularization hyperparameter
    :param validation:              flag indicating whether a validation set should be created by halving the test set
    :param weights_from_lifetime:   lifetime at which kernel learned weights should be saved
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

    # start timer
    time_start = time.process_time()

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    d_Z_all = np.zeros(shape=(M, N_all, D))
    feature_from_repetition = list(range(M))
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
            heapq.heappush(events, (cut_time, m, m, dim, loc, np.zeros(N_all), np.zeros((N_all, D))))
        active_features.append(m)
        active_features_in_tree[m].append(m)

    # iterate through birth times in increasing order
    list_times = []
    list_runtime = []

    w_kernel = np.zeros(M)
    w_kernel_save = np.zeros(M)
    list_kernel_error_train = []
    if validation:
        list_kernel_error_validation = []
    list_kernel_error_test = []

    while len(events) > 0:
        # event = tuple (time, tree, feature, dim, loc), where feature is the index of feature being split
        (birth_time, m, c, dim, loc, q_tilde, d_q_tilde) = heapq.heappop(events)
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
        
        
        d_q_tilde_l = np.copy(d_q_tilde)
        d_q_tilde_l[:, dim] = d_q_tilde[:, dim] - (np.abs(X_all[:, dim] - loc) < 1)
        q_tilde_l = q_tilde - np.sign(X_all[:, dim] - loc) - 1

        d_q_tilde_r = np.copy(d_q_tilde)
        d_q_tilde_r[:, dim] = d_q_tilde[:, dim] + (np.abs(X_all[:, dim] - loc) < 1)
        q_tilde_r = q_tilde + np.sign(X_all[:, dim] - loc) - 1

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

        d_Z_l = compute_derivative(q_tilde_l, d_q_tilde_l)
        d_Z_r = compute_derivative(q_tilde_r, d_q_tilde_r)
        d_Z_all = np.stack([*d_Z_all,d_Z_l])
        d_Z_all = np.stack([*d_Z_all,d_Z_r])
        #print(d_Z_all)

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l, q_tilde_l, d_q_tilde_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r, q_tilde_r, d_q_tilde_r))

        feature_from_repetition.append(m)
        feature_from_repetition.append(m)
        C += 2
        
        # update Mondrian kernel predictions
        w_kernel = np.append(w_kernel, [w_kernel[c], w_kernel[c]])
        w_kernel[c] = 0
        Z_train = Z_all[:N]
        Z_test = Z_all[N:]

        clf = linear_model.SGDRegressor(alpha=delta, fit_intercept=False)
        clf.fit(Z_train, y_train, coef_init=w_kernel)
        w_kernel = clf.coef_
        
        y_hat_train = y_mean + Z_train.dot(w_kernel)
        y_hat_test = y_mean + Z_test.dot(w_kernel)
        if validation:
            error_train, error_validation =\
                errors_regression(y, y_test[:(N_test/2)], y_hat_train, y_hat_test[:(N_test/2)])
            error_train, error_test =\
                errors_regression(y, y_test[(N_test/2):], y_hat_train, y_hat_test[(N_test/2):])
            list_kernel_error_validation.append(error_validation)
        else:
            error_train, error_test = errors_regression(y, y_test, y_hat_train, y_hat_test)
        list_kernel_error_train.append(error_train)
        list_kernel_error_test.append(error_test)

        # save runtime
        list_runtime.append(time.process_time() - time_start)

        # progress indicator in console
        #sys.stdout.write("\rTime: %.2E / %.2E (C = %d, test error = %.3f)" % (birth_time, lifetime_max, C, error_test))
        #sys.stdout.flush()
    
    d_Z_active = d_Z_all[active_features]
    w_kernel_active = np.array(w_kernel)[active_features]
    #H_root = np.tensordot(w_kernel_active, d_Z_active)
    print(d_Z_active.shape)
    d_Z_active = np.swapaxes(d_Z_active, 0,1)
    d_Z_active = np.swapaxes(d_Z_active, 1,2)
    print(d_Z_active.shape)
    print(w_kernel_active.shape)
    H_root = np.matmul(d_Z_active, w_kernel_active)
    H = np.matmul(np.transpose(H_root), H_root)
    eig = jnp.linalg.eig(H)[0]
    print(eig)
    y_diag = jnp.diagonal(H)
    y_diag = jnp.sort(y_diag)[::-1]
    plot_spectrum(eig, y_diag, 'spectrum')

    # this function returns a dictionary with all values of interest stored in it
    results = {'times': list_times, 'runtimes': list_runtime, 'Z': Z_all, 'feature_from_repetition': np.array(feature_from_repetition)}

    results['kernel_train'] = list_kernel_error_train
    results['kernel_test'] = list_kernel_error_test
    if validation:
        results['kernel_validation'] = list_kernel_error_validation
    
    return results


def Mondrian_kernel_features(X, lifetime, M):
    res = evaluate_all_lifetimes(X, None, np.empty((0, X.shape[1])), None, M, lifetime, None)
    Z = np.sqrt(M) * res['Z']   # undo normalization
    return Z, res['feature_from_repetition']


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd

def prepare_training_data(data, n_obs):
    df_train = data.head(n_obs)
    df_test = data.tail(40)

    x_train, y_train, f_train = df_train, df_train.pop("y"), df_train.pop("f")
    x_test, y_test, f_test = df_test, df_test.pop("y"), df_test.pop("f")

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    y_train = y_train.to_numpy().reshape(-1, 1).ravel()
    y_test = y_test.to_numpy().reshape(-1, 1).ravel()
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train) * 10
    x_test = scaler.transform(x_test) * 10
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    data_path = os.path.join("./datasets/")
    
    dataset_name = 'cont' # @param ['cat', 'cont', 'adult', 'heart', 'mi'] 
    outcome_type = 'linear' # @param ['linear', 'rbf', 'matern32', 'complex']
    n_obs = 200 # @param [100, 200, 500, 1000]
    dim_in = 25 # @param [25, 50, 100, 200]
    rep = 1 # @param 

    data_file = f"{outcome_type}_n{n_obs}_d{dim_in}_i{rep}.csv"
    data_file_path = os.path.join(data_path, dataset_name, data_file)
    print(f"Data '{data_file}'", end='\t', flush=True)

    data = pd.read_csv(data_file_path, index_col=0)
    x_train, y_train, x_test, y_test = prepare_training_data(data, n_obs)


    M = 10                      # number of Mondrian trees to use
    lifetime_max = 0.001          # terminal lifetime
    weights_lifetime = 2*1e-6   # lifetime for which weights should be plotted
    delta = 0.001              # ridge regression delta
    evaluate_all_lifetimes(x_train, y_train, x_test, y_test, M, lifetime_max, delta,
                                weights_from_lifetime=weights_lifetime)
    