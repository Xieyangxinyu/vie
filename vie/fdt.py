# standard library imports
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
# local imports
import vie.util as util


class fdt(object):
    """Featurized decision tree."""
    def __init__(self, model, x_train, y_train, c=1.0, sig2=0.01):
        super().__init__()

        self.x_train = x_train
        self.y_train = y_train
        self.c = c
        self.sig2 = sig2
        self.feature = model.tree_.feature
        self.threshold = model.tree_.threshold
        self.children_left = model.tree_.children_left
        self.children_right = model.tree_.children_right
        self.stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))
        try:
            self.hidden_features = self.feature[jnp.where(self.children_left > 0)[0]]
            self.hidden_threshold = self.threshold[jnp.where(self.children_left > 0)[0]]
        except:
            self.hidden_features = self.feature[np.where(self.children_left > 0)[0]]
            self.hidden_threshold = self.threshold[np.where(self.children_left > 0)[0]]
        self.build_map_matrix()
        leaf_id = model.apply(x_train)
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(leaf_id.reshape(-1, 1))
        self.feature0 = enc.transform(leaf_id.reshape(-1, 1)).toarray()

    def train(self):
        feature_soft = self.compute_feature(self.x_train)
        feature_soft = feature_soft * self.feature0
        Sigma_tmp = jnp.diag(1 / (jnp.diag(jnp.matmul(feature_soft.T, feature_soft)) + self.sig2))
        feature_soft = jnp.matmul(feature_soft, Sigma_tmp)
        self.beta = jnp.matmul(feature_soft.T, self.y_train)
        self.Sigma_beta = self.sig2 * Sigma_tmp

    def compute_feature(self, x):
        hidden_features_matrix = x[:, self.hidden_features]
        right_indicator = self.stable_sigmoid(self.c * (hidden_features_matrix - self.hidden_threshold))
        left_indicator = 1. - right_indicator
        soft_indicator = jnp.concatenate([left_indicator, right_indicator], axis=1)
        F_leaf = jnp.multiply(soft_indicator[:, :, np.newaxis], self.map_matrix) + (1.0 - self.map_matrix)
        return jnp.prod(F_leaf, axis=1)

    def build_map_matrix(self):
        # build two dfs, one for storing mapping info for left children, one for right
        column_name = [str(id) for id in jnp.where(self.children_left < 0)[0]]
        row_name = [str(id) for id in jnp.where(self.children_left > 0)[0]]
        df_left = pd.DataFrame(np.zeros((len(row_name), len(column_name))),
                               index=row_name, columns=column_name)
        df_right = pd.DataFrame(np.zeros((len(row_name), len(column_name))),
                                index=row_name, columns=column_name)
        df_list = [df_left, df_right]

        # once reach leaf node, update dfs
        def process_df(path):
            leaf_node = str(path[-1][0])
            flag = path[-1][1]
            for element in path[:-1][::-1]:
                df_list[flag][leaf_node].loc[str(element[0])] = 1.0
                flag = element[1]

        # recursively obtain root to leaf path with direction
        def pre_order(index, flag, path, children_left, children_right):
            path.append((index, flag))
            if children_left[index] < 0:
                process_df(path)
            if children_left[index] > 0:
                pre_order(children_left[index], 0, path, children_left, children_right)
            if children_right[index] > 0:
                pre_order(children_right[index], 1, path, children_left, children_right)
            del path[-1]

        flag = -1
        path = []
        pre_order(0, flag, path, self.children_left, self.children_right)
        self.map_matrix = jnp.concatenate([df_list[0].to_numpy(), df_list[1].to_numpy()], axis=0)



def get_fdt_model(x_train, y_train, c=1.0, sig2=0.01, n_tree=20, compute_psi=True,
                  batch_size=20, n_samp=100, seed=0):

    n_obs = x_train.shape[0]
    max_leaf_nodes = 2 ** (int(np.log2(np.sqrt(n_obs) * np.log(n_obs))) + 1)
    extra = ExtraTreesRegressor(n_estimators=n_tree, max_leaf_nodes=max_leaf_nodes,
                                random_state=seed).fit(x_train, y_train)

    beta_set = np.zeros((extra.n_estimators, max_leaf_nodes))
    betas_set = np.zeros((extra.n_estimators, n_samp, max_leaf_nodes))
    map_matrix_set = np.zeros((extra.n_estimators, 2 * (max_leaf_nodes - 1), max_leaf_nodes))
    feature_set = np.zeros((extra.n_estimators, max_leaf_nodes - 1))
    threshold_set = np.zeros((extra.n_estimators, max_leaf_nodes - 1))

    for j, model in enumerate(extra.estimators_):
        m = fdt(model, x_train, y_train, c=c, sig2=sig2)
        m.train()
        beta_set[j, :] = m.beta
        map_matrix_set[j, :, :] = m.map_matrix
        feature_set[j, :] = m.hidden_features
        threshold_set[j, :] = m.hidden_threshold
        beta_set[j, :] = m.beta
        try:
            beta_samp = np.random.multivariate_normal(m.beta, m.Sigma_beta, size=n_samp)
        except:
            beta_samp = np.random.multivariate_normal(m.beta,
                                                      np.diag(np.diag(m.Sigma_beta)),
                                                      size=n_samp)  # (n_samp, D)
        betas_set[j, :, :] = beta_samp

    beta_set = jnp.asarray(beta_set)
    betas_set = jnp.asarray(betas_set)
    map_matrix_set = jnp.asarray(map_matrix_set)
    feature_set = jnp.asarray(feature_set, dtype=int)
    threshold_set = jnp.asarray(threshold_set)

    stable_sigmoid = lambda x: jnp.exp(jax.nn.log_sigmoid(x))

    def predict(x, map_matrix, feature, threshold, beta):
        hidden_features = jnp.asarray(x)[feature]
        right_indicator = stable_sigmoid(c * (hidden_features - threshold))
        left_indicator = 1. - right_indicator
        soft_indicator = jnp.concatenate([left_indicator, right_indicator], axis=0)
        F_leaf = jnp.multiply(soft_indicator[:, np.newaxis], map_matrix) + (1.0 - map_matrix)
        soft_feature = jnp.prod(F_leaf, axis=0)
        return jnp.dot(soft_feature, beta)

    out_set = {"map_matrix": map_matrix_set, "feature": feature_set, "threshold": threshold_set,
               "beta": beta_set, "betas": betas_set}

    f = jax.jit(jax.vmap(jax.vmap(predict, in_axes=(None, 0, 0, 0, 0), out_axes=0),
                         in_axes=(0, None, None, None, None), out_axes=0))

    grad_fs = jax.jit(jax.vmap(jax.vmap(jax.vmap(jax.grad(predict, argnums=0),
                                                 in_axes=(None, 0, 0, 0, 0), out_axes=0),
                                        in_axes=(0, None, None, None, None), out_axes=0),
                               in_axes=(None, None, None, None, 1), out_axes=0))

    psi_est = np.zeros(x_train.shape[1])

    if compute_psi:
        if n_obs > batch_size:
            X_batches = util.split_into_batches(x_train, batch_size=batch_size)
            psi_est_all = np.array(grad_fs(X_batches[0], map_matrix_set, feature_set, threshold_set, betas_set))
            for batch_id in range(1, X_batches.__len__()):
                X_batch = X_batches[batch_id]
                psi_est_tmp = np.array(grad_fs(X_batch, map_matrix_set, feature_set, threshold_set, betas_set))
                psi_est_all = np.concatenate([psi_est_all, psi_est_tmp], axis=0)
        else:
            psi_est_all = np.array(grad_fs(x_train, map_matrix_set, feature_set, threshold_set, betas_set))

        grad_train = np.mean(psi_est_all ** 2, axis=(0, 1))
        psi_est = np.median(grad_train, axis=0)
    print(f"Shapes: psi_est_all: {psi_est_all.shape}, grad_train: {grad_train.shape}, psi_est: {psi_est.shape}")
    return psi_est_all, f, grad_fs, out_set

