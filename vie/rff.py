# standard library imports
import numpy as np
import tensorflow as tf
# local imports
import vie.util as util
import vie.kernelized as kernel_layers


class rff(object):
    """Random fourier feature."""
    def __init__(self, X, Y, dim_hidden=1024, sig2=0.01, lengthscale=1., seed=None):
        super().__init__()

        self.X = X
        self.Y = Y
        self.dim_in = X.shape[1]
        self.sig2 = sig2
        self.lengthscale = lengthscale
        self.dim_hidden = dim_hidden
        self.seed = seed

        self.rff_layer = kernel_layers.RandomFourierFeatures(
            output_dim=self.dim_hidden,
            kernel_initializer='gaussian',
            scale=self.lengthscale,
            seed=self.seed)
        self.rff_layer.build(input_shape=(None, self.dim_in))
        self.RFF_weight = self.rff_layer.kernel
        self.RFF_bias = self.rff_layer.bias

    def train(self, batch_size=20, epochs=1):
        ### Training and Evaluation ###
        X_batches = util.split_into_batches(self.X, batch_size) * epochs
        Y_batches = util.split_into_batches(self.Y, batch_size) * epochs

        num_steps = X_batches.__len__()
        num_batch = int(num_steps / epochs)
        rff_1 = tf.cast(self.rff_layer(X_batches[0]) * np.sqrt(2. / self.dim_hidden),
                         dtype=tf.float64).numpy()
        weight_cov_val = np.float64(util.compute_inverse(rff_1, sig_sq=self.sig2))
        covl_xy_val = np.float64(np.matmul(rff_1.T, Y_batches[0]))

        for batch_id in range(1, num_batch):
            H_inv = weight_cov_val
            Phi_y = covl_xy_val
            X_batch = X_batches[batch_id]
            Y_batch = Y_batches[batch_id]

            ## update posterior mean/covariance
            try:
                rff_batch = tf.cast(self.rff_layer(X_batch) * np.sqrt(2. / self.dim_hidden),
                                     dtype=tf.float64).numpy()
                weight_cov_val = util.minibatch_woodbury_update(rff_batch, H_inv)
                covl_xy_val = util.minibatch_interaction_update(Phi_y, rff_batch, Y_batch)
            except:
                print("\n================================\n"
                      "Problem occurred at Step {}\n"
                      "================================".format(batch_id))

        self.beta = np.matmul(weight_cov_val, covl_xy_val)
        self.weight_cov_val = weight_cov_val
        self.covl_xy_val = covl_xy_val
        self.Sigma_beta = weight_cov_val * self.sig2

    def compute_feature(self, X):
        return np.sqrt(2. / self.dim_hidden) * np.cos(
            np.matmul(X, self.RFF_weight) + self.RFF_bias)

    def predict(self, X):
        D = self.dim_hidden
        rff_new = np.sqrt(2. / D) * np.cos(np.matmul(X, self.RFF_weight) +
                                            self.RFF_bias)
        pred_mean = np.matmul(rff_new, self.beta)
        pred_cov = np.matmul(np.matmul(rff_new, self.Sigma_beta), rff_new.T)

        return pred_mean.reshape((-1, 1)), pred_cov

    def estimate_psi(self, X, n_samp=100):
        nD_mat = np.sin(np.matmul(X, self.RFF_weight) + self.RFF_bias)
        n, d = X.shape
        D = self.RFF_weight.shape[1]

        psi_mean = np.zeros(self.dim_in)
        psi_var = np.zeros(self.dim_in)
        der_array = np.zeros((n, d, n_samp))
        try:
            beta_samp = np.random.multivariate_normal(self.beta, self.Sigma_beta, size=n_samp).T
        except:
            beta_samp = np.random.multivariate_normal(self.beta,
                                                      np.diag(np.diag(self.Sigma_beta)),
                                                      size=n_samp).T
        # (D, n_samp)
        for r in range(n):
            cur_mat = np.diag(nD_mat[r, :])
            cur_mat_W = np.matmul(self.RFF_weight, cur_mat)  # (d, D)
            cur_W_beta = np.matmul(cur_mat_W, beta_samp)  # (d, n_samp)
            der_array[r, :, :] = cur_W_beta

        der_array = der_array * np.sqrt(2. / D)
        for l in range(self.dim_in):
            grad_samp = der_array[:, l, :].T  # (n_samp, n)
            psi_samp = np.mean(grad_samp ** 2, 1)
            psi_mean[l] = np.mean(psi_samp)
            psi_var[l] = np.var(psi_samp)
        return psi_mean, psi_var
