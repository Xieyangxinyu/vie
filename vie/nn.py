# standard library imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# local imports
import vie.util as util

# Neural network
# @title MetricsCallback
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_train, x_test, y_test, outlier_mse_cutoff=[50, 100], dim_in=25):
        super().__init__()

        self.history = {'psi': []}
        self.history.update({f'y_mse_{thresh}': [] for thresh in outlier_mse_cutoff})
        self.x_train = x_train
        self.x_test = x_test
        self.y_test = y_test
        self.outlier_mse_cutoff = outlier_mse_cutoff

    def estimate_psi(self, sig2=0.01, n_samp=100, batch_size=100):
        W1 = self.model.layers[1].weights[0].numpy()  # d X K
        b1 = self.model.layers[1].bias.numpy()  # vector of length K
        W2 = self.model.layers[2].weights[0].numpy()  # K X 1
        X = self.x_train
        if X.ndim == 1:
            X = X[None, :]
        layer1 = X @ W1 + b1
        phi = layer1 * (layer1 > 0)
        phi_batches = util.split_into_batches(phi, batch_size)
        num_batch = phi_batches.__len__()
        weight_cov_val = np.float64(util.compute_inverse(phi_batches[0], sig_sq=sig2))

        for batch_id in range(1, num_batch):
            H_inv = weight_cov_val
            phi_batch = phi_batches[batch_id]
            weight_cov_val = util.minibatch_woodbury_update(phi_batch, H_inv)
        Sigma_beta = weight_cov_val * sig2

        try:
            beta_samp = np.random.multivariate_normal(W2[:, 0], Sigma_beta, size=n_samp)
        except:
            beta_samp = np.random.multivariate_normal(W2[:, 0], np.diag(np.diag(Sigma_beta)),
                                                      size=n_samp)  # (n_samp, D)

        drelu = (X @ W1 + b1 > 0).astype(float)
        layer2_samp = np.multiply(drelu[:, :, None], beta_samp.T)
        layer2 = np.mean(layer2_samp, axis=2)
        gradient = layer2 @ W1.T
        psi = np.mean(gradient ** 2, 0)
        return psi

    def compute_y_mse(self, outlier_mse_cutoff):
        y_est = self.model.predict(self.x_test, verbose=0).flatten()
        mse = (self.y_test - y_est) ** 2
        # Remove outlier MSEs.
        mse = mse[mse < outlier_mse_cutoff]
        return np.mean(mse)

    def on_epoch_end(self, epoch, logs=None):
        psi = self.estimate_psi()

        for thresh in self.outlier_mse_cutoff:
            self.history[f'y_mse_{thresh}'].append(self.compute_y_mse(thresh))

        self.history['psi'].append(psi)

# @title OutlierRobustMSE
class OutlierRobustMSE(tf.keras.metrics.Metric):
    def __init__(self, outlier_mse_cutoff=10., name='robust_mse', **kwargs):
        name = name + '_' + str(outlier_mse_cutoff)
        super().__init__(name=name, **kwargs)
        self.outlier_mse_cutoff = outlier_mse_cutoff
        self.mse = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight):
        batch_mse = tf.math.square(y_true - y_pred)
        not_outlier = tf.less(batch_mse, self.outlier_mse_cutoff)

        batch_mse = tf.boolean_mask(batch_mse, not_outlier)

        self.mse.update_state(values=batch_mse, sample_weight=sample_weight)

    def result(self):
        return self.mse.result()

    def reset_state(self):
        self.mse.reset_state()

# @title get_nn_model
def get_nn_model(dim_in, dim_hidden=1024, seed=0, lr=1e-3,
                       l1=1e-2, l2=1e-2, outlier_mse_cutoff=[50, 100]):
    inputs = keras.Input(shape=(dim_in,), name="input")
    x = layers.Dense(dim_hidden,
                     kernel_initializer=keras.initializers.RandomNormal(seed=seed),
                     bias_initializer=keras.initializers.RandomUniform(seed=seed),
                     kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2),
                     bias_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2),
                     activation="relu",
                     name="hidden")(inputs)
    outputs = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            OutlierRobustMSE(outlier_mse_cutoff=cutoff)
            for cutoff in outlier_mse_cutoff],
    )
    return model


