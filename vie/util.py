# standard library imports
import numpy as np

def split_into_batches(X, batch_size):
    """Split X into a list of np array, each with row of size batch_size.

    Args:
        X: (np array) n x d input matrix.
        batch_size: (integer) an integer indicating the size of each batch.

    Returns:
        A list of np array, each with row of size batch_size.
    """
    return [X[i:i + batch_size] for i in range(0, len(X), batch_size)]


def compute_inverse(X, sig_sq=1.):
    """Compute inverse of (X^T X + sig_sq * I).

    Args:
        X: (np array) n x d input matrix.
        sig_sq: (numeric) the number specifying the variance of the noise.

    Returns:
        The inverse matrix of (X^T X + sig_sq * I).
    """
    return np.linalg.inv(np.matmul(X.T, X) + sig_sq * np.identity(X.shape[1]))


def minibatch_woodbury_update(X, H_inv):
    """Minibatch update of linear regression posterior covariance
    using Woodbury matrix identity.
    inv(H + X^T X) = H_inv - H_inv X^T inv(I + X H_inv X^T) X H_inv
    Args:
        X: (np array) A n x batch_size matrix of batched observation.
        H_inv: (np array) A D x D matrix of posterior covariance of rff coefficients.
    Returns:
        H_new: (np array) D x D covariance matrix after update.
    """
    batch_size = X.shape[0]

    M0 = np.eye(batch_size, dtype=np.float64) + np.matmul(X, np.matmul(H_inv, X.T))
    M = np.linalg.inv(M0)
    B = np.matmul(X, H_inv)
    H_new = H_inv - np.matmul(B.T, np.matmul(M, B))
    return H_new


def minibatch_interaction_update(Phi_y, rff_output, Y_batch):
    """Minibatch update of interaction between feature matrix and Y.
    Args:
        Phi_y: (np array) A D x 1 matrix of the interaction.
        rff_output: (np array) A n x D matrix of the feature matrix.
        Y_batch: (np array) A n x 1 matrix of the batched response.
    Returns:
        A D x 1 matrix of the updated interaction.
    """
    return Phi_y + np.matmul(rff_output.T, Y_batch)

