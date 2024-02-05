import jax.numpy as jnp
from jax import grad, vmap
from scipy.stats import special_ortho_group
import numpy as np
from numpy import linalg


class SimulatedData:
    def __init__(self, dim_in, active, rotation = None, fun = "poly", noise_sig2 = 0.01, seed = 0):
        self.dim_in = dim_in
        self.active = active
        self.noise_sig2 = noise_sig2
        self.seed = seed
        self.project = jnp.concatenate([jnp.ones(active), jnp.zeros(dim_in - active)])

        if rotation == None:
            self.rotation = np.identity(dim_in)
        elif rotation == "simple":
            self.rotation = np.identity(dim_in)
            self.rotation[1,0] = self.rotation[2,0] = self.rotation[3,1] = self.rotation[4,1] = 1
        elif rotation == "orth":
            self.rotation = special_ortho_group.rvs(dim_in, seed = seed)
        else:
            self.rotation = rotation

        if fun == "poly":
            self.fun = self.poly_fn
        elif fun == "max":
            self.fun = self.max_fn

    def add_noise(self, y):
        r_noise = np.random.RandomState(self.seed)
        noise = r_noise.randn(1)[0] * jnp.sqrt(self.noise_sig2)
        y = y + noise
        return y

    def poly_fn(self, x):
        res = jnp.dot(x, self.rotation)
        y = jnp.dot(res ** 4, self.project)
        return self.add_noise(y)
    
    def max_fn(self, x):
        res = jnp.dot(x, self.rotation)
        res = (res ** 2) * -0.25
        res = jnp.exp(res)
        y = 5 * jnp.max(res * self.project)
        return self.add_noise(y)
    
    def get_true_H(self, x_test):
        n_test = x_test.shape[0]
        W_grad = vmap(grad(self.fun), in_axes=0, out_axes=0)(x_test)
        true_H = jnp.matmul(jnp.transpose(W_grad), W_grad) / n_test
        return true_H