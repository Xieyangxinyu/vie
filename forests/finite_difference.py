import numpy as np
from scipy.stats import special_ortho_group
import jax.numpy as jnp
from jax import grad, vmap
from Mondrian_forest import *
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def estimate_H_finite_diff(x_train, y_train, X_bd_all, M, X_test, history, delta, y_hat_test):
    dim_in = X_test.shape[1]
    N_test = X_test.shape[0]
    importance = []
    for dim in range(dim_in):
        x_eval_pos = deepcopy(X_test)
        x_eval_neg = deepcopy(X_test)
        x_diff = 0.5
        subset_all = []
        x_eval_pos[:,dim] = x_eval_pos[:,dim] + x_diff/2
        x_eval_neg[:,dim] = x_eval_neg[:,dim] - x_diff/2
        
        _, y_eval_pos = evaluate(x_train, y_train, x_eval_pos, M, history, delta)
        _, y_eval_neg = evaluate(x_train, y_train, x_eval_neg, M, history, delta)
        y_diff = y_eval_pos - y_eval_neg
        importance_temp = y_diff/x_diff
        
        importance.append(importance_temp)
    importance = np.vstack(importance)

    H = np.matmul(importance, np.transpose(importance))/N_test
    return H
