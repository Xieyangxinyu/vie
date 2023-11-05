import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from src.Mondrian_forest import evaluate

def estimate_H_finite_diff(X, y_train, M, history, w_trees, step = 0.5):
    X = np.array(X)
    dim_in = X.shape[1]
    N = X.shape[0]
    importance = []
    x_diff = step / 2.0
    for dim in range(dim_in):
        x_eval_pos = deepcopy(X)
        x_eval_neg = deepcopy(X)
        x_eval_pos[:,dim] = x_eval_pos[:,dim] + x_diff
        x_eval_neg[:,dim] = x_eval_neg[:,dim] - x_diff
        
        y_eval_pos = evaluate(y_train, x_eval_pos, M, history, w_trees)
        y_eval_neg = evaluate(y_train, x_eval_neg, M, history, w_trees)
        y_diff = y_eval_pos - y_eval_neg
        importance_temp = y_diff/x_diff
        
        importance.append(importance_temp)
    importance = np.vstack(importance)

    H = np.matmul(importance, np.transpose(importance))/N
    return H


def get_H_estimates(stats, X, y_train, M, sample_range, tries = 10):
    H_est = []
    for i in range(len(sample_range)):
        n_sim = sample_range[i]
        dist = {'n': n_sim, 'H': []}
        for trial in tqdm(range(tries)):
            history, w_trees = \
                stats[i]['history'][trial]
            H_0 = estimate_H_finite_diff(X[:n_sim], y_train, M, history, w_trees)
            dist['H'].append(H_0)
        H_est.append(dist)
    return H_est


