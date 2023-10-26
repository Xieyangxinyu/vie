import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from numpy.linalg import norm, qr, svd, matrix_rank, eig


def plot_trend(stats, x_axis, y_axis, title, xlabel = None, ylabel = None):
    df = pd.DataFrame(stats)

    # Prepare the data for plotting
    plot_data = [df[y_axis][i] for i in range(len(df))]

    # Create the boxplot
    plt.boxplot(plot_data, labels=[str(x[x_axis]) for x in stats])

    # Add titles and labels
    plt.title(title)
    if xlabel is None:
        xlabel = x_axis
    if ylabel is None:
        ylabel = y_axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()


def operator_norm(H, truth):
    norm_truth = norm(truth, ord = 2)
    return norm(truth - H, ord = 2)/norm_truth

def get_eig(A):
    eigenValues, eigenVectors = eig(A)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenVectors

def get_angle_distance(H, truth, active, dim_in):
    dim_proj = active
    proj = np.identity(dim_proj)
    proj = np.pad(proj, [(0, dim_in - dim_proj), (0, dim_in - dim_proj)], mode='constant')

    rank = matrix_rank(truth)
    
    eigenvectors = get_eig(H)
    Q_H, _ = qr(eigenvectors[:, :rank])
    
    eigenvectors = get_eig(truth)
    Q_t, _ = qr(eigenvectors[:, :rank])

    D = np.matmul(np.transpose(Q_H), Q_t)
    _, S, _ = svd(D, full_matrices=True)
    S = np.minimum(S, 1)
    return np.max(np.arccos(S))

def plot_dist(H_estimates, title, true_H, norm_func, sample_range, active, dim_in, tries = 10):
    plot_data = []
    for i in range(len(sample_range)):
        dist = H_estimates[i]
        n_sim = sample_range[i]
        alter_dist = {'n': n_sim, 'dist':[]}
        for trial in range(tries):
            H_0 = dist['H'][trial]
            alter_dist['dist'].append(norm_func(H_0, true_H, active, dim_in))
        plot_data.append(alter_dist)
    plot_trend(plot_data, 'n', 'dist', title, ylabel='Distance from the True H')


def plot_H_estimates(raw_H_estimates, true_H, norm_func, sample_range, active, dim_in):
    plot_dist(raw_H_estimates, 'Distance from True H', true_H, norm_func, sample_range, active, dim_in)