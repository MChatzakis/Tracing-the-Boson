# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_and_build_data():
    """Load data and convert it to the metric system."""
    path_dataset = "INSERT_PATH\\data\\train.csv" #don't forget double backslash
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1)
    y_bs = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, dtype=None, encoding=None, usecols=(1))
    x = np.asarray(data[:, 2:])
    #print(data[1,:])
    y = np.zeros(len(y_bs))
    for i in range(0, len(y_bs)):
        if y_bs[i] == 'b':
            y[i]=1
        elif y_bs[i]== 's':
            y[i]=-1
        else:
            y[i]=-999
    

    return y, x


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]