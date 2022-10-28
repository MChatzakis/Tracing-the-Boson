# -*- coding: utf-8 -*-
"""Project 1

Least squares
"""

import numpy as np

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    # ***************************************************
    
    [N,D] = np.shape(tx)
    w = np.zeros(D)
    w = np.linalg.solve(tx.T @ tx, tx.T) @ y
    mse = 1./(2*N)*((y-tx @ w).T @ (y-tx @ w))
    
    return [w, mse]