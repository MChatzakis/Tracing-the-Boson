# -*- coding: utf-8 -*-
"""Project 1

Least squares
"""

import numpy as np

def logistic_regression(y, X, initial_w, max_iters, gamma):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    losses = []
    w=initial_w

    for iter in range(max_iters):
        # get loss and update w.
        loss, w = gradient_descent_logistic(y, X, w, gamma)

    return loss, w


def gradient_descent_logistic(y, X, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    
    loss = compute_loss_logistic(y, X, w)
    w = w-gamma*compute_gradient_logistic(y, X, w)
    
    return loss, w
    

def compute_loss_logistic(y, X, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss
    """
    
    [N, D]=np.shape(X)
    return 1/N*np.sum(np.log(1+np.exp(X @ w))-np.multiply(y, (X @ w)))

def compute_gradient_logistic(y, X, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        X: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)
    """
    [N, D] = np.shape(X)
    v1=sigmoid(X @ w)
    return 1/N*X.T @ (v1-y)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    e=np.exp(t)
    return np.divide(e, (1+e))

