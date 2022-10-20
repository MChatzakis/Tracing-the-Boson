import numpy as np

from cmath import inf

from utils import *
from helpers import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: The loss of the final computation [value]
        w: The final weights [numpy array of D dimensions]
    """

    loss = -1
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * gradient

        #print("GD Epoch. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    batch_size = 1
    loss = -1
    w = initial_w
    for n_iter in range(max_iters):
        # since batch size is 1 might need to change dat
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            gradient = compute_gradient(y_batch, tx_batch, w)

            w = w - gamma * gradient

            loss = compute_loss(y, tx, w)

        #print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


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

    [N, D] = np.shape(tx)
    w = np.zeros(D)
    w = np.linalg.solve(tx.T @ tx, tx.T) @ y
    mse = 1./(2*N)*((y-tx @ w).T @ (y-tx @ w))

    return [w, mse]


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    w = np.linalg.solve(tx.T.dot(tx) + lambda_*2 *
                        tx.shape[0]*np.identity(tx.shape[1]), tx.T.dot(y))
    return w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w

    for iter in range(max_iters):
        # get loss and update w.
        loss, w = gradient_descent_logistic(y, tx, w, gamma)

    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma_):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    
    w=initial_w

    for iter in range(max_iters):
        # get loss and update w.
        loss, w = reg_gradient_descent_logistic(y, tx, lambda_, w, gamma_)

    return loss, w
