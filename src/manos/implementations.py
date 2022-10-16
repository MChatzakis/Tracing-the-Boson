import numpy as np
from helpers import *


def compute_loss(y, X, w):
    """
    y: Vector of N dimensions (labels) [numpy array]
    w: Vector of D dimensions (weights) [numpy array]
    X: Matrix of NxD dimensions (features)[numpy array]

    Return: MSE(w) [value]
    """

    N = y.shape[0]
    e = y - X.dot(w)

    mse = (np.linalg.norm(e)**2)/(2*N)
    return mse


def compute_gradient(y, X, w):
    """
    y: Vector of N dimensions (labels) [numpy array]
    w: Vector of D dimensions (weights) [numpy array]
    X: Matrix of NxD dimensions (features)[numpy array]

    Return: Gradient L(w) [value]
    """

    N = y.shape[0]
    e = y - X.dot(w)

    XT = X.transpose()

    gradient = -XT.dot(e)/N
    return gradient


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

    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * gradient

        print("GD Epoch. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


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

            # compute a stochastic gradient and loss
            gradient = compute_gradient(y_batch, tx_batch, w)

            # update w through the stochastic gradient update
            w = w - gamma * gradient

            # calculate loss
            loss = compute_loss(y, tx, w)

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w
