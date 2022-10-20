from cmath import inf
import numpy as np
from helpers import *

"""
-----------------
Utility
-----------------
"""


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


def remove_false_lines(y, x):
    ''' this function take y and x and remove the data with the untaked measurement (-999.0)
    Input: y: shape (n, )
           x: shape (n, d)

    Return: ry: shape (nr, ) 
            rx: shape (nr, d) The data without lines containing -999
            n999: shape (1, d) The number of -999 per column
    '''

    [n, d] = np.shape(x)
    n999 = np.zeros(d)
    index = [-1]
    for i in range(0, n):
        if not (-999 in x[i, :]):
            index = np.vstack([index, i])
        else:
            for j in range(0, d):
                if (x[i, j] == -999):
                    n999[j] = n999[j]+1

    index = np.delete(index, 0)

    rx = x[index, :]
    ry = y[index]
    return ry, rx, n999


def remove_false_columns(x):
    ''' this function take x and remove the data with the untaked measurement (-999.0)
    Input: x: shape (n, d)

    Return: rx: shape (nr, d) The data without columns containing -999
    '''
    [n, d] = np.shape(x)
    n999 = np.zeros(d)
    index = [-1]
    for i in range(0, d):
        if not (-999 in x[:, i]):
            index = np.vstack([index, i])

    index = np.delete(index, 0)

    rx = x[:, index]
    return rx


def add_1_column(x):
    ''' Take a x vector and add a 1 column at the beginning for the w0 element
    '''
    [n, d] = np.shape(x)
    v = np.ones([n, 1])
    return np.hstack((v, x))


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

    [N, D] = np.shape(X)
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
    v1 = sigmoid(X @ w)
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
    e = np.exp(t)
    return np.divide(e, (1+e))


"""
-----------------
Actual Implementations
-----------------
"""


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
            gradient = compute_gradient(y_batch, tx_batch, w)

            w = w - gamma * gradient

            loss = compute_loss(y, tx, w)

        #print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


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


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError
