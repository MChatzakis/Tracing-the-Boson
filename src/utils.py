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


def reg_gradient_descent_logistic(y, tx, lambda_, w, gamma_):
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
    
    loss,  gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma_*gradient
    
    return loss, w
    
def penalized_logistic_regression(y, tx, w, lambda_):
    [N, D]=np.shape(tx)
    loss = 1/N*np.sum(np.log(1+np.exp(tx @ w))-np.multiply(y, (tx @ w)))+lambda_* np.linalg.norm(w)**2
    #loss=compute_loss_logistic(y, tx, w)+lambda_* np.linalg.norm(w)**2
    gradient = 1/N*tx.T @ (sigmoid(tx @ w)-y)+2*lambda_*w
    
    return loss, gradient
    

