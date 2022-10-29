"""
-----------------
utils.py
Contains basic utility functions for the implementations

cs433-ML Project 1, EPFL
Ioannis Bantzis, Manos Chatzakis, Maxence Hofer
-----------------
"""


import numpy as np
from helpers import *


"""
-----------------
Computational and Loss functions
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

    w = w-gamma*compute_gradient_logistic(y, X, w)
    loss = compute_loss_logistic(y, X, w)

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
    [N, D] = np.shape(tx)
    loss = 1/N*np.sum(np.log(1+np.exp(tx @ w))-np.multiply(y,
                      (tx @ w)))
    #loss=compute_loss_logistic(y, tx, w)+lambda_* np.linalg.norm(w)**2
    gradient = 1/N*tx.T @ (sigmoid(tx @ w)-y)+2*lambda_*w

    return loss, gradient


"""
-----------------
Data Preprocessing and Handling function
-----------------
"""


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


def separate_4(y, x):
    ''' separate the x matrix and y vector in 4 matrices and 4 vectors according to the PRI_jet_num value
    '''
    [n, d] = np.shape(x)
    index0 = [-1]
    index1 = [-1]
    index2 = [-1]
    index3 = [-1]
    for i in range(n):
        if (x[i, 22] == 0):
            index0 = np.vstack([index0, i])
        elif (x[i, 22] == 1):
            index1 = np.vstack([index1, i])
        elif (x[i, 22] == 2):
            index2 = np.vstack([index2, i])
        elif (x[i, 22] == 3):
            index3 = np.vstack([index3, i])
        else:
            assert ("Error, value of PRI_jet was not in 0:3")

    index0 = np.delete(index0, 0)
    index1 = np.delete(index1, 0)
    index2 = np.delete(index2, 0)
    index3 = np.delete(index3, 0)

    x0 = x[index0, :]
    y0 = y[index0]
    x1 = x[index1, :]
    y1 = y[index1]
    x2 = x[index2, :]
    y2 = y[index2]
    x3 = x[index3, :]
    y3 = y[index3]

    return [x0, y0, x1, y1, x2, y2, x3, y3]


def unit_4(y0, id0, y1, id1, y2, id2, y3, id3):
    y = np.zeros((len(y0)+len(y1)+len(y2)+len(y3)))
    y[id0] = y0[:]
    y[id1] = y1[:]
    y[id2] = y2[:]
    y[id3] = y3[:]

    return y


def remove_false_column_and_average_first(x):
    [n, d] = np.shape(x)
    n999 = np.zeros(d)
    index = [-1]
    for i in range(1, d):
        if not (-999 in x[:, i]):
            index = np.vstack([index, i])
        # else:
        #    n999[i]=np.count_nonzero(x[:, i]==-999)

    index = np.delete(index, 0)
    x0 = np.copy(x[:, 0])
    for i in range(n):
        if x[i, 0] == -999.0:
            x0[i] = 0
    m = np.mean(x0)
    for i in range(n):
        if x[i, 0] == -999.0:
            x0[i] = m

    rx = x[:, index]
    return np.c_[x0, rx]


def average_false_values(x):
    ''' this function take x and remove the data with the untaked measurement (-999.0)
    Input: x: shape (n, d)
    Return: rx: shape (nr, d) The data without columns containing -999
    '''
    [n, d] = np.shape(x)
    rx = x.copy()

    for i in range(n):
        for j in range(d):
            if x[i, j] == -999.0:
                rx[i, j] = 0

    mrx = np.mean(rx, axis=0)
    print(mrx)

    for i in range(n):
        for j in range(d):
            if x[i, j] == -999.0:
                rx[i, j] = mrx[j]

    return rx


def increase_degree(tx, degree):
    [N, D] = np.shape(tx)
    txd = np.zeros([N, D*degree])
    # print(txd)
    for i in range(D):
        for j in range(degree):
            txd[:, D*j+i] = tx[:, i]**(j+1)
    return txd


"""
-----------------
Cross Validation functions
-----------------
"""


def build_k_indices(x, k_fold, seed):
    """build k indices for k-fold.

    Args:
        x:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = x.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data_k_fold(x, y, k_fold, k, seed):
    """split data for k-fold cross validation.
    Args:
        x:  shape=(N,D)
        y:  shape=(N,D)
        k_fold: K in K-fold, i.e. the fold num
        k:  scalar, the k-th fold 
        seed:   the random seed
    """
    k_indices = build_k_indices(x, k_fold, seed)
    test_indices = k_indices[k]

    train_indices = np.delete(k_indices, k).flatten()

    x_te = x[test_indices]
    x_tr = x[train_indices]
    y_te = y[test_indices]
    y_tr = y[train_indices]

    return x_te, x_tr, y_te, y_tr


def linear_prediction(x, w):
    return x.dot(w)


def sigmoid_prediction(x, w):
    return sigmoid(x.dot(w))


def cross_validation_step(x, y, k_fold, k, seed, function, prediction_function, *args):
    x_te, x_tr, y_te, y_tr = split_data_k_fold(x, y, k_fold, k, seed)
    w, _ = function(y_tr, x_tr, *args)
    acc_te = compute_accuracy(y_te, prediction_function(x_te, w), 0.5)
    acc_tr = compute_accuracy(y_tr, prediction_function(x_tr, w), 0.5)
    return acc_tr, acc_te


def cross_validation(x, y, k_fold, seed, function, prediction_function, *args):
    """sample tests:

    cross_validation(tx,y, 4, 1,mean_squared_error_gd, linear_prediction,np.zeros(tx.shape[1]),100,0.001)
    cross_validation(tx,y, 4, 1,mean_squared_error_sgd, linear_prediction,np.zeros(tx.shape[1]),100,0.001)
    cross_validation(tx,y, 4, 1,least_squares, linear_prediction)
    cross_validation(tx,y, 4, 1,ridge_regression, linear_prediction,0.001)
    cross_validation(tx,y, 4, 1,logistic_regression, sigmoid_prediction,np.zeros(tx.shape[1]),100,0.001)
    cross_validation(tx,y, 4, 1,reg_logistic_regression, sigmoid_prediction,0.0001,np.zeros(tx.shape[1]),100,0.001)

    returns (test_accuracy,training_accuracy)

    """

    accs_tr = []
    accs_te = []
    for k in range(k_fold):
        k_acc_tr, k_acc_te = cross_validation_step(
            x, y, k_fold, k, seed, function, prediction_function, *args)
        accs_tr.append(k_acc_tr)
        accs_te.append(k_acc_te)
    acc_tr = np.mean(accs_tr)
    acc_te = np.mean(accs_te)
    return acc_te, acc_tr


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)

    p = np.random.permutation(x.shape[0])
    x, y = x[p], y[p]
    x_tr = x[:int(np.floor(ratio*x.shape[0]))]
    x_te = x[int(np.floor(ratio*x.shape[0])):]
    y_tr = y[:int(np.floor(ratio*y.shape[0]))]
    y_te = y[int(np.floor(ratio*y.shape[0])):]
    return x_tr, x_te, y_tr, y_te
