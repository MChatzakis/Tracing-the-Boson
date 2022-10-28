from itertools import count
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
        #else:
        #    n999[i]=np.count_nonzero(x[:, i]==-999)

    index = np.delete(index, 0)

    rx = x[:, index]
    return rx

def remove_false_column_and_average_first(x):
    [n, d] = np.shape(x)
    n999 = np.zeros(d)
    index = [-1]
    for i in range(1, d):
        if not (-999 in x[:, i]):
            index = np.vstack([index, i])
        #else:
        #    n999[i]=np.count_nonzero(x[:, i]==-999)

    index = np.delete(index, 0)
    x0= np.copy(x[:, 0])
    for i in range(n):
        if x[i, 0]==-999.0:
            x0[i]=0
    m=np.mean(x0)
    for i in range(n):
        if x[i, 0]==-999.0:
            x0[i]=m

    rx = x[:, index]
    return np.c_[x0, rx]

def average_false_values(x):
    ''' this function take x and remove the data with the untaked measurement (-999.0)
    Input: x: shape (n, d)
    Return: rx: shape (nr, d) The data without columns containing -999
    '''
    [n, d] = np.shape(x)
    rx=x.copy()
       

    for i in range(n):
        for j in range(d):
            if x[i, j]==-999.0:
                rx[i, j]=0
    
    mrx=np.mean(rx, axis=0)
    print(mrx)

    for i in range(n):
        for j in range(d):
            if x[i, j]==-999.0:
                rx[i, j]=mrx[j]
    

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
    index0=[-1]
    index1=[-1]
    index2=[-1]
    index3=[-1]
    for i in range(n):
        if (x[i, 22]==0):
            index0 = np.vstack([index0, i])
        elif (x[i, 22]==1):
            index1 = np.vstack([index1, i])
        elif (x[i, 22]==2):
            index2 = np.vstack([index2, i])
        elif (x[i, 22]==3):
            index3 = np.vstack([index3, i])
        else:
            assert("Error, value of PRI_jet was not in 0:3")
    
    index0 = np.delete(index0, 0)
    index1 = np.delete(index1, 0)
    index2 = np.delete(index2, 0)
    index3 = np.delete(index3, 0)

    x0=x[index0, :]
    y0=y[index0]
    x1=x[index1, :]
    y1=y[index1]
    x2=x[index2, :]
    y2=y[index2]
    x3=x[index3, :]
    y3=y[index3]

    return [x0, y0, x1, y1, x2, y2, x3, y3]

def unit_4(y0, id0, y1, id1, y2, id2, y3, id3):
    y=np.zeros((len(y0)+len(y1)+len(y2)+len(y3)))
    y[id0]=y0[:]
    y[id1]=y1[:]
    y[id2]=y2[:]
    y[id3]=y3[:]

    return y

    


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
    """
    
    #e=np.exp(t)
    #return e/(1+e)
    
    s = np.zeros(len(t))
    
    for i in range(len(t)):
        if t[i]>4:
            s[i]=1
        else:
            e=np.exp(t[i])
            s[i]=e/(1+e)
    
    return s
    


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

def increase_degree(tx, degree):
    [N, D]=np.shape(tx)
    txd=np.zeros([N, D*degree])
    #print(txd)
    for i in range(D):
        for j in range(degree):
            txd[:, D*j+i]=tx[:, i]**(j+1)
    return txd
