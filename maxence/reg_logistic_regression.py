"""Project 1

Regularized logistic regression
"""

import numpy as np

def reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma_):
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
    

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    e=np.exp(t)
    return np.divide(e, (1+e))