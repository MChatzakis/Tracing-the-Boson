import numpy as np 

from giannis.implementations import *

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
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def split_data_k_fold(x,y, k_fold, k, seed):
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

    train_indices = np.delete(k_indices,k).flatten()

    x_te = x[test_indices]
    x_tr = x[train_indices]
    y_te = y[test_indices]
    y_tr = y[train_indices]

    return x_te, x_tr, y_te, y_tr


# def cross_validation_step_gd(x,y, k_fold,k,seed, initial_w,max_iters, gamma):
    
#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = mean_squared_error_gd(y_tr, x_tr, initial_w, max_iters, gamma)
#     loss_tr = losses
#     loss_te = compute_loss(y_te, x_te, ws)

#     return loss_tr, loss_te

# def cross_validation_step_sgd(x,y, k_fold,k,seed, initial_w,max_iters, gamma, batch_size=1):
    
#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iters, gamma,batch_size)
#     loss_tr = losses
#     loss_te = compute_loss(y_te, x_te, ws)

#     return loss_tr, loss_te

# def cross_validation_step_least_squares(x,y, k_fold,k,seed):

#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = least_squares(y,x)
#     loss_tr = losses
#     loss_te = compute_loss(y_te, x_te, ws)
    
#     return loss_tr, loss_te

# def cross_validation_step_ridge(x,y, k_fold,k,seed, lambda_):

#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = ridge_regression(y, x, lambda_)
#     loss_tr = losses
#     loss_te = compute_loss(y_te, x_te, ws)
    
#     return loss_tr, loss_te

# def cross_validation_step_logistic(x,y, k_fold,k,seed,initial_w, max_iters, gamma):

#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = logistic_regression(y, x, initial_w, max_iters, gamma)
#     loss_tr = losses
#     loss_te = compute_loss_logistic(y_te, x_te, ws)
    
#     return loss_tr, loss_te

# def cross_validation_step_ridge_logistic(x,y, k_fold,k,seed,initial_w, max_iters, gamma):

#     x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
#     losses, ws = reg_logistic_regression(y, x, initial_w, max_iters, gamma, lambda_)
#     loss_tr = losses
#     loss_te = compute_loss_logistic(y_te, x_te, ws)
    
#     return loss_tr, loss_te

def cross_validation_step(x,y, k_fold, k, seed,function,*args, loss_function):
    x_te, x_tr, y_te, y_tr = split_data_k_fold(x,y, k_fold, k, seed)
    loss_tr, w = function(y_tr, x_tr, *args)
    loss_te = loss_function(y_te, x_te, w)
    
    return loss_tr, loss_te

def cross_validation(x,y, k_fold, k, seed,function,*args, loss_function):
    losses_tr = []
    losses_te = []
    for k in range(k_fold):
        k_loss_tr,k_loss_te = cross_validation_step(x,y, k_fold, k, seed,function,*args, loss_function)
        losses_tr.append(k_loss_tr)
        losses_te.append(k_loss_te)
    loss_tr = np.mean(losses_tr)
    loss_te = np.mean(losses_te) 
    return loss_te, loss_tr   
