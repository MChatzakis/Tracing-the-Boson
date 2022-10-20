import numpy as np

from helpers import *
from implementations import *

DATASET_PATH = "../dataset/train.csv"
yb, input_data, ids = load_csv_data(DATASET_PATH, False)

y = yb
tx = add_1_column(remove_false_columns(input_data))

def gd():
    
    max_iters = 100
    gamma = 0.0000003
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    gd_w, gd_loss = mean_squared_error_gd(y, tx, w_initial, max_iters, gamma)

    print(">>>Gradient Descent (it: " + str(max_iters) + ", g: " + str(gamma) +  ")")
    print("Loss:",gd_loss)
    print("Weights:", gd_w)
    print()

    return


def sgd():
    
    max_iters = 100
    gamma = 0.000000005
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    sgd_w, sgd_loss = mean_squared_error_sgd(
        y, tx, w_initial, max_iters, gamma)

    print(">>>Stochastic Gradient Descent (it: " + str(max_iters) + ", g: " + str(gamma) +  ")")
    print("Loss:",sgd_loss)
    print("Weights:", sgd_w)

    print()

    return


def ls():
    ls_w, ls_loss = least_squares(y, tx)
    
    print(">>>Least Squares")
    print("Loss:",ls_loss)
    print("Weights:", ls_w)
    print()

    return


def rr():
    lambda_ = 0

    rr_w, rr_loss = ridge_regression(y, tx, lambda_)

    print(rr_loss)

    return


def lr():
    max_iters = 100
    gamma = 0.000001
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)
    lr_loss, lr_w = logistic_regression(y, tx, w_initial, max_iters, gamma)
    
    print(">>>Logistic Regression (it: " + str(max_iters) + ", g: " + str(gamma) +  ")")
    print("Loss:",lr_loss)
    print("Weights:", lr_w)
    print()

    return


def rlr():
    max_iters = 100
    gamma = 0.000001
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)
    rlr_loss, rlr_w = reg_logistic_regression(
        y, tx, w_initial, max_iters, gamma)

    print(">>>Reg Logistic Regression (it: " + str(max_iters) + ", g: " + str(gamma) +  ")")
    print("Loss:", rlr_loss)
    print("Weights:", rlr_w)
    print()
    
    return


gd()
sgd()
ls()
#rr()
lr()
#lrl()