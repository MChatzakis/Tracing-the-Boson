import numpy as np
import datetime

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

    gd_loss, gd_w = mean_squared_error_gd(y, tx, w_initial, max_iters, gamma)

    print(gd_loss)

    return


def sgd():
    max_iters = 200
    gamma = 0.000000005
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    sgd_loss, sgd_w = mean_squared_error_sgd(
        y, tx, w_initial, max_iters, gamma)

    print(sgd_loss)

    return


def ls():
    ls_w, ls_loss = least_squares(y, tx)
    print(ls_loss)

    return


def rr():
    lambda_ = 0

    rr_w, rr_loss = ridge_regression(y, tx, lambda_)

    print(rr_w)
    print(rr_loss)

    return


def lr():
    max_iters = 100
    gamma = 0.000001
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)
    lr_loss, lr_w = logistic_regression(y, tx, w_initial, max_iters, gamma)

    print(lr_w)
    print(lr_loss)

    return


def rlr():
    max_iters = 100
    gamma = 0.000001
    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)
    rlr_loss, rlr_w = reg_logistic_regression(
        y, tx, w_initial, max_iters, gamma)

    print(rlr_w)
    print(rlr_loss)
