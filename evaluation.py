"""
-----------------
evaluation.py
Evaluation script to tune parameters through cross validation.

Example:
$ python3 evaluation.py gd 

For usage help:
$ python3 evaluation.py usage

cs433-ML Project 1, EPFL
Ioannis Bantzis, Manos Chatzakis, Maxence Hofer
-----------------
"""

from cgi import test

import numpy as np

import sys
import time

from helpers import *
from implementations import *

DATASET_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
TEST_RESULTS_PATH = "./test_results/"

K_FOLDS = 5
SEED = 42

KEEP_ALL_RESULTS = False
VERBOSE = True

yb, input_data, train_ids = load_csv_data(DATASET_PATH, False)

y = yb
tx = add_1_column(remove_false_columns(input_data))


def gd():
    # Old code
    #max_iters = 100
    #gamma = 0.0000003
    #gd_w, gd_loss = mean_squared_error_gd(y, tx, w_initial, max_iters, gamma)

    #print(">>>Gradient Descent (it: " + str(max_iters) + ", g: " + str(gamma) + ")")
    #print("Loss:", gd_loss)
    #print("Weights:", gd_w)
    # print()

    print("Evaluating Gradient Descent")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10000
    iters = [100]

    gammas = np.linspace(0.000000001, 0.0001, total_gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            test_acc, train_acc = cross_validation(
                tx, y, K_FOLDS, SEED, mean_squared_error_gd, linear_prediction, w_initial, iter, step_size)

            c_res = [iter, step_size, test_acc, train_acc]
            print(c_res)

            if (test_acc > best_acc):
                best_acc = test_acc
                best_result = c_res

            if (KEEP_ALL_RESULTS):
                results.append(c_res)

    print("GD best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def sgd():

    #max_iters = 100
    #gamma = 0.000000005
    #features_num = tx.shape[1]
    #w_initial = np.zeros(features_num)

    # sgd_w, sgd_loss = mean_squared_error_sgd(
    #    y, tx, w_initial, max_iters, gamma)

    # print(">>>Stochastic Gradient Descent (it: " +
    #      str(max_iters) + ", g: " + str(gamma) + ")")
    #print("Loss:", sgd_loss)
    #print("Weights:", sgd_w)
    # print()

    print("Evaluating Stochastic Gradient Descent")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10000
    iters = [100]

    gammas = np.linspace(0.000000001, 0.0000001, total_gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            test_acc, train_acc = cross_validation(tx, y, K_FOLDS, SEED, mean_squared_error_sgd,
                                                   linear_prediction, w_initial, iter, step_size)

            c_res = [iter, step_size, test_acc, train_acc]
            print(c_res)

            if (test_acc > best_acc):
                best_acc = test_acc
                best_result = c_res

            if (KEEP_ALL_RESULTS):
                results.append(c_res)

    print("SGD best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def ls():
    #ls_w, ls_loss = least_squares(y, tx)

    #print(">>>Least Squares")
    #print("Loss:", ls_loss)
    #print("Weights:", ls_w)
    # print()

    #_, test_data, test_ids = load_csv_data(TEST_PATH, False)
    #test_data = add_1_column(remove_false_columns(test_data))
    #evaluate_test_data(test_data, ls_w, test_ids)

    print("Evaluating Least Squares")

    test_acc, train_acc = cross_validation(
        tx, y, K_FOLDS, SEED, least_squares, linear_prediction)

    print("LS best result=[test_acc,train_acc]=", test_acc, train_acc)

    return


def rr():
    #lambda_ = 0
    #rr_w, rr_loss = ridge_regression(y, tx, lambda_)
    # print(rr_loss)
    print("Evaluating Ridge Reggresion")

    total_lambdas = 1000
    lambdas = np.linspace(0.00001, 0.001, total_lambdas)

    results = []
    best_result = []
    best_acc = -1

    for lambda_ in lambdas:
        test_acc, train_acc = cross_validation(
            tx, y, K_FOLDS, SEED, ridge_regression, linear_prediction, lambda_)
        c_res = [lambda_, test_acc, train_acc]
        print(c_res)

        if (test_acc > best_acc):
            best_acc = test_acc
            best_result = c_res

        results.append(c_res)

    print("RR best result=[lambda,test_acc,train_acc]=", best_result)
    return


def lr():
    #max_iters = 100
    #gamma = 0.000001
    #features_num = tx.shape[1]
    #w_initial = np.zeros(features_num)
    #lr_loss, lr_w = logistic_regression(y, tx, w_initial, max_iters, gamma)

    # print(">>>Logistic Regression (it: " +
    #      str(max_iters) + ", g: " + str(gamma) + ")")
    #print("Loss:", lr_loss)
    #print("Weights:", lr_w)
    # print()

    #_, test_data, test_ids = load_csv_data(TEST_PATH, False)
    #test_data = add_1_column(remove_false_columns(test_data))
    #evaluate_test_data(test_data, lr_w, test_ids)

    print("Evaluating Logistic Regression")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10000
    iters = [100]

    gammas = np.linspace(0.0000001, 0.00001, total_gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            test_acc, train_acc = cross_validation(
                tx, y, K_FOLDS, SEED, logistic_regression, sigmoid_prediction, w_initial, iter, step_size)

            c_res = [iter, step_size, test_acc, train_acc]
            print(c_res)

            if (test_acc > best_acc):
                best_acc = test_acc
                best_result = c_res

            if (KEEP_ALL_RESULTS):
                results.append(c_res)

    print("LR best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def rlr():
    #lambda_ = 1e-6
    #max_iters = 100
    #gamma = 0.000001
    #features_num = tx.shape[1]
    #w_initial = np.zeros(features_num)
    # rlr_loss, rlr_w = reg_logistic_regression(
    #    y, tx, lambda_, w_initial, max_iters, gamma)

    # print(">>>Reg Logistic Regression (it: " +
    #      str(max_iters) + ", g: " + str(gamma) + ")")
    #print("Loss:", rlr_loss)
    #print("Weights:", rlr_w)
    # print()

    print("Evaluating Regularized Logistic Regression")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10000
    total_lambdas = 100

    iters = [100]

    lambdas = np.linspace(0.000001, 0.0001, total_lambdas)
    gammas = np.linspace(0.0000001, 0.00001, total_gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            for lambda_ in lambdas:
                test_acc, train_acc = cross_validation(
                    tx, y, K_FOLDS, SEED, reg_logistic_regression, sigmoid_prediction, lambda_, w_initial, iter, step_size)

                c_res = [iter, step_size, lambda_, test_acc, train_acc]
                print(c_res)

                if (test_acc > best_acc):
                    best_acc = test_acc
                    best_result = c_res

                if (KEEP_ALL_RESULTS):
                    results.append(c_res)

    print(
        "RLR best result=[iter,gamma,lambda,test_acc,train_acc]=", best_result)

    return


def evaluate_test_data(test_data_X, weights, ids):

    predicted_y = sigmoid(test_data_X.dot(weights))

    print(predicted_y[0:30])

    labels_y = np.sign(predicted_y-0.5)

    create_csv_submission(ids, labels_y, "LogisticRegression.csv")

    return


def usage():
    print("Cross Validation Evaluation Script. Use:")

    print("'gd'     for mean squared error with gradient descent")
    print("'sgd'    for mean squared error with stochastic gradient descent")
    print("'ls'     for least squares")
    print("'rr'     for ridge regression")
    print("'lr'     for logistic regression")
    print("'rlr'    for regularized logistic regression")

    print("\nExample: python3 evalution.py gd")

    return


# dispatch table ;)
function_options = {
    "gd": gd,
    "sgd": sgd,
    "ls": ls,
    "rr": rr,
    "lr": lr,
    "rlr": rlr,
    "usage": usage
}


def main():

    function_name = sys.argv[1]

    if (function_name == ""):
        print("Define a function to run.")
        sys.exit(-1)

    start = time.time()
    function_options[function_name]()  # first step of becoming a hacker
    end = time.time()

    print("Total time elapsed (seconds):", end - start)

    return


if __name__ == '__main__':
    sys.exit(main())
