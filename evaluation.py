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
import matplotlib.pyplot as plt

import sys
import time

from helpers import *
from implementations import *

DATASET_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"

K_FOLDS = 5
SEED = 42
EXPANSION_DEGREE = 12

KEEP_ALL_RESULTS = False
VERBOSE = False

yb, input_data, train_ids = load_csv_data(DATASET_PATH, False)

y = yb
#tx = increase_degree(add_1_column(remove_false_columns(input_data)), 3)
tx = add_1_column(remove_false_columns(input_data))


def gd():

    # best gamma: 1e-08

    print("Evaluating Gradient Descent")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    iters = [100]

    total_gammas = 10
    # gammas = np.linspace(0.000000001, 0.00000001, total_gammas) #best 1e-08
    gammas = [1e-08]
    print("Gammas to test:", gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            test_acc, train_acc = cross_validation(
                tx,
                y,
                K_FOLDS,
                SEED,
                mean_squared_error_gd,
                linear_prediction,
                w_initial,
                iter,
                step_size,
            )

            c_res = [iter, step_size, test_acc, train_acc]

            best_acc, best_result = improvement_action(
                test_acc, best_acc, c_res, best_result, results
            )

    print("\n>>>GD best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def sgd():

    print("Evaluating Stochastic Gradient Descent")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10
    iters = [100]

    gammas = np.linspace(0.00000000001, 0.0000000001, total_gammas)
    gammas = [1e-11]
    print("Gammas to test:", gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            print("Current gamma test:", step_size)

            test_acc, train_acc = cross_validation(
                tx,
                y,
                K_FOLDS,
                SEED,
                mean_squared_error_sgd,
                linear_prediction,
                w_initial,
                iter,
                step_size,
            )

            c_res = [iter, step_size, test_acc, train_acc]

            best_acc, best_result = improvement_action(
                test_acc, best_acc, c_res, best_result, results
            )

    print("\n>>>SGD best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def ls():

    print("Evaluating Least Squares")

    test_acc, train_acc = cross_validation(
        add_1_column(remove_false_columns(input_data)),
        y,
        K_FOLDS,
        SEED,
        least_squares,
        linear_prediction,
    )

    print("\n>>>LS best result=[test_acc,train_acc]=", test_acc, train_acc)

    return


def rr():

    print("Evaluating Ridge Reggresion")

    total_lambdas = 10
    # lambdas = np.linspace(0.00001, 0.001, total_lambdas) #0.00056
    lambdas = [0.00056]  # best
    print("Lambdas to test:", lambdas)
    results = []
    best_result = []
    best_acc = -1

    test_num = 1
    for lambda_ in lambdas:
        test_acc, train_acc = cross_validation(
            tx, y, K_FOLDS, SEED, ridge_regression, linear_prediction, lambda_
        )

        c_res = [lambda_, test_acc, train_acc]

        print("Test", test_num, "out of ", total_lambdas)
        test_num += 1

        best_acc, best_result = improvement_action(
            test_acc, best_acc, c_res, best_result, results
        )

    print("\n>>>RR best result=[lambda,test_acc,train_acc]=", best_result)
    return


def lr():

    print("Evaluating Logistic Regression")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10
    iters = [100]

    gammas = np.linspace(0.0001, 0.001, total_gammas)
    gammas = [0.0001]
    print("Gammas to test", gammas)

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            test_acc, train_acc = cross_validation(
                tx,
                y,
                K_FOLDS,
                SEED,
                logistic_regression,
                sigmoid_prediction,
                w_initial,
                iter,
                step_size,
            )

            c_res = [iter, step_size, test_acc, train_acc]

            best_acc, best_result = improvement_action(
                test_acc, best_acc, c_res, best_result, results
            )

    print("LR best result=[iter,gamma,test_acc,train_acc]=", best_result)

    return


def rlr():
    # lambda_ = 1e-6
    # max_iters = 100
    # gamma = 0.000001
    # features_num = tx.shape[1]
    # w_initial = np.zeros(features_num)
    # rlr_loss, rlr_w = reg_logistic_regression(
    #    y, tx, lambda_, w_initial, max_iters, gamma)

    # print(">>>Reg Logistic Regression (it: " +
    #      str(max_iters) + ", g: " + str(gamma) + ")")
    # print("Loss:", rlr_loss)
    # print("Weights:", rlr_w)
    # print()

    print("Evaluating Regularized Logistic Regression")

    features_num = tx.shape[1]
    w_initial = np.zeros(features_num)

    total_gammas = 10
    total_lambdas = 10

    iters = [100]

    lambdas = np.linspace(1e-6, 1e-5, total_lambdas)
    gammas = np.linspace(0.000001, 0.00001, total_gammas)

    lambdas = [0.000001]
    gammas = [0.000001]

    results = []
    best_result = []
    best_acc = -1

    for iter in iters:
        for step_size in gammas:
            for lambda_ in lambdas:
                test_acc, train_acc = cross_validation(
                    tx,
                    y,
                    K_FOLDS,
                    SEED,
                    reg_logistic_regression,
                    sigmoid_prediction,
                    lambda_,
                    w_initial,
                    iter,
                    step_size,
                )

                c_res = [iter, step_size, lambda_, test_acc, train_acc]
                print(c_res)

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_result = c_res

                if KEEP_ALL_RESULTS:
                    results.append(c_res)

    print("RLR best result=[iter,gamma,lambda,test_acc,train_acc]=", best_result)

    return


def expansion_rr():
    degrees = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
    ]
    data = {}

    best_degree = -1
    best_test = -1

    for degree in degrees:
        x_e = increase_degree(add_1_column(remove_false_columns(input_data)), degree)
        test_acc, train_acc = cross_validation(
            x_e, y, K_FOLDS, SEED, ridge_regression, linear_prediction, 0.00056
        )

        if test_acc > best_test:
            best_test = test_acc
            best_degree = degree

        data[degree] = test_acc

    print(data)
    print("Best Degree=", best_degree)

    x_d = list(data.keys())
    y_d = list(data.values())

    # plt.bar(range(len(data)), values, tick_label=names)
    plt.plot(x_d, y_d, "-o", color="b")
    plt.xlabel("Expansion Degree")
    plt.ylabel("Test Accuracy")
    plt.title("Expansion Degree Evaluation for Ridge Regression")
    plt.show()

    return


def csv_rr():
    lambda_ = 0.00056
    expan = 11

    x_t = increase_degree(add_1_column(remove_false_columns(input_data)), expan)

    weights, _ = ridge_regression(y, x_t, lambda_)

    _, test_data, test_ids = load_csv_data(TEST_PATH, False)

    x_test = increase_degree(add_1_column(remove_false_columns(test_data)), expan)

    evaluate_test_data(x_test, weights, test_ids, "RidgeRegression.csv")

    return


def evaluate_test_data(test_data_X, weights, ids, name):
    predicted_y = test_data_X.dot(weights)
    labels_y = np.sign(predicted_y - 0.5)

    create_csv_submission(ids, labels_y, name)



def evaluate_partial():
    data_path = "./dataset/train.csv"

    [y, x, ids] = load_csv_data(data_path, sub_sample=False)

    [x0, y0, x1, y1, x2, y2, x3, y3] = separate_4(y, x)

    id = np.arange(len(x[:, 0]))
    [x0, i0, x1, i1, x2, i2, x3, i3] = separate_4(id, x)

    # Loading the partial input feature matrices
    dvals = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    ls = [0.02, 0.12285714, 0.03285714, 0.03285714]
    best_d = [-1,-1,-1,-1]
    
    xs = [x0,x1,x2,x3]
    ys = [y0,y1,y2,y3]
    colors = ["plum","moccasin","lightgray","lightblue"]
    names = ["x0", "x1", "x2", "x3"]
    
    data_d = [{},{},{},{}]
    
    for i in range(len(xs)):
        l = ls[i]
        
        print("Partition", i+1, "of", len(xs))
        best_partition_acc = -1
        
        for d in dvals:
            if(i < 2):
                ptx = increase_degree(add_1_column(remove_false_column_and_average_first(xs[i])), d)
            else:
                ptx = increase_degree(add_1_column(average_false_values(xs[i])), d)
            
            test_acc, train_acc = cross_validation(
                ptx, ys[i], K_FOLDS, SEED, ridge_regression, linear_prediction, l
            )
            
            data_d[i][d] = test_acc
            
            if test_acc > best_partition_acc:
                best_partition_acc = test_acc
                best_d[i] = d
    
    X = np.arange(len(dvals))
    
    for i in range(len(data_d)):
        dict_en = data_d[i]
        x_d = list(dict_en.keys())
        y_d = list(dict_en.values())
        
        plt.plot(x_d, y_d, "-o", color=colors[i], label=names[i])
        #plt.bar(X+0.22*i, y_d, color=colors[i],label=names[i],width=0.22)
    
    plt.xlabel("Partial Expansion Degree")
    plt.ylabel("Partial Test Accuracy")
    plt.title("Expansion Degree Evaluation for Partial Training")
    #plt.xticks(X+0.22*1.5,dvals)
    plt.legend()
    plt.show()
    
    print(best_d) #gives [7, 11, 12, 9]
   


def usage():
    print("Cross Validation Evaluation Script. Use:")

    print("'gd'     for mean squared error with gradient descent")
    print("'sgd'    for mean squared error with stochastic gradient descent")
    print("'ls'     for least squares")
    print("'rr'     for ridge regression")
    print("'lr'     for logistic regression")
    print("'rlr'    for regularized logistic regression")

    print("\nExample: python3 evalution.py gd")

    print(
        "\nTo change the tuning range and other factors, you need to modify the script directly"
    )

    return


def improvement_action(test_acc, best_acc, c_res, best_result, results):

    if VERBOSE == True:
        print(c_res)

    if test_acc > best_acc:
        best_acc = test_acc
        best_result = c_res
        print("Found better accuracy: ", best_result)

    # if (KEEP_ALL_RESULTS):
    #    results.append(c_res)

    return best_acc, best_result


# dispatch table ;)
function_options = {
    "gd": gd,
    "sgd": sgd,
    "ls": ls,
    "rr": rr,
    "lr": lr,
    "rlr": rlr,
    "expansion_rr": expansion_rr,
    "partial":evaluate_partial,
    "usage": usage,
}


def main():

    function_name = sys.argv[1]

    if function_name == "":
        print("Define a function to run.")
        sys.exit(-1)

    start = time.time()
    function_options[function_name]()  # first step of becoming a hacker
    #csv_rr()
    end = time.time()

    print(">>>Train time elapsed (seconds):", end - start)

    return


if __name__ == "__main__":
    sys.exit(main())
