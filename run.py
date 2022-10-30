"""
-----------------
run.py
This file reproduces the best predictions uploaded in AiCrowd competition.

Best predictions are produced by our partial training approach. For this script, we use our optimal
parameters only.

''
python3 run.py
''

cs433-ML Project 1, EPFL
Ioannis Bantzis, Manos Chatzakis, Maxence Hofer
-----------------
"""
import numpy as np

from helpers import *
from utils import *
from implementations import *

# Loading the data
data_path = "./dataset/train.csv"

[y, x, ids] = load_csv_data(data_path, sub_sample=False)

[x0, y0, x1, y1, x2, y2, x3, y3] = separate_4(y, x)

id = np.arange(len(x[:, 0]))
[x0, i0, x1, i1, x2, i2, x3, i3] = separate_4(id, x)

# Loading the partial input feature matrices
d0 = 8
d1 = 11
d2 = 12
d3 = 12

tx0 = increase_degree(add_1_column(remove_false_column_and_average_first(x0)), d0)
tx1 = increase_degree(add_1_column(remove_false_column_and_average_first(x1)), d1)
tx2 = increase_degree(add_1_column(average_false_values(x2)), d2)
tx3 = increase_degree(add_1_column(average_false_values(x3)), d3)

# Lambdas are found using heuristics
lambda_ridge0 = 0.02
lambda_ridge1 = 0.12285714
lambda_ridge2 = 0.03285714
lambda_ridge3 = 0.03285714

# Partial Training
[w0, loss0] = ridge_regression(y0, tx0, lambda_ridge0)
[w1, loss1] = ridge_regression(y1, tx1, lambda_ridge1)
[w2, loss2] = ridge_regression(y2, tx2, lambda_ridge2)
[w3, loss3] = ridge_regression(y3, tx3, lambda_ridge3)

# Disabled the printing of partial and general test accuracy for submission
# calculate_accuracy()

# Producing the csv output file
data_path_test = "./dataset/test.csv"

_, test_data, test_ids = load_csv_data(data_path_test, False)

idt = np.arange(len(test_data[:, 0]))

[x0t, id0, x1t, id1, x2t, id2, x3t, id3] = separate_4(idt, test_data)

tx0t = increase_degree(add_1_column(remove_false_column_and_average_first(x0t)), d0)
tx1t = increase_degree(add_1_column(remove_false_column_and_average_first(x1t)), d1)
tx2t = increase_degree(add_1_column(average_false_values(x2t)), d2)
tx3t = increase_degree(add_1_column(average_false_values(x3t)), d3)

ly0 = np.sign(tx0t @ w0 - 0.5)
ly1 = np.sign(tx1t @ w1 - 0.5)
ly2 = np.sign(tx2t @ w2 - 0.5)
ly3 = np.sign(tx3t @ w3 - 0.5)

labels_y = unit_4(ly0, id0, ly1, id1, ly2, id2, ly3, id3)

print("Creating CSV submission in current directory...")
create_csv_submission(test_ids, labels_y, "./FinalPartialTraining.csv")


def calculate_accuracy():
    ycal0 = np.sign(tx0 @ w0 - 0.5)
    n0 = len(y0)
    r0 = (n0 - np.count_nonzero(ycal0 - (2 * y0 - 1))) / n0
    print("Accuracy 0", (n0 - np.count_nonzero(ycal0 - (2 * y0 - 1))) / n0)

    ycal1 = np.sign(tx1 @ w1 - 0.5)
    n1 = len(y1)
    r1 = (n1 - np.count_nonzero(ycal1 - (2 * y1 - 1))) / n1
    print("Accuracy 1", (n1 - np.count_nonzero(ycal1 - (2 * y1 - 1))) / n1)

    ycal2 = np.sign(tx2 @ w2 - 0.5)
    n2 = len(y2)
    r2 = (n2 - np.count_nonzero(ycal2 - (2 * y2 - 1))) / n2
    print("Accuracy 2", (n2 - np.count_nonzero(ycal2 - (2 * y2 - 1))) / n2)

    ycal3 = np.sign(tx3 @ w3 - 0.5)
    n3 = len(y3)
    r3 = (n3 - np.count_nonzero(ycal3 - (2 * y3 - 1))) / n3
    print("Accuracy 3", (n3 - np.count_nonzero(ycal3 - (2 * y3 - 1))) / n3)

    y_fin = unit_4(ycal0, i0, ycal1, i1, ycal2, i2, ycal3, i3)
    nfin = len(y_fin)
    print("Accuracy (all)", (nfin - np.count_nonzero(y_fin - (2 * y - 1))) / nfin)
