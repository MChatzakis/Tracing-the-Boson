# -*- coding: utf-8 -*-
"""Project 1

Least squares
"""
from helpers import*
import numpy as np
from utils import*
from implementations import*

data_path="C:\\Users\\maxen\\Documents\\Master EPFL\\MA1\\Machine learning\\project1\\data\\train.csv"

[y, x, ids] = load_csv_data(data_path, sub_sample=False)

[ry, rx, n999]=remove_false_lines(y, x)

xc=remove_false_columns(x)

txc=add_1_column(xc)

[nr, d] = np.shape(txc)
gamma_log=8e-5
gamma_ms=1.5e-5
w=np.zeros(d)
max_iters=300
lambda_log=1e-2
lambda_ridge=1e-2

[loss1, w1] = reg_logistic_regression(y, txc, lambda_log, w,  max_iters, gamma_log)
y_cal1=np.sign(sigmoid(txc @  w1)-0.5)

[w2, loss2] = least_squares(y, txc)
y_cal2=np.sign(txc @ w2-0.5)

w3=ridge_regression(y, txc, lambda_ridge)
y_cal3=np.sign(txc @ w3-0.5)
#poly=build_poly_ND(rx, 3)
y_fin= y_cal3
np.sign(y_cal1+y_cal2+y_cal3)

#print(loss2)

print(w3)

#print(y_cal[0:30])

#y_fin=np.sign(y_cal1-0.5)
print(y_fin[0:30]-(2*y[0:30]-1))

print((nr-np.count_nonzero(y_fin-(2*y-1)))/nr)

data_path_test="C:\\Users\\maxen\\Documents\\Master EPFL\\MA1\\Machine learning\\project1\\data\\test.csv"

_, test_data, test_ids  = load_csv_data(data_path_test, False)
test_data = add_1_column(remove_false_columns(test_data))
labels_y = np.sign(np.sign(sigmoid(test_data @ w1)-0.5)+np.sign(test_data @ w2-0.5)+np.sign(test_data @ w3-0.5))
print(np.shape(labels_y), np.shape(test_ids))
create_csv_submission(test_ids, labels_y, "C:\\Users\\maxen\\Documents\\Master EPFL\\MA1\\Machine learning\\project1\\LogisticRegression.csv")
