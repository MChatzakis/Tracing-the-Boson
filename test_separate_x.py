from os import remove
from helpers import*
import numpy as np
from utils import*
from implementations import*



data_path="./dataset/train.csv"

[y, x, ids] = load_csv_data(data_path, sub_sample=False)

[x0, y0, x1, y1, x2, y2, x3, y3]= separate_4(y, x)

id=np.arange(len(x[:,0]))
[x0, i0, x1, i1, x2, i2, x3, i3]=separate_4(id, x)

#[x0r, n90]=remove_false_columns(x0)
#[x1r, n91]=remove_false_columns(x1)
#[x2r, n92]=remove_false_columns(x2)
#[x3r, n93]=remove_false_columns(x3)

#print(n90, np.shape(x0[:, 0]))

''' 
n90: there are false values in 11 columns (1st: 26123/99913, other full false)
n91: there are false values in 8 columns (1st: 7562/77544, other full false)
n92: there are false values in 1 column (2952/50379 false)
n93: there are false values in 1 column (1477/22164 false)
'''


d0=8
d1=11
d2=11
d3=11


tx0 = increase_degree(add_1_column(remove_false_column_and_average_first(x0)), d0)
#add_1_column(remove_false_columns(input_data))
tx1 = increase_degree(add_1_column(remove_false_column_and_average_first(x1)), d1)
tx2 = increase_degree(add_1_column(average_false_values(x2)), d2)
tx3 = increase_degree(add_1_column(average_false_values(x3)), d3)

lambda_ridge0=0.02
lambda_ridge1=0.12285714
lambda_ridge2=0.03285714
lambda_ridge3=0.03285714

[w0, loss0] = ridge_regression(y0, tx0, lambda_ridge0)
[w1, loss1] = ridge_regression(y1, tx1, lambda_ridge1)
[w2, loss2] = ridge_regression(y2, tx2, lambda_ridge2)
[w3, loss3] = ridge_regression(y3, tx3, lambda_ridge3)

#[loss0, w0] = least_squares(y0, tx0)
#[loss1, w1] = least_squares(y1, tx1)
#[loss2, w2] = least_squares(y2, tx2)
#[loss3, w3] = least_squares(y3, tx3)

#max_iters=100
#w00=np.zeros(len(tx0[0, :]))
#w01=np.zeros(len(tx1[0, :]))
#w02=np.zeros(len(tx2[0, :]))
#w03=np.zeros(len(tx3[0, :]))
#print(len(w00), len(w01), len(w02), len(w03))
lambda_log=1e-1
gamma_log0=8e-5
gamma_log1=8e-5
gamma_log2=8e-5
gamma_log3=8e-4
'''
[loss0, w0] = reg_logistic_regression(y0, tx0, lambda_log, w00, max_iters, gamma_log0)
print("0done")
[loss1, w1] = reg_logistic_regression(y1, tx1, lambda_log, w01, max_iters, gamma_log1)
print("1done")
[loss2, w2] = reg_logistic_regression(y2, tx2, lambda_log, w02, max_iters, gamma_log2)
print("2done")
[loss2, w3] = reg_logistic_regression(y3, tx3, lambda_log, w03, max_iters, gamma_log3)
print("3done")
'''


ycal0=np.sign(tx0 @ w0-0.5)
n0=len(y0)
r0=(n0-np.count_nonzero(ycal0-(2*y0-1)))/n0
print((n0-np.count_nonzero(ycal0-(2*y0-1)))/n0)
ycal1=np.sign(tx1 @ w1-0.5)
n1=len(y1)
r1=(n1-np.count_nonzero(ycal1-(2*y1-1)))/n1
print((n1-np.count_nonzero(ycal1-(2*y1-1)))/n1)
ycal2=np.sign(tx2 @ w2-0.5)
n2=len(y2)
r2=(n2-np.count_nonzero(ycal2-(2*y2-1)))/n2
print((n2-np.count_nonzero(ycal2-(2*y2-1)))/n2)
ycal3=np.sign(tx3 @ w3-0.5)
n3=len(y3)
r3=(n3-np.count_nonzero(ycal3-(2*y3-1)))/n3
print((n3-np.count_nonzero(ycal3-(2*y3-1)))/n3)

y_fin=unit_4(ycal0, i0, ycal1, i1, ycal2, i2, ycal3, i3)
nfin=len(y_fin)
print((nfin-np.count_nonzero(y_fin-(2*y-1)))/nfin)


data_path_test="./dataset/test.csv"


_, test_data, test_ids  = load_csv_data(data_path_test, False)

idt=np.arange(len(test_data[:, 0]))

[x0t, id0, x1t, id1, x2t, id2, x3t, id3]= separate_4(idt, test_data)




tx0t = increase_degree(add_1_column(remove_false_column_and_average_first(x0t)), d0)
tx1t = increase_degree(add_1_column(remove_false_column_and_average_first(x1t)), d1)
tx2t = increase_degree(add_1_column(average_false_values(x2t)), d2)
tx3t = increase_degree(add_1_column(average_false_values(x3t)), d3)

ly0 = np.sign(tx0t @ w0 - 0.5)
ly1 = np.sign(tx1t @ w1 - 0.5)
ly2 = np.sign(tx2t @ w2 - 0.5)
ly3 = np.sign(tx3t @ w3 - 0.5)

labels_y=unit_4(ly0, id0, ly1, id1, ly2, id2, ly3, id3)

print((len(id0)+len(id1)+len(id2)+len(id3)))
print(np.shape(labels_y), np.shape(test_ids))
create_csv_submission(test_ids, labels_y, "./test_results/")


