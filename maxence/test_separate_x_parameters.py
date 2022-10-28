from os import remove
from helpers import*
import numpy as np
from utils import*
from implementations import*



data_path="C:\\Users\\maxen\\Documents\\Master EPFL\\MA1\\Machine learning\\project1\\data\\train.csv"

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
deg=np.arange(7, 12)
lambda_=np.linspace(2e-2, 2e-1, 15)
best_degree=np.zeros(4)
best_lambda=np.zeros(4)
best_results=np.zeros(4)

for d in deg:
    print(d)
    for l in lambda_:


        tx0 = increase_degree(add_1_column(remove_false_column_and_average_first(x0)), d)
        tx1 = increase_degree(add_1_column(remove_false_column_and_average_first(x1)), d)
        tx2 = increase_degree(add_1_column(average_false_values(x2)), d)
        tx3 = increase_degree(add_1_column(average_false_values(x3)), d)

        lambda_ridge0=5e-2
        lambda_ridge1=3e-2
        lambda_ridge2=8e-2
        lambda_ridge3=1.5e-1

        [w0, loss0] = ridge_regression(y0, tx0, l)
        [w1, loss1] = ridge_regression(y1, tx1, l)
        [w2, loss2] = ridge_regression(y2, tx2, l)
        [w3, loss3] = ridge_regression(y3, tx3, l)


        ycal0=np.sign(tx0 @ w0-0.5)
        n0=len(y0)
        r0=(n0-np.count_nonzero(ycal0-(2*y0-1)))/n0
        if r0>best_results[0]:
            best_results[0]=r0
            best_lambda[0]=l
            best_degree[0]=d
        #print((n0-np.count_nonzero(ycal0-(2*y0-1)))/n0)
        ycal1=np.sign(tx1 @ w1-0.5)
        n1=len(y1)
        r1=(n1-np.count_nonzero(ycal1-(2*y1-1)))/n1
        if r1>best_results[1]:
            best_results[1]=r1
            best_lambda[1]=l
            best_degree[1]=d
        #print((n1-np.count_nonzero(ycal1-(2*y1-1)))/n1)
        ycal2=np.sign(tx2 @ w2-0.5)
        n2=len(y2)
        r2=(n2-np.count_nonzero(ycal2-(2*y2-1)))/n2
        if r2>best_results[2]:
            best_results[2]=r2
            best_lambda[2]=l
            best_degree[2]=d
        #print((n2-np.count_nonzero(ycal2-(2*y2-1)))/n2)
        ycal3=np.sign(tx3 @ w3-0.5)
        n3=len(y3)
        r3=(n3-np.count_nonzero(ycal3-(2*y3-1)))/n3
        if r3>best_results[3]:
            best_results[3]=r3
            best_lambda[3]=l
            best_degree[3]=d
        #print((n3-np.count_nonzero(ycal3-(2*y3-1)))/n3)

        y_fin=unit_4(ycal0, i0, ycal1, i1, ycal2, i2, ycal3, i3)
        nfin=len(y_fin)
        #print((nfin-np.count_nonzero(y_fin-(2*y-1)))/nfin)
print(best_results, best_lambda, best_degree)

