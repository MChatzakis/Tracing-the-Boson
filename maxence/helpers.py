"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def remove_false_lines(y, x):
    ''' this function take y and x and remove the data with the untaked measurement (-999.0)
    Input: y: shape (n, )
           x: shape (n, d)

    Return: ry: shape (nr, ) 
            rx: shape (nr, d) The data without lines containing -999
            n999: shape (1, d) The number of -999 per column
    '''

    [n, d] = np.shape(x)
    n999=np.zeros(d)
    index=[-1]
    for i in range(0, n):
        if not(-999 in x[i, :]):
            index=np.vstack([index, i])
        else:
            for j in range(0, d):
                if (x[i, j]==-999):
                    n999[j]=n999[j]+1

    index=np.delete(index, 0)

    rx=x[index, :]
    ry=y[index]
    return ry, rx, n999

def remove_false_columns(x):
    ''' this function take x and remove the data with the untaked measurement (-999.0)
    Input: x: shape (n, d)

    Return: rx: shape (nr, d) The data without columns containing -999
    '''
    [n, d] = np.shape(x)
    n999=np.zeros(d)
    index=[-1]
    for i in range(0, d):
        if not(-999 in x[:, i]):
            index=np.vstack([index, i])

    index=np.delete(index, 0)

    rx=x[:, index]
    return rx

def add_1_column(x):
    ''' Take a x vector and add a 1 column at the beginning for the w0 element
    '''
    [n, d] = np.shape(x)
    v=np.ones([n, 1])
    return np.hstack((v, x))