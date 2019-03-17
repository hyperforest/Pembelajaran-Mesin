import numpy as np
import pandas as pd

def dummy(x, unique = 3):
    ret = None
    
    if type(x) == int or type(x) == np.int64:
        ret = np.zeros((unique, ))
        ret[x] = 1
        return ret
    
    else:
        ret = np.zeros((len(x), unique))
        for i in range(len(x)):
            ret[i][x[i]] = 1
        return ret

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def prediksi(X, W, b):
    y = np.zeros((len(X), 3))
    for i in range(len(X)):
        y[i] = feedforward(X[i], W, b)  
    return y

def hitung_akurasi(y_asli, y_pred):
    return (y_asli == y_pred).sum() / len(y_asli)

def feedforward(x, W, b, activations = [sigmoid, softmax]):
    y = x.copy()
    for i in range(len(W)):
        y = np.dot(W[i].T, y) + b[i]
        y = activations[i](y)
    return y

def cartesian_prod(a, b):
    result = np.zeros((a.shape[0], b.shape[0]))
    for i in range(len(result)):
        result[i] = a[i] * b
    return result

def split(df, col_target, val_size = 0.2):
    unique = df[col_target].unique()
    dff = [df[df[col_target] == i] for i in unique]
    
    train = pd.concat([dff[i].iloc[int(len(dff[i]) * val_size):, :]\
                       for i in range(len(unique))])
    val = pd.concat([dff[i].iloc[:int(len(dff[i]) * val_size), :]\
                     for i in range(len(unique))])
    
    X_train = train.drop(col_target, axis = 1).values
    y_train = train.loc[:, col_target].values
    X_val = val.drop(col_target, axis = 1).values
    y_val = val.loc[:, col_target].values
    
    return (X_train, y_train, X_val, y_val)

def get_gradient(x, h, p, t, W):
    db2 = (p - t) * p * (1 - p)
    db1 = h * (1 - h) * np.sum(db2 * W[1], axis = 1)
    dW2 = cartesian_prod(h, db2)
    dW1 = cartesian_prod(x, db1)
    
    return (dW1, db1, dW2, db2)

def update(dW1, db1, dW2, db2, lr, W, b):
    temp1 = W[0] - lr * dW1
    temp2 = b[0] - lr * db1
    temp3 = W[1] - lr * dW2
    temp4 = b[1] - lr * db2
    
    W[0], b[0], W[1], b[1] = temp1, temp2, temp3, temp4

def train_error(p, t):
    return np.sum(0.5 * ((p - t) ** 2))

def val_error(p, t):
    return (0.5 * ((p - t) ** 2)).sum(axis = 1).mean()
