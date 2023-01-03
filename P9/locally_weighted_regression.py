import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../Program 9/tips.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)

def getWeights(X, point, tau):
    w = np.mat(np.eye(X.shape[0]))
    for i, xi in enumerate(X):
        n = np.dot((xi-point),(xi-point).T)
        d = -2*tau*tau
        w[i, i] = np.exp(n/d)
    return w

def predict(X, y, point, tau):
    m = X.shape[0]
    ones = np.ones(m)
    X_ = np.append(X, ones.reshape(m, 1), axis=1)
    point_ = np.array([point, 1])
    w = getWeights(X_, point_, tau)
    theta = np.linalg.pinv(X_.T*(w*X_))*(X_.T*(w*y))
    return point_*theta

def getPred(X, y, tau):
    preds = []
    for item in X:
        preds.append(predict(X, y, item, tau))
    return preds

yPreds = getPred(X, y, 0.8)
m = X.shape[0]
ones = np.ones(m)
X_ = np.append(X, ones.reshape(m, 1), axis=1)
xsort = X_.copy()
xsort.sort(axis=0)
yPreds = np.array(yPreds).reshape(m, 1)
plt.scatter(X, y, color="blue")
plt.plot(xsort[:, 0], yPreds[X_[:, 0].argsort(0)], color="yellow")
plt.show()
