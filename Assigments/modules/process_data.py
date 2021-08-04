import numpy as np


def read_data(exercise, file_name, xcol, ycol):
    file_path = str(exercise) + "/" + str(file_name) + ".txt"
    data = np.loadtxt(file_path, delimiter=",")
    xend = xcol
    yend = xcol + ycol
    X = np.asarray(data[:, 0:xend])
    y = np.asarray(data[:, xend:yend])
    return X, y


def compute_cost(X, y, theta):
    m = len(y)
    ThetaX = np.matmul(X, theta)
    diff = np.subtract(ThetaX, y)

    ssq = np.matmul(np.transpose(diff), diff)
    cost = (1 / (2 * m)) * ssq

    return float(cost)


def gradient_descent(X, y, theta, alpha):
    m = len(y)
    ThetaX = np.matmul(X, theta)
    diff = np.subtract(ThetaX, y)
    new_term = (alpha / m) * np.matmul(np.transpose(X), ThetaX)
    print(new_term)
    theta = np.subtract(theta, new_term)
    print(theta)
    return theta
