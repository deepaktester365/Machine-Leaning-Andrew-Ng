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
    new_term = (alpha / m) * np.matmul(np.transpose(X), diff)
    theta = np.subtract(theta, new_term)
    return theta


def feature_normalize(X):
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0, dtype=np.float64)
    X_norm = (X - mean_x) / std_x
    return X_norm, mean_x, std_x


def normal_eqn(X, y):
    # theta = pinv(X'*X)*(X'*y);
    Xty = np.matmul(np.transpose(X), y)
    Xtx = np.matmul(np.transpose(X), X)
    Xtxinv = np.linalg.inv(Xtx)
    theta = np.matmul(Xtxinv, Xty)
    return theta
