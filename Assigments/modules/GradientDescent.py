import torch


def gradient_descent(X, y, theta, alpha, debug=False):
    # X is the "design matrix"
    # y is the class labels

    m = X.size(dim=1)
    predictions = X.mm(theta.t())

    if debug:
        print("X data:", X)
        print("Theta values:", theta)
        print("Predictions:", predictions)

    sumError = torch.sum(predictions - y)

    delta = (1 / m) * sumError * X

    new_theta = theta - alpha * delta

    return new_theta
