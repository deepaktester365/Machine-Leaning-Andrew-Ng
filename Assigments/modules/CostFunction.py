import torch
import numpy as np


def squared_error_cost_function(X, y, theta, debug=False):
    # X is the "design matrix"
    # y is the class labels

    m = X.size(dim=0)
    predictions = X.mm(theta)

    if debug:
        print("X data:", X)
        print("Theta values:", theta)
        print("Predictions:", predictions)

    sqrErrors = torch.square(predictions - y)

    cost = (1 / (2 * m)) * torch.sum(sqrErrors)

    return cost
